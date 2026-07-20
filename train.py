"""
Training loop for the CNN compressive autoencoder.

Usage:
    python train.py --epochs 100 --batch_size 32

Run `python train.py --help` for all options.

What this script does per epoch:
    1. Train: forward pass through CompressiveAutoencoder (encoder -> STE quantize
       -> decoder), MSE loss, backward, optimizer step. Progress bar via tqdm.
    2. Validate: same forward pass but no gradient tracking, computes MSE loss,
       PSNR, and SSIM on the validation set.
    3. Log: train loss, val loss, val PSNR, val SSIM, and current learning rate to
       TensorBoard, plus a one-line terminal summary every epoch.
    4. Visualize: every --vis_every epochs, saves a side-by-side Original |
       Reconstructed image grid (same fixed validation batch each time, so you
       can visually track the same images improving over training).
    5. Checkpoint: saves a "best" checkpoint whenever val loss improves, and a
       periodic checkpoint every --checkpoint_every epochs regardless.
    6. LR scheduling: ReduceLROnPlateau halves the learning rate if val loss
       plateaus, helping the model escape a stalled loss without manual tuning.
    7. Early stopping: if val loss hasn't improved for --patience epochs, training
       stops early to avoid wasting compute.
    8. Resume: if --resume is given (or --auto_resume finds an existing
       checkpoint), training continues from the saved epoch/optimizer/scheduler
       state instead of starting over.
    9. OOM auto-fallback: if the requested --batch_size causes a CUDA
       out-of-memory error, training automatically retries with
       --fallback_batch_size, then a further reduced size, before giving up.

View TensorBoard with:
    tensorboard --logdir runs
"""

import argparse
import os
import time

import torch
import torch.nn as nn
from src.losses import HybridLoss
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from src.dataloader import get_train_loader, get_val_loader
from src.model import CompressiveAutoencoder
from src.metrics import compute_psnr, compute_ssim
from src.visualization import save_reconstruction_grid


def parse_args():
    parser = argparse.ArgumentParser(description="Train the CNN compressive autoencoder.")
    parser.add_argument("--train_dir", type=str, default="data/DIV2K_train_HR")
    parser.add_argument("--val_dir", type=str, default="data/DIV2K_valid_HR")
    parser.add_argument("--crop_size", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--fallback_batch_size", type=int, default=16,
                         help="Retry with this batch size if --batch_size causes a CUDA OOM error.")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--lr_factor", type=float, default=0.5,
                         help="Factor by which LR is reduced on plateau (ReduceLROnPlateau).")
    parser.add_argument("--lr_patience", type=int, default=5,
                         help="Epochs with no val_loss improvement before LR is reduced.")
    parser.add_argument("--quant_levels", type=int, default=127)
    parser.add_argument("--patience", type=int, default=20,
                         help="Stop early if val loss hasn't improved for this many epochs.")
    parser.add_argument("--checkpoint_every", type=int, default=10,
                         help="Save a periodic checkpoint every N epochs.")
    parser.add_argument("--vis_every", type=int, default=5,
                         help="Save an Original|Reconstructed comparison grid every N epochs.")
    parser.add_argument("--vis_examples", type=int, default=4,
                         help="Number of image pairs shown in each visualization grid.")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    parser.add_argument("--log_dir", type=str, default="runs")
    parser.add_argument("--vis_dir", type=str, default="visualizations")
    parser.add_argument("--run_name", type=str, default=None,
                         help="Name for this run's TensorBoard/checkpoint/visualization subfolder. "
                              "Defaults to a timestamp if not given. Use a FIXED name (not the "
                              "default timestamp) if you want --auto_resume to find this run's "
                              "checkpoints across separate script invocations (e.g. after a Colab "
                              "session disconnect).")
    parser.add_argument("--resume", type=str, default=None,
                         help="Path to a specific checkpoint .pt file to resume from.")
    parser.add_argument("--auto_resume", action="store_true",
                         help="If set and --resume is not given, automatically resume from the "
                              "latest checkpoint found in checkpoint_dir/run_name, if any exists.")
    return parser.parse_args()


def find_latest_checkpoint(checkpoint_dir):
    """Scans checkpoint_dir for .pt files and returns the path with the highest
    recorded epoch number (reading the 'epoch' field inside each file, not
    relying on filenames). Returns None if no valid checkpoints are found."""
    if not os.path.isdir(checkpoint_dir):
        return None

    best_path, best_epoch = None, -1
    for fname in os.listdir(checkpoint_dir):
        if not fname.endswith(".pt"):
            continue
        path = os.path.join(checkpoint_dir, fname)
        try:
            ckpt = torch.load(path, map_location="cpu", weights_only=False)
            epoch = ckpt.get("epoch", -1)
        except Exception:
            continue
        if epoch > best_epoch:
            best_epoch, best_path = epoch, path

    return best_path


def train_one_epoch(
    model,
    loader,
    optimizer,
    criterion,
    scaler,
    device,
):
    model.train()

    total_loss = 0.0
    n_batches = 0

    pbar = tqdm(loader, desc="Train", leave=False)

    for batch in pbar:

        batch = batch.to(device)

        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast(
            "cuda",
            enabled=(device.type == "cuda"),
        ):

            reconstruction, _ = model(batch)

            loss, mse_loss, ssim = criterion(
                reconstruction,
                batch,
            )

        scaler.scale(loss).backward()

        scaler.unscale_(optimizer)

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            1.0,
        )

        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        n_batches += 1

        pbar.set_postfix(
            loss=f"{loss.item():.4f}",
            mse=f"{mse_loss.item():.4f}",
            ssim=f"{ssim.item():.4f}",
        )

    return total_loss / n_batches


@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    total_psnr = 0.0
    total_ssim = 0.0
    n_batches = 0
    pbar = tqdm(loader, desc="Val", leave=False)
    for batch in pbar:
        batch = batch.to(device)
        with torch.amp.autocast(
            "cuda",
            enabled=(device.type == "cuda"),
        ):

            reconstruction, _ = model(batch)

            loss, mse_loss, _ = criterion(
                reconstruction,
                batch,
            )

            psnr = compute_psnr(
                reconstruction,
                batch,
            )

            ssim = compute_ssim(
                reconstruction,
                batch,
            )

        total_loss += loss.item()
        total_psnr += psnr.item()
        total_ssim += ssim.item()
        n_batches += 1

        pbar.set_postfix(
            loss=f"{loss.item():.4f}",
            mse=f"{mse_loss.item():.4f}",
            psnr=f"{psnr.item():.2f} dB",
            ssim=f"{ssim.item():.4f}",
        )

    return (
        total_loss / n_batches,
        total_psnr / n_batches,
        total_ssim / n_batches,
    )


def save_checkpoint(path, model, optimizer, scheduler, epoch, val_loss, val_psnr, val_ssim, epochs_without_improvement):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "val_loss": val_loss,
        "val_psnr": val_psnr,
        "val_ssim": val_ssim,
        "epochs_without_improvement": epochs_without_improvement,
    }, path)


def run_training(args, batch_size):
    """Runs the full training loop at a given batch_size. Raises
    torch.cuda.OutOfMemoryError if the GPU runs out of memory, so the caller
    (main()) can catch it and retry with a smaller batch size."""

    run_name = args.run_name or time.strftime("%Y%m%d_%H%M%S")
    checkpoint_dir = os.path.join(args.checkpoint_dir, run_name)
    log_dir = os.path.join(args.log_dir, run_name)
    vis_dir = os.path.join(args.vis_dir, run_name)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Run name: {run_name}  |  batch_size: {batch_size}")
    print(f"Checkpoints: {checkpoint_dir}")
    print(f"TensorBoard logs: {log_dir}  (view with: tensorboard --logdir {args.log_dir})")
    print(f"Visualizations: {vis_dir}")

    train_loader = get_train_loader(
        data_dir=args.train_dir, crop_size=args.crop_size,
        batch_size=batch_size, num_workers=args.num_workers,
    )
    val_loader = get_val_loader(
        data_dir=args.val_dir, crop_size=args.crop_size,
        batch_size=batch_size, num_workers=max(1, args.num_workers // 2),
    )
    print(f"Train batches/epoch: {len(train_loader)}  |  Val batches/epoch: {len(val_loader)}")

    # Fixed batch used for visualization every --vis_every epochs, so the SAME
    # images are compared across epochs -- makes visual progress easy to track.
    vis_batch = next(iter(val_loader)).to(device)

    model = CompressiveAutoencoder(quant_levels=args.quant_levels).to(device)
    optimizer = torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=1e-4)
    criterion = HybridLoss(
    alpha=0.8,
    beta=0.2,
)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=args.epochs,
    eta_min=1e-6,
)
   # Automatic Mixed Precision (AMP)
    scaler = torch.amp.GradScaler(
    "cuda",
    enabled=(device.type == "cuda")
)

    # --- Resume logic ---
    start_epoch = 1
    best_val_loss = float("inf")
    epochs_without_improvement = 0
    resume_path = args.resume
    if resume_path is None and args.auto_resume:
        resume_path = find_latest_checkpoint(checkpoint_dir)
        if resume_path:
            print(f"Auto-resume: found existing checkpoint at {resume_path}")

    if resume_path and os.path.isfile(resume_path):
        ckpt = torch.load(resume_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if ckpt.get("scheduler_state_dict") is not None:
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        start_epoch = ckpt["epoch"] + 1
        epochs_without_improvement = ckpt.get("epochs_without_improvement", 0)

        # best_val_loss must come from best.pt specifically, NOT from resume_path.
        # resume_path is the checkpoint with the highest epoch number (correct for
        # restoring model/optimizer weights -- we want the most recent state), but
        # that is not necessarily the checkpoint with the lowest val_loss. Reading
        # best_val_loss from the wrong checkpoint could cause a worse epoch to be
        # incorrectly saved as the new "best".
        best_ckpt_path = os.path.join(checkpoint_dir, "best.pt")
        if os.path.isfile(best_ckpt_path):
            best_ckpt = torch.load(best_ckpt_path, map_location="cpu", weights_only=False)
            best_val_loss = best_ckpt.get("val_loss", float("inf"))
        else:
            best_val_loss = ckpt.get("val_loss", float("inf"))

        print(f"Resumed model/optimizer state from epoch {ckpt['epoch']}. "
              f"True best_val_loss so far: {best_val_loss:.5f}. "
              f"epochs_without_improvement: {epochs_without_improvement}. "
              f"Continuing from epoch {start_epoch}.")
    elif resume_path:
        print(f"Warning: --resume path '{resume_path}' not found. Starting from scratch.")

    writer = SummaryWriter(log_dir=log_dir)

    for epoch in range(start_epoch, args.epochs + 1):
        epoch_start = time.time()

        train_loss = train_one_epoch(model,train_loader,optimizer,criterion,scaler,device,)
        val_loss, val_psnr, val_ssim = validate(model, val_loader, criterion, device)
        scheduler.step()

        current_lr = optimizer.param_groups[0]["lr"]
        elapsed = time.time() - epoch_start

        # --- TensorBoard logging ---
        writer.add_scalar("Loss/train", train_loss, epoch)
        writer.add_scalar("Loss/val", val_loss, epoch)
        writer.add_scalar("PSNR/val", val_psnr, epoch)
        writer.add_scalar("SSIM/val", val_ssim, epoch)
        writer.add_scalar("LR", current_lr, epoch)

        print(
            f"Epoch {epoch:4d}/{args.epochs} | "
            f"train_loss={train_loss:.5f} | val_loss={val_loss:.5f} | "
            f"val_psnr={val_psnr:.2f}dB | val_ssim={val_ssim:.4f} | "
            f"lr={current_lr:.2e} | {elapsed:.1f}s"
        )

        # --- Periodic visualization ---
        if epoch % args.vis_every == 0 or epoch == start_epoch:
            model.eval()
            with torch.no_grad():
                vis_reconstruction, _ = model(vis_batch)
            save_path = save_reconstruction_grid(
                vis_batch, vis_reconstruction, epoch, vis_dir,
                writer=writer, n_examples=args.vis_examples,
            )
            print(
                f"\n{'=' * 40}\n"
                f"Epoch: {epoch}\n"
                f"Train Loss: {train_loss:.5f}\n"
                f"Validation Loss: {val_loss:.5f}\n"
                f"PSNR: {val_psnr:.2f} dB\n"
                f"SSIM: {val_ssim:.4f}\n"
                f"Saved reconstruction grid: {save_path}\n"
                f"{'=' * 40}\n"
            )

        # --- Checkpointing ---
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_without_improvement = 0
            save_checkpoint(
                os.path.join(checkpoint_dir, "best.pt"),
                model, optimizer, scheduler, epoch, val_loss, val_psnr, val_ssim,
                epochs_without_improvement,
            )
            print(f"  -> New best val_loss ({val_loss:.5f}), saved best.pt")
        else:
            epochs_without_improvement += 1

        if epoch % args.checkpoint_every == 0:
            save_checkpoint(
                os.path.join(checkpoint_dir, f"epoch_{epoch}.pt"),
                model, optimizer, scheduler, epoch, val_loss, val_psnr, val_ssim,
                epochs_without_improvement,
            )

        # --- Early stopping ---
        if epochs_without_improvement >= args.patience:
            print(
                f"\nNo improvement in val_loss for {args.patience} epochs. "
                f"Stopping early at epoch {epoch}."
            )
            break

    writer.close()
    print(f"\nTraining complete. Best val_loss: {best_val_loss:.5f}")
    print(f"Best checkpoint: {os.path.join(checkpoint_dir, 'best.pt')}")


def main():
    args = parse_args()

    # Batch sizes to try, in order, on CUDA OOM. De-duplicated, largest first.
    candidate_batch_sizes = []
    for bs in [args.batch_size, args.fallback_batch_size, max(4, args.fallback_batch_size // 2)]:
        if bs not in candidate_batch_sizes:
            candidate_batch_sizes.append(bs)

    last_error = None
    for bs in candidate_batch_sizes:
        try:
            run_training(args, bs)
            return
        except torch.cuda.OutOfMemoryError as e:
            print(f"\nCUDA out of memory at batch_size={bs}. Clearing cache and retrying "
                  f"with a smaller batch size...")
            torch.cuda.empty_cache()
            last_error = e
            continue

    print("\nTraining failed at every attempted batch size due to GPU memory limits.")
    print(f"Tried: {candidate_batch_sizes}")
    if last_error is not None:
        raise last_error


if __name__ == "__main__":
    main()
