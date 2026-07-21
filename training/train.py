"""
Main training entrypoint.

Usage:
    python -m training.train --config configs/config.yaml
    python -m training.train --config configs/config.yaml --resume checkpoints/latest.pt

Pipeline:
    1. Load config
    2. Build DIV2K train/val DataLoaders
    3. Build CompressionAutoencoder + CompressionLoss
    4. Build AdamW optimizer + cosine LR scheduler
    5. Loop epochs:
        - train_one_epoch(): forward, combined loss, backward, step
        - evaluate() on val set (training/validate.py)
        - log to TensorBoard (logs/) + console
        - save checkpoints/latest.pt every epoch, checkpoints/best.pt on
          new best val loss, checkpoints/epoch_N.pt every save_every_n_epochs
"""

import argparse
from pathlib import Path

import torch
import yaml
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.tensorboard import SummaryWriter
from torch.amp import autocast, GradScaler
from tqdm import tqdm

from src.model import CompressionAutoencoder
from src.losses import CompressionLoss
from src.dataset import get_dataloaders
from training.validate import evaluate


def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def get_device(preferred: str) -> str:
    if preferred == "cuda" and not torch.cuda.is_available():
        print("CUDA requested but not available — falling back to CPU. "
              "Training will be significantly slower.")
        return "cpu"
    return preferred


def train_one_epoch(model, loader, optimizer, loss_fn, device, epoch, writer,
                     scaler, grad_clip_norm):
    model.train()
    running_totals = {}
    pbar = tqdm(loader, desc=f"Epoch {epoch} [train]")
    use_amp = device == "cuda"

    for step, batch in enumerate(pbar):
        x = batch.to(device, non_blocking=True)

        optimizer.zero_grad()

        # Mixed precision: most ops run in float16 on GPU, roughly halving
        # activation memory and speeding up training on T4/A100-class GPUs.
        # This is what actually fixes the CUDA OOM at our current batch
        # size/resolution — no-ops safely on CPU (use_amp=False there).
        with autocast(device_type="cuda", enabled=use_amp):
            out = model(x)
            losses = loss_fn(out, x)

        scaler.scale(losses["total"]).backward()

        # Gradient clipping: must unscale first when using AMP, since
        # gradients are scaled up internally by GradScaler and clipping
        # against the wrong magnitude would clip too aggressively (or not
        # at all). This guards against the occasional large gradient spike
        # that can otherwise destabilize training, particularly early on
        # and particularly under mixed precision.
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_norm)

        scaler.step(optimizer)
        scaler.update()

        for k, v in losses.items():
            running_totals[k] = running_totals.get(k, 0.0) + v.item()

        pbar.set_postfix(loss=f"{losses['total'].item():.4f}")

        global_step = epoch * len(loader) + step
        if step % 50 == 0:
            for k, v in losses.items():
                writer.add_scalar(f"train/{k}", v.item(), global_step)

    return {k: v / len(loader) for k, v in running_totals.items()}


def save_checkpoint(path: Path, model, optimizer, scheduler, epoch, best_val_psnr):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "best_val_psnr": best_val_psnr,
    }, path)


def load_checkpoint(path: Path, model, optimizer, scheduler):
    ckpt = torch.load(path, map_location="cpu")
    model.load_state_dict(ckpt["model_state_dict"])
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    scheduler.load_state_dict(ckpt["scheduler_state_dict"])
    return ckpt["epoch"] + 1, ckpt["best_val_psnr"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument("--resume", type=str, default=None,
                         help="Path to a checkpoint to resume from")
    args = parser.parse_args()

    config = load_config(args.config)
    train_cfg = config["training"]

    device = get_device(train_cfg["device"])
    print(f"Using device: {device}")

    train_loader, val_loader = get_dataloaders(config)
    print(f"Train batches/epoch: {len(train_loader)}  |  Val batches: {len(val_loader)}")

    model = CompressionAutoencoder(config).to(device)
    loss_fn = CompressionLoss(train_cfg["loss_weights"]).to(device)

    optimizer = AdamW(
        model.parameters(),
        lr=train_cfg["learning_rate"],
        weight_decay=train_cfg["weight_decay"],
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=train_cfg["num_epochs"])

    start_epoch = 0
    best_val_psnr = float("-inf")
    if args.resume:
        start_epoch, best_val_psnr = load_checkpoint(
            Path(args.resume), model, optimizer, scheduler
        )
        print(f"Resumed from {args.resume} at epoch {start_epoch}")

    checkpoint_dir = Path(train_cfg["checkpoint_dir"])
    writer = SummaryWriter(log_dir=train_cfg["log_dir"])
    scaler = GradScaler(device="cuda", enabled=(device == "cuda"))

    for epoch in range(start_epoch, train_cfg["num_epochs"]):
        train_metrics = train_one_epoch(
            model, train_loader, optimizer, loss_fn, device, epoch, writer,
            scaler, train_cfg.get("grad_clip_norm", 1.0)
        )
        scheduler.step()

        val_metrics = evaluate(model, val_loader, device,
                                config["model"]["quantization_bits"])

        print(f"Epoch {epoch}: "
              f"train_loss={train_metrics['total']:.4f}  "
              f"val_psnr={val_metrics['psnr']:.2f}dB  "
              f"val_ssim={val_metrics['ssim']:.4f}  "
              f"val_ratio={val_metrics['avg_compression_ratio']:.2f}x")

        for k, v in val_metrics.items():
            writer.add_scalar(f"val/{k}", v, epoch)
        writer.add_scalar("train/lr", scheduler.get_last_lr()[0], epoch)

        save_checkpoint(checkpoint_dir / "latest.pt", model, optimizer,
                         scheduler, epoch, best_val_psnr)

        # We track "best" by val PSNR (higher is better) rather than a raw
        # loss value, since PSNR is the metric you'll actually report/care
        # about downstream — a lower combined loss doesn't always mean a
        # visually better reconstruction once perceptual/rate terms mix in.
        if val_metrics["psnr"] > best_val_psnr:
            best_val_psnr = val_metrics["psnr"]
            save_checkpoint(checkpoint_dir / "best.pt", model, optimizer,
                             scheduler, epoch, best_val_psnr)
            print(f"  New best PSNR: {best_val_psnr:.2f}dB -> saved best.pt")

        if (epoch + 1) % train_cfg["save_every_n_epochs"] == 0:
            save_checkpoint(checkpoint_dir / f"epoch_{epoch+1}.pt", model,
                             optimizer, scheduler, epoch, best_val_psnr)

    writer.close()
    print("Training complete.")


if __name__ == "__main__":
    main()