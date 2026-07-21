"""
Validation / evaluation utilities.

Called from train.py at the end of each epoch, and standalone for
final model evaluation against a checkpoint.
"""

import torch
from pytorch_msssim import ssim as ssim_fn, ms_ssim as ms_ssim_fn


def _psnr(recon: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Peak Signal-to-Noise Ratio, in dB. Higher = better. Images in [0,1],
    so max_val = 1.0. Computed per-batch-average MSE for stability when a
    batch happens to contain a near-perfect (MSE~0) reconstruction."""
    mse = torch.mean((recon - target) ** 2)
    if mse == 0:
        return torch.tensor(100.0)  # cap rather than divide by zero -> inf
    return 10 * torch.log10(1.0 / mse)


@torch.no_grad()
def evaluate(model, dataloader, device: str, quantization_bits: int = 8) -> dict:
    """
    Runs the model over the full dataloader in eval mode and returns
    averaged metrics.

    Returns:
        dict with keys: "psnr", "ssim", "ms_ssim", "avg_compression_ratio"
    """
    model.eval()

    total_psnr = 0.0
    total_ssim = 0.0
    total_ms_ssim = 0.0
    total_ratio = 0.0
    num_batches = 0

    for batch in dataloader:
        x = batch.to(device)
        out = model(x)
        recon = out["reconstruction"].clamp(0, 1)  # guard against tiny
                                                     # float overshoot past
                                                     # [0,1] before metrics

        total_psnr += _psnr(recon, x).item()
        total_ssim += ssim_fn(recon, x, data_range=1.0, size_average=True).item()
        total_ms_ssim += ms_ssim_fn(recon, x, data_range=1.0, size_average=True).item()

        # Compression ratio at this batch's resolution: original uint8 RGB
        # bits vs quantized latent bits (pre-entropy-coding, matches the
        # same calculation used in src/model.py's __main__ check)
        b, c, h, w = x.shape
        latent = out["latent"]
        original_bits = c * h * w * 8
        compressed_bits = latent.numel() // b * quantization_bits
        total_ratio += original_bits / compressed_bits

        num_batches += 1

    model.train()  # restore training mode for the caller

    return {
        "psnr": total_psnr / num_batches,
        "ssim": total_ssim / num_batches,
        "ms_ssim": total_ms_ssim / num_batches,
        "avg_compression_ratio": total_ratio / num_batches,
    }


if __name__ == "__main__":
    # Quick check — run as a module from the project root:
    #   python -m training.validate
    import yaml
    from src.model import CompressionAutoencoder
    from src.dataset import get_dataloaders

    with open("configs/config.yaml") as f:
        config = yaml.safe_load(f)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = CompressionAutoencoder(config).to(device)

    _, val_loader = get_dataloaders(config)
    metrics = evaluate(model, val_loader, device,
                        config["model"]["quantization_bits"])

    print("Validation metrics (untrained model, sanity check only):")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")