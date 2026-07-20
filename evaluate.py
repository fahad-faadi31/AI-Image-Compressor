"""
Full evaluation script for the AI Image Compression project.

This script:
1. Loads the trained model.
2. Evaluates the complete validation dataset.
3. Computes:
    - MSE
    - PSNR
    - SSIM
4. Saves a report inside results/
"""

import json
import os

import torch
import torch.nn.functional as F

from src.dataloader import get_val_loader
from src.metrics import compute_psnr, compute_ssim
from src.model import CompressiveAutoencoder


def main():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Using device: {device}")

    os.makedirs("results", exist_ok=True)

    # -------------------------
    # Load model
    # -------------------------

    model = CompressiveAutoencoder().to(device)

    checkpoint = torch.load(
        "checkpoints/best.pt",
        map_location=device,
        weights_only=False,
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    print("Model loaded successfully.\n")

    # -------------------------
    # Validation loader
    # -------------------------

    val_loader = get_val_loader()

    total_mse = 0.0
    total_psnr = 0.0
    total_ssim = 0.0

    num_batches = 0

    print("Running evaluation...\n")

    with torch.no_grad():

        for images in val_loader:

            images = images.to(device)

            reconstruction, quantized = model(images)

            mse = F.mse_loss(reconstruction, images)

            psnr = compute_psnr(reconstruction, images)

            ssim = compute_ssim(reconstruction, images)

            total_mse += mse.item()
            total_psnr += psnr.item()
            total_ssim += ssim.item()

            num_batches += 1

            print(
                f"Batch {num_batches:02d}"
                f" | MSE {mse.item():.6f}"
                f" | PSNR {psnr.item():.2f} dB"
                f" | SSIM {ssim.item():.4f}"
            )

    avg_mse = total_mse / num_batches
    avg_psnr = total_psnr / num_batches
    avg_ssim = total_ssim / num_batches

    print("\n==============================")
    print("Evaluation Complete")
    print("==============================")
    print(f"Average MSE  : {avg_mse:.6f}")
    print(f"Average PSNR : {avg_psnr:.2f} dB")
    print(f"Average SSIM : {avg_ssim:.4f}")

    # -------------------------
    # Save txt report
    # -------------------------

    with open("results/report.txt", "w") as f:

        f.write("AI Image Compression Evaluation\n")
        f.write("=" * 40 + "\n\n")

        f.write(f"Average MSE  : {avg_mse:.6f}\n")
        f.write(f"Average PSNR : {avg_psnr:.2f} dB\n")
        f.write(f"Average SSIM : {avg_ssim:.4f}\n")

    # -------------------------
    # Save json report
    # -------------------------

    report = {
        "average_mse": avg_mse,
        "average_psnr": avg_psnr,
        "average_ssim": avg_ssim,
    }

    with open("results/report.json", "w") as f:
        json.dump(report, f, indent=4)

    print("\nReports saved successfully.")
    print("results/report.txt")
    print("results/report.json")


if __name__ == "__main__":
    main()