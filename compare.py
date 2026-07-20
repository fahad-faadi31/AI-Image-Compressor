"""
compare.py

Compare the original image and the reconstructed image.

Outputs:
- MSE
- PSNR
- SSIM
- Original file size
- Compressed file size
- Compression ratio

Also saves a side-by-side comparison figure.
"""

import os

import matplotlib.pyplot as plt
import torch
from PIL import Image
from torchvision import transforms

from src.metrics import compute_psnr, compute_ssim


# ==========================================================
# CONFIG
# ==========================================================

ORIGINAL_IMAGE = "sample.jpeg"          # change if needed
RECONSTRUCTED_IMAGE = "reconstructed.png"
COMPRESSED_FILE = "compressed.bin"

SAVE_FIGURE = "comparison.png"

transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])


# ==========================================================
# LOAD IMAGES
# ==========================================================

original = Image.open(ORIGINAL_IMAGE).convert("RGB")
reconstructed = Image.open(RECONSTRUCTED_IMAGE).convert("RGB")

original_tensor = transform(original).unsqueeze(0)
reconstructed_tensor = transform(reconstructed).unsqueeze(0)


# ==========================================================
# METRICS
# ==========================================================

mse = torch.mean(
    (original_tensor - reconstructed_tensor) ** 2
).item()

psnr = compute_psnr(
    reconstructed_tensor,
    original_tensor,
).item()

ssim = compute_ssim(
    reconstructed_tensor,
    original_tensor,
).item()


# ==========================================================
# FILE SIZES
# ==========================================================

original_bytes = os.path.getsize(ORIGINAL_IMAGE)
compressed_bytes = os.path.getsize(COMPRESSED_FILE)

ratio = original_bytes / compressed_bytes


# ==========================================================
# PRINT RESULTS
# ==========================================================

print("\n========== COMPARISON ==========\n")

print(f"MSE              : {mse:.6f}")
print(f"PSNR             : {psnr:.2f} dB")
print(f"SSIM             : {ssim:.4f}")

print()

print(f"Original Size    : {original_bytes:,} bytes")
print(f"Compressed Size  : {compressed_bytes:,} bytes")
print(f"Compression Ratio: {ratio:.2f}:1")

print("\n===============================\n")


# ==========================================================
# SAVE SIDE BY SIDE
# ==========================================================

fig, ax = plt.subplots(1, 2, figsize=(10, 5))

ax[0].imshow(original)
ax[0].set_title("Original")
ax[0].axis("off")

ax[1].imshow(reconstructed)
ax[1].set_title("Reconstructed")
ax[1].axis("off")

plt.tight_layout()

plt.savefig(SAVE_FIGURE)

print(f"Comparison figure saved as '{SAVE_FIGURE}'")