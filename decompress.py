"""
decompress.py

Decompress a compressed latent file produced by compress.py
and reconstruct the image using the trained decoder.

Pipeline:

compressed.bin
      ↓
zlib decompress
      ↓
Quantized Latent
      ↓
Decoder
      ↓
reconstructed.png
"""

import torch
from torchvision.utils import save_image

from src.model import CompressiveAutoencoder
from src.entropy import EntropyCoder


# =====================================================
# CONFIG
# =====================================================

CHECKPOINT = "checkpoints/best.pt"
COMPRESSED_FILE = "compressed.bin"
OUTPUT_IMAGE = "reconstructed.png"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Shape of latent produced by encoder
LATENT_SHAPE = (1, 64, 8, 8)


# =====================================================
# LOAD MODEL
# =====================================================

print(f"Using device: {DEVICE}")

model = CompressiveAutoencoder().to(DEVICE)

checkpoint = torch.load(
    CHECKPOINT,
    map_location=DEVICE,
    weights_only=False,
)

model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

print("Model loaded successfully.")


# =====================================================
# LOAD COMPRESSED FILE
# =====================================================

coder = EntropyCoder()

compressed_bytes = coder.load(COMPRESSED_FILE)

print("Compressed file loaded.")


# =====================================================
# DECOMPRESS LATENT
# =====================================================

latent = coder.decompress(
    compressed_bytes,
    LATENT_SHAPE,
)

latent = latent.to(DEVICE)

print("Latent recovered.")
print("Latent Shape :", tuple(latent.shape))


# =====================================================
# RECONSTRUCT IMAGE
# =====================================================

with torch.no_grad():

    reconstructed = model.decode(latent)

print("Image reconstructed.")


# =====================================================
# SAVE IMAGE
# =====================================================

save_image(reconstructed.cpu(), OUTPUT_IMAGE)

print(f"Reconstructed image saved as '{OUTPUT_IMAGE}'")


# =====================================================
# INFO
# =====================================================

print("\n========== DONE ==========")
print("Decompression completed successfully.")
print("==========================")