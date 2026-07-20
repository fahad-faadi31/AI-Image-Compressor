"""
compress.py

Compress a single image using the trained neural codec.

Pipeline

Image
   ↓
Encoder
   ↓
Quantizer
   ↓
zlib
   ↓
compressed.bin
"""

import os

import torch
from PIL import Image
from torchvision import transforms

from src.model import CompressiveAutoencoder
from src.entropy import EntropyCoder


# -----------------------------
# CONFIG
# -----------------------------
IMAGE_PATH = "sample.jpeg"          # change later if needed
CHECKPOINT = "checkpoints/best.pt"
OUTPUT_FILE = "compressed.bin"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# -----------------------------
# Load model
# -----------------------------
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


# -----------------------------
# Image transform
# -----------------------------
transform = transforms.Compose([
    transforms.ToTensor(),
])


# -----------------------------
# Load image
# -----------------------------
img = Image.open(IMAGE_PATH).convert("RGB")

width, height = img.size

print(f"Original Size : {width} x {height}")

img = img.resize((128, 128))

tensor = transform(img).unsqueeze(0).to(DEVICE)


# -----------------------------
# Encode
# -----------------------------
with torch.no_grad():

    latent = model.encode(tensor)

print("Encoding completed.")

print("Latent Shape :", tuple(latent.shape))


# -----------------------------
# Compress
# -----------------------------
coder = EntropyCoder()

compressed = coder.compress(latent)

coder.save(compressed, OUTPUT_FILE)

print("Compressed file saved.")


# -----------------------------
# Statistics
# -----------------------------
original_bytes = tensor.numel()

compressed_bytes = len(compressed)

ratio = original_bytes / compressed_bytes

print("\n========== RESULTS ==========")

print(f"Original Bytes   : {original_bytes:,}")

print(f"Compressed Bytes : {compressed_bytes:,}")

print(f"Compression Ratio: {ratio:.2f}:1")

print("=============================")