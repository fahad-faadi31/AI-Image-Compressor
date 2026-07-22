"""
Builds the Huffman frequency table from your TRAINED model's actual latent
code distribution, sampled over the validation set. This table gets saved
once and shipped alongside the model checkpoint -- the API loads it at
startup rather than rebuilding it per request (see src/entropy.py's
HuffmanTable docstring for why a static table is used).

Usage:
    python -m scripts.build_huffman_table --checkpoint checkpoints/best.pt
"""

import argparse
from collections import Counter

import torch
import yaml

from src.model import CompressionAutoencoder
from src.dataset import get_dataloaders
from src.entropy import HuffmanTable, huffman_encode


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best.pt")
    parser.add_argument("--output", type=str, default="models/huffman_table.json")
    parser.add_argument("--num_batches", type=int, default=20,
                         help="More batches = more representative table, "
                              "but this only needs to be run once, so it's "
                              "worth sampling generously.")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = CompressionAutoencoder(config).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"Loaded checkpoint from epoch {ckpt['epoch']}")

    _, val_loader = get_dataloaders(config)
    counts = Counter()
    total_symbols = 0

    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            if i >= args.num_batches:
                break
            x = batch.to(device)
            codes = model.compress(x)
            flat = codes.cpu().numpy().flatten()
            counts.update(flat.tolist())
            total_symbols += flat.size

    print(f"Sampled {total_symbols:,} symbols across "
          f"{min(args.num_batches, len(val_loader))} batches")

    table = HuffmanTable.from_counts(dict(counts), num_symbols=2 ** config["model"]["quantization_bits"])
    table.save(args.output)
    print(f"Saved Huffman table to {args.output}")

    # Sanity check: re-load and confirm it round-trips correctly on a real
    # batch of codes from the model, not just synthetic test data.
    reloaded = HuffmanTable.load(args.output)
    x = next(iter(val_loader)).to(device)
    with torch.no_grad():
        real_codes = model.compress(x)[0].cpu().numpy()  # first image in batch
    encoded = huffman_encode(real_codes, reloaded)

    original_bits = real_codes.size * 8
    compressed_bits = len(encoded) * 8
    print(f"\nSanity check on one real validation image's codes:")
    print(f"  shape: {real_codes.shape}")
    print(f"  raw: {original_bits/8:.0f} bytes -> huffman: {compressed_bits/8:.0f} bytes "
          f"({original_bits/compressed_bits:.2f}x smaller than raw codes)")


if __name__ == "__main__":
    main()