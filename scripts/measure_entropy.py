"""
Diagnostic: measures how compressible your TRAINED model's latent codes
actually are, before we build the real entropy coder.

If codes were uniformly distributed across all 256 values (8 bits), the
theoretical entropy would be exactly 8.0 bits/symbol -- meaning entropy
coding could NOT help at all (nothing to exploit). If the codes are skewed
(some values much more common than others -- which is what the `rate` loss
term during training was pushing toward), the entropy will be lower than
8.0, and that gap directly tells us how much extra compression entropy
coding can realistically buy us on top of the current fixed 6x ratio.

Usage:
    python -m scripts.measure_entropy --checkpoint checkpoints/best.pt
"""

import argparse
from collections import Counter

import numpy as np
import torch
import yaml

from src.model import CompressionAutoencoder
from src.dataset import get_dataloaders


def compute_entropy(counts: Counter, total: int) -> float:
    """Shannon entropy in bits, from a symbol frequency count."""
    entropy = 0.0
    for count in counts.values():
        p = count / total
        entropy -= p * np.log2(p)
    return entropy


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best.pt")
    parser.add_argument("--num_batches", type=int, default=10,
                         help="How many validation batches to sample (more = more accurate)")
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
    bits = config["model"]["quantization_bits"]
    levels = 2 ** bits

    counts = Counter()
    total_symbols = 0

    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            if i >= args.num_batches:
                break
            x = batch.to(device)
            codes = model.compress(x)  # integer codes, shape (B, C, H, W)
            flat = codes.cpu().numpy().flatten()
            counts.update(flat.tolist())
            total_symbols += flat.size

    measured_entropy = compute_entropy(counts, total_symbols)
    uniform_entropy = bits  # what entropy WOULD be if codes were uniform

    print(f"\nSampled {total_symbols:,} latent code symbols "
          f"across {min(args.num_batches, len(val_loader))} validation batches")
    print(f"Distinct values used: {len(counts)} / {levels} possible")
    print(f"\nMeasured entropy:  {measured_entropy:.3f} bits/symbol")
    print(f"Uniform (no-gain) entropy: {uniform_entropy:.3f} bits/symbol")

    savings_pct = (1 - measured_entropy / uniform_entropy) * 100
    theoretical_extra_ratio = uniform_entropy / measured_entropy
    print(f"\nEntropy coding could theoretically reduce code size by "
          f"~{savings_pct:.1f}%")
    print(f"That would take current 6.00x raw ratio to roughly "
          f"~{6.0 * theoretical_extra_ratio:.2f}x with real entropy coding")

    # Show the most/least common values -- useful sanity check
    most_common = counts.most_common(5)
    print(f"\nTop 5 most common code values (value: count): {most_common}")


if __name__ == "__main__":
    main()