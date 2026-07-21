"""
Top-level compression model: wires Encoder -> Quantizer -> Decoder.

This is the module both training/train.py and api/inference.py import from.

Contract:
    CompressionAutoencoder(config).forward(x) -> dict with keys:
        "reconstruction":    (B, 3, H, W)     reconstructed image
        "latent":            (B, C, H/8, W/8) pre-quantization latent
        "latent_quantized":  (B, C, H/8, W/8) quantized latent fed to decoder
        "latent_codes":      (B, C, H/8, W/8) integer codes in [0, 2^bits - 1],
                              used later for actual byte-size / entropy coding
                              (not differentiable — for logging/inference only)
"""

import torch
import torch.nn as nn

from src.encoder import Encoder
from src.decoder import Decoder


class Quantizer(nn.Module):
    """Straight-through quantizer: rounds in forward, identity gradient in backward.

    Steps:
        0. GroupNorm the raw encoder output first. This is a stability fix:
           without it, if the encoder's raw output magnitude grows large
           (e.g. from a gradient spike early in training), tanh() saturates
           toward +-1 almost everywhere, and its gradient (1 - tanh(z)^2)
           collapses to near zero — which silently kills gradient flow back
           into the encoder. Once that happens the latent codes freeze and
           stop responding to the input image at all, even though the
           decoder may still keep training. GroupNorm (not BatchNorm) is
           used because it normalizes per-sample, so it behaves identically
           whether we're training with batch_size=8 or serving one image at
           a time in the API.
        1. Bound the normalized output to [-1, 1] with tanh — quantization
           needs a fixed range to define discrete levels over.
        2. Scale to [0, levels-1] and round to the nearest integer level.
        3. Straight-through trick: use the *rounded* value in the forward
           computation graph, but make its local gradient equal to the
           gradient of the *unrounded* scaled value (i.e. identity), so
           backprop isn't blocked by the zero-gradient round() operation.
        4. Rescale back to [-1, 1] so the decoder always sees the same
           input range no matter how many bits we quantize to.
    """

    def __init__(self, bits: int, latent_channels: int):
        super().__init__()
        self.bits = bits
        self.levels = 2 ** bits
        num_groups = min(8, latent_channels)
        self.pre_quant_norm = nn.GroupNorm(num_groups, latent_channels)

    def _bound(self, z: torch.Tensor) -> torch.Tensor:
        z_normed = self.pre_quant_norm(z)
        return torch.tanh(z_normed)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        z_bounded = self._bound(z)
        scaled = (z_bounded + 1) / 2 * (self.levels - 1)
        rounded = torch.round(scaled)
        # Straight-through estimator: forward value = rounded,
        # backward gradient = gradient as if we'd used `scaled` directly.
        ste = scaled + (rounded - scaled).detach()
        z_quantized = ste / (self.levels - 1) * 2 - 1
        return z_quantized

    @torch.no_grad()
    def get_codes(self, z: torch.Tensor) -> torch.Tensor:
        """Integer codes only, no gradient — what actually gets stored/sent
        over the wire at inference time (before entropy coding)."""
        z_bounded = self._bound(z)
        scaled = (z_bounded + 1) / 2 * (self.levels - 1)
        return torch.round(scaled).to(torch.int32)


class CompressionAutoencoder(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        m = config["model"]
        self.latent_channels = m["latent_channels"]
        self.encoder = Encoder(
            base_channels=m["base_channels"],
            latent_channels=m["latent_channels"],
            num_residual_blocks=m["num_residual_blocks"],
            use_attention=m["use_attention"],
        )
        self.quantizer = Quantizer(bits=m["quantization_bits"],
                                    latent_channels=m["latent_channels"])
        self.decoder = Decoder(
            base_channels=m["base_channels"],
            latent_channels=m["latent_channels"],
            num_residual_blocks=m["num_residual_blocks"],
        )

    def forward(self, x: torch.Tensor) -> dict:
        latent = self.encoder(x)
        latent_quantized = self.quantizer(latent)
        reconstruction = self.decoder(latent_quantized)
        return {
            "reconstruction": reconstruction,
            "latent": latent,
            "latent_quantized": latent_quantized,
        }

    @torch.no_grad()
    def compress(self, x: torch.Tensor) -> torch.Tensor:
        """Inference-time path: image -> integer codes only (no reconstruction).
        Used by api/inference.py's /compress endpoint."""
        latent = self.encoder(x)
        return self.quantizer.get_codes(latent)

    @torch.no_grad()
    def decompress(self, codes: torch.Tensor) -> torch.Tensor:
        """Inference-time path: integer codes -> reconstructed image.
        Used by api/inference.py's /decompress endpoint."""
        levels = self.quantizer.levels
        z_quantized = codes.to(torch.float32) / (levels - 1) * 2 - 1
        return self.decoder(z_quantized)


if __name__ == "__main__":
    # Quick end-to-end check — run as a module from the project root:
    #   python -m src.model
    import yaml

    with open("configs/config.yaml") as f:
        config = yaml.safe_load(f)

    model = CompressionAutoencoder(config)
    x = torch.rand(2, 3, 256, 256)

    # Training-path forward
    out = model(x)
    print("Training forward pass:")
    print(f"  reconstruction: {tuple(out['reconstruction'].shape)}")
    print(f"  latent_quantized range: [{out['latent_quantized'].min():.3f}, "
          f"{out['latent_quantized'].max():.3f}]  (expected within [-1, 1])")

    # Verify gradients actually flow through the quantizer (the whole point
    # of the straight-through estimator)
    loss = out["reconstruction"].mean()
    loss.backward()
    enc_grad = model.encoder.stem.weight.grad
    assert enc_grad is not None and enc_grad.abs().sum() > 0, \
        "Gradient did not reach the encoder — STE is broken"
    print("  gradient reached encoder through quantizer: PASSED")

    # Inference-path compress/decompress
    codes = model.compress(x)
    print(f"\nInference compress: codes shape {tuple(codes.shape)}, "
          f"dtype {codes.dtype}, range [{codes.min().item()}, {codes.max().item()}]")
    recon = model.decompress(codes)
    print(f"Inference decompress: reconstruction shape {tuple(recon.shape)}")

    # Rough compression ratio at this bit depth (before entropy coding,
    # which would shrink this further)
    bits = config["model"]["quantization_bits"]
    latent_elems = codes.numel() // codes.shape[0]
    original_bits = 3 * 256 * 256 * 8  # uint8 RGB
    compressed_bits = latent_elems * bits
    print(f"\nRaw compression ratio (pre-entropy-coding): "
          f"{original_bits / compressed_bits:.2f}x")