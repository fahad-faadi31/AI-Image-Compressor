"""
Encoder network: image -> latent representation.

Contract:
    Input:  tensor of shape (B, 3, H, W), float32, values in [0, 1],
            H and W must be divisible by 8 (3 downsampling stages).
    Output: tensor of shape (B, latent_channels, H/8, W/8) (pre-quantization).

Architecture:
    Stem:        3 -> base_channels                       (H,   W)
    Stage 1:     stride-2 conv, base -> base*2             (H/2, W/2)  + residual blocks
    Stage 2:     stride-2 conv, base*2 -> base*4            (H/4, W/4)  + residual blocks
    Stage 3:     stride-2 conv, base*4 -> base*4             (H/8, W/8)  + residual blocks
    Attention:   SE block (optional, config-driven)
    Head:        1x1 conv, base*4 -> latent_channels

Why 3 stages / /8 downsampling specifically: matches tile_size=256 in
configs/config.yaml giving a 32x32 latent grid per tile — small enough for
real compression, large enough to preserve texture detail. If tile_size
changes, this ratio stays the same since it's stride-based, not hardcoded
to 256.
"""

import torch
import torch.nn as nn

from src.blocks import ResidualBlock, SEBlock


class DownBlock(nn.Module):
    """Stride-2 downsample followed by N residual blocks at the new width."""

    def __init__(self, in_channels: int, out_channels: int, num_residual_blocks: int):
        super().__init__()
        self.downsample = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                                     stride=2, padding=1)
        self.res_blocks = nn.Sequential(
            *[ResidualBlock(out_channels) for _ in range(num_residual_blocks)]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.downsample(x)
        x = self.res_blocks(x)
        return x


class Encoder(nn.Module):
    def __init__(self, base_channels: int, latent_channels: int,
                 num_residual_blocks: int, use_attention: bool):
        super().__init__()

        self.stem = nn.Conv2d(3, base_channels, kernel_size=3, padding=1)

        # Channel widths per stage: base -> base*2 -> base*4 -> base*4
        # (widen for the first two downsamples, hold steady on the third —
        # keeps parameter count reasonable while still growing capacity
        # where spatial resolution, and thus signal, is highest)
        c1, c2, c3 = base_channels * 2, base_channels * 4, base_channels * 4

        self.stage1 = DownBlock(base_channels, c1, num_residual_blocks)
        self.stage2 = DownBlock(c1, c2, num_residual_blocks)
        self.stage3 = DownBlock(c2, c3, num_residual_blocks)

        self.attention = SEBlock(c3) if use_attention else nn.Identity()

        self.head = nn.Conv2d(c3, latent_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        if h % 8 != 0 or w % 8 != 0:
            raise ValueError(
                f"Encoder input H and W must be divisible by 8, got ({h}, {w}). "
                f"This should be guaranteed by src/tiling.py padding upstream."
            )

        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.attention(x)
        x = self.head(x)
        return x


if __name__ == "__main__":
    # Quick shape check — run as a module from the project root:
    #   python -m src.encoder
    # (NOT `python src/encoder.py` — that breaks the `from src.blocks import`
    # above, since src/ wouldn't be on the import path as a package)
    enc = Encoder(base_channels=64, latent_channels=32,
                   num_residual_blocks=4, use_attention=True)
    dummy = torch.rand(2, 3, 256, 256)
    out = enc(dummy)
    print(f"Input:  {tuple(dummy.shape)}")
    print(f"Output: {tuple(out.shape)}  (expected: (2, 32, 32, 32))")
    n_params = sum(p.numel() for p in enc.parameters())
    print(f"Parameters: {n_params:,}")