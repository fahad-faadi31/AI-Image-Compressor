"""
Decoder network: (dequantized) latent -> reconstructed image.

Contract:
    Input:  tensor of shape (B, latent_channels, H/8, W/8), float32.
    Output: tensor of shape (B, 3, H, W), float32, values in [0, 1].

Architecture (mirrors Encoder in reverse):
    Head:        1x1 conv, latent_channels -> base*4          (H/8, W/8)
                 + residual blocks
    Stage 1:     PixelShuffle upsample, base*4 -> base*4        (H/4, W/4)  + residual blocks
    Stage 2:     PixelShuffle upsample, base*4 -> base*2        (H/2, W/2)  + residual blocks
    Stage 3:     PixelShuffle upsample, base*2 -> base            (H,   W)    + residual blocks
    Output:      3x3 conv, base -> 3, then Sigmoid to bound to [0, 1]

Why PixelShuffle instead of ConvTranspose2d: transposed convolutions have
uneven kernel overlap that produces visible checkerboard artifacts —
exactly the kind of reconstruction flaw a compression product can't have.
PixelShuffle (sub-pixel convolution) rearranges channels into spatial
resolution instead, avoiding that failure mode.
"""

import torch
import torch.nn as nn

from src.blocks import ResidualBlock


class UpBlock(nn.Module):
    """PixelShuffle-based 2x upsample followed by N residual blocks."""

    def __init__(self, in_channels: int, out_channels: int, num_residual_blocks: int):
        super().__init__()
        # Conv expands channels by 4x so PixelShuffle(2) can trade that
        # channel factor for a 2x spatial upsample: (C*4, H, W) -> (C, 2H, 2W)
        self.expand = nn.Conv2d(in_channels, out_channels * 4, kernel_size=3, padding=1)
        self.shuffle = nn.PixelShuffle(upscale_factor=2)
        self.res_blocks = nn.Sequential(
            *[ResidualBlock(out_channels) for _ in range(num_residual_blocks)]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.expand(x)
        x = self.shuffle(x)
        x = self.res_blocks(x)
        return x


class Decoder(nn.Module):
    def __init__(self, base_channels: int, latent_channels: int,
                 num_residual_blocks: int):
        super().__init__()

        # Mirrors Encoder's c1, c2, c3 in reverse
        c1, c2, c3 = base_channels * 2, base_channels * 4, base_channels * 4

        self.head = nn.Conv2d(latent_channels, c3, kernel_size=1)
        self.head_res_blocks = nn.Sequential(
            *[ResidualBlock(c3) for _ in range(num_residual_blocks)]
        )

        self.stage1 = UpBlock(c3, c2, num_residual_blocks)
        self.stage2 = UpBlock(c2, c1, num_residual_blocks)
        self.stage3 = UpBlock(c1, base_channels, num_residual_blocks)

        self.output_conv = nn.Conv2d(base_channels, 3, kernel_size=3, padding=1)
        self.output_act = nn.Sigmoid()  # bounds output to [0, 1], matching dataset.py

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        x = self.head(z)
        x = self.head_res_blocks(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.output_conv(x)
        x = self.output_act(x)
        return x


if __name__ == "__main__":
    # Quick shape check — run as a module from the project root:
    #   python -m src.decoder
    dec = Decoder(base_channels=64, latent_channels=32, num_residual_blocks=4)
    dummy_latent = torch.rand(2, 32, 32, 32)
    out = dec(dummy_latent)
    print(f"Input:  {tuple(dummy_latent.shape)}")
    print(f"Output: {tuple(out.shape)}  (expected: (2, 3, 256, 256))")
    print(f"Output range: [{out.min():.3f}, {out.max():.3f}]  (expected within [0, 1])")
    n_params = sum(p.numel() for p in dec.parameters())
    print(f"Parameters: {n_params:,}")