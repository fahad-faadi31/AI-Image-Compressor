"""
U-Net style Encoder for AI Image Compression.

Architecture:
Input:
    3 x 128 x 128

Feature extraction:
    64  x 128 x 128  -> skip 1
    128 x 64  x 64   -> skip 2
    256 x 32  x 32   -> skip 3
    256 x 16  x 16   -> skip 4

Bottleneck:
    128 x 8 x 8

The encoder returns:
    latent, skips

The skip features are passed to the decoder to preserve
fine details and improve reconstruction quality.
"""

import torch
import torch.nn as nn

from src.blocks import ResidualBlock, DownsampleBlock


class Encoder(nn.Module):

    def __init__(self):
        super().__init__()

        # Initial feature extraction
        self.initial = nn.Sequential(
            nn.Conv2d(
                3,
                64,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.GroupNorm(8, 64),
            nn.LeakyReLU(0.1, inplace=True),
        )

        # 128x128 -> 64x64
        self.down1 = DownsampleBlock(64, 128)

        # 64x64 -> 32x32
        self.down2 = DownsampleBlock(128, 256)

        # 32x32 -> 16x16
        self.down3 = DownsampleBlock(256, 256)

        # 16x16 -> 8x8
        self.down4 = DownsampleBlock(256, 256)

        # Residual refinement blocks
        self.res1 = ResidualBlock(128)
        self.res2 = ResidualBlock(256)
        self.res3 = ResidualBlock(256)
        self.res4 = ResidualBlock(256)

        # Bottleneck compression
        self.bottleneck = nn.Sequential(
            nn.Conv2d(
                256,
                128,
                kernel_size=1,
                stride=1,
            ),
            nn.Tanh(),
        )

    def forward(self, x):

        skips = []

        # 3x128x128
        x = self.initial(x)

        # 64x128x128
        skips.append(x)

        # 128x64x64
        x = self.down1(x)
        x = self.res1(x)
        skips.append(x)

        # 256x32x32
        x = self.down2(x)
        x = self.res2(x)
        skips.append(x)

        # 256x16x16
        x = self.down3(x)
        x = self.res3(x)
        skips.append(x)

        # 256x8x8
        x = self.down4(x)
        x = self.res4(x)

        # 128x8x8 latent
        latent = self.bottleneck(x)

        return latent, skips


if __name__ == "__main__":

    print("========== ENCODER TEST ==========")

    encoder = Encoder()

    dummy = torch.randn(
        2,
        3,
        128,
        128,
    )

    latent, skips = encoder(dummy)

    print("Input :", dummy.shape)
    print("Latent :", latent.shape)

    print("\nSkip features:")

    for i, skip in enumerate(skips):
        print(f"Skip {i+1}: {skip.shape}")

    assert latent.shape == (
        2,
        128,
        8,
        8,
    )

    print("\nEncoder test passed.")
    print("=================================")