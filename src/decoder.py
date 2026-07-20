"""
U-Net style Decoder for AI Image Compression.

Input:
    latent: 128 x 8 x 8

Uses encoder skip connections:

    skip4: 256 x 16 x 16
    skip3: 256 x 32 x 32
    skip2: 128 x 64 x 64
    skip1: 64  x 128 x 128

Output:
    3 x 128 x 128
"""

import torch
import torch.nn as nn

from src.blocks import ResidualBlock, UpsampleBlock


class Decoder(nn.Module):

    def __init__(self):
        super().__init__()

        # 128x8x8 -> 256x8x8
        self.bottleneck = nn.Sequential(
            nn.Conv2d(
                128,
                256,
                kernel_size=3,
                padding=1,
            ),
            nn.GroupNorm(16, 256),
            nn.LeakyReLU(0.1, inplace=True),
        )


        # 256x8x8 -> 256x16x16
        self.up1 = UpsampleBlock(256, 256)

        self.conv1 = nn.Sequential(
            nn.Conv2d(
                256 + 256,
                256,
                kernel_size=3,
                padding=1,
            ),
            nn.GroupNorm(16, 256),
            nn.LeakyReLU(0.1, inplace=True),
            ResidualBlock(256),
        )


        # 256x16x16 -> 256x32x32
        self.up2 = UpsampleBlock(256, 256)

        self.conv2 = nn.Sequential(
            nn.Conv2d(
                256 + 256,
                256,
                kernel_size=3,
                padding=1,
            ),
            nn.GroupNorm(16, 256),
            nn.LeakyReLU(0.1, inplace=True),
            ResidualBlock(256),
        )


        # 256x32x32 -> 128x64x64
        self.up3 = UpsampleBlock(256, 128)

        self.conv3 = nn.Sequential(
            nn.Conv2d(
                128 + 128,
                128,
                kernel_size=3,
                padding=1,
            ),
            nn.GroupNorm(8, 128),
            nn.LeakyReLU(0.1, inplace=True),
            ResidualBlock(128),
        )


        # 128x64x64 -> 64x128x128
        self.up4 = UpsampleBlock(128, 64)

        self.conv4 = nn.Sequential(
            nn.Conv2d(
                64 + 64,
                64,
                kernel_size=3,
                padding=1,
            ),
            nn.GroupNorm(8, 64),
            nn.LeakyReLU(0.1, inplace=True),
            ResidualBlock(64),
        )


        # Final reconstruction layer
        self.output = nn.Sequential(
            nn.Conv2d(
                64,
                3,
                kernel_size=3,
                padding=1,
            ),
            nn.Sigmoid(),
        )


    def forward(self, latent, skips):

        x = self.bottleneck(latent)


        # skip4: 256x16x16
        x = self.up1(x)

        x = torch.cat(
            [x, skips[3]],
            dim=1,
        )

        x = self.conv1(x)


        # skip3: 256x32x32
        x = self.up2(x)

        x = torch.cat(
            [x, skips[2]],
            dim=1,
        )

        x = self.conv2(x)


        # skip2: 128x64x64
        x = self.up3(x)

        x = torch.cat(
            [x, skips[1]],
            dim=1,
        )

        x = self.conv3(x)


        # skip1: 64x128x128
        x = self.up4(x)

        x = torch.cat(
            [x, skips[0]],
            dim=1,
        )

        x = self.conv4(x)


        return self.output(x)



if __name__ == "__main__":

    print("========== DECODER TEST ==========")

    decoder = Decoder()

    latent = torch.randn(
        2,
        128,
        8,
        8,
    )

    skips = [
        torch.randn(2, 64, 128, 128),
        torch.randn(2, 128, 64, 64),
        torch.randn(2, 256, 32, 32),
        torch.randn(2, 256, 16, 16),
    ]


    output = decoder(
        latent,
        skips,
    )


    print("Latent :", latent.shape)
    print("Output :", output.shape)


    assert output.shape == (
        2,
        3,
        128,
        128,
    )

    print("\nDecoder test passed.")
    print("=================================")