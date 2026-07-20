"""
Reusable CNN building blocks.

Version 2.1
-----------
Improvements:
- GroupNorm instead of BatchNorm
- LeakyReLU activations
- Residual blocks with normalization
- Better training stability for image compression
"""

import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    """
    Conv -> GroupNorm -> LeakyReLU
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        stride=1,
        padding=1,
    ):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding,
                bias=False,
            ),
            nn.GroupNorm(
                num_groups=8,
                num_channels=out_channels,
            ),
            nn.LeakyReLU(
                negative_slope=0.1,
                inplace=True,
            ),
        )

    def forward(self, x):
        return self.block(x)


class ResidualBlock(nn.Module):
    """
    Residual Block

        x
         │
     Conv
         │
      GroupNorm
         │
     LeakyReLU
         │
     Conv
         │
      GroupNorm
         │
        +
         │
     LeakyReLU
         │
       Output
    """

    def __init__(self, channels):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(
                channels,
                channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),

            nn.GroupNorm(
                num_groups=8,
                num_channels=channels,
            ),

            nn.LeakyReLU(
                negative_slope=0.1,
                inplace=True,
            ),

            nn.Conv2d(
                channels,
                channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),

            nn.GroupNorm(
                num_groups=8,
                num_channels=channels,
            ),
        )

        self.activation = nn.LeakyReLU(
            negative_slope=0.1,
            inplace=True,
        )

    def forward(self, x):

        out = self.block(x)

        out = out + x

        out = self.activation(out)

        return out


class DownsampleBlock(nn.Module):
    """
    Strided convolution with GroupNorm.
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.block = nn.Sequential(

            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False,
            ),

            nn.GroupNorm(
                num_groups=8,
                num_channels=out_channels,
            ),

            nn.LeakyReLU(
                negative_slope=0.1,
                inplace=True,
            ),
        )

    def forward(self, x):
        return self.block(x)


class UpsampleBlock(nn.Module):
    """
    Nearest Neighbor Upsampling
    followed by Conv + GroupNorm + LeakyReLU.
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.block = nn.Sequential(

            nn.Upsample(
                scale_factor=2,
                mode="nearest",
            ),

            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),

            nn.GroupNorm(
                num_groups=8,
                num_channels=out_channels,
            ),

            nn.LeakyReLU(
                negative_slope=0.1,
                inplace=True,
            ),
        )

    def forward(self, x):
        return self.block(x)


if __name__ == "__main__":

    x = torch.randn(2, 64, 32, 32)

    print("Testing ResidualBlock...")
    block = ResidualBlock(64)
    y = block(x)
    print("ResidualBlock:", y.shape)

    print("Testing DownsampleBlock...")
    down = DownsampleBlock(64, 128)
    y = down(x)
    print("Downsample:", y.shape)

    print("Testing UpsampleBlock...")
    up = UpsampleBlock(128, 64)
    y = up(torch.randn(2, 128, 16, 16))
    print("Upsample:", y.shape)

    print("\nAll block tests passed successfully.")