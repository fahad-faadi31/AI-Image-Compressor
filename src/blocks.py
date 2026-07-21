"""
Shared network building blocks used by both Encoder and Decoder.
Kept separate so we don't duplicate the same residual/attention code twice.
"""

import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    """Standard pre-activation residual block, stride 1, channels unchanged.

    Using GroupNorm (not BatchNorm): compression models often run inference
    with batch_size=1 (single image), where BatchNorm's running statistics
    behave poorly. GroupNorm is batch-size independent.
    """

    def __init__(self, channels: int, num_groups: int = 8):
        super().__init__()
        self.norm1 = nn.GroupNorm(num_groups, channels)
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(num_groups, channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.conv1(self.act(self.norm1(x)))
        out = self.conv2(self.act(self.norm2(out)))
        return out + residual


class SEBlock(nn.Module):
    """Squeeze-and-Excitation channel attention.

    Cheap (a couple of small linear layers) — lets the model learn to
    weight *which channels* matter most for a given image region, e.g.
    prioritizing edge/texture channels over flat-color channels. This is
    the "lightweight attention" from our design discussion, not full
    self-attention.
    """

    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        hidden = max(channels // reduction, 4)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, hidden),
            nn.SiLU(inplace=True),
            nn.Linear(hidden, channels),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.shape
        weights = self.pool(x).view(b, c)
        weights = self.fc(weights).view(b, c, 1, 1)
        return x * weights