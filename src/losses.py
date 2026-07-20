"""
Loss functions for the Image Compression project.

This module provides:

1. MSE Loss
2. SSIM Loss
3. Hybrid Loss = alpha*MSE + beta*(1-SSIM)

The hybrid loss encourages both
- pixel accuracy
- structural similarity

which usually produces noticeably sharper reconstructions than
using MSE alone.
"""

import torch
import torch.nn as nn

from src.metrics import compute_ssim


class HybridLoss(nn.Module):

    def __init__(
        self,
        alpha=0.8,
        beta=0.2,
    ):
        super().__init__()

        self.alpha = alpha
        self.beta = beta

        self.mse = nn.MSELoss()

    def forward(
        self,
        prediction,
        target,
    ):

        mse_loss = self.mse(
            prediction,
            target,
        )

        ssim_value = compute_ssim(
            prediction,
            target,
        )

        ssim_loss = 1.0 - ssim_value

        total_loss = (

            self.alpha * mse_loss

            +

            self.beta * ssim_loss

        )

        return total_loss, mse_loss, ssim_value


if __name__ == "__main__":

    pred = torch.rand(
        4,
        3,
        128,
        128,
    )

    target = torch.rand(
        4,
        3,
        128,
        128,
    )

    criterion = HybridLoss()

    loss, mse, ssim = criterion(
        pred,
        target,
    )

    print()

    print("========== LOSS TEST ==========")

    print(f"Hybrid Loss : {loss:.6f}")
    print(f"MSE         : {mse:.6f}")
    print(f"SSIM        : {ssim:.6f}")

    print()

    print("Loss test passed.")

    print("===============================")