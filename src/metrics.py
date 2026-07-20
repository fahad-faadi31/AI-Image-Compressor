"""
Evaluation metrics for the compression pipeline.

PSNR (Peak Signal-to-Noise Ratio) is derived directly from MSE:
    PSNR = 10 * log10(MAX^2 / MSE)
where MAX is the maximum possible pixel value. Since our pipeline normalizes
images to [0, 1] (see src/transforms.py and the Sigmoid decoder output), MAX = 1,
which simplifies the formula to:
    PSNR = 10 * log10(1 / MSE) = -10 * log10(MSE)

Higher PSNR = better reconstruction (less error). This is why we chose MSE as
the training loss in Phase 0 -- minimizing MSE directly maximizes PSNR.

SSIM (Structural Similarity Index) compares local luminance, contrast, and
structure between two images using a sliding Gaussian window, rather than a
simple per-pixel error. It correlates better with perceived visual quality than
PSNR/MSE, since it accounts for structural information rather than treating
every pixel error identically. Implemented from scratch here (standard
windowed-Gaussian formulation) rather than pulling in an external library, to
stay consistent with the project's philosophy of understanding every component.
"""

import math
import torch
import torch.nn.functional as F


def compute_psnr(pred, target, max_val=1.0, eps=1e-10):
    """
    Args:
        pred, target: tensors of the same shape, e.g. [B, 3, H, W].
        max_val: maximum possible pixel value (1.0 for our [0,1] normalized images).
        eps: small constant to avoid log(0) / division by zero if reconstruction
            is ever pixel-perfect (MSE = 0), which is astronomically unlikely but
            would otherwise crash the run.

    Returns:
        Scalar tensor: mean PSNR (in dB) across the batch.
    """
    mse = torch.mean((pred - target) ** 2, dim=[1, 2, 3])  # per-image MSE
    psnr = 10 * torch.log10((max_val ** 2) / (mse + eps))
    return psnr.mean()


def _gaussian_1d(window_size, sigma):
    coords = torch.arange(window_size, dtype=torch.float32) - window_size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    return g / g.sum()


def _create_ssim_window(window_size, channels, sigma=1.5):
    """Builds a (channels, 1, window_size, window_size) Gaussian window for
    depthwise convolution -- each channel is filtered independently."""
    g_1d = _gaussian_1d(window_size, sigma).unsqueeze(1)     # [window_size, 1]
    g_2d = g_1d @ g_1d.t()                                    # [window_size, window_size], outer product
    window = g_2d.unsqueeze(0).unsqueeze(0)                   # [1, 1, window_size, window_size]
    return window.expand(channels, 1, window_size, window_size).contiguous()


@torch.no_grad()
def compute_ssim(pred, target, window_size=11, max_val=1.0):
    """
    Args:
        pred, target: tensors [B, C, H, W], values in [0, max_val].
        window_size: size of the Gaussian sliding window (11 is the standard
            choice from the original SSIM paper).
        max_val: dynamic range of pixel values (1.0 for our [0,1] images).

    Returns:
        Scalar tensor: mean SSIM across the batch (range roughly [-1, 1], where
        1.0 = identical images).
    """
    channels = pred.shape[1]
    window = _create_ssim_window(window_size, channels).to(pred.device)
    padding = window_size // 2

    mu_pred = F.conv2d(pred, window, padding=padding, groups=channels)
    mu_target = F.conv2d(target, window, padding=padding, groups=channels)

    mu_pred_sq = mu_pred ** 2
    mu_target_sq = mu_target ** 2
    mu_pred_target = mu_pred * mu_target

    sigma_pred_sq = F.conv2d(pred * pred, window, padding=padding, groups=channels) - mu_pred_sq
    sigma_target_sq = F.conv2d(target * target, window, padding=padding, groups=channels) - mu_target_sq
    sigma_pred_target = F.conv2d(pred * target, window, padding=padding, groups=channels) - mu_pred_target

    # Stabilizing constants from the original SSIM paper, scaled to our dynamic range.
    c1 = (0.01 * max_val) ** 2
    c2 = (0.03 * max_val) ** 2

    numerator = (2 * mu_pred_target + c1) * (2 * sigma_pred_target + c2)
    denominator = (mu_pred_sq + mu_target_sq + c1) * (sigma_pred_sq + sigma_target_sq + c2)
    ssim_map = numerator / denominator

    return ssim_map.mean()

