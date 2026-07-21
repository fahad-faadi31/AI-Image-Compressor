"""
Combined loss for training the compression autoencoder.

L_total = w_l1        * L1(recon, target)
        + w_ms_ssim    * (1 - MS-SSIM(recon, target))
        + w_perceptual * VGG_perceptual(recon, target)
        + w_rate       * RateProxy(latent)

Each weight comes from configs/config.yaml -> training.loss_weights.

Why each term exists:
    - L1: pixel-level fidelity, robust to outliers (vs MSE which
      over-penalizes rare large errors and encourages blur)
    - MS-SSIM: matches human perception of structural similarity across
      multiple scales better than pixel losses alone
    - Perceptual (VGG features): keeps textures/edges looking sharp instead
      of the blurry "safe average" that pure pixel losses produce —
      compares deep feature activations, not raw pixels
    - Rate proxy: penalizes latent magnitude (mean-squared value of the
      PRE-quantization latent). This is a standard proxy for entropy under
      a fixed quantization step size: a latent with smaller, more
      concentrated values ends up using fewer of the available
      quantization levels, which is what actually makes the codes more
      compressible once we add real entropy coding (e.g. arithmetic
      coding) in the API layer. It is an approximation, not exact entropy
      — an exact learned entropy model (as in Balle et al.) is a valid
      future upgrade but adds real complexity for a first version.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_msssim import ms_ssim
from torchvision.models import vgg16, VGG16_Weights


class VGGPerceptualLoss(nn.Module):
    """Compares recon vs target in VGG16 feature space instead of pixel space.

    Uses early-to-mid layers (up to relu3_3) — deep enough to capture
    texture/edge structure, shallow enough to stay fast and avoid
    high-level semantic features we don't need for a compression task.
    """

    LAYER_CUTOFF = 16  # index into vgg16.features corresponding to relu3_3

    # VGG was trained on ImageNet-normalized inputs — this normalization is
    # internal to the perceptual loss only. It does NOT affect the [0,1]
    # convention used everywhere else in the pipeline (dataset, model I/O).
    IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

    def __init__(self):
        super().__init__()
        weights = VGG16_Weights.IMAGENET1K_V1
        vgg = vgg16(weights=weights).features[: self.LAYER_CUTOFF]
        vgg.eval()
        for p in vgg.parameters():
            p.requires_grad = False
        self.vgg = vgg
        self.register_buffer("mean", self.IMAGENET_MEAN)
        self.register_buffer("std", self.IMAGENET_STD)

    def _normalize(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean) / self.std

    def forward(self, recon: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        recon_feat = self.vgg(self._normalize(recon))
        target_feat = self.vgg(self._normalize(target))
        return F.l1_loss(recon_feat, target_feat)


class CompressionLoss(nn.Module):
    def __init__(self, weights: dict, use_perceptual: bool = True):
        super().__init__()
        self.weights = weights
        self.perceptual = VGGPerceptualLoss() if use_perceptual else None

    def forward(self, output: dict, target: torch.Tensor) -> dict:
        recon = output["reconstruction"]
        latent = output["latent"]

        l1 = F.l1_loss(recon, target)

        # ms_ssim returns similarity in [0,1] (1 = identical); we minimize
        # (1 - similarity) so the loss decreases as reconstructions improve
        ms_ssim_val = ms_ssim(recon, target, data_range=1.0, size_average=True)
        ms_ssim_loss = 1.0 - ms_ssim_val

        rate = latent.pow(2).mean()

        total = (
            self.weights["l1"] * l1
            + self.weights["ms_ssim"] * ms_ssim_loss
            + self.weights["rate"] * rate
        )

        components = {
            "l1": l1.detach(),
            "ms_ssim_loss": ms_ssim_loss.detach(),
            "rate": rate.detach(),
        }

        if self.perceptual is not None:
            perceptual = self.perceptual(recon, target)
            total = total + self.weights["perceptual"] * perceptual
            components["perceptual"] = perceptual.detach()

        components["total"] = total
        return components


if __name__ == "__main__":
    # Quick check — run as a module from the project root:
    #   python -m src.losses
    # NOTE: first run downloads pretrained VGG16 weights (needs internet).
    import yaml
    from src.model import CompressionAutoencoder

    with open("configs/config.yaml") as f:
        config = yaml.safe_load(f)

    model = CompressionAutoencoder(config)
    loss_fn = CompressionLoss(config["training"]["loss_weights"])

    x = torch.rand(2, 3, 256, 256)
    out = model(x)
    losses = loss_fn(out, x)

    print("Loss components:")
    for k, v in losses.items():
        print(f"  {k}: {v.item():.4f}")

    # Confirm gradients flow all the way back through the combined loss
    losses["total"].backward()
    enc_grad = model.encoder.stem.weight.grad
    assert enc_grad is not None and enc_grad.abs().sum() > 0
    print("\ngradient reached encoder through combined loss: PASSED")