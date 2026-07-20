"""
Complete Image Compression Model.

Encoder
    ↓
Quantizer
    ↓
Decoder

The encoder now returns:

    latent, skips

The decoder uses both:

    reconstruction = decoder(latent, skips)
"""

import torch
import torch.nn as nn

from src.encoder import Encoder
from src.decoder import Decoder
from src.quantizer import Quantizer


class CompressiveAutoencoder(nn.Module):

    def __init__(self, quant_levels=127):
        super().__init__()

        self.encoder = Encoder()
        self.quantizer = Quantizer(levels=quant_levels)
        self.decoder = Decoder()

    def forward(self, x):

        # Encoder
        latent, skips = self.encoder(x)

        # Quantization
        latent_q = self.quantizer(latent)

        # Dequantization
        latent_dq = self.quantizer.dequantize(latent_q)

        # Decoder
        reconstruction = self.decoder(
            latent_dq,
            skips,
        )

        return reconstruction, latent_q

    def encode(self, x):

        latent, _ = self.encoder(x)

        latent_q = self.quantizer(latent)

        return latent_q

    def decode(self, latent_q):

        latent = self.quantizer.dequantize(latent_q)

        raise RuntimeError(
            "decode() cannot be called alone because the new "
            "U-Net decoder requires encoder skip features.\n"
            "Use model.forward(image) instead, or redesign the "
            "decoder for deployment without skip connections."
        )


if __name__ == "__main__":

    print("========== MODEL TEST ==========\n")

    model = CompressiveAutoencoder()

    dummy = torch.randn(
        2,
        3,
        128,
        128,
    )

    reconstruction, latent = model(dummy)

    print("Input Shape         :", dummy.shape)
    print("Latent Shape        :", latent.shape)
    print("Output Shape        :", reconstruction.shape)

    print()

    params = sum(
        p.numel()
        for p in model.parameters()
    )

    print(f"Parameters : {params:,}")

    assert reconstruction.shape == dummy.shape
    assert latent.shape == (2, 128, 8, 8)

    print("\nModel test passed.")
    print("===============================")