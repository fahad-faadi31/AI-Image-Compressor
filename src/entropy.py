"""
Entropy coding using zlib.

This module is responsible for converting the quantized latent tensor
into a real compressed binary stream and back again.

Pipeline:

Latent Tensor
      ↓
int8 numpy
      ↓
bytes
      ↓
zlib.compress()
      ↓
compressed bytes


Reverse:

compressed bytes
      ↓
zlib.decompress()
      ↓
numpy int8
      ↓
torch tensor
"""

import zlib
import numpy as np
import torch


class EntropyCoder:
    def __init__(self, compression_level=9):
        """
        compression_level:
            1 = fastest
            9 = best compression
        """
        self.level = compression_level

    def compress(self, latent_tensor):
        """
        Compress a quantized latent tensor.

        Args:
            latent_tensor : torch.Tensor

        Returns:
            compressed_bytes
        """

        latent = latent_tensor.detach().cpu().numpy().astype(np.int8)

        raw_bytes = latent.tobytes()

        compressed = zlib.compress(raw_bytes, self.level)

        return compressed

    def decompress(self, compressed_bytes, shape):
        """
        Recover latent tensor.

        Args:
            compressed_bytes
            shape

        Returns:
            torch.FloatTensor
        """

        raw = zlib.decompress(compressed_bytes)

        latent = np.frombuffer(raw, dtype=np.int8)

        latent = latent.reshape(shape)

        latent = torch.from_numpy(latent.astype(np.float32))

        return latent

    @staticmethod
    def save(compressed_bytes, path):
        with open(path, "wb") as f:
            f.write(compressed_bytes)

    @staticmethod
    def load(path):
        with open(path, "rb") as f:
            return f.read()