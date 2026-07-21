"""
Inference logic: model loading (once, at startup) + compress/decompress
functions called by api/main.py route handlers.

Contract:
    load_model(checkpoint_path, config) -> CompressionAutoencoder (eval mode, on device)
    compress_image(model, image_bytes: bytes) -> dict (see api/main.py /compress docstring)
    decompress_image(model, payload) -> bytes (reconstructed image file)

Design notes for the "deployment" step:
    - Model is loaded ONCE at API startup (FastAPI lifespan event), not
      per-request — reloading a checkpoint per request would be far too slow.
    - Uses src/tiling.py for images above tiling.min_size_for_tiling.
    - Runs under torch.no_grad() / torch.inference_mode().

NOTE: Stub only. Implemented in the "API development" step, after the model
and tiling modules are working.
"""


def load_model(checkpoint_path: str, config: dict):
    raise NotImplementedError


def compress_image(model, image_bytes: bytes) -> dict:
    raise NotImplementedError


def decompress_image(model, payload) -> bytes:
    raise NotImplementedError
