# AI Image Compressor

Neural network-based image compression, served as an API. Encoder compresses
an image to a compact latent representation; decoder reconstructs it with
minimal quality loss. Supports arbitrary input resolution (small images
through 4K/8K) via an adaptive tiling pipeline — no fixed input size
requirement for API consumers.

**Status: scaffold stage.** Module interfaces are defined; implementations
are being built incrementally. See TODOs in each file for what's next.

## Project structure

```
AI-Image-Compressor/
├── dataset/            # DIV2K train/val images (not committed — see .gitignore)
├── models/             # exported/traced models for serving
├── checkpoints/         # training checkpoints (.pt)
├── training/
│   ├── train.py        # main training loop
│   └── validate.py     # PSNR/SSIM/compression-ratio evaluation
├── src/
│   ├── encoder.py       # Encoder network
│   ├── decoder.py       # Decoder network
│   ├── model.py         # CompressionAutoencoder (encoder+quantizer+decoder)
│   ├── dataset.py       # DIV2K Dataset + augmentation
│   ├── losses.py        # combined reconstruction/perceptual/rate loss
│   └── tiling.py         # adaptive tiling for arbitrary image sizes
├── api/
│   ├── main.py           # FastAPI app, /compress and /decompress endpoints
│   └── inference.py     # model loading + compress/decompress logic
├── configs/
│   └── config.yaml       # single source of truth for all hyperparameters
├── tests/                # unit tests (tiling logic, loss shapes, API contract)
├── requirements.txt
├── Dockerfile
└── README.md
```

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Build order

1. Project scaffold (this step) — done
2. Dataset pipeline (`src/dataset.py`)
3. Model architecture (`src/encoder.py`, `decoder.py`, `model.py`)
4. Loss functions (`src/losses.py`)
5. Training loop (`training/train.py`, `validate.py`)
6. Tiling pipeline (`src/tiling.py`)
7. API (`api/main.py`, `inference.py`)
8. Docker + deployment

## Running the API (once implemented)

```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000
# Swagger UI: http://localhost:8000/docs
```
