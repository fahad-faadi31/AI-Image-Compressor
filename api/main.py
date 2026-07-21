"""
FastAPI application entrypoint.

Run (once implemented):
    uvicorn api.main:app --host 0.0.0.0 --port 8000

Swagger docs auto-generated at /docs, ReDoc at /redoc.

Endpoints (contracts locked now, implemented in the "API development" step):

POST /compress
    Input:  multipart/form-data image file (jpg/png/webp/bmp/tiff)
    Output: JSON with base64 (or URL to) compressed image +
            { original_size_bytes, compressed_size_bytes, compression_ratio,
              original_dimensions, final_dimensions, psnr, ssim }

POST /decompress
    Input:  the compressed representation returned by /compress
    Output: reconstructed image file

GET /health
    Simple liveness check for load balancers / container orchestration.
"""

from fastapi import FastAPI

app = FastAPI(
    title="AI Image Compressor API",
    description="Neural network-based image compression service.",
    version="0.1.0",
)


@app.get("/health")
def health():
    return {"status": "ok"}


# TODO: POST /compress   -> api/inference.py: compress_image()
# TODO: POST /decompress -> api/inference.py: decompress_image()
