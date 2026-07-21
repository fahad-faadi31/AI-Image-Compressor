# Placeholder Dockerfile — finalized in the "Deployment" step once the
# model and API are implemented (needs to decide CPU vs CUDA base image,
# multi-stage build to keep image size down, etc.)

FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
