# ---------------------------------------------------------------------------
# Insurance SLM — RunPod Serverless Worker
#
# Build (requires Docker Desktop running):
#   docker buildx build --platform linux/amd64 \
#     -t <your-dockerhub-username>/insurance-slm:latest \
#     -f deploy/runpod.Dockerfile . --push
#
# Then in https://console.runpod.io/serverless/new-endpoint:
#   Container Image  → <your-dockerhub-username>/insurance-slm:latest
#   Container Disk   → 20 GB  (model downloads to /runpod-volume/cache on first start)
#   GPU              → 1× A40 or RTX 4090 (24 GB VRAM)
#   Min Workers      → 0  (scale to zero when idle)
#   Max Workers      → 1–3
# ---------------------------------------------------------------------------

FROM python:3.11-slim-bookworm

WORKDIR /app

# libgomp1 is required by bitsandbytes at runtime
RUN apt-get update && apt-get install -y --no-install-recommends \
        curl \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# PyTorch CUDA 12.1 wheels (separate layer for caching)
RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir \
        torch==2.2.2 \
        torchvision==0.17.2 \
        --index-url https://download.pytorch.org/whl/cu121

# Inference + RunPod SDK
RUN pip install --no-cache-dir \
        runpod==1.7.3 \
        transformers==4.44.2 \
        peft==0.12.0 \
        bitsandbytes==0.42.0 \
        accelerate==0.33.0 \
        sentencepiece==0.2.0 \
        protobuf==4.25.4

# Copy all serve/ files directly into /app (model_loader.py + runpod_handler.py)
COPY serve/ ./

# Adapter directory — empty by default; bake a trained adapter in by copying
# models/insurance-phi35-lora/ before building, or point ADAPTER_PATH at a
# RunPod network volume mount where you've uploaded the adapter.
RUN mkdir -p /app/adapter /runpod-volume/cache

# ---------------------------------------------------------------------------
# Runtime configuration — override via RunPod endpoint environment variables
# ---------------------------------------------------------------------------
ENV BASE_MODEL_ID=microsoft/Phi-3.5-mini-instruct
ENV ADAPTER_PATH=/app/adapter
ENV MODEL_CACHE_DIR=/runpod-volume/cache
ENV USE_4BIT=true
ENV MERGE_ON_LOAD=false

# RunPod serverless entry point
CMD ["python", "-u", "runpod_handler.py"]
