# ---------------------------------------------------------------------------
# Insurance SLM — Inference Server
#
# Uses python:3.11-slim as the base (small, reliable with Depot) and
# installs PyTorch + CUDA wheels via pip. This avoids the large pre-pulled
# NVIDIA/PyTorch base image layers that trigger Depot digest mismatches.
# ---------------------------------------------------------------------------

FROM python:3.11-slim-bookworm

WORKDIR /app

# System dependencies
# libgomp1 is required by bitsandbytes at runtime
RUN apt-get update && apt-get install -y --no-install-recommends \
        git \
        curl \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Install PyTorch CUDA 12.1 wheels first (isolated step for caching)
RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir \
        torch==2.2.2 \
        torchvision==0.17.2 \
        --index-url https://download.pytorch.org/whl/cu121

# Install remaining inference dependencies
RUN pip install --no-cache-dir \
        transformers==4.44.2 \
        peft==0.12.0 \
        bitsandbytes==0.42.0 \
        accelerate==0.33.0 \
        sentencepiece==0.2.0 \
        protobuf==4.25.4 \
        fastapi==0.112.2 \
        "uvicorn[standard]==0.30.6" \
        pydantic==2.8.2

# Copy inference server code
COPY serve/ ./
COPY models/ /app/models/

# If a trained adapter exists locally at models/insurance-phi35-lora, it will be
# baked into the image and served automatically. If not, the server falls back to
# the base model.
RUN mkdir -p /app/models /models/cache

# ---------------------------------------------------------------------------
# Runtime configuration — override in fly.toml [env] or fly secrets
# ---------------------------------------------------------------------------
ENV BASE_MODEL_ID=microsoft/Phi-3.5-mini-instruct
ENV ADAPTER_PATH=/app/models/insurance-phi35-lora
ENV MODEL_CACHE_DIR=/models/cache
ENV USE_4BIT=true
ENV MERGE_ON_LOAD=false
ENV PORT=8080

EXPOSE 8080

# Healthcheck — fly.io polls this before marking the machine healthy
HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080", "--workers", "1"]
