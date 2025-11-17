# Syntax=docker/dockerfile:1
FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    TF_CPP_MIN_LOG_LEVEL=2 \
    CUDA_VISIBLE_DEVICES="" \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        git \
        curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace
VOLUME ["/workspace"]

COPY requirements.txt ./
RUN pip install -r requirements.txt

COPY . .
RUN chmod +x ./repro.sh || true

# Default command opens a shell; override with `make train` etc.
CMD ["bash"]
