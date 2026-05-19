# IndexTTS2 + Wyoming bridge for Home Assistant.
#
# The project source is bind-mounted at runtime to /app, so this image carries
# only the venv. Rebuild only when pyproject.toml / uv.lock change; iterate on
# wyoming_indextts.py / streaming_v2.py / etc. with a simple container restart.
#
# Mirrors install.sh's three preconditions:
#   - cu128 torch wheels (devel CUDA image keeps nvcc around for DeepSpeed /
#     Triton / BigVGAN JIT-compile at first inference)
#   - python3.10-dev headers for the same JIT path
#   - prebuilt flash-attn 2.8.3 wheel matching cp310 / torch2.8 / cu12 / cxx11abiTRUE
FROM nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    UV_LINK_MODE=copy \
    PATH=/opt/venv/bin:/root/.local/bin:$PATH

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.10 \
        python3.10-dev \
        python3.10-venv \
        python3-pip \
        build-essential \
        git \
        curl \
        ca-certificates \
        ffmpeg \
        libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

RUN curl -LsSf https://astral.sh/uv/install.sh | sh

# Install deps into /opt/venv. --no-install-project skips building the local
# `indextts` package — the source lives at /app via bind mount and Python finds
# it because the orchestrator + uvicorn both run with cwd=/app.
WORKDIR /build
COPY pyproject.toml uv.lock ./
RUN uv sync \
        --all-extras \
        --no-install-project \
        --no-install-package flash-attn

# flash-attn 2.8.3 has no CUDA-13 wheels and source builds need CUDA 12 nvcc.
# Same prebuilt wheel install.sh pins.
RUN uv pip install --python /opt/venv/bin/python \
        'https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3%2Bcu12torch2.8cxx11abiTRUE-cp310-cp310-linux_x86_64.whl'

# Build-time sanity check (CUDA runtime isn't active yet — only verify imports).
RUN /opt/venv/bin/python -c "import torch, flash_attn; print('torch', torch.__version__, '+ flash-attn', flash_attn.__version__)"

WORKDIR /app
EXPOSE 8000 10200

ENTRYPOINT ["/opt/venv/bin/python", "scripts/serve_ha.py"]
