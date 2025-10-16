FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-venv python3-pip git ca-certificates curl && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements first for caching
COPY requirements.txt /app/requirements.txt

# Install CUDA-enabled PyTorch matching CUDA 12.1 and project deps
RUN python3 -m venv /opt/venv && \
    . /opt/venv/bin/activate && \
    pip install --upgrade pip && \
    pip install --index-url https://download.pytorch.org/whl/cu121 torch torchvision torchaudio && \
    pip install -r /app/requirements.txt && \
    pip cache purge

ENV PATH="/opt/venv/bin:$PATH"

# Copy project
COPY . /app

# Default command prints GPU info and hints
CMD ["bash", "-lc", "python - << 'PY'\nimport torch\nprint('CUDA available:', torch.cuda.is_available())\nprint('torch:', torch.__version__, ' cuda:', torch.version.cuda)\nprint('device count:', torch.cuda.device_count())\nif torch.cuda.is_available():\n    print('device name:', torch.cuda.get_device_name(0))\nPY"]

