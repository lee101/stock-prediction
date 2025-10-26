# syntax=docker/dockerfile:1.7-labs

ARG CUDA_VERSION=12.4.1
ARG BUILD_IMAGE="nvidia/cuda:${CUDA_VERSION}-cudnn-devel-ubuntu22.04"
ARG RUNTIME_IMAGE="nvidia/cuda:${CUDA_VERSION}-cudnn-runtime-ubuntu22.04"

FROM ${BUILD_IMAGE} AS build

ARG DEBIAN_FRONTEND=noninteractive

SHELL ["/bin/bash", "-euxo", "pipefail", "-c"]

ENV UV_CACHE_DIR=/workspace/.uvcache \
    PATH="/root/.local/bin:${PATH}"

RUN --mount=type=cache,target=/var/cache/apt \
    apt-get update && apt-get install -y --no-install-recommends \
        python3 \
        python3-venv \
        python3-dev \
        ca-certificates \
        curl \
        git \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN curl -LsSf https://astral.sh/uv/install.sh | sh
RUN python3 -m venv /opt/venv

ENV PATH="/opt/venv/bin:${PATH}" \
    UV_LINK_MODE=copy

WORKDIR /workspace

COPY pyproject.toml uv.lock ./
COPY marketsimulator/pyproject.toml marketsimulator/
COPY . .

RUN mkdir -p trainingdata trainingdatadaily trainingdatahourly compiled_models hyperparams

RUN uv pip install --python /opt/venv/bin/python --no-cache-dir --editable .

FROM ${RUNTIME_IMAGE} AS runtime

ARG DEBIAN_FRONTEND=noninteractive

SHELL ["/bin/bash", "-euxo", "pipefail", "-c"]

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility \
    UV_CACHE_DIR=/workspace/.uvcache \
    PATH="/opt/venv/bin:${PATH}"

RUN --mount=type=cache,target=/var/cache/apt \
    apt-get update && apt-get install -y --no-install-recommends \
        python3 \
        python3-venv \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

COPY --from=build /opt/venv /opt/venv
COPY marketsimulator ./marketsimulator
COPY analysis ./analysis
COPY analysis_runner_funcs ./analysis_runner_funcs
COPY utils ./utils
COPY src ./src
COPY traininglib ./traininglib
COPY training ./training
COPY fal_marketsimulator ./fal_marketsimulator
COPY falmarket ./falmarket
COPY fal_utils ./fal_utils
COPY gymrl ./gymrl
COPY toto ./toto
COPY runpodmarket ./runpodmarket

RUN mkdir -p trainingdata trainingdatadaily trainingdatahourly compiled_models hyperparams run_logs

CMD ["python", "-m", "marketsimulator.runner"]
