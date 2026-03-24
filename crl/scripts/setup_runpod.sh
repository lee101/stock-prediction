#!/bin/bash
set -e

echo "=== CRL RunPod Setup ==="

# detect GPU
if nvidia-smi &>/dev/null; then
    echo "GPU detected:"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
fi

# install libtorch if not present
TORCH_DIR=/usr/local/lib/libtorch
if [ ! -d "$TORCH_DIR" ]; then
    echo "Downloading libtorch (CUDA 12.4)..."
    cd /tmp
    wget -q "https://download.pytorch.org/libtorch/cu124/libtorch-cxx11-abi-shared-with-deps-2.6.0%2Bcu124.zip" -O libtorch.zip
    unzip -q libtorch.zip -d /usr/local/lib/
    rm libtorch.zip
    echo "libtorch installed to $TORCH_DIR"
else
    echo "libtorch already installed at $TORCH_DIR"
fi

# build
cd "$(dirname "$0")/.."
echo "Building CRL..."
make clean
make TORCH_DIR=$TORCH_DIR -j$(nproc)

echo "Running tests..."
make test

echo "=== Setup complete ==="
echo "Binaries: crl_train crl_autoresearch crl_evaluate crl_benchmark"

# verify data symlink
if [ -L data ] && [ -d data ]; then
    echo "Data dir OK: $(ls data/*.bin 2>/dev/null | wc -l) .bin files"
else
    echo "WARNING: data symlink missing. Create: ln -sf ../pufferlib_market/data data"
fi
