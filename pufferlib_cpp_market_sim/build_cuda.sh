#!/bin/bash
# Ultra-Fast Market Simulator Build Script with CUDA

set -e  # Exit on error

echo "========================================="
echo "Building Ultra-Fast Market Simulator"
echo "With CUDA Kernels + PufferLib V-trace"
echo "========================================="
echo

# Check CUDA
if ! command -v nvcc &> /dev/null; then
    echo "ERROR: nvcc not found. Please install CUDA Toolkit."
    exit 1
fi

echo "CUDA Version:"
nvcc --version | grep "release"
echo

# Check GPU
if ! command -v nvidia-smi &> /dev/null; then
    echo "ERROR: nvidia-smi not found. NVIDIA drivers not installed?"
    exit 1
fi

echo "GPU Information:"
nvidia-smi --query-gpu=name,compute_cap,memory.total --format=csv,noheader
echo

# Detect CUDA architecture
COMPUTE_CAP=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d '.')
echo "Detected CUDA Compute Capability: $COMPUTE_CAP"

# Map to CMake architecture
# 8.0 = 80, 8.6 = 86, 8.9 = 89, 9.0 = 90
CUDA_ARCH=$COMPUTE_CAP
echo "Using CMake CUDA Architecture: $CUDA_ARCH"
echo

# Check PufferLib extension
PUFFERLIB_SO="../external/pufferlib-3.0.0/pufferlib/_C.cpython-314-x86_64-linux-gnu.so"
if [ ! -f "$PUFFERLIB_SO" ]; then
    echo "WARNING: PufferLib CUDA extension not found!"
    echo "Building PufferLib extension..."
    cd ../external/pufferlib-3.0.0/pufferlib
    python3.14 -m pip install -e . --no-build-isolation
    cd - > /dev/null

    if [ ! -f "$PUFFERLIB_SO" ]; then
        echo "ERROR: Failed to build PufferLib extension"
        exit 1
    fi
fi

echo "PufferLib extension: OK"
ls -lh "$PUFFERLIB_SO"
echo

# Clean previous build
if [ -d "build" ]; then
    echo "Cleaning previous build..."
    rm -rf build
fi

mkdir -p build
cd build

# Configure
echo "Configuring CMake with CUDA support..."
cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CUDA_ARCHITECTURES=$CUDA_ARCH \
    -DCMAKE_CUDA_FLAGS="-O3 --use_fast_math" \
    -DCMAKE_CXX_FLAGS="-O3 -march=native"

echo
echo "Building..."
make -j$(nproc)

echo
echo "========================================="
echo "Build Complete!"
echo "========================================="
echo

# List built targets
echo "Built targets:"
ls -lh train_market train_market_fast test_market libmarket_sim.so

echo
echo "To run:"
echo "  cd build"
echo "  ./train_market_fast"
echo
echo "To profile:"
echo "  nsys profile ./train_market_fast"
echo "  ncu ./train_market_fast"
echo

# Quick test
echo "Running quick validation..."
if ./test_market > test_output.log 2>&1; then
    echo "✓ Test passed!"
else
    echo "✗ Test failed. Check test_output.log"
    exit 1
fi

echo
echo "Ready to train! Run: ./train_market_fast"
