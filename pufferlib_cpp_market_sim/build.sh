#!/bin/bash

set -e

echo "=== Building PufferLib3 C++ Market Simulator ==="

# Create build directory
mkdir -p build
cd build

# Configure with CMake
echo "Configuring with CMake..."
cmake .. -DCMAKE_BUILD_TYPE=Release

# Build
echo "Building..."
make -j$(nproc)

echo ""
echo "Build complete!"
echo "Executables:"
echo "  - train_market: ./build/train_market"
echo "  - test_market: ./build/test_market"
