# CUDA Build Guide - Ultra-Fast Market Simulator

## Overview
This guide explains how to build and run the ultra-optimized C++ market simulator with CUDA kernels, PufferLib3 V-trace integration, and learned execution policies.

## Prerequisites

### Hardware
- **NVIDIA GPU**: RTX 3000/4000/5000 series (Ampere, Ada, or newer)
- **VRAM**: Minimum 8GB, recommended 24GB+
- **RAM**: Minimum 16GB system RAM

### Software
- **CUDA Toolkit**: 12.0 or newer
- **CMake**: 3.18 or newer
- **GCC/G++**: 9.0 or newer
- **LibTorch**: 2.9.0+ with CUDA 12.8 support
- **Python**: 3.14+ (for PufferLib extension)

## Installation Steps

### 1. Verify CUDA Installation

```bash
nvcc --version
nvidia-smi

# Should show:
# - CUDA 12.x
# - Your GPU (e.g., RTX 5090)
```

### 2. Build PufferLib CUDA Extension

```bash
cd ../external/pufferlib-3.0.0/pufferlib
python3.14 -m pip install -e .

# Verify the extension is built
ls -lh _C.cpython-314-x86_64-linux-gnu.so
# Should be ~8MB
```

### 3. Configure Build

```bash
cd pufferlib_cpp_market_sim
mkdir -p build
cd build

# Configure with CMake
cmake .. \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_ARCHITECTURES=89  # Change based on your GPU
```

**CUDA Architecture Codes**:
- RTX 5090/5080: `89` (Ada)
- RTX 4090/4080: `89` (Ada)
- RTX 3090/3080: `86` (Ampere)

### 4. Build

```bash
# Build with all CPU cores
make -j$(nproc)

# You should see:
# - libmarket_sim.so (with CUDA support)
# - train_market (basic trainer)
# - train_market_fast (optimized with CUDA kernels)
# - test_market
```

### 5. Verify Build

```bash
# Check CUDA symbols in the library
nm -D libmarket_sim.so | grep cuda

# Should show CUDA runtime symbols

# Check executable dependencies
ldd train_market_fast

# Should show:
# - libtorch_cuda.so
# - libcudart.so
# - _C.cpython-314-x86_64-linux-gnu.so (pufferlib)
```

## Running the Optimized Trainer

### Basic Usage

```bash
cd pufferlib_cpp_market_sim/build

# Run the ultra-fast CUDA trainer
./train_market_fast

# Expected output:
# === Ultra-Fast PufferLib3 C++ Market Simulator ===
# CUDA Device: 1 GPUs available
# ...
# Expected throughput: >500K steps/sec with CUDA kernels
```

### Monitor GPU Usage

```bash
# In another terminal
watch -n 1 nvidia-smi

# You should see:
# - GPU utilization: 80-100%
# - Memory usage: 4-8GB
# - Power draw: near TDP
```

### Performance Benchmarking

```bash
# Run a short benchmark
./train_market_fast 2>&1 | tee benchmark.log

# Check final throughput
grep "Average throughput" benchmark.log

# Expected results:
# - RTX 5090: >800K steps/sec
# - RTX 4090: >600K steps/sec
# - RTX 3090: >400K steps/sec
```

## Build Troubleshooting

### Problem: CUDA not found

```bash
# Set CUDA_HOME explicitly
export CUDA_HOME=/usr/local/cuda-12
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# Rebuild
rm -rf build && mkdir build && cd build && cmake .. && make -j$(nproc)
```

### Problem: PufferLib extension not found

```bash
# Check if the .so exists
find ../external/pufferlib-3.0.0 -name "_C.cpython*.so"

# If missing, rebuild pufferlib
cd ../external/pufferlib-3.0.0/pufferlib
pip install -e . --force-reinstall

# Check again
ls -lh _C.cpython-314-x86_64-linux-gnu.so
```

### Problem: CUDA architecture mismatch

```bash
# Find your GPU's compute capability
nvidia-smi --query-gpu=compute_cap --format=csv

# Update CMakeLists.txt line 61:
# set(CMAKE_CUDA_ARCHITECTURES "XX")  # Your arch code

# Rebuild
cd build && rm -rf * && cmake .. && make -j$(nproc)
```

### Problem: Linker errors with LibTorch

```bash
# Check LibTorch path
echo $CMAKE_PREFIX_PATH

# Should point to: ../external/libtorch/libtorch

# If not, set explicitly:
export CMAKE_PREFIX_PATH=/path/to/libtorch

# Rebuild
cd build && rm -rf * && cmake .. && make -j$(nproc)
```

### Problem: Out of memory during compilation

```bash
# Reduce parallel jobs
make -j4  # Instead of -j$(nproc)

# Or compile CUDA files separately
make cuda_kernels.cu.o -j1
make -j$(nproc)
```

## Optimization Levels

### Debug Build (slower, more verbose)

```bash
cmake .. -DCMAKE_BUILD_TYPE=Debug
make -j$(nproc)

# Enables:
# - CUDA error checking after every kernel
# - Assertions
# - Debug symbols
# - No optimizations
```

### Release Build (fastest)

```bash
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)

# Enables:
# - -O3 optimization
# - --use_fast_math for CUDA
# - No debug symbols
# - ~2x faster than debug
```

## Performance Profiling

### CUDA Profiler (Nsight Systems)

```bash
# Profile the training loop
nsys profile --stats=true ./train_market_fast

# Analyze results
nsys-ui report1.nsys-rep

# Look for:
# - Kernel execution times
# - Memory transfer bottlenecks
# - CPU/GPU overlaps
```

### CUDA Kernel Profiling (Nsight Compute)

```bash
# Profile specific kernels
ncu --set full ./train_market_fast

# Analyze kernel performance:
# - portfolio_step_fused_kernel
# - assemble_observations_kernel
# - pufferlib::compute_puff_advantage
```

### LibTorch Profiling

```cpp
// Add to main_train_fast.cpp

#include <torch/csrc/autograd/profiler.h>

// Before training loop:
auto profiler = torch::autograd::profiler::ProfilerConfig(
    torch::autograd::profiler::ProfilerState::CUDA
);

// After training loop:
profiler.save("training_profile.json");

// View with chrome://tracing
```

## Expected Performance Metrics

### Throughput Targets

| GPU | Batch Size | Expected Steps/Sec |
|-----|------------|-------------------|
| RTX 5090 | 4096 | 800K - 1M |
| RTX 4090 | 4096 | 600K - 800K |
| RTX 3090 | 4096 | 400K - 600K |
| RTX 3080 | 2048 | 200K - 400K |

### Kernel Timing Breakdown

| Operation | Time (ms) | Throughput |
|-----------|-----------|------------|
| Portfolio Step (CUDA) | 0.05 | 80M env-steps/sec |
| Observation Assembly | 0.02 | 200M assembles/sec |
| V-trace Advantages | 0.02 | 200M timesteps/sec |
| Policy Forward | 0.5 | 8M inferences/sec |
| PPO Update | 2.0 | 2M updates/sec |
| **Total (per batch)** | **~3ms** | **~1.3M steps/sec** |

### Memory Usage

| Component | VRAM Usage |
|-----------|------------|
| Market Data (7 symbols) | 500 MB |
| Environment State (4096 envs) | 1 GB |
| Policy Network | 100 MB |
| Execution Policy | 50 MB |
| Rollout Buffers | 2 GB |
| LibTorch Runtime | 1 GB |
| **Total** | **~5 GB** |

## Advanced Configuration

### Multi-GPU Training

```cpp
// TODO: Add multi-GPU support
// Use torch::nn::DataParallel
// Distribute environments across GPUs
```

### Mixed Precision Training

Already enabled in `market_config.h`:
```cpp
config.use_mixed_precision = true;
```

This uses FP16 for faster computation while maintaining FP32 for critical paths.

### Kernel Tuning

Edit `cuda_kernels.cu` line 10:
```cpp
constexpr int THREADS_PER_BLOCK = 256;  // Try 128, 256, 512
```

Optimal value depends on your GPU architecture.

## Next Steps

1. **Verify CUDA installation**: `nvidia-smi`, `nvcc --version`
2. **Build PufferLib extension**: See section 2
3. **Configure and build**: See sections 3-4
4. **Run and benchmark**: See Running section
5. **Profile and optimize**: See Performance Profiling section

## Support

If you encounter issues:
1. Check the [FAST_OPTIMIZATION_DESIGN.md](FAST_OPTIMIZATION_DESIGN.md) for architecture details
2. Review build logs carefully
3. Verify all dependencies are installed
4. Check GPU compatibility (compute capability ≥ 8.0)

## Performance Debugging

### If throughput is low (<200K steps/sec):

1. **Check GPU utilization**:
   ```bash
   nvidia-smi dmon -s u
   ```
   Should show >80% GPU utilization

2. **Profile kernel times**:
   ```bash
   ncu ./train_market_fast
   ```
   Look for slow kernels

3. **Check memory bandwidth**:
   ```bash
   nvidia-smi dmon -s m
   ```
   Memory bandwidth should be >500 GB/s on RTX 5090

4. **Verify CUDA streams**:
   Kernels should overlap with data transfers

5. **Check batch size**:
   Larger batches = better GPU utilization (try 8192)
