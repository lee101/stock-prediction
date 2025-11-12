#!/bin/bash
# Setup environment variables for stable GPU training

# Reduce aggressive CUDA memory bursts
export CUDA_DEVICE_MAX_CONNECTIONS=1
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

# Disable TF32 for better numerical stability (optional)
# export NVIDIA_TF32_OVERRIDE=0

# Enable CUDA launch blocking for better error reporting (DEBUG ONLY - slows training)
# export CUDA_LAUNCH_BLOCKING=1

# Source this file before training:
# source scripts/setup_training_env.sh

echo "Training environment configured:"
echo "  CUDA_DEVICE_MAX_CONNECTIONS=$CUDA_DEVICE_MAX_CONNECTIONS"
echo "  PYTORCH_CUDA_ALLOC_CONF=$PYTORCH_CUDA_ALLOC_CONF"
