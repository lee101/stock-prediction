#!/bin/bash
# GPU training wrapper script

# Activate virtual environment
source /media/lee/crucial2/code/stock/.venv/bin/activate

# Set CUDA paths
export CUDA_HOME=/usr/local/cuda-12.0
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# Set PyTorch CUDA libraries
VENV_PATH=/media/lee/crucial2/code/stock/.venv
export LD_LIBRARY_PATH=$VENV_PATH/lib/python3.12/site-packages/nvidia/nvjitlink/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$VENV_PATH/lib/python3.12/site-packages/nvidia/cublas/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$VENV_PATH/lib/python3.12/site-packages/nvidia/cudnn/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$VENV_PATH/lib/python3.12/site-packages/nvidia/nccl/lib:$LD_LIBRARY_PATH

# Preload nvjitlink to resolve symbol issues
export LD_PRELOAD=$VENV_PATH/lib/python3.12/site-packages/nvidia/nvjitlink/lib/libnvJitLink.so.12

# Set CUDA device
export CUDA_VISIBLE_DEVICES=0
export CUDA_LAUNCH_BLOCKING=1

# Run the training script
echo "🚀 Starting GPU training..."
echo "Environment setup complete."
echo "CUDA_HOME: $CUDA_HOME"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "---"

cd /media/lee/crucial2/code/stock/hftraining
python train_hf.py "$@"