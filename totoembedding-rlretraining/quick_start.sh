#!/bin/bash

# Quick Start Script for Toto RL Training with HuggingFace Style

echo "=================================================="
echo "Toto RL Training with HuggingFace Optimizations"
echo "=================================================="

# Default configuration
CONFIG_FILE="config/hf_rl_config.json"
OPTIMIZER="gpro"
EPOCHS=100
BATCH_SIZE=32

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --optimizer)
            OPTIMIZER="$2"
            shift 2
            ;;
        --epochs)
            EPOCHS="$2"
            shift 2
            ;;
        --batch-size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --unfreeze)
            UNFREEZE="--unfreeze-embeddings"
            shift
            ;;
        --distributed)
            DISTRIBUTED="--distributed"
            shift
            ;;
        --debug)
            DEBUG="--debug"
            shift
            ;;
        # --wandb option removed; TensorBoard is used by default
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

echo ""
echo "Configuration:"
echo "  Optimizer: $OPTIMIZER"
echo "  Epochs: $EPOCHS"
echo "  Batch Size: $BATCH_SIZE"
echo "  Config File: $CONFIG_FILE"
echo "  Toto Embeddings: ${UNFREEZE:-Frozen}"
echo "  Training Mode: ${DISTRIBUTED:-Single GPU}"
echo ""

# Create necessary directories
mkdir -p models/hf_rl
mkdir -p logs/hf_rl
mkdir -p config

# Check if config file exists
if [ ! -f "$CONFIG_FILE" ]; then
    echo "Config file not found. Creating default config..."
    python -c "
from hf_rl_trainer import HFRLConfig
import json
config = HFRLConfig()
with open('$CONFIG_FILE', 'w') as f:
    json.dump(config.__dict__, f, indent=2)
print('Default config created at $CONFIG_FILE')
"
fi

# Launch training
echo "Starting training..."
python launch_hf_training.py \
    --config-file "$CONFIG_FILE" \
    --optimizer "$OPTIMIZER" \
    --num-epochs "$EPOCHS" \
    --batch-size "$BATCH_SIZE" \
    $UNFREEZE \
    $DISTRIBUTED \
    $DEBUG

echo ""
echo "Training completed!"
echo "- Logs (TensorBoard): logs/hf_rl/"
echo "- Models: models/hf_rl/"
echo ""
echo "To view training curves:"
echo "  tensorboard --logdir logs/hf_rl --port 6006"
