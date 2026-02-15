#!/bin/bash
# Run extended stock training on remote RTX 5090 machine
# Usage: ./run_remote.sh

set -e

REMOTE="administrator@93.127.141.100"
REMOTE_DIR="/nvme0n1-disk/code/stock-prediction"
LOCAL_DIR="$(cd "$(dirname "$0")/../.." && pwd)"

echo "=== Extended Stock Training (200M steps) ==="
echo "Remote: $REMOTE:$REMOTE_DIR"
echo ""

# Step 1: Sync code
echo "1. Syncing codebase..."
rsync -avz --exclude='.git' --exclude='*.pyc' --exclude='__pycache__' \
    --exclude='*.egg-info' --exclude='.venv' --exclude='venv' \
    --exclude='wandb' --exclude='checkpoints' --exclude='outputs' \
    "$LOCAL_DIR/" "$REMOTE:$REMOTE_DIR/"

# Step 2: Build C extension on remote
echo ""
echo "2. Building C extension..."
ssh "$REMOTE" "cd $REMOTE_DIR && source ~/.bashrc && \
    cd pufferlib_market && pip install -e . --quiet"

# Step 3: Export data (if needed)
echo ""
echo "3. Checking/exporting training data..."
ssh "$REMOTE" "cd $REMOTE_DIR && source ~/.bashrc && \
    if [ ! -f pufferlib_market/data/extended_stocks.bin ]; then \
        echo 'Exporting extended stocks data...'; \
        python -m experiments.extended_stock_corr_200M.export_data \
            --output pufferlib_market/data/extended_stocks.bin \
            --forecast-cache alpacanewccrosslearning/forecast_cache/mega24_novol_baseline_20260206_0038_lb2400; \
    else \
        echo 'Data already exists'; \
    fi"

# Step 4: Launch training in tmux
echo ""
echo "4. Launching training in tmux session 'stock_train'..."
ssh "$REMOTE" "cd $REMOTE_DIR && source ~/.bashrc && \
    tmux kill-session -t stock_train 2>/dev/null || true && \
    tmux new-session -d -s stock_train && \
    tmux send-keys -t stock_train 'cd $REMOTE_DIR && source ~/.bashrc && \
        python -m experiments.extended_stock_corr_200M.train_extended \
            --data-path pufferlib_market/data/extended_stocks.bin \
            --checkpoint-dir experiments/extended_stock_corr_200M/checkpoints \
            2>&1 | tee experiments/extended_stock_corr_200M/train.log' Enter"

echo ""
echo "=== Training launched! ==="
echo "Monitor with: ssh $REMOTE 'tmux attach -t stock_train'"
echo "Or: ssh $REMOTE 'tail -f $REMOTE_DIR/experiments/extended_stock_corr_200M/train.log'"
