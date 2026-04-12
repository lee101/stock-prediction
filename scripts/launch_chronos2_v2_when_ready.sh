#!/usr/bin/env bash
# ============================================================================
# Auto-launch v2 Chronos2 training with full cache once:
#   1) v1 training (PID 408556) finishes
#   2) Full cache (.cache/chronos2_train_data_full.npz) is ready
#
# Run: bash scripts/launch_chronos2_v2_when_ready.sh &
# ============================================================================

set -euo pipefail

PROJ_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJ_DIR"

V1_PID=408556
CACHE_PATH=".cache/chronos2_train_data_full.npz"
LOG_FILE="chronos2_finetune_v2.log"
OUTPUT_DIR="chronos2_finetuned/stocks_all_v2"

echo "[$(date -u +%H:%M:%SZ)] Watcher: waiting for v1 (PID $V1_PID) and cache..."

# Wait for v1 training to finish
while kill -0 $V1_PID 2>/dev/null; do
    sleep 30
done
echo "[$(date -u +%H:%M:%SZ)] v1 training finished."

# Wait for full cache to be ready
while [[ ! -f "$CACHE_PATH" ]]; do
    echo "[$(date -u +%H:%M:%SZ)] Waiting for cache $CACHE_PATH ..."
    sleep 30
done
echo "[$(date -u +%H:%M:%SZ)] Cache ready: $CACHE_PATH"

# Activate venv
source .venv/bin/activate

echo "[$(date -u +%H:%M:%SZ)] Launching v2 training (full cache, 50k steps)..."
nohup python chronos2_full_finetune.py \
    --cache-path     "$CACHE_PATH" \
    --output-dir     "$OUTPUT_DIR" \
    --num-steps      50000 \
    --batch-size     256 \
    --context-length 512 \
    --finetune-mode  full \
    --torch-dtype    bfloat16 \
    --learning-rate  5e-5 \
    --seed           42 \
    > "$LOG_FILE" 2>&1 &
V2_PID=$!
echo "[$(date -u +%H:%M:%SZ)] v2 training launched as PID $V2_PID → $LOG_FILE"
