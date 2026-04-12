#!/usr/bin/env bash
# ============================================================================
# Train full Chronos2 on all stock/crypto data — RunPod launcher
#
# Usage (from project root):
#   bash scripts/train_chronos2_full_runpod.sh [--steps 50000] [--muon] [--lora]
#
# Environment vars:
#   R2_ENDPOINT, R2_BUCKET, R2_ACCESS_KEY, R2_SECRET_KEY — for checkpoint upload
#   RUNPOD_POD_ID — auto-set on RunPod pods
#
# The script:
#   1. Activates virtual env
#   2. Builds/updates data cache if stale
#   3. Runs chronos2_full_finetune.py
#   4. Uploads checkpoint to R2
#   5. Prints final MAE
# ============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJ_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJ_DIR"

# --- Defaults ---
NUM_STEPS=50000
BATCH_SIZE=256
CONTEXT=512
LR=5e-5
FINETUNE_MODE=full
LORA_R=32
USE_MUON=false
R2_PREFIX="chronos2/finetune/stocks_all_v1"
CACHE_PATH=".cache/chronos2_train_data.npz"
OUTPUT_DIR=""   # auto-named if empty

# --- Parse flags ---
while [[ $# -gt 0 ]]; do
    case $1 in
        --steps)        NUM_STEPS=$2;        shift 2 ;;
        --batch)        BATCH_SIZE=$2;       shift 2 ;;
        --context)      CONTEXT=$2;          shift 2 ;;
        --lr)           LR=$2;               shift 2 ;;
        --lora)         FINETUNE_MODE=lora;  shift   ;;
        --muon)         USE_MUON=true;       shift   ;;
        --r2-prefix)    R2_PREFIX=$2;        shift 2 ;;
        --output-dir)   OUTPUT_DIR=$2;       shift 2 ;;
        --cache)        CACHE_PATH=$2;       shift 2 ;;
        *) echo "Unknown flag: $1"; exit 1 ;;
    esac
done

# --- Activate venv ---
if [[ -f ".venv/bin/activate" ]]; then
    source .venv/bin/activate
elif [[ -f ".venv312/bin/activate" ]]; then
    source .venv312/bin/activate
else
    echo "ERROR: No .venv found" >&2
    exit 1
fi

echo "=== Chronos2 full fine-tune ==="
echo "  steps=$NUM_STEPS batch=$BATCH_SIZE ctx=$CONTEXT lr=$LR"
echo "  mode=$FINETUNE_MODE muon=$USE_MUON"
echo "  cache=$CACHE_PATH"
echo "  r2=$R2_PREFIX"
echo "=============================="

# --- Build argument list ---
ARGS=(
    --daily-data-dir   trainingdata
    --hourly-data-dirs binance_spot_hourly
    --num-steps        $NUM_STEPS
    --batch-size       $BATCH_SIZE
    --context-length   $CONTEXT
    --learning-rate    $LR
    --finetune-mode    $FINETUNE_MODE
    --lora-r           $LORA_R
    --torch-dtype      bfloat16
    --cache-path       "$CACHE_PATH"
    --r2-prefix        "$R2_PREFIX"
    --num-workers      16
)

if [[ -n "$OUTPUT_DIR" ]]; then
    ARGS+=(--output-dir "$OUTPUT_DIR")
fi

if $USE_MUON; then
    ARGS+=(--use-muon)
fi

# --- Run ---
echo "Starting training at $(date -u +%Y-%m-%dT%H:%M:%SZ)"
python chronos2_full_finetune.py "${ARGS[@]}"
STATUS=$?

echo "Finished at $(date -u +%Y-%m-%dT%H:%M:%SZ) (exit $STATUS)"
exit $STATUS
