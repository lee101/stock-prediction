#!/usr/bin/env bash
# ============================================================================
# Launch Chronos2 v3 training — larger/longer run with improved augmentation
#
# v3 improvements over v2:
#   - 100k steps (vs 50k) — more training compute
#   - Freq subsampling augmentation (prob=0.15) — multi-timescale robustness
#   - Stronger amplitude jitter (log_std=0.45 vs 0.30)
#   - Muon optimizer
#   - R2 checkpoint upload
#
# Usage: bash scripts/launch_chronos2_v3.sh
# ============================================================================
set -euo pipefail
PROJ_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJ_DIR"

# Map CLOUDFLARE_R2_* -> R2_*
[[ -z "${R2_ACCESS_KEY:-}" && -n "${CLOUDFLARE_R2_ACCESS_KEY_ID:-}" ]] && \
    export R2_ACCESS_KEY="$CLOUDFLARE_R2_ACCESS_KEY_ID"
[[ -z "${R2_SECRET_KEY:-}" && -n "${CLOUDFLARE_R2_SECRET_ACCESS_KEY:-}" ]] && \
    export R2_SECRET_KEY="$CLOUDFLARE_R2_SECRET_ACCESS_KEY"
export R2_BUCKET="${R2_BUCKET:-models}"

source .venv/bin/activate

echo "[$(date -u +%H:%M:%SZ)] Launching v3 training..."
nohup python chronos2_full_finetune.py \
    --cache-path      .cache/chronos2_train_data_full.npz \
    --output-dir      chronos2_finetuned/stocks_all_v3 \
    --num-steps       100000 \
    --batch-size      256 \
    --context-length  512 \
    --finetune-mode   full \
    --torch-dtype     bfloat16 \
    --learning-rate   5e-5 \
    --amp-log-std     0.45 \
    --noise-frac      0.003 \
    --dropout-rate    0.03 \
    --use-muon \
    --r2-prefix       chronos2/finetune/stocks_all_v3/finetuned-ckpt \
    --seed            123 \
    > chronos2_finetune_v3.log 2>&1 &
V3_PID=$!
echo "[$(date -u +%H:%M:%SZ)] v3 training launched as PID $V3_PID -> chronos2_finetune_v3.log"
echo "$V3_PID" > .chronos2_v3_pid
