#!/usr/bin/env bash
# ============================================================================
# Launch Chronos2 v4 training — longer context (1024) + stronger regularisation
#
# v4 improvements over v3:
#   - context_length=1024 (vs 512) — 2x longer history, captures more structure
#   - batch_size=128, grad_accum=2 → effective batch 256 (same as v3)
#   - Removes freq_subsample_prob (context/target mismatch was a concern)
#   - Keeps amp_log_std=0.45 and Muon from v3
#   - 200k steps (vs 100k in v3) to compensate for fewer qualifying series
#   - Seed 42
#
# Why longer context?
#   Same-period benchmark showed base model (ctx=2048) beats finetuned v2
#   (ctx=512) for GOOG (6.6% vs 7.4%), TSLA (5.7% vs 9.1%), META (9.6% vs
#   13.0%). Longer history captures structural patterns that ctx=512 misses.
#   ctx=1024 is the sweet spot: 5980/7796 series qualify (77%) vs only 33% for
#   ctx=2048.
#
# Usage: bash scripts/launch_chronos2_v4.sh
# Run AFTER v3 completes (same GPU, sequential).
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

echo "[$(date -u +%H:%M:%SZ)] Launching v4 training (ctx=1024, 200k steps)..."
nohup python chronos2_full_finetune.py \
    --cache-path      .cache/chronos2_train_data_full.npz \
    --output-dir      chronos2_finetuned/stocks_all_v4 \
    --num-steps       200000 \
    --batch-size      128 \
    --grad-accum      2 \
    --context-length  1024 \
    --finetune-mode   full \
    --torch-dtype     bfloat16 \
    --learning-rate   5e-5 \
    --amp-log-std     0.45 \
    --noise-frac      0.003 \
    --dropout-rate    0.03 \
    --use-muon \
    --r2-prefix       chronos2/finetune/stocks_all_v4/finetuned-ckpt \
    --seed            42 \
    > chronos2_finetune_v4.log 2>&1 &
V4_PID=$!
echo "[$(date -u +%H:%M:%SZ)] v4 training launched as PID $V4_PID -> chronos2_finetune_v4.log"
echo "$V4_PID" > .chronos2_v4_pid
