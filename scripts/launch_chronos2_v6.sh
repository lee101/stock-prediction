#!/usr/bin/env bash
# ============================================================================
# Launch Chronos2 v6 training — ctx=1024, all augmentations including crash robustness
#
# v6 improvements over v5:
#   - outlier_inject_prob=0.10: randomly inject crash/spike bars (robustness)
#   - outlier_magnitude=5.0: 5x local std for injected outliers
#   - Keeps channel_dropout=0.15, time_warp=0.15 from v5
#   - ctx=1024, batch=128, grad_accum=2, 200k steps
#   - Same Muon + amp_log_std=0.45 regularization
#
# Run after v5 completes (or in parallel if GPU allows).
# ============================================================================
set -euo pipefail
PROJ_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJ_DIR"

[[ -z "${R2_ACCESS_KEY:-}" && -n "${CLOUDFLARE_R2_ACCESS_KEY_ID:-}" ]] && \
    export R2_ACCESS_KEY="$CLOUDFLARE_R2_ACCESS_KEY_ID"
[[ -z "${R2_SECRET_KEY:-}" && -n "${CLOUDFLARE_R2_SECRET_ACCESS_KEY:-}" ]] && \
    export R2_SECRET_KEY="$CLOUDFLARE_R2_SECRET_ACCESS_KEY"
export R2_BUCKET="${R2_BUCKET:-models}"

source .venv/bin/activate

echo "[$(date -u +%H:%M:%SZ)] Launching v6 training (ctx=1024, outlier_inject=0.10, all augs)..."
nohup python chronos2_full_finetune.py \
    --cache-path              .cache/chronos2_train_data_full.npz \
    --output-dir              chronos2_finetuned/stocks_all_v6 \
    --num-steps               200000 \
    --batch-size              128 \
    --grad-accum              2 \
    --context-length          1024 \
    --finetune-mode           full \
    --torch-dtype             bfloat16 \
    --learning-rate           5e-5 \
    --amp-log-std             0.45 \
    --noise-frac              0.003 \
    --dropout-rate            0.03 \
    --freq-subsample-prob     0.15 \
    --channel-dropout-prob    0.15 \
    --time-warp-prob          0.15 \
    --outlier-inject-prob     0.10 \
    --outlier-magnitude       5.0 \
    --use-muon \
    --r2-prefix               chronos2/finetune/stocks_all_v6/finetuned-ckpt \
    --seed                    44 \
    > chronos2_finetune_v6.log 2>&1 &
V6_PID=$!
echo "[$(date -u +%H:%M:%SZ)] v6 training launched as PID $V6_PID -> chronos2_finetune_v6.log"
echo "$V6_PID" > .chronos2_v6_pid
