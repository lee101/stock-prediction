#!/usr/bin/env bash
# ============================================================================
# Launch Chronos2 v8 training — ctx=1024, full augmentation suite + trend injection
#
# v8 improvements over v7:
#   - trend_inject_prob=0.15: randomly add smooth linear trend to context.
#     Forces model to learn relative patterns regardless of overall drift direction.
#     Simulates trending/sideways markets; reduces trend-following bias.
#   - Keeps all v7 augmentations: gap_inject=0.15, channel_dropout=0.15,
#     time_warp=0.15, outlier_inject=0.10, freq_subsample=0.15, Muon optimizer
#   - Same lr=3e-5 as v7 (good convergence at 200k steps)
#   - ctx=1024, batch=128, grad_accum=2, 200k steps, seed=46
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

echo "[$(date -u +%H:%M:%SZ)] Launching v8 training (ctx=1024, trend_inject=0.15, all augs)..."
nohup python chronos2_full_finetune.py \
    --cache-path              .cache/chronos2_train_data_full.npz \
    --output-dir              chronos2_finetuned/stocks_all_v8 \
    --num-steps               200000 \
    --batch-size              128 \
    --grad-accum              2 \
    --context-length          1024 \
    --finetune-mode           full \
    --torch-dtype             bfloat16 \
    --learning-rate           3e-5 \
    --amp-log-std             0.45 \
    --noise-frac              0.003 \
    --dropout-rate            0.03 \
    --freq-subsample-prob     0.15 \
    --channel-dropout-prob    0.15 \
    --time-warp-prob          0.15 \
    --outlier-inject-prob     0.10 \
    --outlier-magnitude       5.0 \
    --gap-inject-prob         0.15 \
    --gap-magnitude-frac      0.05 \
    --trend-inject-prob       0.15 \
    --trend-magnitude-frac    0.10 \
    --use-muon \
    --r2-prefix               chronos2/finetune/stocks_all_v8/finetuned-ckpt \
    --seed                    46 \
    > chronos2_finetune_v8.log 2>&1 &
V8_PID=$!
echo "[$(date -u +%H:%M:%SZ)] v8 training launched as PID $V8_PID -> chronos2_finetune_v8.log"
echo "$V8_PID" > .chronos2_v8_pid
