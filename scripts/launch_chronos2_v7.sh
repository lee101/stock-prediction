#!/usr/bin/env bash
# ============================================================================
# Launch Chronos2 v7 training — ctx=1024, all augmentations + gap injection
#
# v7 improvements over v6:
#   - gap_inject_prob=0.15: randomly inject overnight price gaps (level shifts)
#     Simulates earnings gap-ups/downs, macro shocks, overnight news events.
#   - Slightly reduced learning rate (3e-5 vs 5e-5) for better late-stage
#     convergence at 200k steps.
#   - Keeps all v6 augmentations: channel_dropout=0.15, time_warp=0.15,
#     outlier_inject=0.10, freq_subsample=0.15, Muon optimizer
#   - ctx=1024, batch=128, grad_accum=2, 200k steps
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

echo "[$(date -u +%H:%M:%SZ)] Launching v7 training (ctx=1024, gap_inject=0.15, all augs)..."
nohup python chronos2_full_finetune.py \
    --cache-path              .cache/chronos2_train_data_full.npz \
    --output-dir              chronos2_finetuned/stocks_all_v7 \
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
    --use-muon \
    --r2-prefix               chronos2/finetune/stocks_all_v7/finetuned-ckpt \
    --seed                    45 \
    > chronos2_finetune_v7.log 2>&1 &
V7_PID=$!
echo "[$(date -u +%H:%M:%SZ)] v7 training launched as PID $V7_PID -> chronos2_finetune_v7.log"
echo "$V7_PID" > .chronos2_v7_pid
