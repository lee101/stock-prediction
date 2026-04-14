#!/usr/bin/env bash
# ============================================================================
# Launch Chronos2 v12 training — prediction_length=2 + full augmentations
#
# v12 improvements over v11:
#   - prediction_length=2: train model to forecast BOTH day-1 and day-2 explicitly
#     so the multi-step continuation signal (step2_weight) is genuinely learned
#     rather than using a secondary output from a single-step model.
#   - All v11 augmentations retained: struct_break=0.10, return_momentum=0.10,
#     earnings_shock=0.10, trend_inject=0.15, gap_inject=0.15, etc.
#   - Seed=73 for diversity from v11
#   - ctx=1024, 200k steps, Muon optimizer
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

echo "[$(date -u +%H:%M:%SZ)] Launching v12 training (ctx=1024, pred_len=2, full augs)..."
nohup python chronos2_full_finetune.py \
    --cache-path              .cache/chronos2_train_data_full.npz \
    --output-dir              chronos2_finetuned/stocks_all_v12 \
    --num-steps               200000 \
    --batch-size              128 \
    --grad-accum              2 \
    --context-length          1024 \
    --prediction-length       2 \
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
    --vol-regime-prob         0.15 \
    --vol-regime-max-mult     4.0 \
    --mean-reversion-prob     0.10 \
    --mean-reversion-amplitude 0.03 \
    --earnings-shock-prob     0.10 \
    --earnings-shock-magnitude 0.15 \
    --struct-break-prob       0.10 \
    --struct-break-level-frac 0.08 \
    --struct-break-vol-mult   3.0 \
    --return-momentum-prob    0.10 \
    --return-momentum-blend   0.4 \
    --use-muon \
    --r2-prefix               chronos2/finetune/stocks_all_v12/finetuned-ckpt \
    --seed                    73 \
    > chronos2_finetune_v12.log 2>&1 &
V12_PID=$!
echo "[$(date -u +%H:%M:%SZ)] v12 training launched as PID $V12_PID -> chronos2_finetune_v12.log"
echo "$V12_PID" > .chronos2_v12_pid
