#!/usr/bin/env bash
# ============================================================================
# Launch Chronos2 v13 training — washout + parabolic augmentations
#
# v13 improvements over v12:
#   - washout_prob=0.10: drop+recovery pattern (stop-loss cascade simulation)
#   - parabolic_trend_prob=0.10: power-law trend (blow-off top / capitulation)
#   - prediction_length=2 retained from v12
#   - All v11/v12 augmentations retained
#   - Seed=17 for diversity from v11/v12
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

echo "[$(date -u +%H:%M:%SZ)] Launching v13 training (ctx=1024, pred_len=2, washout+parabolic augs)..."
nohup python chronos2_full_finetune.py \
    --cache-path              .cache/chronos2_train_data_full.npz \
    --output-dir              chronos2_finetuned/stocks_all_v13 \
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
    --washout-prob            0.10 \
    --washout-magnitude-frac  0.12 \
    --parabolic-trend-prob    0.10 \
    --parabolic-trend-magnitude-frac 0.15 \
    --use-muon \
    --r2-prefix               chronos2/finetune/stocks_all_v13/finetuned-ckpt \
    --seed                    17 \
    > chronos2_finetune_v13.log 2>&1 &
V13_PID=$!
echo "[$(date -u +%H:%M:%SZ)] v13 training launched as PID $V13_PID -> chronos2_finetune_v13.log"
echo "$V13_PID" > .chronos2_v13_pid
