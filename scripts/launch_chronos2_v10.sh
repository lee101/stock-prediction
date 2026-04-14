#!/usr/bin/env bash
# ============================================================================
# Launch Chronos2 v10 training — ctx=1024, earnings shock augmentation
#
# v10 improvements over v9:
#   - earnings_shock_prob=0.10: inject sudden large price moves (±5-15%) at a
#     random position in context, followed by either momentum continuation
#     (50% follow-through fading over 3 bars) or partial mean-reversion
#     (30% pullback fading over 3 bars). Simulates earnings/news events.
#     Distinct from gap_inject (persistent level shift) and outlier_inject
#     (isolated extreme bar that reverts immediately).
#   - Keeps all v9 augmentations: trend_inject=0.15, gap_inject=0.15,
#     channel_dropout=0.15, time_warp=0.15, outlier_inject=0.10,
#     freq_subsample=0.15, vol_regime=0.15, mean_reversion=0.10
#   - Same lr=3e-5 as v7-v9, 200k steps, ctx=1024, seed=59
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

echo "[$(date -u +%H:%M:%SZ)] Launching v10 training (ctx=1024, earnings_shock=0.10)..."
nohup python chronos2_full_finetune.py \
    --cache-path              .cache/chronos2_train_data_full.npz \
    --output-dir              chronos2_finetuned/stocks_all_v10 \
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
    --vol-regime-prob         0.15 \
    --vol-regime-max-mult     4.0 \
    --mean-reversion-prob     0.10 \
    --mean-reversion-amplitude 0.03 \
    --earnings-shock-prob     0.10 \
    --earnings-shock-magnitude 0.15 \
    --use-muon \
    --r2-prefix               chronos2/finetune/stocks_all_v10/finetuned-ckpt \
    --seed                    59 \
    > chronos2_finetune_v10.log 2>&1 &
V10_PID=$!
echo "[$(date -u +%H:%M:%SZ)] v10 training launched as PID $V10_PID -> chronos2_finetune_v10.log"
echo "$V10_PID" > .chronos2_v10_pid
