#!/usr/bin/env bash
# ============================================================================
# Launch Chronos2 v9 training — ctx=1024, volatility regime + mean-reversion augs
#
# v9 improvements over v8:
#   - vol_regime_prob=0.15: split context at random mid-point, scale second half
#     by a random vol multiplier (0.25–4×, log-uniform). Simulates GARCH-like
#     volatility clustering. Teaches the model to adapt uncertainty to recent vol.
#   - mean_reversion_prob=0.10: overlay a damped sinusoidal oscillation on context.
#     Simulates range-bound / mean-reverting regimes; improves short-term accuracy.
#   - Keeps all v8 augmentations: trend_inject=0.15, gap_inject=0.15,
#     channel_dropout=0.15, time_warp=0.15, outlier_inject=0.10, freq_subsample=0.15
#   - Same lr=3e-5 as v7/v8, 200k steps, ctx=1024, seed=47
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

echo "[$(date -u +%H:%M:%SZ)] Launching v9 training (ctx=1024, vol_regime=0.15, mean_reversion=0.10)..."
nohup python chronos2_full_finetune.py \
    --cache-path              .cache/chronos2_train_data_full.npz \
    --output-dir              chronos2_finetuned/stocks_all_v9 \
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
    --use-muon \
    --r2-prefix               chronos2/finetune/stocks_all_v9/finetuned-ckpt \
    --seed                    47 \
    > chronos2_finetune_v9.log 2>&1 &
V9_PID=$!
echo "[$(date -u +%H:%M:%SZ)] v9 training launched as PID $V9_PID -> chronos2_finetune_v9.log"
echo "$V9_PID" > .chronos2_v9_pid
