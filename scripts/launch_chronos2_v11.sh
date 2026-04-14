#!/usr/bin/env bash
# ============================================================================
# Launch Chronos2 v11 training — ctx=1024, structural break + return momentum
#
# v11 improvements over v10:
#   - struct_break_prob=0.10: inject a structural break (simultaneous level
#     shift + volatility regime change) at a random split point. Distinct from
#     gap_inject (level-only) and vol_regime (vol-only) — simulates macro/sector
#     regime changes that alter both price level and volatility.
#   - return_momentum_prob=0.10: blend context with a smoothed (momentum) or
#     negated-returns (mean-reversion) version, creating artificial serial
#     autocorrelation. Distinct from mean_reversion (sinusoidal overlay) and
#     trend_inject (linear drift) — directly manipulates return autocorrelation.
#   - Keeps all v10 augmentations: earnings_shock=0.10, trend_inject=0.15,
#     gap_inject=0.15, channel_dropout=0.15, time_warp=0.15, outlier_inject=0.10,
#     freq_subsample=0.15, vol_regime=0.15, mean_reversion=0.10
#   - Same lr=3e-5, 200k steps, ctx=1024, seed=61
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

echo "[$(date -u +%H:%M:%SZ)] Launching v11 training (ctx=1024, struct_break=0.10, return_momentum=0.10)..."
nohup python chronos2_full_finetune.py \
    --cache-path              .cache/chronos2_train_data_full.npz \
    --output-dir              chronos2_finetuned/stocks_all_v11 \
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
    --struct-break-prob       0.10 \
    --struct-break-level-frac 0.08 \
    --struct-break-vol-mult   3.0 \
    --return-momentum-prob    0.10 \
    --return-momentum-blend   0.4 \
    --use-muon \
    --r2-prefix               chronos2/finetune/stocks_all_v11/finetuned-ckpt \
    --seed                    61 \
    > chronos2_finetune_v11.log 2>&1 &
V11_PID=$!
echo "[$(date -u +%H:%M:%SZ)] v11 training launched as PID $V11_PID -> chronos2_finetune_v11.log"
echo "$V11_PID" > .chronos2_v11_pid
