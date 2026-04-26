#!/bin/bash
# Reproduce the 120d XGB+Cat skew-gated sweep.
#
# This is the narrow follow-up to track1_oos120d_xgbcat. The useful filter is
# cross-sectional ret_5d skew >= 1.0, which removes the bad windows that made
# the no-gate stress36x cell p10-negative.
set -euo pipefail

cd /nvme0n1-disk/code/stock-prediction
source .venv/bin/activate

OUT=${1:-analysis/xgbnew_daily/track1_oos120d_xgbcat/sweep_skewgate_repro}
mkdir -p "$OUT" logs/track1

XGB=$(ls analysis/xgbnew_daily/track1_oos120d_xgb/alltrain_seed*.pkl)
CAT=$(ls analysis/xgbnew_daily/track1_oos120d_cat/alltrain_seed*.pkl)
PATHS=$(printf '%s\n%s\n' "$XGB" "$CAT" | paste -sd,)

python -m xgbnew.sweep_ensemble_grid \
    --symbols-file symbol_lists/stocks_wide_1000_v1.txt \
    --model-paths "$PATHS" \
    --oos-start 2025-12-18 \
    --oos-end 2026-04-17 \
    --window-days 30 \
    --stride-days 7 \
    --leverage-grid 2.0,2.5,2.75,3.0 \
    --min-score-grid 0.62 \
    --top-n-grid 2 \
    --hold-through \
    --min-dollar-vol 50000000 \
    --inference-min-dolvol-grid 50000000 \
    --inference-min-vol-grid 0.12 \
    --inference-max-vol-grid 0.0 \
    --fill-buffer-bps-grid -1 \
    --fee-regimes deploy,stress36x \
    --regime-cs-skew-min-grid=1.0 \
    --output-dir "$OUT" \
    --verbose \
    2>&1 | tee logs/track1/xgbcat_skewgate_120d.log
