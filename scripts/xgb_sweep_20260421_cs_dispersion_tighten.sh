#!/bin/bash
# Tighten cs-dispersion gate on 2025H1 fold: at the sweet-spot thresholds
# (iqr=0.042 skew=1.0), try lower leverage and higher ms to cut tail
# costs and see if we can pass HARD RULE #1 neg-rate at stress36x.

set -euo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate

OUTDIR=analysis/xgbnew_daily/sweep_20260421_cs_dispersion_tighten
mkdir -p "$OUTDIR"

python -m xgbnew.sweep_ensemble_grid \
  --symbols-file symbol_lists/stocks_wide_1000_v1.txt \
  --data-root trainingdata \
  --model-paths analysis/xgbnew_daily/oos2024_ensemble_gpu_fresh/alltrain_seed0.pkl,\
analysis/xgbnew_daily/oos2024_ensemble_gpu_fresh/alltrain_seed7.pkl,\
analysis/xgbnew_daily/oos2024_ensemble_gpu_fresh/alltrain_seed42.pkl,\
analysis/xgbnew_daily/oos2024_ensemble_gpu_fresh/alltrain_seed73.pkl,\
analysis/xgbnew_daily/oos2024_ensemble_gpu_fresh/alltrain_seed197.pkl \
  --train-start 2020-01-01 --train-end 2024-12-31 \
  --oos-start   2025-01-01 --oos-end   2026-04-20 \
  --window-days 30 --stride-days 5 \
  --leverage-grid 1.0,1.5,2.0 \
  --min-score-grid 0.60,0.70,0.80 \
  --top-n-grid 1 \
  --hold-through \
  --fee-regimes deploy,stress36x \
  --min-dollar-vol 50000000 \
  --inference-min-vol-grid 0.10 \
  --regime-cs-iqr-max-grid 0.042 \
  --regime-cs-skew-min-grid 1.0 \
  --output-dir "$OUTDIR" \
  --verbose 2>&1 | tee "$OUTDIR/sweep.log"
