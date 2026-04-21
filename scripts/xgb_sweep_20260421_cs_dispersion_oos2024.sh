#!/bin/bash
# Cross-fold robustness check: same regime-dispersion gate thresholds
# against the fresh oos2024 ensemble (train 2020 → 2024-12-31, OOS
# 2025-01-01 → 2026-04-20, 15 months including both pre-tariff calm
# and tariff crash). If the sweet-spot cells (iqr=0.042, skew=1.0)
# carry across a different training cutoff, the edge is structural.
# If only the 2025H1 fold shows it, it's likely an overfit to the
# pack of crash-window days specifically.

set -euo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate

OUTDIR=analysis/xgbnew_daily/sweep_20260421_cs_dispersion_oos2024
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
  --leverage-grid 2.0 \
  --min-score-grid 0.60 \
  --top-n-grid 1 \
  --hold-through \
  --fee-regimes deploy,stress36x \
  --min-dollar-vol 50000000 \
  --inference-min-vol-grid 0.10 \
  --regime-cs-iqr-max-grid 0.0,0.040,0.042,0.045,0.048,0.050 \
  --regime-cs-skew-min-grid=-1e9,0.0,0.5,1.0 \
  --output-dir "$OUTDIR" \
  --verbose 2>&1 | tee "$OUTDIR/sweep.log"
