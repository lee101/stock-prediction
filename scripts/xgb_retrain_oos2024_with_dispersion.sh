#!/bin/bash
# Retrain oos2024-cutoff ensemble WITH cs_iqr/cs_skew as learned day-level
# features (in addition to the existing DAILY_FEATURE_COLS). Same 5 seeds
# as oos2024_ensemble_gpu_fresh so results are directly comparable.

set -euo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate

OUTDIR=analysis/xgbnew_daily/oos2024_ensemble_disp_gpu_fresh
mkdir -p "$OUTDIR"

python -m xgbnew.train_alltrain_ensemble \
  --symbols-file symbol_lists/stocks_wide_1000_v1.txt \
  --data-root trainingdata \
  --train-start 2020-01-01 --train-end 2024-12-31 \
  --seeds 0,7,42,73,197 \
  --n-estimators 400 --max-depth 5 --learning-rate 0.03 \
  --device cuda \
  --min-dollar-vol 5000000 \
  --include-dispersion \
  --out-dir "$OUTDIR" \
  --verbose 2>&1 | tee "$OUTDIR/train.log"
