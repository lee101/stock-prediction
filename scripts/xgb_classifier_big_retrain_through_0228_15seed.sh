#!/usr/bin/env bash
# Scaled-up CLASSIFIER retrain-through-0228 with 15 seeds + include-dispersion
# + bigger trees + slower LR. The 5-seed classifier already clears +23.61%/mo
# stress36x on the 50d heldout at lev=2 ms=0.60 tn=2 (see memory). Testing
# if more compute + 15-seed bonferroni + wider feature set pushes this past
# the +27%/mo bar.
set -euo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate

OUT_DIR="analysis/xgbnew_daily/retrain_through_2026_02_28_ensemble_big"
mkdir -p "$OUT_DIR"

python -m xgbnew.train_alltrain_ensemble \
  --symbols-file symbol_lists/stocks_wide_1000_v1.txt \
  --data-root trainingdata \
  --train-start 2020-01-01 --train-end 2026-02-28 \
  --min-dollar-vol 0 \
  --seeds 0,1,2,3,4,5,6,7,8,9,10,11,42,73,197 \
  --n-estimators 1200 --max-depth 7 --learning-rate 0.02 \
  --include-dispersion \
  --device cuda \
  --out-dir "$OUT_DIR" \
  --verbose 2>&1 | tee "$OUT_DIR/train.log"
