#!/usr/bin/env bash
# Add 10 seeds to existing 5-seed heldout2025h2_cat ensemble for clean
# 15-seed bonferroni. Tests whether CatBoost survives where XGB was
# refuted at 15-seed on both folds.
set -euo pipefail
cd "$(dirname "$0")/.."
# shellcheck disable=SC1091
source .venv/bin/activate

OUT_DIR="analysis/xgbnew_daily/heldout2025h2_cat"
mkdir -p "$OUT_DIR"

python -m xgbnew.train_ensemble_family \
  --family cat \
  --symbols-file symbol_lists/stocks_wide_1000_v1.txt \
  --data-root trainingdata \
  --train-start 2020-01-01 --train-end 2025-06-30 \
  --min-dollar-vol 0 \
  --seeds 1,2,3,4,5,6,8,9,10,11 \
  --n-estimators 400 --max-depth 5 --learning-rate 0.03 \
  --device cuda \
  --out-dir "$OUT_DIR" \
  2>&1 | tee -a "$OUT_DIR/train_extra10.log"
