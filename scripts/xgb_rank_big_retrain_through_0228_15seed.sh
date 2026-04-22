#!/usr/bin/env bash
# Scaled-up ranker: 3× trees, deeper, slower LR, include cross-sectional
# dispersion features. Same retrain-through-0228 cutoff. Tests whether
# more compute + wider feature set + crash-absorbing cutoff combines to
# clear the +27%/mo stress36x bar on the 50d held-out.
set -euo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate

OUT_DIR="analysis/xgbnew_daily/retrain_through_0228_ensemble_xgbrank_big"
mkdir -p "$OUT_DIR"

python -m xgbnew.train_ensemble_family \
  --family xgb_rank \
  --device cuda \
  --symbols-file symbol_lists/stocks_wide_1000_v1.txt \
  --train-start 2020-01-01 --train-end 2026-02-28 \
  --seeds 0,1,2,3,4,5,6,7,8,9,10,11,42,73,197 \
  --n-estimators 1200 --max-depth 7 --learning-rate 0.02 \
  --ranker-deciles 10 \
  --include-dispersion \
  --out-dir "$OUT_DIR" \
  --verbose 2>&1 | tee "$OUT_DIR/train.log"
