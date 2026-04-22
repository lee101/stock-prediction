#!/usr/bin/env bash
# Train 15-seed XGB ranker with train_end=2026-02-28 (absorbs early-Feb
# tariff-crash entry). Evaluated on held-out 2026-03-01 → 2026-04-20 (50d).
# Classifier on this cutoff hit +23.61%/mo median on this heldout
# (see project_xgb_retrain_through_0228_first_signal.md). If the ranker's
# structural advantage compounds with retrain-through, this may clear
# the +27%/mo stress36x bar.
set -euo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate

OUT_DIR="analysis/xgbnew_daily/retrain_through_0228_ensemble_xgbrank"
mkdir -p "$OUT_DIR"

python -m xgbnew.train_ensemble_family \
  --family xgb_rank \
  --device cuda \
  --symbols-file symbol_lists/stocks_wide_1000_v1.txt \
  --train-start 2020-01-01 --train-end 2026-02-28 \
  --seeds 0,1,2,3,4,5,6,7,8,9,10,11,42,73,197 \
  --n-estimators 400 --max-depth 5 --learning-rate 0.03 \
  --ranker-deciles 10 \
  --out-dir "$OUT_DIR" \
  --verbose 2>&1 | tee "$OUT_DIR/train.log"
