#!/usr/bin/env bash
# Train 5-seed XGB ranker on oos2024 cutoff (2024-12-31). Second independent
# fold for validating the ranker's true-OOS edge.
set -euo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate

OUT_DIR="analysis/xgbnew_daily/oos2024_ensemble_xgbrank"
mkdir -p "$OUT_DIR"

python -m xgbnew.train_ensemble_family \
  --family xgb_rank \
  --device cuda \
  --symbols-file symbol_lists/stocks_wide_1000_v1.txt \
  --train-start 2020-01-01 --train-end 2024-12-31 \
  --seeds 0,7,42,73,197 \
  --n-estimators 400 \
  --max-depth 5 \
  --learning-rate 0.03 \
  --ranker-deciles 10 \
  --out-dir "$OUT_DIR" \
  --verbose 2>&1 | tee "$OUT_DIR/train.log"
