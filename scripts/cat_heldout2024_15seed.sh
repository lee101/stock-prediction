#!/usr/bin/env bash
# Train 15-seed CAT ensemble at fold 2 cutoff (train_end=2024-12-31).
# No existing CAT heldout2024 — this is fresh training. OOS window matches
# XGB's oos2024 fold: 2025-01-01 → 2026-04-20.
set -euo pipefail
cd "$(dirname "$0")/.."
# shellcheck disable=SC1091
source .venv/bin/activate

OUT_DIR="analysis/xgbnew_daily/heldout2024_cat"
mkdir -p "$OUT_DIR"

python -m xgbnew.train_ensemble_family \
  --family cat \
  --symbols-file symbol_lists/stocks_wide_1000_v1.txt \
  --data-root trainingdata \
  --train-start 2020-01-01 --train-end 2024-12-31 \
  --min-dollar-vol 0 \
  --seeds 0,1,2,3,4,5,6,7,8,9,10,11,42,73,197 \
  --n-estimators 400 --max-depth 5 --learning-rate 0.03 \
  --device cuda \
  --out-dir "$OUT_DIR" \
  2>&1 | tee -a "$OUT_DIR/train.log"
