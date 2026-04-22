#!/usr/bin/env bash
# Train 15-seed XGB ranker with goodness-weighted per-day training
# sample weights (weight = day's max |target_oc|, clipped at 5%).
# Same fold 1 cutoff (2020 → 2025-06-30) as the baseline 15-seed ranker
# so we can H2H against analysis/xgbnew_daily/oos2025h1_ensemble_xgbrank
# on the sweep grid.
set -euo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate

OUT_DIR="analysis/xgbnew_daily/oos2025h1_ensemble_xgbrank_gw"
mkdir -p "$OUT_DIR"

python -m xgbnew.train_ensemble_family \
  --family xgb_rank \
  --device cuda \
  --symbols-file symbol_lists/stocks_wide_1000_v1.txt \
  --train-start 2020-01-01 --train-end 2025-06-30 \
  --seeds 0,1,2,3,4,5,6,7,8,9,10,11,42,73,197 \
  --n-estimators 400 --max-depth 5 --learning-rate 0.03 \
  --ranker-deciles 10 \
  --ranker-sample-weight abs_target \
  --ranker-sample-weight-clip 0.05 \
  --out-dir "$OUT_DIR" \
  --verbose 2>&1 | tee "$OUT_DIR/train.log"
