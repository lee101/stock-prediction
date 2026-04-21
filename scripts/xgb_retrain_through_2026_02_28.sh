#!/usr/bin/env bash
# Retrain-through with earlier cutoff for larger held-out OOS.
# train: 2020-01-01 → 2026-02-28 (absorbs early-Feb tariff crash entry)
# OOS:   2026-03-01 → 2026-04-20 (~35 trading days, ~15 stride-2 windows)
set -euo pipefail
cd "$(dirname "$0")/.."
# shellcheck disable=SC1091
source .venv/bin/activate

python -m xgbnew.train_alltrain_ensemble \
  --symbols-file symbol_lists/stocks_wide_1000_v1.txt \
  --data-root trainingdata \
  --train-start 2020-01-01 \
  --train-end 2026-02-28 \
  --min-dollar-vol 0 \
  --seeds 0,7,42,73,197 \
  --n-estimators 400 --max-depth 5 --learning-rate 0.03 \
  --device cuda \
  --out-dir analysis/xgbnew_daily/retrain_through_2026_02_28_ensemble \
  --verbose
