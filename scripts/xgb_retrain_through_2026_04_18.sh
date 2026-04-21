#!/usr/bin/env bash
# Retrain-through with the most-recent practical cutoff (task: hourly monitor 2026-04-21 16:00 UTC).
# train: 2020-01-01 -> 2026-04-18 (includes full tariff crash Feb-Apr + the first few post-crash days)
# OOS:   2026-04-19 -> 2026-04-20 (weekend — only 1-2 trading days if any; probe only)
# Successor to 2026-04-05 retrain which showed 0 positive-median cells on 10-day heldout.
# Hypothesis: with more crash-era training data, model may calibrate closer to current regime.
set -euo pipefail
cd "$(dirname "$0")/.."
# shellcheck disable=SC1091
source .venv/bin/activate

python -m xgbnew.train_alltrain_ensemble \
  --symbols-file symbol_lists/stocks_wide_1000_v1.txt \
  --data-root trainingdata \
  --train-start 2020-01-01 \
  --train-end 2026-04-18 \
  --min-dollar-vol 0 \
  --seeds 0,7,42,73,197 \
  --n-estimators 400 --max-depth 5 --learning-rate 0.03 \
  --device cuda \
  --out-dir analysis/xgbnew_daily/retrain_through_2026_04_18_ensemble \
  --verbose
