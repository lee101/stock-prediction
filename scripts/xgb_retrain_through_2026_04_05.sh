#!/usr/bin/env bash
# Retrain-through with latest practical cutoff (task: hourly monitor 2026-04-21).
# train: 2020-01-01 → 2026-04-05 (includes full tariff crash Feb-Apr)
# OOS:   2026-04-06 → 2026-04-17 (~9 trading days, ~4 stride-2 windows)
# Small held-out N — this is a regime-absorption probe.
set -euo pipefail
cd "$(dirname "$0")/.."
# shellcheck disable=SC1091
source .venv/bin/activate

python -m xgbnew.train_alltrain_ensemble \
  --symbols-file symbol_lists/stocks_wide_1000_v1.txt \
  --data-root trainingdata \
  --train-start 2020-01-01 \
  --train-end 2026-04-05 \
  --min-dollar-vol 0 \
  --seeds 0,7,42,73,197 \
  --n-estimators 400 --max-depth 5 --learning-rate 0.03 \
  --device cuda \
  --out-dir analysis/xgbnew_daily/retrain_through_2026_04_05_ensemble \
  --verbose
