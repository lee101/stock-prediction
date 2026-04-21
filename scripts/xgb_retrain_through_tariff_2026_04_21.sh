#!/usr/bin/env bash
# Retrain-through experiment (2026-04-21, task #114).
#
# Five inference-side levers have failed on true-OOS (2025-07 → 2026-04).
# The OOS is regime-INVERTED not noise. Next architectural axis: train
# ensemble through 2026-03-20 to absorb the tariff crash, hold out the
# last 30 calendar days as true-OOS. If sweep lands a positive-median
# cell with neg ≤10%, monthly retrain is the right cadence. If still
# negative, the tariff regime is not learnable from history.
#
# Same 5-seed champion hyperparameters. Output:
#   analysis/xgbnew_daily/retrain_through_2026_03_20_ensemble/
set -euo pipefail
cd "$(dirname "$0")/.."
# shellcheck disable=SC1091
source .venv/bin/activate

SYMS="symbol_lists/stocks_wide_1000_v1.txt"
SEEDS="0,7,42,73,197"
TRAIN_END="2026-03-20"
OUT="analysis/xgbnew_daily/retrain_through_2026_03_20_ensemble"

echo "[$(date -u +%H:%M:%SZ)] training retrain-through (train_end=${TRAIN_END}) -> ${OUT}"
python -m xgbnew.train_alltrain_ensemble \
  --symbols-file "$SYMS" \
  --data-root trainingdata \
  --train-start 2020-01-01 \
  --train-end "$TRAIN_END" \
  --min-dollar-vol 0 \
  --seeds "$SEEDS" \
  --n-estimators 400 --max-depth 5 --learning-rate 0.03 \
  --device cuda \
  --out-dir "$OUT" \
  --verbose

echo "[$(date -u +%H:%M:%SZ)] retrain-through ensemble trained: ${OUT}"
