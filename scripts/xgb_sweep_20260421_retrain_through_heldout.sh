#!/usr/bin/env bash
# Sweep the retrain-through ensemble on held-out last-30d (task #114).
#
# Training: 2020-01-01 → 2026-03-20 (absorbs tariff crash start).
# OOS:      2026-03-21 → 2026-04-20 (last 30d, ~22 trading days).
#
# Window=5d stride=2d → ~9 windows over the OOS period (small N is
# fine — this is a regime-learnability probe, not a PnL estimate).
# Grid targets reasonable defaults + deployed values for reference.
set -euo pipefail
cd "$(dirname "$0")/.."
# shellcheck disable=SC1091
source .venv/bin/activate

OUT_DIR="analysis/xgbnew_daily/sweep_20260421_retrain_through_heldout"
mkdir -p "$OUT_DIR"

MODELS="analysis/xgbnew_daily/retrain_through_2026_03_20_ensemble/alltrain_seed0.pkl"
MODELS+=",analysis/xgbnew_daily/retrain_through_2026_03_20_ensemble/alltrain_seed7.pkl"
MODELS+=",analysis/xgbnew_daily/retrain_through_2026_03_20_ensemble/alltrain_seed42.pkl"
MODELS+=",analysis/xgbnew_daily/retrain_through_2026_03_20_ensemble/alltrain_seed73.pkl"
MODELS+=",analysis/xgbnew_daily/retrain_through_2026_03_20_ensemble/alltrain_seed197.pkl"

python -m xgbnew.sweep_ensemble_grid \
  --symbols-file symbol_lists/stocks_wide_1000_v1.txt \
  --model-paths "$MODELS" \
  --train-start 2020-01-01 --train-end 2026-03-20 \
  --oos-start 2026-03-21 --oos-end 2026-04-20 \
  --window-days 5 --stride-days 2 \
  --leverage-grid "1.0,1.5,2.0" \
  --min-score-grid "0.55,0.60,0.65,0.70,0.75,0.80,0.85" \
  --top-n-grid "1" \
  --hold-through \
  --fee-regimes "deploy,stress36x" \
  --min-dollar-vol 50000000 \
  --inference-min-vol-grid "0.10" \
  --output-dir "$OUT_DIR" \
  --verbose 2>&1 | tee "$OUT_DIR/stdout.log"
