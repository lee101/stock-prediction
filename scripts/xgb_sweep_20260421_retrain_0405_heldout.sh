#!/usr/bin/env bash
# Sweep retrain_through_2026_04_05 ensemble on the small (~8 window) held-out.
# train: 2020-01-01 → 2026-04-05, OOS: 2026-04-06 → 2026-04-17.
# Small N; this is a regime-absorption probe with MORE crash data than 0228.
set -euo pipefail
cd "$(dirname "$0")/.."
# shellcheck disable=SC1091
source .venv/bin/activate

OUT_DIR="analysis/xgbnew_daily/sweep_20260421_retrain_through_0405_heldout"
mkdir -p "$OUT_DIR"

MODELS="analysis/xgbnew_daily/retrain_through_2026_04_05_ensemble/alltrain_seed0.pkl"
MODELS+=",analysis/xgbnew_daily/retrain_through_2026_04_05_ensemble/alltrain_seed7.pkl"
MODELS+=",analysis/xgbnew_daily/retrain_through_2026_04_05_ensemble/alltrain_seed42.pkl"
MODELS+=",analysis/xgbnew_daily/retrain_through_2026_04_05_ensemble/alltrain_seed73.pkl"
MODELS+=",analysis/xgbnew_daily/retrain_through_2026_04_05_ensemble/alltrain_seed197.pkl"

python -m xgbnew.sweep_ensemble_grid \
  --symbols-file symbol_lists/stocks_wide_1000_v1.txt \
  --model-paths "$MODELS" \
  --train-start 2020-01-01 --train-end 2026-04-05 \
  --oos-start 2026-04-06 --oos-end 2026-04-17 \
  --window-days 3 --stride-days 2 \
  --leverage-grid "1.0,1.5,2.0" \
  --min-score-grid "0.55,0.60,0.65,0.70,0.75,0.80,0.85" \
  --top-n-grid "1" \
  --hold-through \
  --fee-regimes "deploy,stress36x" \
  --min-dollar-vol 50000000 \
  --inference-min-vol-grid "0.10" \
  --regime-cs-iqr-max-grid=0,0.042 \
  --regime-cs-skew-min-grid=-1000000000,1.0 \
  --output-dir "$OUT_DIR" \
  --verbose
