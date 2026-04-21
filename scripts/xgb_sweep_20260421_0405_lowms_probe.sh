#!/usr/bin/env bash
# Probe: on retrain_through_2026_04_05 ensemble, do we find ANY alpha at lower min_score gates?
# Previous sweep (ms in 0.55-0.85) showed 0 pos-med cells on 10-day heldout.
# This extends down to ms=0.30,0.35,0.40,0.45,0.50 and also up top-n to 2,3 to test diversification.
set -euo pipefail
cd "$(dirname "$0")/.."
# shellcheck disable=SC1091
source .venv/bin/activate

OUT_DIR="analysis/xgbnew_daily/sweep_20260421_retrain_through_0405_lowms"
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
  --min-score-grid "0.30,0.35,0.40,0.45,0.50,0.55,0.60" \
  --top-n-grid "1,2,3" \
  --hold-through \
  --fee-regimes "deploy,stress36x" \
  --min-dollar-vol 50000000 \
  --inference-min-vol-grid "0.10" \
  --output-dir "$OUT_DIR" \
  --verbose
