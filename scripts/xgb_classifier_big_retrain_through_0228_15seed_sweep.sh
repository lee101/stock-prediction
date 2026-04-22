#!/usr/bin/env bash
# Sweep BIG 15-seed classifier (retrain-through 2026-02-28, n=1200 d=7 +disp)
# on heldout 2026-03-01 → 2026-04-20. Baseline 5-seed retrain-through-0228
# classifier hit +23.61%/mo stress36x at lev=2 ms=0.60 tn=2 (below +27% bar).
# Tests if bigger trees + wider features + 15-seed bonferroni push it over.
set -euo pipefail
cd "$(dirname "$0")/.."
# shellcheck disable=SC1091
source .venv/bin/activate

OUT_DIR="analysis/xgbnew_daily/sweep_20260422_classifier_big_retrain_through_0228_w5s2"
mkdir -p "$OUT_DIR"

MODELS=$(ls analysis/xgbnew_daily/retrain_through_2026_02_28_ensemble_big/alltrain_seed*.pkl | paste -sd,)

python -m xgbnew.sweep_ensemble_grid \
  --symbols-file symbol_lists/stocks_wide_1000_v1.txt \
  --model-paths "$MODELS" \
  --train-start 2020-01-01 --train-end 2026-02-28 \
  --oos-start 2026-03-01 --oos-end 2026-04-20 \
  --window-days 5 --stride-days 2 \
  --leverage-grid "1.0,1.5,2.0" \
  --min-score-grid "0.55,0.60,0.65,0.70,0.75,0.80,0.85" \
  --top-n-grid "1,2" \
  --hold-through \
  --fee-regimes "deploy,stress36x" \
  --min-dollar-vol 50000000 \
  --inference-min-vol-grid "0.10" \
  --output-dir "$OUT_DIR" \
  --verbose 2>&1 | tee "$OUT_DIR/stdout.log"
