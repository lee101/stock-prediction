#!/usr/bin/env bash
# Clean 15-seed bonferroni on PLAIN retrain-through-0228 classifier (n=400 d=5).
# Same architecture as the +23.61%/mo 5-seed baseline — just adds 10 seeds.
# If this collapses too, 5-seed {0,7,42,73,197} tail-bias is confirmed for
# the classifier family (was already shown for MuonMLP and ranker).
set -euo pipefail
cd "$(dirname "$0")/.."
# shellcheck disable=SC1091
source .venv/bin/activate

OUT_DIR="analysis/xgbnew_daily/sweep_20260422_classifier_plain_retrain_through_0228_15seed"
mkdir -p "$OUT_DIR"

MODELS=$(ls analysis/xgbnew_daily/retrain_through_2026_02_28_ensemble/alltrain_seed*.pkl | paste -sd,)

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
