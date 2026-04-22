#!/usr/bin/env bash
# Sweep 15-seed XGB ranker (retrain-through 2026-02-28) on held-out
# 2026-03-01 → 2026-04-20 (50 days, ~10 stride-5 windows). Compares the
# same seed set × cutoff with classifier retrain-through-0228 at
# sweep_20260421_retrain_through_0228_heldout.
set -euo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate

OUT_DIR="analysis/xgbnew_daily/sweep_20260422_xgbrank15_retrain_through_0228_heldout"
mkdir -p "$OUT_DIR"

MODELS=$(ls analysis/xgbnew_daily/retrain_through_0228_ensemble_xgbrank/alltrain_seed*.pkl | paste -sd,)

python -m xgbnew.sweep_ensemble_grid \
  --symbols-file symbol_lists/stocks_wide_1000_v1.txt \
  --model-paths "$MODELS" \
  --train-start 2020-01-01 --train-end 2026-02-28 \
  --oos-start 2026-03-01 --oos-end 2026-04-20 \
  --window-days 15 --stride-days 5 \
  --leverage-grid "1.0,1.5,2.0" \
  --min-score-grid "0.0" \
  --top-n-grid "1,2" \
  --hold-through \
  --fee-regimes "deploy,stress36x" \
  --min-dollar-vol 50000000 \
  --inference-min-vol-grid "0.10" \
  --output-dir "$OUT_DIR" \
  --verbose 2>&1 | tee "$OUT_DIR/stdout.log"
