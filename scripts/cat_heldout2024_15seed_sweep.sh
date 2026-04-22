#!/usr/bin/env bash
# 15-seed bonferroni sweep on CAT heldout2024 fold (train_end=2024-12-31,
# OOS 2025-01-01 → 2026-04-20). Matches XGB oos2024 grid.
set -euo pipefail
cd "$(dirname "$0")/.."
# shellcheck disable=SC1091
source .venv/bin/activate

OUT_DIR="analysis/xgbnew_daily/sweep_20260422_cat_heldout2024_15seed"
mkdir -p "$OUT_DIR"

MODELS=$(ls analysis/xgbnew_daily/heldout2024_cat/alltrain_seed*.pkl | paste -sd,)

python -m xgbnew.sweep_ensemble_grid \
  --symbols-file symbol_lists/stocks_wide_1000_v1.txt \
  --model-paths "$MODELS" \
  --train-start 2020-01-01 --train-end 2024-12-31 \
  --oos-start 2025-01-01 --oos-end 2026-04-20 \
  --window-days 10 --stride-days 5 \
  --leverage-grid "1.0,1.5,2.0" \
  --min-score-grid "0.55,0.60,0.70,0.85" \
  --top-n-grid "1,2" \
  --hold-through \
  --fee-regimes "deploy,stress36x" \
  --min-dollar-vol 50000000 \
  --inference-min-vol-grid "0.10" \
  --output-dir "$OUT_DIR" \
  --verbose 2>&1 | tee "$OUT_DIR/stdout.log"
