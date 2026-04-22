#!/usr/bin/env bash
# Sweep 5-seed XGB ranker (oos2024 cutoff 2024-12-31) on true-OOS
# 2025-01-01 → 2026-04-20 to validate the ranker's edge on a second fold.
set -euo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate

OUT_DIR="analysis/xgbnew_daily/sweep_20260422_xgbrank_oos2024"
mkdir -p "$OUT_DIR"

MODELS=$(ls analysis/xgbnew_daily/oos2024_ensemble_xgbrank/alltrain_seed*.pkl | paste -sd,)

python -m xgbnew.sweep_ensemble_grid \
  --symbols-file symbol_lists/stocks_wide_1000_v1.txt \
  --model-paths "$MODELS" \
  --train-start 2020-01-01 --train-end 2024-12-31 \
  --oos-start 2025-01-01 --oos-end 2026-04-20 \
  --window-days 15 --stride-days 5 \
  --leverage-grid "1.0,2.0,3.0" \
  --min-score-grid "0.0" \
  --top-n-grid "1,2,3" \
  --hold-through \
  --fee-regimes "deploy,stress36x" \
  --min-dollar-vol 50000000 \
  --inference-min-vol-grid "0.10" \
  --output-dir "$OUT_DIR" \
  --verbose 2>&1 | tee "$OUT_DIR/stdout.log"
