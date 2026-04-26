#!/usr/bin/env bash
# Dense ms × vol × lev grid around deployed live config, on the frozen
# 2024-cutoff TRUE-OOS ensemble. Deployed = lev=2.0, ms=0.85, vol=0.12.
# Goal: find strict-dominance cells (better on every metric) for review.
set -euo pipefail
cd "$(dirname "$0")/.."
# shellcheck disable=SC1091
source .venv/bin/activate

OUT_DIR="analysis/xgbnew_daily/sweep_20260421_frontier_refine"
mkdir -p "$OUT_DIR"

MODELS="analysis/xgbnew_daily/oos2024_ensemble_gpu/alltrain_seed0.pkl"
MODELS+=",analysis/xgbnew_daily/oos2024_ensemble_gpu/alltrain_seed7.pkl"
MODELS+=",analysis/xgbnew_daily/oos2024_ensemble_gpu/alltrain_seed42.pkl"
MODELS+=",analysis/xgbnew_daily/oos2024_ensemble_gpu/alltrain_seed73.pkl"
MODELS+=",analysis/xgbnew_daily/oos2024_ensemble_gpu/alltrain_seed197.pkl"

python -m xgbnew.sweep_ensemble_grid \
  --symbols-file symbol_lists/stocks_wide_1000_v1.txt \
  --model-paths "$MODELS" \
  --train-start 2020-01-01 --train-end 2024-12-31 \
  --oos-start 2025-01-01 --oos-end 2026-04-20 \
  --window-days 14 --stride-days 7 \
  --leverage-grid "1.75,2.0,2.25,2.5" \
  --min-score-grid "0.82,0.84,0.85,0.86,0.87,0.88,0.90" \
  --top-n-grid "1" \
  --hold-through \
  --fee-regimes "deploy,stress36x" \
  --min-dollar-vol 50000000 \
  --inference-min-vol-grid "0.10,0.11,0.12,0.13,0.14" \
  --output-dir "$OUT_DIR" \
  --verbose 2>&1 | tee "$OUT_DIR/stdout.log"
