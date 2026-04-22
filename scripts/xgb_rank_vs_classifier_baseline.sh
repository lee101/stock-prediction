#!/usr/bin/env bash
# H2H: classifier-XGB (oos2025h1_ensemble_gpu_fresh) vs ranker-XGB
# (oos2025h1_ensemble_xgbrank). Same universe, window, hold-through, fees.
# The classifier uses ms=0.55-0.85 (conviction gate on P(up));
# the ranker uses ms=0.0 (fires every day because score is rank-percentile).
set -euo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate

OUT_DIR="analysis/xgbnew_daily/sweep_20260422_classifier_baseline_oos2025h1"
mkdir -p "$OUT_DIR"

MODELS=$(ls analysis/xgbnew_daily/oos2025h1_ensemble_gpu_fresh/alltrain_seed*.pkl | paste -sd,)

python -m xgbnew.sweep_ensemble_grid \
  --symbols-file symbol_lists/stocks_wide_1000_v1.txt \
  --model-paths "$MODELS" \
  --train-start 2020-01-01 --train-end 2025-06-30 \
  --oos-start 2025-07-01 --oos-end 2026-04-20 \
  --window-days 15 --stride-days 5 \
  --leverage-grid "1.0,2.0,3.0" \
  --min-score-grid "0.0,0.55,0.60,0.70,0.85" \
  --top-n-grid "1,2" \
  --hold-through \
  --fee-regimes "deploy,stress36x" \
  --min-dollar-vol 50000000 \
  --inference-min-vol-grid "0.10" \
  --output-dir "$OUT_DIR" \
  --verbose 2>&1 | tee "$OUT_DIR/stdout.log"
