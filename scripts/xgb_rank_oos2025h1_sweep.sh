#!/usr/bin/env bash
# Sweep 5-seed XGB ranker on true-OOS 2025-07 → 2026-04-20.
# Same symbols, same window, same grid as the classification XGB baseline
# at oos2025h1_ensemble_gpu_fresh so we get a direct H2H.
set -euo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate

OUT_DIR="analysis/xgbnew_daily/sweep_20260422_xgbrank_oos2025h1"
mkdir -p "$OUT_DIR"

MODELS=$(ls analysis/xgbnew_daily/oos2025h1_ensemble_xgbrank/alltrain_seed*.pkl | paste -sd,)
echo "[xgbrank_oos2025h1] -> $OUT_DIR"

# Ranker output is already rank-percentile normalized in [0,1], so ms thresholds
# mean "top-X% percentile pick" not "P(up)>X". Scan a useful range.
python -m xgbnew.sweep_ensemble_grid \
  --symbols-file symbol_lists/stocks_wide_1000_v1.txt \
  --model-paths "$MODELS" \
  --train-start 2020-01-01 --train-end 2025-06-30 \
  --oos-start 2025-07-01 --oos-end 2026-04-20 \
  --window-days 15 --stride-days 5 \
  --leverage-grid "1.0,2.0,3.0" \
  --min-score-grid "0.0,0.95,0.98,0.99,0.995" \
  --top-n-grid "1,2" \
  --hold-through \
  --fee-regimes "deploy,stress36x" \
  --min-dollar-vol 50000000 \
  --inference-min-vol-grid "0.10" \
  --output-dir "$OUT_DIR" \
  --verbose 2>&1 | tee "$OUT_DIR/stdout.log"
