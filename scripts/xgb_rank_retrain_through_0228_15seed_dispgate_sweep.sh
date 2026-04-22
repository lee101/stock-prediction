#!/usr/bin/env bash
# Dispgate variant: same 15-seed ranker @ retrain-through-0228, swept
# over regime_cs_iqr_max × regime_cs_skew_min. On fold 1 oos2025h1 the
# ranker dispgate stacked ~+5-9pp stress36x (see memory), so stacking it
# onto the retrain-through cutoff may clear the +27% bar.
set -euo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate

OUT_DIR="analysis/xgbnew_daily/sweep_20260422_xgbrank15_retrain_through_0228_dispgate"
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
  --regime-cs-iqr-max-grid "0.035,0.042,0.050,1.0" \
  --regime-cs-skew-min-grid "0.0,0.5,1.0,-1000000000" \
  --output-dir "$OUT_DIR" \
  --verbose 2>&1 | tee "$OUT_DIR/stdout.log"
