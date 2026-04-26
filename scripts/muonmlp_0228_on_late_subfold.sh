#!/usr/bin/env bash
# Same 0228-cutoff MuonMLP ensemble, but evaluated ONLY on the last 22-day
# subfold 2026-03-21→04-20 (the same window where 0320-cutoff failed with
# 0/432 pos-median). Isolates whether 0228's +29.71 stress edge lives in
# the first 20 days (2026-03-01→03-20) or is a model-wide property.
set -euo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate

OUT_DIR="analysis/xgbnew_daily/sweep_20260422_rt0228_on_late_subfold"
mkdir -p "$OUT_DIR"

MODELS=$(ls analysis/xgbnew_daily/retrain_through_2026_02_28_mlp_muon/alltrain_seed*.pkl | paste -sd,)
echo "[rt0228_late_subfold] N=15 -> $OUT_DIR"

python -m xgbnew.sweep_ensemble_grid \
  --symbols-file symbol_lists/stocks_wide_1000_v1.txt \
  --model-paths "$MODELS" \
  --train-start 2020-01-01 --train-end 2026-02-28 \
  --oos-start 2026-03-21 --oos-end 2026-04-20 \
  --window-days 5 --stride-days 2 \
  --leverage-grid "1.0,1.5,2.0,3.0" \
  --min-score-grid "0.55,0.60,0.65" \
  --top-n-grid "1,2" \
  --hold-through \
  --fee-regimes "deploy,stress36x" \
  --min-dollar-vol 50000000 \
  --inference-min-vol-grid "0.10" \
  --regime-cs-iqr-max-grid=0,0.042,0.06 \
  --regime-cs-skew-min-grid=-1000000000,0.0,1.0 \
  --output-dir "$OUT_DIR" \
  --verbose 2>&1 | tee "$OUT_DIR/stdout.log"
