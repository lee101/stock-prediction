#!/usr/bin/env bash
# Sweep 15-seed MuonMLP (cutoff 2026-02-28) on held-out 2026-03-01 → 2026-04-20.
# Matches the XGB retrain-through-0228 grid for head-to-head comparison.
# Includes dispersion gate knobs since that was the best lever for XGB.
set -euo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate

OUT_DIR="analysis/xgbnew_daily/sweep_20260422_retrain_through_0228_mlp_muon"
mkdir -p "$OUT_DIR"

MODELS=$(ls analysis/xgbnew_daily/retrain_through_2026_02_28_mlp_muon/alltrain_seed*.pkl | paste -sd,)
N=$(ls analysis/xgbnew_daily/retrain_through_2026_02_28_mlp_muon/alltrain_seed*.pkl | wc -l)
echo "[retrain_through_0228_mlp_muon] N=$N models -> $OUT_DIR"

python -m xgbnew.sweep_ensemble_grid \
  --symbols-file symbol_lists/stocks_wide_1000_v1.txt \
  --model-paths "$MODELS" \
  --train-start 2020-01-01 --train-end 2026-02-28 \
  --oos-start 2026-03-01 --oos-end 2026-04-20 \
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
