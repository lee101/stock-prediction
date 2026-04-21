#!/bin/bash
# Stack cs-dispersion features + cs-dispersion gate on oos2024.
# Tests: does gating 'bad' days on top of a dispersion-aware model
# compound the edge? Or does the model already internalize it?
set -euo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate

OUTDIR=analysis/xgbnew_daily/sweep_20260421_disp_plus_gate
mkdir -p "$OUTDIR"
ENS=oos2024_ensemble_disp_gpu_fresh

python -m xgbnew.sweep_ensemble_grid \
  --symbols-file symbol_lists/stocks_wide_1000_v1.txt \
  --data-root trainingdata \
  --model-paths analysis/xgbnew_daily/${ENS}/alltrain_seed0.pkl,\
analysis/xgbnew_daily/${ENS}/alltrain_seed7.pkl,\
analysis/xgbnew_daily/${ENS}/alltrain_seed42.pkl,\
analysis/xgbnew_daily/${ENS}/alltrain_seed73.pkl,\
analysis/xgbnew_daily/${ENS}/alltrain_seed197.pkl \
  --train-start 2020-01-01 --train-end 2024-12-31 \
  --oos-start   2025-01-01 --oos-end   2026-04-20 \
  --window-days 30 --stride-days 5 \
  --leverage-grid 1.0,1.5,2.0 \
  --min-score-grid 0.60,0.70 \
  --top-n-grid 1 \
  --hold-through \
  --fee-regimes deploy,stress36x \
  --min-dollar-vol 50000000 \
  --inference-min-vol-grid 0.10 \
  --regime-cs-iqr-max-grid 0.0,0.042 \
  --regime-cs-skew-min-grid=-1e9,1.0 \
  --output-dir "$OUTDIR" \
  --verbose 2>&1 | tee "$OUTDIR/sweep.log"
