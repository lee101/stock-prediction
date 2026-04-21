#!/usr/bin/env bash
# Invert-scores diagnostic: does the fresh held-out XGB ensemble
# rank-order OOS winners vs losers, or is it noise on this regime?
#
# Method: replace blended scores with 1 − scores so the sim picks the
# BOTTOM-N (originally worst-rated) symbols. Long-only sim is still
# used. Interpretation:
#   - positive median → model rank-orders correctly → proper short-side
#     sim with borrow costs etc. would profit on these names.
#   - negative median in BOTH regular and inverted sweeps → the ensemble
#     has no directional edge; no short-side lever to chase.
#
# Scores on inverted distribution cap around 0.5 (originally 0.5 becomes
# 0.5; originally 0.9 becomes 0.1). So we sweep low ms values.
set -euo pipefail
cd "$(dirname "$0")/.."
# shellcheck disable=SC1091
source .venv/bin/activate

OUT_DIR="analysis/xgbnew_daily/sweep_20260421_invert_diag"
mkdir -p "$OUT_DIR"

MODELS="analysis/xgbnew_daily/oos2025h1_ensemble_gpu_fresh/alltrain_seed0.pkl"
MODELS+=",analysis/xgbnew_daily/oos2025h1_ensemble_gpu_fresh/alltrain_seed7.pkl"
MODELS+=",analysis/xgbnew_daily/oos2025h1_ensemble_gpu_fresh/alltrain_seed42.pkl"
MODELS+=",analysis/xgbnew_daily/oos2025h1_ensemble_gpu_fresh/alltrain_seed73.pkl"
MODELS+=",analysis/xgbnew_daily/oos2025h1_ensemble_gpu_fresh/alltrain_seed197.pkl"

# Small grid — we just want to see the SIGN of median, across a few
# operating points. If the sign is wrong (negative), we stop here.
#   lev {1.0, 2.0} × ms {0.0, 0.50, 0.55} × top_n {1} × fee {deploy}
# = 6 cells.
python -m xgbnew.sweep_ensemble_grid \
  --symbols-file symbol_lists/stocks_wide_1000_v1.txt \
  --model-paths "$MODELS" \
  --train-start 2020-01-01 --train-end 2025-06-30 \
  --oos-start 2025-07-01 --oos-end 2026-04-20 \
  --window-days 14 --stride-days 5 \
  --leverage-grid "1.0,2.0" \
  --min-score-grid "0.0,0.50,0.55" \
  --top-n-grid "1" \
  --hold-through \
  --fee-regimes "deploy" \
  --min-dollar-vol 50000000 \
  --inference-min-vol-grid "0.10" \
  --invert-scores \
  --output-dir "$OUT_DIR" \
  --verbose 2>&1 | tee "$OUT_DIR/stdout.log"
