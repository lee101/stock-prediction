#!/bin/bash
# Cross-sectional momentum-rank filter sweep on fresh 2025H1-cutoff ensemble.
#
# Hypothesis from scripts/xgb_diagnose_regime_inversion.py counterfactual:
#   - NO FILTER:                          top-1/day mean target_oc = −0.18%/day
#   - drop top-25% ret_20d:               +0.00%/day  (hot names underperform)
#   - drop bottom-25% ret_5d:             +0.12%/day  (weak-recent underperform)
#   - drop BOTH extremes:                 +0.25%/day  (stacked)
#
# This sweep runs the full windowed backtest (30d/5d stride) on the fresh
# held-out ensemble at the current best-cell config:
#   lev=2.0 ms=0.60 (fresh ensemble caps ~0.67), hold_through=1,
#   min_vol_20d=0.10, min_dolvol=50M
# and sweeps the two rank-pct axes Cartesian-style:
#   max_ret_20d_rank_pct ∈ {1.0, 0.75, 0.50, 0.25}
#   min_ret_5d_rank_pct  ∈ {0.0, 0.25, 0.50, 0.75}
# → 4 × 4 × 2 fee_regimes = 32 cells.
#
# Goal: find a momentum-rank combo where TRUE-OOS windowed median ≥ 0 and
# neg-rate < 13/38 (current baseline on no-filter fresh ensemble).

set -euo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate

OUTDIR=analysis/xgbnew_daily/sweep_20260421_momentum_ranks
mkdir -p "$OUTDIR"

python -m xgbnew.sweep_ensemble_grid \
  --symbols-file symbol_lists/stocks_wide_1000_v1.txt \
  --data-root trainingdata \
  --model-paths analysis/xgbnew_daily/oos2025h1_ensemble_gpu_fresh/alltrain_seed0.pkl,\
analysis/xgbnew_daily/oos2025h1_ensemble_gpu_fresh/alltrain_seed7.pkl,\
analysis/xgbnew_daily/oos2025h1_ensemble_gpu_fresh/alltrain_seed42.pkl,\
analysis/xgbnew_daily/oos2025h1_ensemble_gpu_fresh/alltrain_seed73.pkl,\
analysis/xgbnew_daily/oos2025h1_ensemble_gpu_fresh/alltrain_seed197.pkl \
  --train-start 2020-01-01 --train-end 2025-06-30 \
  --oos-start   2025-07-01 --oos-end   2026-04-20 \
  --window-days 30 --stride-days 5 \
  --leverage-grid 2.0 \
  --min-score-grid 0.60 \
  --top-n-grid 1 \
  --hold-through \
  --fee-regimes deploy,stress36x \
  --min-dollar-vol 50000000 \
  --inference-min-vol-grid 0.10 \
  --max-ret-20d-rank-pct-grid 1.0,0.75,0.50,0.25 \
  --min-ret-5d-rank-pct-grid  0.0,0.25,0.50,0.75 \
  --output-dir "$OUTDIR" \
  --verbose 2>&1 | tee "$OUTDIR/sweep.log"
