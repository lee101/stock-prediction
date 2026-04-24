#!/bin/bash
# Targeted trailing-stop sweep around the goodness-winning neighborhood.
#
# Baseline winner (entry-based stops only): wmax=0.15 L_tar=2 ra=1 hd=10
# asl=15 gave med/mo +5.98%, w21dDD −26.56%, frac21>25 = 5.1%.
# The 21d DD still breaches the user's 25% comfort zone.
#
# Grid: wmax=0.15 × L_tar=2 × ra=1 × hd=10 × asl=15 ×
#       ats={0,8,12} × pts={0,10,15} = 9 cells (1 cached baseline, 8 new).
# Also test L_tar=3 at same other knobs to see if leverage + trailing
# buys meaningful extra median monthly.
set -euo pipefail
cd /nvme0n1-disk/code/stock-prediction
source .venv/bin/activate
exec env PYTHONPATH=. python -m cvar_portfolio.sweep_vanilla_hparam \
  --symbols symbol_lists/stocks_wide_1000_v1.txt \
  --data-root trainingdata \
  --start 2023-06-01 --end 2026-04-18 \
  --max-symbols 100 --min-avg-dol-vol 5000000 \
  --num-scen 1500 --fit-type gaussian --api cvxpy \
  --out analysis/cvar_portfolio/sweep_trailing_stop_v1 \
  --w-max-grid 0.15 \
  --ltar-grid 2.0 3.0 \
  --risk-aversion-grid 1.0 \
  --confidence-grid 0.95 \
  --auto-lever \
  --fee-bps 10 --slip-bps 5 \
  --hold-days-grid 10 \
  --per-asset-stop-loss-grid 15 \
  --portfolio-stop-loss-grid 0 \
  --per-asset-trailing-stop-grid 0 8 12 \
  --portfolio-trailing-stop-grid 0 10 15 \
  --sort-by goodness_score
