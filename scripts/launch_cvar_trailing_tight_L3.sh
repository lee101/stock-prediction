#!/bin/bash
# Follow-up: tighter trailing stops for L_tar=3 after sweep_trailing_stop_v1
# showed pts=10 and ats=8 were too loose at higher leverage (all L=3
# cells breached user's 25% rolling 21d DD bar).
#
# Grid: wmax=0.15 × L=3 × ra=1 × hd=10 × asl=15 × ats={0,5,6} × pts={0,5,7}
# = 9 cells. Aim: +6-8%/mo median while keeping w21dDD > -25% and
# frac21>25 near zero.
set -euo pipefail
cd /nvme0n1-disk/code/stock-prediction
source .venv/bin/activate
exec env PYTHONPATH=. python -m cvar_portfolio.sweep_vanilla_hparam \
  --symbols symbol_lists/stocks_wide_1000_v1.txt \
  --data-root trainingdata \
  --start 2023-06-01 --end 2026-04-18 \
  --max-symbols 100 --min-avg-dol-vol 5000000 \
  --num-scen 1500 --fit-type gaussian --api cvxpy \
  --out analysis/cvar_portfolio/sweep_trailing_tight_L3 \
  --w-max-grid 0.15 \
  --ltar-grid 3.0 \
  --risk-aversion-grid 1.0 \
  --confidence-grid 0.95 \
  --auto-lever \
  --fee-bps 10 --slip-bps 5 \
  --hold-days-grid 10 \
  --per-asset-stop-loss-grid 15 \
  --portfolio-stop-loss-grid 0 \
  --per-asset-trailing-stop-grid 0 5 6 \
  --portfolio-trailing-stop-grid 0 5 7 \
  --sort-by goodness_score
