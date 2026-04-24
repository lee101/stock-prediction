#!/bin/bash
# Phase-2 wide-momentum sweep: push toward user's +25-30%/mo target.
# Phase-1 (memory/project_cvar_wide_momentum_phase1_2026_04_24.md) found
# goodness-optimal = K=30 hd=5 L=1 ats=10 (+5.01%/mo, w21dDD -19.88%),
# raw-median-optimal = K=50 hd=3 L=1 no-stop (+5.93%/mo sortino 2.45).
# Both ~4-5× below target.
#
# Oracle on 807-sym × hd=1 × top-3 = +30.3%/mo. Closing the gap requires
# shorter hd + smaller K (more concentration) + better-tuned momentum.
#
# Phase-2 grid:
#   K ∈ {5, 10, 20}          # high concentration
#   hd ∈ {1, 3}              # daily rebal + keep hd=3 as anchor
#   L = 1.0                  # L=2 breaches DD on wider universe
#   momentum_lookback ∈ {5, 10, 20}  # shorter windows = more responsive signal
#   ats ∈ {0, 10}            # trailing on/off
#   pts = 0                  # phase-1 showed pts=15% is inert
# = 3 × 2 × 3 × 2 = 36 cells. ~70-90 min on CPU.
set -euo pipefail
cd /nvme0n1-disk/code/stock-prediction
source .venv/bin/activate
exec env PYTHONPATH=. python -m cvar_portfolio.sweep_wide_momentum \
  --symbols symbol_lists/stocks_wide_1000_v1.txt \
  --data-root trainingdata \
  --start 2023-06-01 --end 2026-04-18 \
  --max-symbols 1000 --min-avg-dol-vol 1000000 \
  --num-scen 1500 --fit-type gaussian --api cvxpy \
  --out analysis/cvar_portfolio/sweep_wide_momentum_phase2 \
  --w-max-grid 0.15 \
  --ltar-grid 1.0 \
  --risk-aversion-grid 1.0 \
  --confidence-grid 0.95 \
  --fee-bps 10 --slip-bps 5 \
  --hold-days-grid 1 3 \
  --per-asset-stop-loss-grid 15 \
  --portfolio-stop-loss-grid 0 \
  --per-asset-trailing-stop-grid 0 10 \
  --portfolio-trailing-stop-grid 0 \
  --topk-grid 5 10 20 \
  --momentum-lookback-grid 5 10 20 \
  --sort-by goodness_score
