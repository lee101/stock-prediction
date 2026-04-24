#!/bin/bash
# Phase-3 CVaR wide-momentum refinement around phase-1 champion.
#
# Phase-1 winner (memory/project_cvar_wide_momentum_phase1_2026_04_24.md):
#   K=30 hd=5 L=1.0 ats=10 mom=20 = +5.01%/mo, w21dDD −19.88%, good +0.75.
# Phase-2 (memory/project_cvar_wide_momentum_phase2_2026_04_24.md):
#   hd=1 × K∈{5,10,20} REFUTED — fees burn 14-39% of gross; K<30 noisy.
#
# Phase-3 hypothesis: the phase-1 grid was coarse. Exploring:
#   - K ∈ {25, 30, 35, 40} (neighborhood of K=30)
#   - hd ∈ {4, 5, 6}       (neighborhood of hd=5)
#   - L ∈ {1.0, 1.2, 1.5}  (L=2 breached DD on wide panel, but 1.2-1.5 unexplored)
#   - ats ∈ {7, 8, 10, 12} (neighborhood of ats=10)
#   - mom_lb = 20          (phase-2 confirmed this is the sweet spot)
#   - pts = 0              (phase-1 showed pts=15% is inert)
# = 4 × 3 × 3 × 4 = 144 cells. ~4-5h on CPU. Kicked off in background.
#
# Goal: push median past +5.01%/mo toward +10%/mo while holding w21dDD ≥ −25%.
set -euo pipefail
cd /nvme0n1-disk/code/stock-prediction
source .venv/bin/activate
exec env PYTHONPATH=. python -m cvar_portfolio.sweep_wide_momentum \
  --symbols symbol_lists/stocks_wide_1000_v1.txt \
  --data-root trainingdata \
  --start 2023-06-01 --end 2026-04-18 \
  --max-symbols 1000 --min-avg-dol-vol 1000000 \
  --num-scen 1500 --fit-type gaussian --api cvxpy \
  --out analysis/cvar_portfolio/sweep_wide_momentum_phase3 \
  --w-max-grid 0.15 \
  --ltar-grid 1.0 1.2 1.5 \
  --risk-aversion-grid 1.0 \
  --confidence-grid 0.95 \
  --fee-bps 10 --slip-bps 5 \
  --hold-days-grid 4 5 6 \
  --per-asset-stop-loss-grid 15 \
  --portfolio-stop-loss-grid 0 \
  --per-asset-trailing-stop-grid 7 8 10 12 \
  --portfolio-trailing-stop-grid 0 \
  --topk-grid 25 30 35 40 \
  --momentum-lookback-grid 20 \
  --sort-by goodness_score
