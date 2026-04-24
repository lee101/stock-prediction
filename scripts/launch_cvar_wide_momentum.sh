#!/bin/bash
# Wider-universe momentum-top-K CVaR sweep to break past the 100-sym oracle
# ceiling. See memory/project_cvar_panel_physical_ceiling_2026_04_24.md:
# 807-sym × hd=1 × top-3 oracle reaches +30.32%/mo (−37.7% DD), while
# 100-sym LP champion caps at +5.98%/mo. Even with some alpha erosion vs
# oracle, wider + shorter horizon should unlock +10-20%/mo territory.
#
# Phase-1 orientation grid (conservative, just to locate the frontier):
#   universe = top-K by past-20d log-return (recomputed each rebalance)
#   K ∈ {20, 30, 50}
#   hd ∈ {3, 5}  (daily rebalance is expensive; save for phase 2)
#   L_tar ∈ {1, 2}  (unleveraged + mild leverage)
#   w_max = 0.15 (single-name cap)
#   ats ∈ {0, 10}  (per-asset trailing 10%)
#   pts ∈ {0, 15}  (portfolio trailing 15%)
# = 3 × 2 × 2 × 2 × 2 = 48 cells. Each solves LP on K≤50 names so is fast.
set -euo pipefail
cd /nvme0n1-disk/code/stock-prediction
source .venv/bin/activate
exec env PYTHONPATH=. python -m cvar_portfolio.sweep_wide_momentum \
  --symbols symbol_lists/stocks_wide_1000_v1.txt \
  --data-root trainingdata \
  --start 2023-06-01 --end 2026-04-18 \
  --max-symbols 1000 --min-avg-dol-vol 1000000 \
  --num-scen 1500 --fit-type gaussian --api cvxpy \
  --out analysis/cvar_portfolio/sweep_wide_momentum \
  --w-max-grid 0.15 \
  --ltar-grid 1.0 2.0 \
  --risk-aversion-grid 1.0 \
  --confidence-grid 0.95 \
  --auto-lever \
  --fee-bps 10 --slip-bps 5 \
  --hold-days-grid 3 5 \
  --per-asset-stop-loss-grid 15 \
  --portfolio-stop-loss-grid 0 \
  --per-asset-trailing-stop-grid 0 10 \
  --portfolio-trailing-stop-grid 0 15 \
  --topk-grid 20 30 50 \
  --momentum-lookback-grid 20 \
  --sort-by goodness_score
