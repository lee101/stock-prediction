#!/bin/bash
# Combined-LP sweep: top-K_mom ∪ top-K_xgb in a single CVaR LP.
# Hypothesis: a single LP over the union universe should match-or-beat the
# post-hoc blend champion (good +6.37, ann +88%, maxDD −20.65%).
#
# Grid mirrors phase-3 momentum + xgb top-K champions:
#   K_mom ∈ {25}        — phase-3 momentum sweet spot
#   K_xgb ∈ {15}        — xgb-topk wide champion
#   mom_lb ∈ {20}        — phase-3 lookback
#   ms ∈ {0.0}           — no min-score gate
#   ak ∈ {0.0, 0.005}    — alpha tilt scale (0=universe-only, 0.005=tilt+rank)
#   hd ∈ {3, 5}          — xgb-3d vs mom-5d hold
#   ats ∈ {10, 12}       — per-asset trailing stops
#   pts ∈ {0}            — apply post-hoc trailing stop separately
# = 1×1×1×1×2×2×2×1 = 8 cells.
#
# Each cell ~10-15 min CPU on the 1000-sym wide panel. Total ~1.5-2h.
set -euo pipefail
cd /nvme0n1-disk/code/stock-prediction
source .venv/bin/activate
exec env PYTHONPATH=. python -m scripts.sweep_combined_lp_universe \
  --symbols symbol_lists/stocks_wide_1000_v1.txt \
  --data-root trainingdata \
  --start 2023-06-01 --end 2026-04-18 \
  --max-symbols 1000 --min-avg-dol-vol 1000000 \
  --num-scen 1500 --fit-type gaussian --api cvxpy \
  --out analysis/cvar_portfolio/sweep_combined_lp_universe_34mo \
  --xgb-score-cache analysis/cvar_portfolio/xgb_panel_scores_wide_oos2025h1_fresh_2023start.parquet \
  --k-mom-grid 25 \
  --k-xgb-grid 15 \
  --mom-lookback-grid 20 \
  --min-score-grid 0.0 \
  --alpha-k-grid 0.0 0.005 \
  --w-max-grid 0.15 \
  --ltar-grid 1.0 \
  --risk-aversion-grid 1.0 \
  --confidence-grid 0.95 \
  --fee-bps 10 --slip-bps 5 \
  --hold-days-grid 3 5 \
  --per-asset-stop-loss-grid 15 \
  --portfolio-stop-loss-grid 0 \
  --per-asset-trailing-stop-grid 10 12 \
  --portfolio-trailing-stop-grid 0
