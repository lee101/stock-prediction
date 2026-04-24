#!/bin/bash
# Daily-rebal small-K XGB sweep on 34mo window — push for headline-median lift.
#
# Panel ceiling scan suggested +25-30%/mo achievable only via 800-sym
# universe + hd=1-3 + top-3-5. Current 34mo blend champion is +6.53%/mo at
# L=1.25 — 4× below target. This sweep tests if small-K + daily/2-day
# rebalance on the wide panel can lift the headline median.
#
# Grid:
#   K   ∈ {3, 5, 10}     (smaller than current 15-25 frontier)
#   hd  ∈ {1, 2, 3}      (1=daily, 2=alt-day, 3=current sweet spot)
#   ats ∈ {8, 10}
#   ms  ∈ {0.0, 0.55}    (filter weak picks at hd=1 to reduce noise+fees)
# = 3 × 3 × 2 × 2 = 36 cells. ~50 min on CPU (smaller universe-size, faster solves).
set -euo pipefail
cd /nvme0n1-disk/code/stock-prediction
source .venv/bin/activate
exec env PYTHONPATH=. python -m cvar_portfolio.sweep_wide_xgb \
  --symbols symbol_lists/stocks_wide_1000_v1.txt \
  --data-root trainingdata \
  --start 2023-06-01 --end 2026-04-18 \
  --max-symbols 1000 --min-avg-dol-vol 1000000 \
  --num-scen 1500 --fit-type gaussian --api cvxpy \
  --out analysis/cvar_portfolio/sweep_wide_xgb_34mo_smallk \
  --xgb-score-cache analysis/cvar_portfolio/xgb_panel_scores_wide_oos2025h1_fresh_2023start.parquet \
  --w-max-grid 0.15 \
  --ltar-grid 1.0 \
  --risk-aversion-grid 1.0 \
  --confidence-grid 0.95 \
  --fee-bps 10 --slip-bps 5 \
  --hold-days-grid 1 2 3 \
  --per-asset-stop-loss-grid 15 \
  --portfolio-stop-loss-grid 0 \
  --per-asset-trailing-stop-grid 8 10 \
  --portfolio-trailing-stop-grid 0 \
  --topk-grid 3 5 10 \
  --min-score-grid 0.0 0.55 \
  --sort-by goodness_score
