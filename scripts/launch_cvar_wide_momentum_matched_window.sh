#!/bin/bash
# Matched-window momentum baseline for head-to-head vs XGB-top-K sweep.
#
# XGB sweep ran on 2024-07-01 → 2026-04-18 (22mo). Phase-3 momentum champion
# was measured on 2023-06-01 → 2026-04-18 (34mo). To compare apples-to-apples,
# re-run momentum phase-3 champion neighborhood on the same 22mo window.
#
# Phase-3 momentum champion: K=25 hd=5 L=1.0 ats=12 mom_lb=20 = +6.11%/mo.
# This matched-window sweep covers K ∈ {15, 25, 35} × hd ∈ {3, 5, 10}
# × ats ∈ {10, 12} = 18 cells so we can compare XGB-top-K cells directly.
set -euo pipefail
cd /nvme0n1-disk/code/stock-prediction
source .venv/bin/activate
exec env PYTHONPATH=. python -m cvar_portfolio.sweep_wide_momentum \
  --symbols symbol_lists/stocks_wide_1000_v1.txt \
  --data-root trainingdata \
  --start 2024-07-01 --end 2026-04-18 \
  --max-symbols 1000 --min-avg-dol-vol 1000000 \
  --num-scen 1500 --fit-type gaussian --api cvxpy \
  --out analysis/cvar_portfolio/sweep_wide_momentum_matched_2024_07 \
  --w-max-grid 0.15 \
  --ltar-grid 1.0 \
  --risk-aversion-grid 1.0 \
  --confidence-grid 0.95 \
  --fee-bps 10 --slip-bps 5 \
  --hold-days-grid 3 5 10 \
  --per-asset-stop-loss-grid 15 \
  --portfolio-stop-loss-grid 0 \
  --per-asset-trailing-stop-grid 10 12 \
  --portfolio-trailing-stop-grid 0 \
  --topk-grid 15 25 35 \
  --momentum-lookback-grid 20 \
  --sort-by goodness_score
