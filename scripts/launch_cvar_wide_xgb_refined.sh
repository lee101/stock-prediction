#!/bin/bash
# Refined XGB sweep around K=15 hd=3 ats=10 champion discovered 2026-04-24.
#
# Phase-1 XGB sweep winner was K=15 hd=3 ats=10 ms=0 = +1.61/−1.44/−6.49 sortino+4.08.
# Blend grid found that this cell × MOM-K35-hd3-ats12 @ 0.5/0.5 gives good +5.53.
#
# Goal: push the XGB median higher without breaking the clean tail so the blend
# can reach the +25-30%/mo target. Tighten the neighborhood:
#   K ∈ {10, 15, 20}         (10 untested; 20 interpolates between 15 and 25)
#   hd ∈ {2, 3, 4}           (2 daily+1, 4 one past the champion)
#   ats ∈ {8, 10, 12}        (8 tighter — may clip the tail AND median)
#   wmax ∈ {0.15, 0.25}      (0.25 allows larger concentrated positions)
#   ms ∈ {0.0, 0.55}         (0.55 filters junk at larger K where ms is active)
# = 3 × 3 × 3 × 2 × 2 = 108 cells. ~2.5h on CPU.
set -euo pipefail
cd /nvme0n1-disk/code/stock-prediction
source .venv/bin/activate
exec env PYTHONPATH=. python -m cvar_portfolio.sweep_wide_xgb \
  --symbols symbol_lists/stocks_wide_1000_v1.txt \
  --data-root trainingdata \
  --start 2024-07-01 --end 2026-04-18 \
  --max-symbols 1000 --min-avg-dol-vol 1000000 \
  --num-scen 1500 --fit-type gaussian --api cvxpy \
  --out analysis/cvar_portfolio/sweep_wide_xgb_refined_oos2025h1 \
  --xgb-score-cache analysis/cvar_portfolio/xgb_panel_scores_wide_oos2025h1_fresh.parquet \
  --w-max-grid 0.15 0.25 \
  --ltar-grid 1.0 \
  --risk-aversion-grid 1.0 \
  --confidence-grid 0.95 \
  --fee-bps 10 --slip-bps 5 \
  --hold-days-grid 2 3 4 \
  --per-asset-stop-loss-grid 15 \
  --portfolio-stop-loss-grid 0 \
  --per-asset-trailing-stop-grid 8 10 12 \
  --portfolio-trailing-stop-grid 0 \
  --topk-grid 10 15 20 \
  --min-score-grid 0.0 0.55 \
  --sort-by goodness_score
