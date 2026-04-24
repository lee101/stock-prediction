#!/bin/bash
# XGB top-K universe_fn on wide panel: head-to-head vs momentum-top-K champion.
#
# Phase-3 champion (momentum-top-K): K=25 hd=5 L=1.0 ats=12 = +6.11%/mo.
# This sweep replaces the universe_fn with an XGB-scored top-K ranker, using
# `oos2025h1_ensemble_gpu_fresh` (10-seed classifier, train_end=2025-06-30).
#
# Window: 2024-07-01 → 2026-04-18 (matches the XGB score cache coverage).
# Pre-2025-07-01 is XGB-IS; post-2025-07-01 is true OOS.
#
# Focused grid around phase-3 champion:
#   K ∈ {15, 25, 35}          # smaller K because XGB is a better signal than momentum
#   hd ∈ {3, 5, 10}           # include hd=10 because XGB scores are more stable
#   ats ∈ {10, 12}            # neighborhood of phase-3 winner
#   min_score ∈ {0.0, 0.52}   # default 0.0; 0.52 ~ ensemble median based on cache
# = 3 × 3 × 2 × 2 = 36 cells. ~40-50 min on CPU.
#
# A separate momentum baseline cell at (K=25, hd=5, ats=12) on the SAME
# 2024-07 window will be run outside this sweep for direct comparison.
set -euo pipefail
cd /nvme0n1-disk/code/stock-prediction
source .venv/bin/activate
exec env PYTHONPATH=. python -m cvar_portfolio.sweep_wide_xgb \
  --symbols symbol_lists/stocks_wide_1000_v1.txt \
  --data-root trainingdata \
  --start 2024-07-01 --end 2026-04-18 \
  --max-symbols 1000 --min-avg-dol-vol 1000000 \
  --num-scen 1500 --fit-type gaussian --api cvxpy \
  --out analysis/cvar_portfolio/sweep_wide_xgb_topk_oos2025h1 \
  --xgb-score-cache analysis/cvar_portfolio/xgb_panel_scores_wide_oos2025h1_fresh.parquet \
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
  --min-score-grid 0.0 0.52 \
  --sort-by goodness_score
