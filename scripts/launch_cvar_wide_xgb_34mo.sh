#!/bin/bash
# Top-refined XGB cells on 34mo window 2023-06→2026-04 for direct blend against
# phase-3 momentum champion (which was measured on the same 34mo window).
#
# 2023-06→2025-07 is IS for the XGB ensemble (train_end=2025-06-30); 2025-07+
# is genuinely OOS. This sweep is primarily to validate the 22mo XGB champion
# cells AND get matched-window daily returns for the blend grid vs phase-3.
#
# Top refined cells (K ∈ {10, 15} × hd=3 × ats ∈ {8, 10} × ms ∈ {0, 0.55}):
#   2 × 1 × 2 × 2 = 8 cells (w/ ms grid de-duped). ~13 min on CPU.
set -euo pipefail
cd /nvme0n1-disk/code/stock-prediction
source .venv/bin/activate
exec env PYTHONPATH=. python -m cvar_portfolio.sweep_wide_xgb \
  --symbols symbol_lists/stocks_wide_1000_v1.txt \
  --data-root trainingdata \
  --start 2023-06-01 --end 2026-04-18 \
  --max-symbols 1000 --min-avg-dol-vol 1000000 \
  --num-scen 1500 --fit-type gaussian --api cvxpy \
  --out analysis/cvar_portfolio/sweep_wide_xgb_34mo \
  --xgb-score-cache analysis/cvar_portfolio/xgb_panel_scores_wide_oos2025h1_fresh_2023start.parquet \
  --w-max-grid 0.15 \
  --ltar-grid 1.0 \
  --risk-aversion-grid 1.0 \
  --confidence-grid 0.95 \
  --fee-bps 10 --slip-bps 5 \
  --hold-days-grid 3 \
  --per-asset-stop-loss-grid 15 \
  --portfolio-stop-loss-grid 0 \
  --per-asset-trailing-stop-grid 8 10 \
  --portfolio-trailing-stop-grid 0 \
  --topk-grid 10 15 \
  --min-score-grid 0.0 0.55 \
  --sort-by goodness_score
