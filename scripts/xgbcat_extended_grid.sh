#!/bin/bash
# Expanded XGB+CAT blend grid around the +42.91%/mo ms=0.60 lev=2.0 hit.
# Tests robustness across lev ∈ {1.5, 2.0, 2.5}, ms ∈ {0.55-0.65}, top_n ∈ {1, 2}.
set -euo pipefail
source .venv/bin/activate

OUT=analysis/xgbnew_daily/track1_oos120d_xgbcat/sweep_out_extended
mkdir -p "$OUT"

XGB=$(ls analysis/xgbnew_daily/track1_oos120d_xgb/alltrain_seed*.pkl)
CAT=$(ls analysis/xgbnew_daily/track1_oos120d_cat/alltrain_seed*.pkl)
PATHS=$(echo -e "${XGB}\n${CAT}" | paste -sd,)

python -m xgbnew.sweep_ensemble_grid \
    --symbols-file symbol_lists/stocks_wide_1000_v1.txt \
    --model-paths "$PATHS" \
    --oos-start 2025-12-18 --oos-end 2026-04-17 \
    --window-days 30 --stride-days 7 \
    --leverage-grid 1.0,1.5,2.0,2.5,3.0 \
    --min-score-grid 0.55,0.58,0.60,0.62,0.65 \
    --top-n-grid 1,2,3 \
    --hold-through \
    --min-dollar-vol 50000000 \
    --inference-min-vol-grid 0.12 \
    --fee-regimes deploy,stress36x \
    --output-dir "$OUT" \
    2>&1 | tee logs/track1/sweep_xgbcat_extended.log
