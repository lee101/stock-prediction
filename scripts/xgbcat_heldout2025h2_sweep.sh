#!/bin/bash
# TRUE-OOS validation of XGB+CAT cross-family blend edge.
# Trained on 2020-01-01 → 2025-06-30 → OOS 2025-07-01 → 2025-12-17 (~170d never seen).
# Tests whether the alltrain ms=0.62 edge (+27.60%/mo 0/14 neg) survives
# held-out validation, or was data contamination.
set -euo pipefail
source .venv/bin/activate

OUT=analysis/xgbnew_daily/heldout2025h2_xgbcat/sweep_out
mkdir -p "$OUT" logs/track1

XGB=$(ls analysis/xgbnew_daily/heldout2025h2_xgb/alltrain_seed*.pkl)
CAT=$(ls analysis/xgbnew_daily/heldout2025h2_cat/alltrain_seed*.pkl)
PATHS=$(echo -e "${XGB}\n${CAT}" | paste -sd,)

echo "[heldout2025h2] blend paths:"
echo "$PATHS" | tr ',' '\n'
echo ""

# 15d windows 5d stride = 32 windows over the 170d OOS span -- matches the
# stride config that validated ms=0.62 as structural (not noise) on alltrain.
python -m xgbnew.sweep_ensemble_grid \
    --symbols-file symbol_lists/stocks_wide_1000_v1.txt \
    --model-paths "$PATHS" \
    --oos-start 2025-07-01 --oos-end 2025-12-17 \
    --window-days 15 --stride-days 5 \
    --leverage-grid 1.0,2.0,3.0 \
    --min-score-grid 0.55,0.58,0.60,0.61,0.62,0.63,0.65 \
    --top-n-grid 1,2 \
    --hold-through \
    --min-dollar-vol 50000000 \
    --inference-min-vol-grid 0.12 \
    --fee-regimes deploy,stress36x \
    --output-dir "$OUT" \
    2>&1 | tee logs/track1/sweep_heldout2025h2_xgbcat.log
