#!/bin/bash
# 15-seed MuonMLP × dispersion gate sweep
# Tests whether disp gate rescues the seed-averaged 15-seed MuonMLP
# (which collapsed to +10.49 without gate vs +23 for seed-lucky 5-seed).
set -euo pipefail
source .venv/bin/activate

MUON5=$(ls analysis/xgbnew_daily/heldout2025h2_mlp_muon/alltrain_seed*.pkl)
MUON10=$(ls analysis/xgbnew_daily/heldout2025h2_mlp_muon_xt/alltrain_seed*.pkl)
ALL=$(echo -e "${MUON5}\n${MUON10}")
PATHS=$(echo -e "$ALL" | paste -sd,)
N=$(echo -e "$ALL" | wc -l)

OUT=analysis/xgbnew_daily/heldout2025h2_muonmlp_15seed_disp/sweep_out
mkdir -p "$OUT" logs/track1
echo "[muonmlp_15seed_disp] $N models"

python -m xgbnew.sweep_ensemble_grid \
    --symbols-file symbol_lists/stocks_wide_1000_v1.txt \
    --model-paths "$PATHS" \
    --oos-start 2025-07-01 --oos-end 2025-12-17 \
    --window-days 15 --stride-days 5 \
    --leverage-grid 1.0,2.0,3.0 \
    --min-score-grid 0.55,0.58,0.60,0.62 \
    --top-n-grid 1,2 \
    --hold-through \
    --min-dollar-vol 50000000 \
    --inference-min-vol-grid 0.12 \
    --regime-cs-iqr-max-grid 0.0,0.042,0.060,0.080 \
    --regime-cs-skew-min-grid=-1000.0,0.0,0.5,1.0 \
    --fee-regimes deploy,stress36x \
    --output-dir "$OUT" \
    2>&1 | tee logs/track1/sweep_heldout2025h2_muonmlp_15seed_disp.log
