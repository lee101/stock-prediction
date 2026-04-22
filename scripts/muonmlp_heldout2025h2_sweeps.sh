#!/bin/bash
# Held-out OOS sweeps:
#   1. MuonMLP solo (5 seeds)
#   2. XGB+CAT+MuonMLP blend (15 seeds)
# Same OOS/grid/fee-regimes as xgbcat_heldout2025h2_sweep.sh so we can diff cleanly.
set -euo pipefail
source .venv/bin/activate

which=${1:-all}
SYMS=symbol_lists/stocks_wide_1000_v1.txt
MUON=$(ls analysis/xgbnew_daily/heldout2025h2_mlp_muon/alltrain_seed*.pkl)

mkdir -p logs/track1

run_sweep() {
    local tag=$1
    local paths=$2
    local out=analysis/xgbnew_daily/heldout2025h2_${tag}/sweep_out
    mkdir -p "$out"
    echo "[heldout2025h2:${tag}] $(echo -e "$paths" | wc -l) models"
    python -m xgbnew.sweep_ensemble_grid \
        --symbols-file "$SYMS" \
        --model-paths "$(echo -e "$paths" | paste -sd,)" \
        --oos-start 2025-07-01 --oos-end 2025-12-17 \
        --window-days 15 --stride-days 5 \
        --leverage-grid 1.0,2.0,3.0 \
        --min-score-grid 0.55,0.58,0.60,0.61,0.62,0.63,0.65 \
        --top-n-grid 1,2 \
        --hold-through \
        --min-dollar-vol 50000000 \
        --inference-min-vol-grid 0.12 \
        --fee-regimes deploy,stress36x \
        --output-dir "$out" \
        2>&1 | tee "logs/track1/sweep_heldout2025h2_${tag}.log"
}

if [[ "$which" == "all" || "$which" == "solo" ]]; then
    run_sweep muonmlp "$MUON"
fi

if [[ "$which" == "all" || "$which" == "blend" ]]; then
    XGB=$(ls analysis/xgbnew_daily/heldout2025h2_xgb/alltrain_seed*.pkl)
    CAT=$(ls analysis/xgbnew_daily/heldout2025h2_cat/alltrain_seed*.pkl)
    ALL=$(echo -e "${XGB}\n${CAT}\n${MUON}")
    run_sweep xgbcatmuon "$ALL"
fi
