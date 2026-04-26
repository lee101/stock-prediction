#!/bin/bash
# Per-seed check on 10 new MuonMLP seeds at best non-disp cell (lev=3 ms=0.60 tn=2)
# to confirm the 5-seed blend was tail-lucky.
set -euo pipefail
source .venv/bin/activate

OUT=analysis/xgbnew_daily/heldout2025h2_mlp_muon_xt/per_seed_sweep
mkdir -p "$OUT"

for seed in 1 2 3 4 5 6 8 9 10 11; do
    echo "=== seed $seed ==="
    python -m xgbnew.sweep_ensemble_grid \
        --symbols-file symbol_lists/stocks_wide_1000_v1.txt \
        --model-paths "analysis/xgbnew_daily/heldout2025h2_mlp_muon_xt/alltrain_seed${seed}.pkl" \
        --oos-start 2025-07-01 --oos-end 2025-12-17 \
        --window-days 15 --stride-days 5 \
        --leverage-grid 3.0 \
        --min-score-grid 0.60 \
        --top-n-grid 2 \
        --hold-through \
        --min-dollar-vol 50000000 \
        --inference-min-vol-grid 0.12 \
        --fee-regimes deploy \
        --output-dir "${OUT}/seed${seed}" \
        2>&1 | tail -3
done
