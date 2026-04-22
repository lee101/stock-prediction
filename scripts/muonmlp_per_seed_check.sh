#!/bin/bash
# Per-seed MuonMLP backtest on heldout2025h2 at the best deploy cell
# (lev=3 ms=0.60 top_n=2). Bonferroni-style check that +23.18%/mo median
# isn't driven by a single lucky seed.
set -euo pipefail
source .venv/bin/activate

OUT=analysis/xgbnew_daily/heldout2025h2_mlp_muon/per_seed_sweep
mkdir -p "$OUT" logs/track1

for seed in 0 7 42 73 197; do
    echo ""
    echo "=== seed $seed ==="
    python -m xgbnew.sweep_ensemble_grid \
        --symbols-file symbol_lists/stocks_wide_1000_v1.txt \
        --model-paths "analysis/xgbnew_daily/heldout2025h2_mlp_muon/alltrain_seed${seed}.pkl" \
        --oos-start 2025-07-01 --oos-end 2025-12-17 \
        --window-days 15 --stride-days 5 \
        --leverage-grid 1.0,2.0,3.0 \
        --min-score-grid 0.55,0.58,0.60 \
        --top-n-grid 1,2 \
        --hold-through \
        --min-dollar-vol 50000000 \
        --inference-min-vol-grid 0.12 \
        --fee-regimes deploy \
        --output-dir "${OUT}/seed${seed}" \
        2>&1 | tail -20
done
