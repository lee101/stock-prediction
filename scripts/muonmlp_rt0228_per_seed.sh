#!/bin/bash
# Per-seed check on 15-seed retrain-through-0228 MuonMLP at best cell.
# Goal: detect whether +66.03 deploy / +29.71 stress at lev=2.0 ms=0.60 tn=2 skew≥0
# is driven by a few star seeds or seed-robust.
set -euo pipefail
source .venv/bin/activate

OUT=analysis/xgbnew_daily/retrain_through_2026_02_28_mlp_muon/per_seed_sweep
mkdir -p "$OUT"

for seed in 0 1 2 3 4 5 6 7 8 9 10 11 42 73 197; do
    echo "=== seed $seed ==="
    python -m xgbnew.sweep_ensemble_grid \
        --symbols-file symbol_lists/stocks_wide_1000_v1.txt \
        --model-paths "analysis/xgbnew_daily/retrain_through_2026_02_28_mlp_muon/alltrain_seed${seed}.pkl" \
        --train-start 2020-01-01 --train-end 2026-02-28 \
        --oos-start 2026-03-01 --oos-end 2026-04-20 \
        --window-days 5 --stride-days 2 \
        --leverage-grid 2.0 \
        --min-score-grid 0.60 \
        --top-n-grid 2 \
        --hold-through \
        --fee-regimes "deploy,stress36x" \
        --min-dollar-vol 50000000 \
        --inference-min-vol-grid 0.10 \
        --regime-cs-skew-min-grid=0.0 \
        --regime-cs-iqr-max-grid=0.0 \
        --output-dir "${OUT}/seed${seed}" \
        2>&1 | tail -3
done
