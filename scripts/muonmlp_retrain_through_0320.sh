#!/bin/bash
# Companion: train 15-seed MuonMLP with cutoff 2026-03-20 (absorbs more of crash).
# Heldout 2026-03-21 → 2026-04-20 (22 trading days). Tests whether the retrain-
# through edge is specific to 0228 cutoff or generalizes across cutoffs.
set -euo pipefail
source .venv/bin/activate

OUT=analysis/xgbnew_daily/retrain_through_2026_03_20_mlp_muon
mkdir -p "$OUT" logs/track1

python -m xgbnew.train_ensemble_family \
    --family mlp_muon --device cuda \
    --symbols-file symbol_lists/stocks_wide_1000_v1.txt \
    --train-start 2020-01-01 --train-end 2026-03-20 \
    --val-frac 0.1 \
    --seeds 0,1,2,3,4,5,6,7,8,9,10,11,42,73,197 \
    --out-dir "$OUT" \
    --muon-hidden 128 --muon-blocks 3 --mlp-dropout 0.10 \
    --muon-lr 0.02 --muon-momentum 0.95 \
    --mlp-lr 3e-4 --mlp-weight-decay 1e-4 \
    --mlp-batch 16384 --mlp-epochs 40 --mlp-patience 6 \
    2>&1 | tee logs/track1/train_retrain_through_0320_mlp_muon.log
