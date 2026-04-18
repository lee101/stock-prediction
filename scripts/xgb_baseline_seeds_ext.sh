#!/usr/bin/env bash
# Extended seed sweep to confirm baseline_s2 DD improvement isn't noise.
# Seeds 3, 4, 5, 6 at the same champion cell (lev=1.0, no gate, top_n=1,
# n_est=400, depth=5, lr=0.03, 846 sym, pandas path).
#
# Context: baseline_s2 in the DD-reduction sweep beat seed=0 on:
#   Δsortino +0.39, Δworst_dd -4.48pt, Δneg 0, Δp10 +0.09, Δmed -0.65.
# This script adds 4 more seeds so we can tell if seed=2 is a lucky
# spike or the DD-improvement axis is robust to the seed.
set -euo pipefail
cd /nvme0n1-disk/code/stock-prediction
source .venv/bin/activate

STAMP=$(date +%Y%m%d_%H%M%S)
OUT=analysis/xgbnew_dd_sweep
LOG=logs/xgb_dd_seedext_${STAMP}.log
mkdir -p "$OUT" logs

COMMON=(
    --symbols-file symbol_lists/stocks_wide_1000_v1.txt
    --data-root trainingdata
    --train-start 2021-01-01 --train-end 2023-12-31
    --oos-start 2024-01-02
    --n-estimators 400 --max-depth 5 --learning-rate 0.03
    --top-n 1 --xgb-weight 1.0 --leverage 1.0
    --fill-buffer-bps 5.0 --fee-rate 2.78e-05
    --device cuda -v
)

for seed in 3 4 5 6; do
    sub_out="$OUT/baseline_s${seed}"
    mkdir -p "$sub_out"
    echo "=============================================================" | tee -a "$LOG"
    echo "[$(date +%T)] start seed=$seed" | tee -a "$LOG"
    echo "=============================================================" | tee -a "$LOG"
    python -u -m xgbnew.eval_multiwindow "${COMMON[@]}" \
        --random-state "$seed" \
        --output-dir "$sub_out" 2>&1 | tee -a "$LOG"
    echo "[$(date +%T)] done seed=$seed" | tee -a "$LOG"
done

echo "All seed-ext runs complete." | tee -a "$LOG"
ls -lht "$OUT"/baseline_s*/multiwindow_*.json 2>/dev/null | head -20 | tee -a "$LOG"
