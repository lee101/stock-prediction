#!/usr/bin/env bash
# Packing-grid sweep: top_n in {1,2,3,4} x allocation_mode in {equal, softmax, score_norm}
# at seed=2 (DD-winner) champion cell (n_est=400 depth=5 lr=0.03).
#
# Hypothesis: for top_n>1, packing concentrates exposure on the highest-
# score pick while still capturing decorrelation from the tail. Targets
# Δsortino≥0 AND Δworst_dd≤0 vs plain top_n=1.
#
# Baseline row = top_n=1 (packing is a no-op for K=1).
set -euo pipefail
cd /nvme0n1-disk/code/stock-prediction
source .venv/bin/activate

STAMP=$(date +%Y%m%d_%H%M%S)
OUT=analysis/xgbnew_packing_grid
LOG=logs/xgb_packing_grid_${STAMP}.log
mkdir -p "$OUT" logs

COMMON=(
    --symbols-file symbol_lists/stocks_wide_1000_v1.txt
    --data-root trainingdata
    --train-start 2021-01-01 --train-end 2023-12-31
    --oos-start 2024-01-02
    --n-estimators 400 --max-depth 5 --learning-rate 0.03
    --xgb-weight 1.0 --leverage 1.0
    --fill-buffer-bps 5.0 --fee-rate 2.78e-05
    --random-state 2
    --device cuda -v
)

# Baseline — explicit top_n=1 for comparability (packing no-op).
sub="$OUT/top1_equal"
mkdir -p "$sub"
echo "[`date +%T`] start top_n=1 equal (baseline)" | tee -a "$LOG"
python -u -m xgbnew.eval_multiwindow "${COMMON[@]}" \
    --top-n 1 --allocation-mode equal \
    --output-dir "$sub" 2>&1 | tee -a "$LOG"

for TOP_N in 2 3 4; do
    for MODE in equal softmax score_norm; do
        sub="$OUT/top${TOP_N}_${MODE}"
        mkdir -p "$sub"
        echo "============================================" | tee -a "$LOG"
        echo "[`date +%T`] start top_n=${TOP_N} mode=${MODE}" | tee -a "$LOG"
        echo "============================================" | tee -a "$LOG"
        python -u -m xgbnew.eval_multiwindow "${COMMON[@]}" \
            --top-n "$TOP_N" \
            --allocation-mode "$MODE" \
            --output-dir "$sub" 2>&1 | tee -a "$LOG"
        echo "[`date +%T`] done top_n=${TOP_N} mode=${MODE}" | tee -a "$LOG"
    done
done

echo "All packing-grid runs complete." | tee -a "$LOG"
ls -lht "$OUT"/top*/multiwindow_*.json 2>/dev/null | head -20 | tee -a "$LOG"
