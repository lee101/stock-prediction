#!/usr/bin/env bash
# DD-reduction candidate sweep for XGB top_n=1 champion.
#
# Launches four back-to-back evals on the same 34-window OOS grid, pinned to
# the **pandas** feature path (no --fast-features) so deltas are comparable
# to the established baseline at
# analysis/xgbnew_leverage_sensitivity/multiwindow_20260418_062758.json.
#
# Outputs:
#   analysis/xgbnew_dd_sweep/{baseline,ma50,ma20,voltarget015}_YYYYMMDD_HHMMSS.json
#   logs/xgb_dd_sweep_<timestamp>.log
#
# Each sub-run is ~15 min on GPU. Use:
#   bash scripts/xgb_dd_reduction_sweep.sh
#   tail -F logs/xgb_dd_sweep_*.log
set -euo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate

STAMP=$(date +%Y%m%d_%H%M%S)
OUT=analysis/xgbnew_dd_sweep
LOG=logs/xgb_dd_sweep_${STAMP}.log
mkdir -p "$OUT" logs

COMMON=(
    --symbols-file symbol_lists/stocks_wide_1000_v1.txt
    --data-root trainingdata
    --train-start 2021-01-01 --train-end 2023-12-31
    --oos-start 2024-01-02
    --n-estimators 400 --max-depth 5 --learning-rate 0.03
    --top-n 1 --xgb-weight 1.0 --leverage 1.0
    --random-state 0
    --fill-buffer-bps 5.0 --fee-rate 2.78e-05
    --device cuda
    -v
)

run_one() {
    local tag="$1"; shift
    local extra=("$@")
    local sub_out="$OUT/$tag"
    mkdir -p "$sub_out"
    echo "=============================================================" | tee -a "$LOG"
    echo "[$(date +%T)] start tag=$tag  extra=${extra[*]}" | tee -a "$LOG"
    echo "=============================================================" | tee -a "$LOG"
    python -u -m xgbnew.eval_multiwindow "${COMMON[@]}" \
        "${extra[@]}" \
        --output-dir "$sub_out" 2>&1 | tee -a "$LOG"
    echo "[$(date +%T)] done  tag=$tag" | tee -a "$LOG"
}

# Re-baseline (pandas path, lev=1.0, seed=0, no gate) — same build as the
# comparison. Prevents stale-artifact comparisons.
run_one baseline
run_one ma50           --regime-gate-window 50
run_one ma20           --regime-gate-window 20
run_one voltarget015   --vol-target-ann 0.15

echo "All runs complete. Latest results:" | tee -a "$LOG"
ls -lht "$OUT"/*/multiwindow_*.json 2>/dev/null | head -20 | tee -a "$LOG"
echo "" | tee -a "$LOG"
echo "Analyze each with:" | tee -a "$LOG"
echo "  python xgbnew/analyze_sweep.py $OUT/<tag>/multiwindow_*.json" | tee -a "$LOG"
