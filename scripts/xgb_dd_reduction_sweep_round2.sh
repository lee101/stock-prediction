#!/usr/bin/env bash
# Round-2 DD-reduction sweep for the XGB top_n=1 champion.
#
# Runs after round-1 (`scripts/xgb_dd_reduction_sweep.sh`) finishes. Picks up
# the winners of round-1 and stacks them:
#   * ma50 × lev=1.25          — gate + the validated leverage knee
#   * voltarget010             — tighter vol cap
#   * voltarget020             — looser vol cap
#   * ma50_voltarget015        — combined best gate + best sizer
#   * baseline_s1 / baseline_s2 — seed replicas for DD-stddev
#
# Pinned pandas path, same 34-window OOS grid, same fee/fill-buffer as
# round-1 so rows slot straight into `xgboptimiztions.md`.
#
# Launch:
#   WAIT_PID=<round1_pid> bash scripts/xgb_dd_reduction_sweep_round2.sh
#   (omit WAIT_PID to launch immediately)
#
# tail -F logs/xgb_dd_sweep_round2_*.log

set -euo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate

WAIT_PID="${WAIT_PID:-0}"
if [[ "$WAIT_PID" != "0" ]]; then
    echo "[round2] waiting for PID $WAIT_PID to finish before starting..."
    while kill -0 "$WAIT_PID" 2>/dev/null; do
        sleep 30
    done
    echo "[round2] PID $WAIT_PID exited, starting round 2"
fi

STAMP=$(date +%Y%m%d_%H%M%S)
OUT=analysis/xgbnew_dd_sweep
LOG=logs/xgb_dd_sweep_round2_${STAMP}.log
mkdir -p "$OUT" logs

COMMON=(
    --symbols-file symbol_lists/stocks_wide_1000_v1.txt
    --data-root trainingdata
    --train-start 2021-01-01 --train-end 2023-12-31
    --oos-start 2024-01-02
    --n-estimators 400 --max-depth 5 --learning-rate 0.03
    --top-n 1 --xgb-weight 1.0
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

# Leverage knee + gate
run_one ma50_lev125         --regime-gate-window 50 --leverage 1.25 --random-state 0
# Vol-target grid
run_one voltarget010        --vol-target-ann 0.10   --leverage 1.0  --random-state 0
run_one voltarget020        --vol-target-ann 0.20   --leverage 1.0  --random-state 0
# Stacked gate + sizing
run_one ma50_voltarget015   --regime-gate-window 50 --vol-target-ann 0.15 --leverage 1.0 --random-state 0
# Seed replicas of the baseline to bound DD stddev
run_one baseline_s1         --leverage 1.0 --random-state 1
run_one baseline_s2         --leverage 1.0 --random-state 2

echo "All round-2 runs complete. Latest results:" | tee -a "$LOG"
ls -lht "$OUT"/*/multiwindow_*.json 2>/dev/null | head -30 | tee -a "$LOG"
echo "" | tee -a "$LOG"
echo "Analyze each with:" | tee -a "$LOG"
echo "  python xgbnew/analyze_sweep.py $OUT/<tag>/multiwindow_*.json" | tee -a "$LOG"
