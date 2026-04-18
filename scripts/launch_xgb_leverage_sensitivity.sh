#!/usr/bin/env bash
# Leverage sensitivity sweep on the XGB champion config.
#
# Prerequisite: seed-Bonferroni sweep has finished AND passed — i.e.
# null-max Sharpe significantly below observed on the seed axis. Check
# with:
#   python xgbnew/analyze_sweep.py analysis/xgbnew_seed_bonferroni/multiwindow_*.json
#
# This sweep cross-cuts leverage (1.0 → 3.0) × 3 seeds on the champion
# config (top_n=1, n_est=400, depth=5, lr=0.03). GPU-trained for speed.
# Fill buffer stays at realism default (5 bps) to match deploy numbers.
#
# Usage:
#   bash scripts/launch_xgb_leverage_sensitivity.sh
#
# Output:
#   analysis/xgbnew_leverage_sensitivity/multiwindow_*.json
#   logs/xgb_leverage_sensitivity_<timestamp>.log
set -euo pipefail

cd "$(dirname "$0")/.."

source .venv/bin/activate

mkdir -p analysis/xgbnew_leverage_sensitivity logs

LOG="logs/xgb_leverage_sensitivity_$(date +%Y%m%d_%H%M%S).log"

nohup python -u -m xgbnew.eval_multiwindow \
    --symbols-file symbol_lists/stocks_wide_1000_v1.txt \
    --data-root trainingdata \
    --train-start 2021-01-01 --train-end 2023-12-31 \
    --oos-start 2024-01-02 \
    --n-estimators 400 --max-depth 5 --learning-rate 0.03 \
    --top-n 1 --xgb-weight 1.0 \
    --leverage-grid "1.0,1.25,1.5,2.0,2.5,3.0" \
    --random-state-grid "0,1,2" \
    --fill-buffer-bps 5.0 --fee-rate 2.78e-05 \
    --device cuda \
    --output-dir analysis/xgbnew_leverage_sensitivity \
    -v \
    > "$LOG" 2>&1 &

echo "Launched PID=$!"
echo "Log: $LOG"
echo ""
echo "Tail with:  tail -F $LOG"
echo "Analyze when done:"
echo "  python xgbnew/analyze_sweep.py analysis/xgbnew_leverage_sensitivity/multiwindow_*.json"
