#!/bin/bash
# Auto-launch cross-feature sweep after L-block (PID 543189) finishes
set -euo pipefail
cd /nvme0n1-disk/code/stock-prediction
source .venv313/bin/activate

LBLOCK_PID=543189
echo "Waiting for L-block (PID $LBLOCK_PID) to finish..."
while kill -0 "$LBLOCK_PID" 2>/dev/null; do
    sleep 30
done
echo "L-block finished. Launching cross-feature sweep..."

python3 -u -m pufferlib_market.autoresearch_rl \
    --stocks \
    --train-data pufferlib_market/data/stocks11_daily_train_2012_cross.bin \
    --val-data pufferlib_market/data/stocks11_daily_val_2012_cross.bin \
    --holdout-data pufferlib_market/data/stocks11_daily_val_2012_cross.bin \
    --time-budget 600 \
    --max-timesteps-per-sample 700 \
    --max-trials 8 \
    --start-from 101 \
    --leaderboard autoresearch_stocks11_cross.csv \
    --checkpoint-root pufferlib_market/checkpoints/stocks11_cross \
    --holdout-eval-steps 90 \
    --holdout-n-windows 20 \
    --fee-rate-override 0.001 \
    --periods-per-year 252 \
    --max-steps-override 252 \
    2>&1 | tee autoresearch_stocks11_cross.log
