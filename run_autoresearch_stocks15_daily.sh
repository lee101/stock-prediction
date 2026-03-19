#!/bin/bash
set -e
source /nvme0n1-disk/code/stock-prediction/.venv313/bin/activate
cd /nvme0n1-disk/code/stock-prediction

echo "=== Autoresearch: stocks15 daily (periods_per_year=252) ==="

# Check data exists
if [ ! -f pufferlib_market/data/stocks15_daily_train.bin ]; then
    echo "ERROR: stocks15_daily_train.bin not found. Run export_stocks15_daily.sh first."
    exit 1
fi

PYTHONPATH="$PWD/PufferLib:$PWD" python -u -m pufferlib_market.autoresearch_rl \
  --train-data pufferlib_market/data/stocks15_daily_train.bin \
  --val-data pufferlib_market/data/stocks15_daily_val.bin \
  --time-budget 300 \
  --max-trials 35 \
  --periods-per-year 252 \
  --max-steps-override 90 \
  --fee-rate-override 0.001 \
  --leaderboard pufferlib_market/autoresearch_stocks15_daily_leaderboard.csv \
  --checkpoint-root pufferlib_market/checkpoints/autoresearch_stocks15_daily

echo ""
echo "=== Leaderboard ==="
cat pufferlib_market/autoresearch_stocks15_daily_leaderboard.csv
