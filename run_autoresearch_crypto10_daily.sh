#!/bin/bash
set -e
source /nvme0n1-disk/code/stock-prediction/.venv313/bin/activate
cd /nvme0n1-disk/code/stock-prediction

echo "=== Autoresearch: crypto10 daily (periods_per_year=365) ==="

# Check data exists
if [ ! -f pufferlib_market/data/crypto10_daily_train.bin ]; then
    echo "ERROR: crypto10_daily_train.bin not found"
    exit 1
fi

PYTHONPATH="$PWD/PufferLib:$PWD" python -u -m pufferlib_market.autoresearch_rl \
  --train-data pufferlib_market/data/crypto10_daily_train.bin \
  --val-data pufferlib_market/data/crypto10_daily_val.bin \
  --time-budget 300 \
  --max-trials 35 \
  --periods-per-year 365 \
  --max-steps-override 80 \
  --fee-rate-override 0.001 \
  --leaderboard pufferlib_market/autoresearch_crypto10_daily_leaderboard.csv \
  --checkpoint-root pufferlib_market/checkpoints/autoresearch_crypto10_daily

echo ""
echo "=== Leaderboard ==="
cat pufferlib_market/autoresearch_crypto10_daily_leaderboard.csv
