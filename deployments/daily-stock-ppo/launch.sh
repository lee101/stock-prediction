#!/usr/bin/env bash
set -euo pipefail

# 3-model ensemble PPO daily stock trader
# Runs once daily at ~9:35 AM ET, signals via Alpaca live API
# Ensemble: random_mut_2201 + random_mut_8597 + random_mut_5526 (softmax_avg)
# Val: 0/50 negative windows, med=+14.94%/90d, p10=+7.64%/90d

cd /nvme0n1-disk/code/stock-prediction
source .venv313/bin/activate

exec python -u trade_daily_stock_prod.py \
    --daemon \
    --live \
    --data-source alpaca \
    --allocation-pct 25.0 \
    "$@"
