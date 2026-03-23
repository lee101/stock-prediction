#!/usr/bin/env bash
set -euo pipefail

# Ensemble PPO daily stock trader
# Runs once daily at ~9:35 AM ET, signals via Alpaca live API
# Ensemble: random_mut_2201 + random_mut_8597 (softmax_avg)
# Val: 0/50 negative windows, med=+13.57%/90d

cd /nvme0n1-disk/code/stock-prediction
source .venv313/bin/activate

exec python -u trade_daily_stock_prod.py \
    --daemon \
    --live \
    --data-source alpaca \
    --allocation-pct 25.0 \
    "$@"
