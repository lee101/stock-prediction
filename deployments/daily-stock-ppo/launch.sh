#!/usr/bin/env bash
set -euo pipefail

# Ensemble PPO daily stock trader: tp05_s123 + tp05_s15 (softmax_avg)
# Runs once daily at ~9:35 AM ET, signals via Alpaca live API
# Val: 0/50 neg, med=+28.76%/90d, p10=+16.37%/90d, worst=+10.17% @ 5bps

cd /nvme0n1-disk/code/stock-prediction
source .venv313/bin/activate

exec python -u trade_daily_stock_prod.py \
    --daemon \
    --live \
    --data-source alpaca \
    --allocation-pct 25.0 \
    "$@"
