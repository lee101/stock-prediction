#!/usr/bin/env bash
set -euo pipefail

# Standalone PPO daily stock trader: tp05_s123 (h=1024, trade_penalty=0.05)
# Runs once daily at ~9:35 AM ET, signals via Alpaca live API
# Val: 0/50 negative windows, med=+16.52%/90d, p10=+10.45%/90d, worst=+5.62%

cd /nvme0n1-disk/code/stock-prediction
source .venv313/bin/activate

exec python -u trade_daily_stock_prod.py \
    --daemon \
    --live \
    --data-source alpaca \
    --allocation-pct 25.0 \
    "$@"
