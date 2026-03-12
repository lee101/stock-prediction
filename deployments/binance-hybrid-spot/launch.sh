#!/usr/bin/env bash
set -euo pipefail

cd /home/lee/code/stock

exec /home/lee/code/stock/.venv313/bin/python -u \
  rl-trading-agent-binance/trade_binance_live.py \
  --live \
  --model gemini-3.1-flash-lite-preview \
  --symbols BTCUSD ETHUSD SOLUSD \
  --execution-mode margin \
  --leverage 5 \
  --interval 3600 \
  "$@"
