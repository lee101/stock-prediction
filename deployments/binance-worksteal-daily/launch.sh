#!/usr/bin/env bash
set -euo pipefail

cd /home/lee/code/stock

exec /home/lee/code/stock/.venv313/bin/python -u \
  binance_worksteal/trade_live.py \
  --live \
  --daemon \
  --dip-pct 0.20 \
  --profit-target 0.15 \
  --stop-loss 0.10 \
  --max-positions 5 \
  --sma-filter 20 \
  --trailing-stop 0.03 \
  --symbols BTCUSD ETHUSD SOLUSD DOGEUSD AVAXUSD LINKUSD AAVEUSD LTCUSD \
            XRPUSD DOTUSD UNIUSD NEARUSD APTUSD ICPUSD SHIBUSD ADAUSD \
            FILUSD ARBUSD OPUSD INJUSD SUIUSD TIAUSD SEIUSD ATOMUSD \
            ALGOUSD BCHUSD BNBUSD TRXUSD PEPEUSD MATICUSD \
  "$@"
