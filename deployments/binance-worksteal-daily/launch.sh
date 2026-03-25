#!/usr/bin/env bash
set -euo pipefail

cd /home/lee/code/stock

# Universe file takes precedence if it exists; --symbols is the fallback.
# --max-symbols caps the total to prevent runaway API calls.
UNIVERSE_FILE="binance_worksteal/universe_v2.yaml"
UNIVERSE_ARG=""
SYMBOLS_ARG=""

if [ -f "$UNIVERSE_FILE" ]; then
  UNIVERSE_ARG="--universe-file $UNIVERSE_FILE"
else
  SYMBOLS_ARG="--symbols BTCUSD ETHUSD SOLUSD DOGEUSD AVAXUSD LINKUSD AAVEUSD LTCUSD \
            XRPUSD DOTUSD UNIUSD NEARUSD APTUSD ICPUSD SHIBUSD ADAUSD \
            FILUSD ARBUSD OPUSD INJUSD SUIUSD TIAUSD SEIUSD ATOMUSD \
            ALGOUSD BCHUSD BNBUSD TRXUSD PEPEUSD MATICUSD"
fi

exec /home/lee/code/stock/.venv313/bin/python -u \
  binance_worksteal/trade_live.py \
  --live \
  --daemon \
  --dip-pct 0.18 \
  --profit-target 0.20 \
  --stop-loss 0.15 \
  --max-positions 5 \
  --sma-filter 20 \
  --trailing-stop 0.03 \
  --entry-proximity-bps 3000 \
  --entry-poll-hours 4 \
  --health-report-hours 6 \
  --dip-pct-fallback 0.18 0.15 0.12 \
  --max-symbols 100 \
  $UNIVERSE_ARG \
  $SYMBOLS_ARG \
  "$@"
