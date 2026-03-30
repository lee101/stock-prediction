#!/usr/bin/env bash
set -euo pipefail

cd /home/lee/code/stock

# Universe file takes precedence if it exists; otherwise use a replayable
# static fallback list. The safety cap prevents runaway API calls.
UNIVERSE_FILE="binance_worksteal/universe_v2.yaml"
COMMON_ARGS=(
  --live
  --daemon
  --dip-pct 0.18
  --profit-target 0.20
  --stop-loss 0.15
  --max-positions 5
  --sma-filter 20
  --trailing-stop 0.03
  --entry-proximity-bps 3000
  --entry-poll-hours 4
  --health-report-hours 6
  --dip-pct-fallback 0.18 0.15 0.12
  --max-symbols 100
)

if [ -f "$UNIVERSE_FILE" ]; then
  exec /home/lee/code/stock/.venv313/bin/python -u \
    binance_worksteal/trade_live.py \
    "${COMMON_ARGS[@]}" \
    --universe-file "$UNIVERSE_FILE" \
    "$@"
else
  exec /home/lee/code/stock/.venv313/bin/python -u \
    binance_worksteal/trade_live.py \
    "${COMMON_ARGS[@]}" \
    --symbols SOLUSD LINKUSD DOTUSD UNIUSD SHIBUSD ADAUSD SUIUSD ATOMUSD \
    "$@"
fi
