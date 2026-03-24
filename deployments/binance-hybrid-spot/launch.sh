#!/usr/bin/env bash
set -euo pipefail

cd /home/lee/code/stock

set -a; source /home/lee/code/stock/.env.binance-hybrid 2>/dev/null || true; set +a

exec /home/lee/code/stock/.venv313/bin/python -u \
  rl-trading-agent-binance/trade_binance_live.py \
  --live \
  --model gemini-3.1-flash-lite-preview \
  --symbols BTCUSD ETHUSD SOLUSD LTCUSD AVAXUSD DOGEUSD LINKUSD ADAUSD UNIUSD AAVEUSD ALGOUSD DOTUSD SHIBUSD XRPUSD MATICUSD \
  --execution-mode margin \
  --leverage 0.5 \
  --interval 3600 \
  --fallback-mode chronos2 \
  --rl-checkpoint /home/lee/code/stock/pufferlib_market/checkpoints/crypto15_v2/gpu0/c15_tp03_slip5_s7/best.pt \
  "$@"
