#!/usr/bin/env bash
set -euo pipefail

cd /home/lee/code/stock

exec /home/lee/code/stock/.venv313/bin/python -u \
  rl-trading-agent-binance/trade_binance_live.py \
  --live \
  --model gemini-3.1-flash-lite-preview \
  --symbols BTCUSD ETHUSD SOLUSD DOGEUSD AAVEUSD LINKUSD \
  --execution-mode margin \
  --leverage 5 \
  --interval 3600 \
  --fallback-mode chronos2 \
  --rl-checkpoint /home/lee/code/stock/pufferlib_market/checkpoints/mixed23_a40_sweep/robust_reg_tp005_ent/best.pt \
  "$@"
