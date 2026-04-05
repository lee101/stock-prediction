#!/usr/bin/env bash
set -euo pipefail

cd /home/lee/code/stock

set -a; source /home/lee/code/stock/.env.binance-hybrid 2>/dev/null || true; set +a

# Switched to robust_reg_tp005_ent (2026-04-06): +191.4% vs dd002 +116.6%, Sort 19.82 vs 15.35
# Slippage robust: 0bps=+181.6%, 5bps=+191.4%, 10bps=+252.8%, 20bps=+194.3%
# Previous: robust_reg_tp005_dd002 (2026-03-31): +54.6% return, Sortino 2.85, DD 26.9%
exec /home/lee/code/stock/.venv313/bin/python -u \
  rl-trading-agent-binance/trade_binance_live.py \
  --live \
  --model gemini-3.1-flash-lite-preview \
  --symbols BTCUSD ETHUSD SOLUSD DOGEUSD AAVEUSD LINKUSD \
  --execution-mode margin \
  --leverage 0.5 \
  --interval 3600 \
  --fallback-mode chronos2 \
  --rl-checkpoint /home/lee/code/stock/pufferlib_market/checkpoints/mixed23_a40_sweep/robust_reg_tp005_ent/best.pt \
  "$@"
