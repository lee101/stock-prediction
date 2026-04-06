#!/usr/bin/env bash
set -euo pipefail

cd /home/lee/code/stock

set -a; source /home/lee/code/stock/.env.binance-hybrid 2>/dev/null || true; set +a

# Switched to robust_champion (2026-04-06): +216% return, Sort=25.06, 65% WR, 54 trades/period
# Slippage robust: 0bps=+131% Sort=18.13, 5bps=+216% Sort=25.06, 10bps=+217% Sort=25.30, 20bps=+204% Sort=24.40
# 50-window holdout: 82% positive@30bar, 100% positive@60bar, median Sort=3.28 cross-seed
# Previous: robust_reg_tp005_ent (+191%, Sort=19.82), dd002 (+117%, Sort=15.35)
exec /home/lee/code/stock/.venv313/bin/python -u \
  rl_trading_agent_binance/trade_binance_live.py \
  --live \
  --model gemini-3.1-flash-lite-preview \
  --symbols BTCUSD ETHUSD SOLUSD DOGEUSD AAVEUSD LINKUSD \
  --execution-mode margin \
  --leverage 0.5 \
  --interval 3600 \
  --fallback-mode chronos2 \
  --rl-checkpoint /home/lee/code/stock/pufferlib_market/checkpoints/a100_scaleup/robust_champion/best.pt \
  "$@"
