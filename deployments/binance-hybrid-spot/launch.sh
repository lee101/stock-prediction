#!/usr/bin/env bash
set -euo pipefail

cd /home/lee/code/stock

set -a; source /home/lee/code/stock/.env.binance-hybrid 2>/dev/null || true; set +a

# REVERTED 2026-04-07 to robust_reg_tp005_dd002
# Reason: robust_champion "+216% Sort=25.06" was evaluated at decision_lag=0 (LOOKAHEAD BIAS).
# At realistic decision_lag=2 on mixed23_latest_val (50 windows, 30h, slippage sweep 0-20bps):
#   robust_champion:          med_ret=-2.82% med_sort=-0.50 (FAILS lag>=2 protocol)
#   robust_reg_tp005_dd002:   med_ret=+0.01% med_sort=+0.76 (passes)
#   robust_reg_tp005_ent:     med_ret=-5.45% med_sort=-2.09
# Running process still holds old checkpoint until: sudo supervisorctl restart binance-hybrid-spot
exec /home/lee/code/stock/.venv313/bin/python -u \
  rl_trading_agent_binance/trade_binance_live.py \
  --live \
  --model gemini-3.1-flash-lite-preview \
  --symbols BTCUSD ETHUSD SOLUSD DOGEUSD AAVEUSD LINKUSD \
  --execution-mode margin \
  --leverage 0.5 \
  --interval 3600 \
  --fallback-mode chronos2 \
  --rl-checkpoint /home/lee/code/stock/pufferlib_market/checkpoints/mixed23_a40_sweep/robust_reg_tp005_dd002/best.pt \
  "$@"
