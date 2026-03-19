#!/bin/bash
# Trading monitor - runs via at/cron during market hours
# DO NOT COMMIT - contains sudo password

cd /home/lee/code/stock
source .venv/bin/activate

echo "ilu" | sudo -S claude --dangerously-skip-permissions -p "Check the live stock trading bot status:
1. Read /tmp/pufferlib_stocks_live.log - check recent activity
2. Run: source .venv/bin/activate && python -c \"
from alpaca.trading.client import TradingClient
from env_real import ALP_KEY_ID_PROD, ALP_SECRET_KEY_PROD
api = TradingClient(ALP_KEY_ID_PROD, ALP_SECRET_KEY_PROD, paper=False)
print('Positions:')
for p in api.get_all_positions(): print(f'  {p.symbol}: {p.qty}')
print('Orders:')
for o in api.get_orders(): print(f'  {o.side} {o.symbol} @ {o.limit_price or o.type}')
\"
3. Are orders at reasonable prices? Is the bot trading correctly?
4. Is SOL/USD sell order still active?
5. Any issues to fix?
Keep response brief." 2>&1 | tee -a /tmp/claude_trading_monitor.log
