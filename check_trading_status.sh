#!/bin/bash
# Check trading status script - runs via cron during market hours

cd /home/lee/code/stock

echo "=== Trading Status Check $(date) ==="

# Check if trading process is running
if pgrep -f "trade_ppo_stocks" > /dev/null; then
    echo "Trading bot: RUNNING"
else
    echo "Trading bot: NOT RUNNING"
fi

# Check recent log entries
echo ""
echo "=== Recent Log Entries ==="
tail -20 /var/log/supervisor/pufferlib-stocks-live.log 2>/dev/null || echo "No log file"

# Check positions
echo ""
echo "=== Current Positions ==="
source .venv/bin/activate
python -c "
from alpaca.trading.client import TradingClient
from env_real import ALP_KEY_ID_PROD, ALP_SECRET_KEY_PROD
api = TradingClient(ALP_KEY_ID_PROD, ALP_SECRET_KEY_PROD, paper=False)
account = api.get_account()
print(f'Equity: \${float(account.equity):,.2f}')
print(f'Buying Power: \${float(account.buying_power):,.2f}')
print(f'Day P/L: \${float(account.equity) - float(account.last_equity):,.2f}')
print()
for pos in api.get_all_positions():
    pnl = float(pos.unrealized_pl)
    print(f'{pos.symbol}: {pos.qty} @ \${float(pos.avg_entry_price):.2f} (P/L: \${pnl:+.2f})')
"
