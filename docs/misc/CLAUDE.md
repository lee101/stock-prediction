- use uv pip NEVER pip
- any docs go in docs/

## Log Files
- `trade_stock_e2e.log` - main trading loop, position management, strategy execution
- `alpaca_cli.log` - Alpaca API calls (orders, positions, account)
- `logs/{symbol}_{side}_{mode}_watcher.log` - individual watcher logs (e.g., `logs/btcusd_buy_entry_watcher.log`)
