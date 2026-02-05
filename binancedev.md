# Binance Dev Utilities

Updated: 2026-02-03

This note captures the local CLI utilities for inspecting Binance spot accounts.
All commands assume the repo root and an active Python environment (e.g. `.venv313`).

## Setup

```bash
source .venv313/bin/activate
export BINANCE_API_KEY="..."
export BINANCE_SECRET="..."

# Binance.US (if applicable)
export BINANCE_TLD="us"
```

## Account Inspection

Balances:
```bash
python -m binance_cli balances
python -m binance_cli balances --asset BTC,ETH,USDT
python -m binance_cli balance BTC
```

Account value (USDT):
```bash
python -m binance_cli account-value
python -m binance_cli account-value --hide-assets
```

Open orders:
```bash
python -m binance_cli open-orders
python -m binance_cli open-orders --symbol BTCUSDT
python -m binance_cli open-orders --symbol BTCUSD,ETHUSD
```

Executed trades (last 24h by default):
```bash
python -m binance_cli recent-trades --days 1
python -m binance_cli recent-trades --symbol BTCUSD,ETHUSD --days 2
```

Previous-day PnL (spot snapshot, USDT estimated via BTC price):
```bash
python -m binance_cli daily-pnl
```

Holdings snapshot + prev-day comparison (stores daily snapshots in a local DB):
```bash
python -m binance_cli holdings-summary
python -m binance_cli holdings-summary --top 0        # show all holdings
python -m binance_cli holdings-summary --db-path /tmp/binance_holdings.db
```

## Notes
- `daily-pnl` uses Binance spot account snapshots and converts BTC totals to USDT
  using the latest BTCUSDT price. If snapshots are unavailable, the command will
  error with an actionable message.
- `open-orders` hits the live Binance API. If you are region‑restricted, set
  `BINANCE_TLD=us` or `BINANCE_BASE_ENDPOINT` as needed.
- `holdings-summary` writes snapshots to `strategy_state/binance_holdings_snapshots.db`
  (safe to cron daily). The command compares against the latest snapshot from the
  previous UTC day when available.
