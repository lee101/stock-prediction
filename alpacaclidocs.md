# Trading CLI Reference

## Quick Status Command
```bash
python scripts/binance_status.py
```
Prints balances, 24h PnL, selector position, open orders, and recent order history in table form.

---

## Current Account State (2026-02-07 ~21:50 UTC)

### Architecture
**Selector bot** (single process) replaces 3 independent bots. Picks best-edge trade across BTCUSD/ETHUSD/SOLUSD, holds one position at a time. Trades on FDUSD zero-fee pairs.

### Binance Balances
| Asset | Amount | Value (USDT) |
|-------|--------|-------------|
| SOL | 58.303 | $4,978.49 |
| USDT | 2.88 | $2.88 |
| BTC | 0.000009 | $0.63 |
| U | 0.56 | $0.56 |
| FDUSD | 0.04 | $0.04 |
| **Total** | | **$4,982.61** |

### Open Position (Selector)
- **SOLUSD** long 58.302 @ $87.34 (opened 2026-02-07 06:48 UTC)
- Exit watcher active: sell @ $86.32, qty=58.302

### Open Orders
None.

### Recent Order History
| Time (UTC) | Pair | Side | Qty | Price | Status |
|------------|------|------|-----|-------|--------|
| 02-07 19:51 | SOLFDUSD | BUY | 58.302 | 87.34 | FILLED |
| 02-07 16:44 | BTCFDUSD | SELL | 0.04548 | 70,521.40 | FILLED |
| 02-07 16:02 | SOLU | SELL | 21.435 | 87.46 | FILLED |
| 02-07 12:45 | SOLU | BUY | 21.435 | 87.60 | FILLED |
| 02-07 12:32 | ETHU | SELL | 0.90530 | 2,075.52 | FILLED |
| 02-07 12:17 | ETHU | BUY | 0.90530 | 2,058.65 | FILLED |
| 02-07 11:42 | ETHU | SELL | 0.90010 | 2,066.27 | FILLED |
| 02-07 10:31 | ETHU | BUY | 0.90010 | 2,052.73 | FILLED |
| 02-07 10:24 | BTCU | BUY | 0.04553 | 110,367.92 | FILLED |

### 24h PnL
- Delta: -0.00778 BTC (~-$533 USDT)

### Running Bots (Supervisor)
| Process | Status | Uptime |
|---------|--------|--------|
| binanceexp1-selector | RUNNING | 5h |
| binanceexp1-cache-refresh | RUNNING | 11h |
| binanceexp1-btcusd | STOPPED | (replaced by selector) |
| binanceexp1-ethusd | STOPPED | (replaced by selector) |
| binanceexp1-solusd | STOPPED | (replaced by selector) |

### Selector Bot Config
```
--symbols BTCUSD,ETHUSD,SOLUSD
--checkpoints BTCUSD=btcusd_h1only_ft20_20260207/ep016,ETHUSD=ethusd_h1only_ft20_20260207/ep020,SOLUSD=solusd_h1only_ft10_20260207/ep010
--intensity-scale 5.0 --risk-weight 0.0 --min-edge 0.0 --max-hold-hours 6
--offset-map ETHUSD=0.0003,SOLUSD=0.0005 --default-offset 0.0
--cache-only --cycle-minutes 5 --poll-seconds 30 --expiry-minutes 90
```

---

## Binance CLI Tools

### 1. `binanceexp1.trade_binance_selector` -- Multi-Asset Selector Bot (ACTIVE)
```bash
python -m binanceexp1.trade_binance_selector \
  --symbols BTCUSD,ETHUSD,SOLUSD \
  --checkpoints BTCUSD=ckpt1.pt,ETHUSD=ckpt2.pt,SOLUSD=ckpt3.pt \
  --horizon 1 --sequence-length 96 \
  --intensity-scale 5.0 --risk-weight 0.0 \
  --min-edge 0.0 --max-hold-hours 6 \
  --default-offset 0.0 --offset-map ETHUSD=0.0003,SOLUSD=0.0005 \
  --min-gap-pct 0.0003 --poll-seconds 30 \
  --expiry-minutes 90 --cycle-minutes 5 \
  --cache-only --log-metrics \
  --metrics-log-path metrics.csv \
  --state-path strategy_state/selector_state.json
```
- Picks best-edge symbol each cycle, holds one position at a time
- Reentry allowed after hold limit
- State persisted to `selector_state.json`

### 2. `binanceexp1.trade_binance_hourly` -- Per-Symbol Trading Bot (OLD)
```bash
python -m binanceexp1.trade_binance_hourly \
  --symbols BTCUSD --checkpoint path.pt \
  --horizon 1 --sequence-length 96 \
  --intensity-scale 2.0 --price-offset-pct 0.0 \
  --min-gap-pct 0.0003 --poll-seconds 30 \
  --expiry-minutes 90 --cycle-minutes 5 \
  --cache-only --log-metrics --metrics-log-path metrics.csv
```
Per-symbol checkpoints: `--checkpoints BTCUSD=ckpt1.pt,ETHUSD=ckpt2.pt`

### 3. `binanceneural.binance_watcher_cli` -- Order Placement & Monitoring
```bash
python -m binanceneural.binance_watcher_cli SYMBOL --side {buy|sell} \
  --limit-price PRICE --target-qty QTY --mode {entry|exit} \
  --expiry-minutes 90 --poll-seconds 30 \
  --config-path STATE.json --price-tolerance 0.0008
```
- Spawned by trading bots as subprocesses
- Price tolerance band: market must be within 0.08% of limit for order placement
- Symbol remapping: SOLUSD -> SOLFDUSD via `BINANCE_DEFAULT_QUOTE=FDUSD`

### 4. `scripts/refresh_binanceexp1_caches.py` -- Cache Refresh Daemon
```bash
python scripts/refresh_binanceexp1_caches.py
```
- Infinite loop, 5-min intervals
- Refreshes price CSVs + Chronos2 forecast caches (`h1` and `h24`) for `SOLUSD`, `LINKUSD`, `UNIUSD`, `BTCUSD`, and `ETHUSD`
- Run via supervisor alongside --cache-only bots

### 5. `scripts/state_inspector_cli.py` -- Trade State Inspector
```bash
python scripts/state_inspector_cli.py overview [--limit N]
python scripts/state_inspector_cli.py symbol SYMBOL [--side {buy|sell}]
python scripts/state_inspector_cli.py history [--symbol SYM] [--side SIDE] [--limit N]
python scripts/state_inspector_cli.py strategies [--date YYYY-MM-DD] [--symbol SYM] [--days N]
```
Global options: `--state-suffix SUFFIX`, `--state-dir PATH`

State files: `trade_outcomes{suffix}.json`, `active_trades{suffix}.json`, `trade_history{suffix}.json`, `trade_learning{suffix}.json`

### 6. `scripts/audit_exit_watchers.py` -- Exit Coverage Audit
```bash
python scripts/audit_exit_watchers.py [--fix] [--dry-run] \
  [--fallback-bps 10] [--default-expiry-minutes 1440]
```
- Shows which positions have active exit watchers/orders
- `--fix` spawns missing exit watchers

### 7. `scripts/kill_all_watchers.py` -- Kill Watcher Processes
```bash
python scripts/kill_all_watchers.py [--cancel-orders] [--yes]
```
- Kills watcher processes only; does NOT close positions
- `--cancel-orders` also cancels their open orders

### 8. `scripts/cancel_multi_orders.py` -- Cancel Duplicate Orders
```bash
python scripts/cancel_multi_orders.py
```
- Runs in loop (5-min intervals), cancels oldest duplicate orders per symbol

### 9. `scripts/binance_api_smoketest.py` -- API Connectivity Test
```bash
python scripts/binance_api_smoketest.py \
  [--symbol SOLUSDT] [--side sell] [--notional 1.0] \
  [--price-multiplier 1.15] [--skip-order] [--sleep-seconds 1.0]
```
Exit codes: 0=ok, 2=price fail, 3=account fail, 4=bad qty, 5=low balance, 6=order fail, 7=no orderId, 8=cancel fail

---

## Backtesting & Training

### 10. `binanceexp1.run_experiment` -- Single-Asset Backtest
```bash
python -m binanceexp1.run_experiment \
  --symbol BTCUSD --checkpoint path.pt \
  --horizon 1 --intensity-scale 2.0 \
  --price-offset-pct 0.0 --cache-only \
  --validation-days 30
```

### 11. `binanceexp1.sweep` -- Parameter Grid Search
```bash
python -m binanceexp1.sweep \
  --checkpoint path.pt --symbol BTCUSD \
  --horizon 1 --cache-only \
  --intensity 0.6 0.8 1.0 1.5 2.0 2.5 \
  --offset 0.0 0.0002 0.0005
```

### 12. `binanceexp1.run_multiasset_selector` -- Selector Backtester
```bash
python -m binanceexp1.run_multiasset_selector \
  --checkpoints BTCUSD=ckpt1.pt,ETHUSD=ckpt2.pt,SOLUSD=ckpt3.pt \
  --intensity-map BTCUSD=5.0,ETHUSD=5.0,SOLUSD=5.0 \
  --offset-map BTCUSD=0.0,ETHUSD=0.0003,SOLUSD=0.0005 \
  --risk-weight 0.0 --min-edge 0.0 --max-hold-hours 6 \
  --initial-cash 10000 --cache-only
```

---

## Automation Pipelines

### 13. `scripts/binance_auto_pipeline.py`
```bash
python scripts/binance_auto_pipeline.py \
  --symbols BTCUSD,ETHUSD,SOLUSD \
  --update-data --run-lora --run-cache --run-policy --run-selector
```

### 14. `scripts/binance_zero_fee_full_auto.py`
```bash
python scripts/binance_zero_fee_full_auto.py \
  --pair-list fdusd --update-data --run-lora --run-cache --run-policy --run-selector
```

---

## `binance_wrapper` Python API (`src.binan.binance_wrapper`)

| Function | Description |
|----------|-------------|
| `get_account_balances()` | All account balances |
| `get_asset_balance(asset)` | Single asset balance dict |
| `get_asset_free_balance(asset)` | Free balance float |
| `get_asset_total_balance(asset)` | Free + locked float |
| `get_account_value_usdt()` | Total account value in USDT with per-asset breakdown |
| `get_account_snapshots_spot(limit=5)` | Daily account snapshots |
| `get_prev_day_pnl_usdt()` | Previous day PnL in BTC and USDT |
| `get_open_orders(symbol=None)` | Open orders (all or per symbol) |
| `get_all_orders(symbol)` | Full order history for symbol |
| `get_my_trades(symbol, limit=500)` | Trade fill history |
| `get_symbol_price(symbol)` | Current price |
| `get_symbol_filters(symbol)` | Exchange filters (lot size, tick, min notional) |
| `get_min_notional(symbol)` | Minimum order notional |
| `create_order(symbol, side, quantity, price=None)` | Place limit/market order |
| `create_market_buy_quote(symbol, quote_qty)` | Market buy by quote amount |
| `create_all_in_order(symbol, side, price=None)` | Full balance order |
| `cancel_all_orders()` | Cancel all open orders |

```python
from src.binan import binance_wrapper as bw

bw.get_account_value_usdt()
# {'total_usdt': 4982.61, 'assets': [...]}

bw.get_prev_day_pnl_usdt()
# {'delta_usdt': -533.10, 'delta_btc': -0.00778, ...}

bw.get_open_orders()
bw.get_my_trades('SOLFDUSD', limit=10)
bw.get_all_orders('BTCFDUSD')
```

---

## Environment Variables
| Var | Purpose | Example |
|-----|---------|---------|
| `BINANCE_DEFAULT_QUOTE` | Quote currency remapping | `FDUSD` (SOLUSD->SOLFDUSD) |
| `TRADE_STATE_SUFFIX` | State file suffix for multi-deployment | `_paper` |
| `PAPER` | Alpaca paper trading flag | `1` |

## Supervisor Configs
- `supervisor/binanceexp1-selector.conf` (active)
- `supervisor/binanceexp1-cache-refresh.conf` (active)
- `supervisor/binanceexp1-btcusd.conf` (stopped, autostart=false)
- `supervisor/binanceexp1-ethusd.conf` (stopped, autostart=false)
- `supervisor/binanceexp1-solusd.conf` (stopped, autostart=false)

## Key File Locations
- `binanceexp1/trade_binance_selector.py` -- live selector bot
- `binanceexp1/trade_binance_hourly.py` -- old per-symbol bot
- `binanceexp1/run_multiasset_selector.py` -- selector backtester
- `binanceexp1/run_experiment.py` -- single-asset training/backtest
- `binanceexp1/sweep.py` -- parameter grid search
- `newnanoalpacahourlyexp/marketsimulator/selector.py` -- core selector sim engine
- `scripts/refresh_binanceexp1_caches.py` -- cache refresh daemon
- `strategy_state/selector_state.json` -- selector position state
- `src/binan/binance_wrapper.py` -- Binance API wrapper
