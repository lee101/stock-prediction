# May 10 Binance Production Audit

Audit time:

- Local: `2026-05-10T21:17:57+12:00`
- UTC: `2026-05-10T09:17:57+00:00`

## Binance Trading Writers

No Binance live trading writer process was running.

Verification:

- `src.binance_live_process_audit.audit_running_binance_live_processes()`
  returned `ok=True`, `reason=no Binance live writer processes found`, and no
  processes.
- A targeted stop sweep over known Binance trading signatures matched
  `0` running trading processes:
  - `trade_binance_live.py`
  - `scripts/binance_hourly_xgb_margin_trader.py`
  - `binance_worksteal.trade_live`
  - `binanceleveragesui.trade_margin_meta`
  - `trade_crypto30_daily.py`
  - `binanceexp1.trade_binance_selector`
  - `binanceexp1.trade_binance_hourly`

No live Binance orders were placed or canceled during this audit.

## Binance Processes Still Running

These Binance-related processes were running, but they are not classified as
live trading writers:

- `pid=1954`: `scripts/refresh_binanceexp1_caches.py`
  - Cache refresh process, running since `2026-04-27`.
- `pid=2139774` and child `pid=2139995`: `go run ./cmd/binancego dashboard ...`
  - Local dashboard/backtest UI, started `2026-05-10 20:38:53 +12`.

These were left running because they are not live trading writers.

## Supervisor Access

Direct `supervisorctl status` failed with a permission error. The documented
sudo path could not be used because `SUDO_PASSWORD` was not set in this shell:

- `sudo -n supervisorctl status` reported: `sudo: a password is required`
- `SUDO_PASSWORD_set=` was empty.

Supervisor config files still exist for several Binance-related services,
including:

- `binance-hourly-xgb-margin-pack`
- `binance-hybrid-spot`
- `binance-meta-margin`
- `binance-worksteal-daily`
- `binanceexp1-selector`
- `binanceexp1-solusd`
- `crypto30-daily`
- cache/collector services such as `binance-5min-collector` and
  `binance-rl-cache-refresh`

Because supervisor access was unavailable, supervisor-managed service states
could not be changed directly in this audit. Process-level verification still
showed no live Binance writer process.

## Account Snapshot

`binance_status.py` at `2026-05-10 09:17 UTC`:

- Spot total: about `$0.82`.
- Cross-margin net value: about `$3,009.19`.
- Combined approximate total: about `$3,010.01`.
- Free/net `USDT`: `$1,158.15`.
- Realized trading PnL state file: missing / unavailable.

Actionable margin positions:

- `ETH`: about `$606.78`, covered by open `SELL`.
- `AAVE`: about `$557.43`, covered by open `SELL`.
- `DOGE`: about `$238.89`, covered by open `SELL`.
- `SOL`: about `$230.19`, covered by open `SELL`.
- `BTC`: about `$216.94`, covered by open `SELL`.

Exit coverage:

- Actionable positions: `5`
- Covered: `5`
- Partial: `0`
- Missing: `0`
- Dust positions below `$12`: `7`
- Liability dust below `$12`: `6`

Open orders:

- `5` open margin orders, all covered long exits:
  - `BTCUSDT` `SELL` `0.00268 @ 95827.26`
  - `ETHUSDT` `SELL` `0.2604 @ 2442.06`
  - `DOGEUSDT` `SELL` `2193 @ 0.13464`
  - `SOLUSDT` `SELL` `2.438 @ 101.93`
  - `AAVEUSDT` `SELL` `5.758 @ 116.61`

## Current Decision

All Binance production trading writers are stopped at the process level. The
account is not actively compounding; it is holding five actionable covered long
positions, idle USDT, and dust. No production Binance strategy should be
restarted until a candidate clears the documented promotion gates.
