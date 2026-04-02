# Alpaca Prod Verify

Use this to verify the daily stock trader before calling the prod path healthy.

Start every session with:

```bash
source ~/.secretbashrc
source .venv313/bin/activate
```

The script defaults to paper mode, but pass `--paper` explicitly when verifying. Paper mode uses the paper Alpaca keys and does not need `ALLOW_ALPACA_LIVE_TRADING=1`.

Minimal checks:

1. Preflight config

```bash
python trade_daily_stock_prod.py --check-config --paper
```

This should catch missing checkpoints, bad credentials, and invalid server config.

2. Print the resolved runtime config

```bash
python trade_daily_stock_prod.py \
  --print-config \
  --paper \
  --symbols AAPL MSFT NVDA
```

Confirm the checkpoints, symbols, data source, and execution backend are what you expect.

3. One-shot dry run

```bash
python trade_daily_stock_prod.py \
  --once \
  --dry-run \
  --print-payload \
  --paper \
  --symbols AAPL MSFT NVDA
```

Check `action`, `symbol`, `confidence`, `value_estimate`, `allow_open`, `allow_open_reason`, `bars_fresh`, and `latest_bar_timestamp`.

4. Hit the real Alpaca paper account

Run this during market hours if you want to exercise the actual order path. This talks to Alpaca paper data and the Alpaca paper trading account. Because `--dry-run` is not set, it may submit paper orders.

```bash
python trade_daily_stock_prod.py \
  --once \
  --paper \
  --data-source alpaca \
  --execution-backend alpaca \
  --print-payload \
  --symbols AAPL MSFT NVDA
```

For a continuous paper run around market open, replace `--once` with `--daemon`.

After it runs, verify the result in both places:

```bash
tail -n 20 strategy_state/daily_stock_rl_signals.jsonl
```

and in the Alpaca paper account UI or account activity feed. The plan, fills, and resulting positions should all look reasonable for the account.

5. Optional local regression checks

Backtest:

```bash
python trade_daily_stock_prod.py \
  --backtest \
  --paper \
  --data-source local \
  --data-dir trainingdata \
  --symbols AAPL MSFT NVDA
```

Trading-server parity:

```bash
python trade_daily_stock_prod.py \
  --backtest \
  --compare-server-parity \
  --paper \
  --data-source local \
  --data-dir trainingdata \
  --symbols AAPL MSFT NVDA
```

Useful logs:

```bash
journalctl -u <systemd-unit-name> -n 200 --no-pager
tail -n 20 strategy_state/daily_stock_rl_signals.jsonl
```

Healthy enough to proceed means:

- `--check-config` passes
- the dry-run payload is sane
- the paper-account run reaches Alpaca and behaves reasonably in that account
- the signal log matches what the paper account shows
- backtest and parity checks finish without exceptions

Live-only guardrails:

- set `ALLOW_ALPACA_LIVE_TRADING=1` before any live run
- exactly one live writer may control the Alpaca account
- automatic exits must not realize a loss unless they are force-exit paths
