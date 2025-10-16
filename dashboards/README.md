# Dashboards Module

This package keeps a lightweight record of vanity metrics and Alpaca spreads in SQLite.

## Collector

Run the collector daemon to poll shelf snapshots, spreads, and log-derived metrics. Defaults come from `dashboards/config.toml` if present.

```bash
python -m dashboards.collector_daemon --interval 300
```

Use `--once` for a single run or append `--symbol` / `--shelf` overrides.

## CLI

Inspect stored data directly from the terminal.

Show the latest spread samples and render an ASCII chart:

```bash
python -m dashboards.cli spreads --symbol AAPL --limit 120 --chart
```

List recent snapshots for the tracked shelf file and summarise the newest entry:

```bash
python -m dashboards.cli shelves --summary
```

Inspect numeric metrics extracted from `trade_stock_e2e.log` and `alpaca_cli.log` (or any paths configured under `[logs]`):

```bash
python -m dashboards.cli metrics --metric current_qty --symbol AAPL --chart
```

## Configuration

Optionally create `dashboards/config.toml` (or `config.json`) to override defaults:

```toml
collection_interval_seconds = 120
shelf_files = ["positions_shelf.json"]
spread_symbols = ["AAPL", "NVDA", "TSLA", "BTCUSD"]
[logs]
trade = "trade_stock_e2e.log"
alpaca = "alpaca_cli.log"
```

Delete the database (`dashboards/metrics.db`) if you want to reset stored history.
