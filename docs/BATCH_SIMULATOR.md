# Batch Market Simulator

This tool runs the market simulator across multiple stock pairs to find the best performing strategies.

## Features

- **Data Download**: Downloads historical data from Alpaca Markets API to `trainingdata/` directory
- **Batch Simulation**: Runs market simulator for multiple symbols in parallel
- **PnL Tracking**: Saves PnL over time and performance metrics to `strategytraining/` directory
- **Symbol Sources**:
  - Default: Uses symbols from `trade_stock_e2e.py`
  - Alpaca API: Can fetch all tradable symbols from Alpaca Markets
  - Custom: Specify your own list of symbols

## Usage

### Basic Usage (Default Symbols)

Run with default symbols from `trade_stock_e2e.py`:

```bash
python batch_run_market_simulator.py
```

### Specify Custom Symbols

```bash
python batch_run_market_simulator.py --symbols AAPL MSFT NVDA TSLA
```

### Use All Alpaca Tradable Symbols

Fetch all tradable US equities from Alpaca:

```bash
python batch_run_market_simulator.py --use-all-alpaca --asset-class us_equity
```

Fetch all tradable crypto:

```bash
python batch_run_market_simulator.py --use-all-alpaca --asset-class crypto
```

### Download Only (No Simulation)

Download historical data without running simulations:

```bash
python batch_run_market_simulator.py --download-only --data-years 10
```

### Skip Download (Use Existing Data)

Run simulations using previously downloaded data:

```bash
python batch_run_market_simulator.py --skip-download
```

### Limit Number of Symbols (Testing)

Test with first 5 symbols only:

```bash
python batch_run_market_simulator.py --limit 5
```

## Command Line Options

| Option | Default | Description |
|--------|---------|-------------|
| `--symbols` | From `trade_stock_e2e.py` | Specific symbols to simulate |
| `--use-all-alpaca` | False | Use all tradable symbols from Alpaca API |
| `--asset-class` | `us_equity` | Asset class when using `--use-all-alpaca` (`us_equity` or `crypto`) |
| `--download-only` | False | Only download data, skip simulation |
| `--skip-download` | False | Skip data download, use existing data |
| `--simulation-days` | 30 | Number of trading days to simulate |
| `--data-years` | 10 | Years of historical data to download |
| `--initial-cash` | 100000 | Initial cash for simulation |
| `--output-dir` | `strategytraining/batch_results` | Output directory for results |
| `--data-dir` | `trainingdata` | Directory for training data |
| `--run-name` | Auto-generated | Name for this batch run |
| `--limit` | None | Limit number of symbols to process |
| `--force-kronos` | True | Use Kronos-only forecasting |

## Output Structure

### Directory Structure

```
strategytraining/batch_results/
├── batch_run_YYYYMMDD_HHMMSS_full_results.json    # Full detailed results
├── batch_run_YYYYMMDD_HHMMSS_summary.csv          # Summary table
├── pnl_timeseries/
│   ├── AAPL_pnl.csv                               # PnL over time for AAPL
│   ├── MSFT_pnl.csv                               # PnL over time for MSFT
│   └── ...
└── plots/
    ├── AAPL/
    │   ├── equity_curve.png
    │   └── symbol_contributions.png
    └── ...

trainingdata/
├── AAPL.csv
├── MSFT.csv
└── ...
```

### Output Files

#### Summary CSV

Ranked table of all symbols by performance:

| symbol | final_equity | total_return | total_return_pct | sharpe_ratio | max_drawdown_pct | win_rate | profit_factor | trades | fees |
|--------|--------------|--------------|------------------|--------------|------------------|----------|---------------|--------|------|
| AAPL   | 125000       | 25000        | 25.00            | 2.5          | -5.2             | 0.65     | 2.1           | 45     | 120  |

#### Full Results JSON

Detailed results including:
- Daily snapshots (equity, cash, positions)
- PnL over time
- All performance metrics
- Trade executions

#### PnL Timeseries CSV

Time-based PnL tracking for each symbol:

| timestamp | day | phase | pnl | pnl_pct | equity |
|-----------|-----|-------|-----|---------|--------|
| 2024-01-01T09:30:00 | 0 | open | 0 | 0.0 | 100000 |
| 2024-01-01T16:00:00 | 0 | close | 1500 | 1.5 | 101500 |

## Performance Metrics

The batch runner calculates and saves the following metrics for each symbol:

- **Total Return**: Absolute and percentage return
- **Sharpe Ratio**: Risk-adjusted return (annualized)
- **Max Drawdown**: Maximum peak-to-trough decline
- **Win Rate**: Percentage of profitable trades
- **Profit Factor**: Gross profit / gross loss
- **Trades Executed**: Total number of trades
- **Fees Paid**: Total transaction costs

## Example Workflows

### Find Best Stocks for 10-Year Backtest

```bash
# 1. Download 10 years of data for default symbols
python batch_run_market_simulator.py --download-only --data-years 10

# 2. Run simulations with 60-day trading period
python batch_run_market_simulator.py --skip-download --simulation-days 60

# 3. Review results
cat strategytraining/batch_results/batch_run_*/summary.csv | head -20
```

### Test Strategy on All Alpaca Stocks

```bash
# Run on first 50 US equities as a test
python batch_run_market_simulator.py \
    --use-all-alpaca \
    --asset-class us_equity \
    --limit 50 \
    --simulation-days 30 \
    --data-years 5
```

### Compare Crypto vs Equities

```bash
# Run crypto symbols
python batch_run_market_simulator.py \
    --symbols BTCUSD ETHUSD UNIUSD LINKUSD \
    --run-name crypto_comparison \
    --simulation-days 30

# Run equity symbols
python batch_run_market_simulator.py \
    --symbols SPY QQQ AAPL MSFT \
    --run-name equity_comparison \
    --simulation-days 30
```

## Notes

- Default symbols are the high-performing ones from `trade_stock_e2e.py`
- Data is downloaded as hourly bars for more granular backtesting
- Simulations use the same forecasting pipeline as live trading
- Results are automatically ranked by total return percentage
- Use `--force-kronos` for faster simulations with Kronos-only forecasting

## Troubleshooting

### "No data received for symbol"

Some symbols may not have historical data available on Alpaca. The script will skip these and continue.

### Memory Issues

If processing many symbols, use `--limit` to process in batches:

```bash
# Process first 100 symbols
python batch_run_market_simulator.py --use-all-alpaca --limit 100

# Process next 100 (modify script to add --offset if needed)
```

### Rate Limiting

Alpaca has rate limits on API calls. The script includes small delays between downloads. If you hit rate limits, use `--download-only` first, then run simulations separately with `--skip-download`.
