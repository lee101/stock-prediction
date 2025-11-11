# Strategy Training - Position Sizing Dataset Collection

This directory contains tools for collecting comprehensive position sizing datasets by running market simulations over rolling time windows.

## Overview

The position sizing dataset is built by:
1. Loading historical OHLCV data for multiple symbols (stocks, crypto, ETFs)
2. Creating rolling 7-day windows across a full year of data
3. Running market simulations for each window
4. Collecting trades, positions, PnL, and performance metrics
5. Aggregating all data into structured datasets for ML training

## Key Features

- **Multiple Asset Classes**: Stocks (252 trading days), Crypto (365 days), ETFs
- **Rolling Windows**: 7-day windows with configurable stride (default: 7 days, ~52 weeks)
- **Realistic Simulation**: Uses marketsimulator with slippage, fees, and position sizing
- **Comprehensive Metrics**: Trades, positions, equity curves, Sharpe ratios, drawdowns
- **Scalable**: Process 100+ symbols in parallel

## Data Structure

The collected dataset consists of three main files:

### 1. Trades (`*_trades.parquet`)
Individual trade records with:
- Entry/exit timestamps and prices
- Position size
- PnL (absolute and percentage)
- Duration in bars
- Symbol and window metadata

### 2. Window Summaries (`*_summaries.parquet`)
Per-window performance metrics:
- Total return
- Sharpe ratio
- Max drawdown
- Number of trades
- Win rate
- Average PnL

### 3. Position Snapshots (`*_positions.parquet`)
Time-series of position changes:
- Timestamp
- Action (open/close)
- Position size
- Price
- Current capital and equity

## Usage

### Collect Dataset

Basic usage - process all symbols:
```bash
python strategytraining/collect_position_sizing_dataset.py
```

Process specific symbols:
```bash
python strategytraining/collect_position_sizing_dataset.py \
    --symbols AAPL MSFT AMZN BTC-USD ETH-USD \
    --dataset-name tech_crypto_dataset
```

Quick test with limited symbols:
```bash
python strategytraining/collect_position_sizing_dataset.py \
    --max-symbols 5 \
    --dataset-name test_dataset
```

Custom window settings:
```bash
python strategytraining/collect_position_sizing_dataset.py \
    --window-days 14 \
    --stride-days 7 \
    --dataset-name long_window_dataset
```

### Analyze Dataset

Quick analysis:
```bash
python strategytraining/analyze_dataset.py \
    strategytraining/datasets/position_sizing_dataset_20250131_120000
```

Export full report:
```bash
python strategytraining/analyze_dataset.py \
    strategytraining/datasets/position_sizing_dataset_20250131_120000 \
    --export analysis_report.json
```

Analyze specific symbol:
```bash
python strategytraining/analyze_dataset.py \
    strategytraining/datasets/position_sizing_dataset_20250131_120000 \
    --symbol AAPL
```

## Command-Line Options

### collect_position_sizing_dataset.py

| Option | Default | Description |
|--------|---------|-------------|
| `--data-dir` | `trainingdata/train` | Directory containing training data |
| `--output-dir` | `strategytraining/datasets` | Output directory for dataset |
| `--window-days` | `7` | Window size in days |
| `--stride-days` | `7` | Stride between windows in days |
| `--max-symbols` | `None` | Maximum number of symbols to process |
| `--symbols` | `None` | Specific symbols to process (space-separated) |
| `--dataset-name` | `position_sizing_dataset` | Name for the output dataset |

### analyze_dataset.py

| Option | Description |
|--------|-------------|
| `dataset_path` | Path to dataset (without suffix) |
| `--export` | Export full report to JSON file |
| `--symbol` | Analyze specific symbol |

## Data Requirements

### Input Data Format
CSV files with columns:
- `timestamp`: ISO format datetime
- `Open`, `High`, `Low`, `Close`: Price data
- `Volume`: Trading volume

### Minimum Requirements
- At least 2000 data points (hourly bars) per symbol
- Continuous timestamps (gaps are handled but reduce effective windows)

## Trading Calendar

The system automatically handles different trading calendars:

**Stocks/ETFs** (252 trading days):
- ~7 hourly bars per day (market hours only)
- 7-day window ≈ 49 hourly bars
- 52 rolling windows per year

**Crypto** (365 days):
- 24 hourly bars per day (24/7 trading)
- 7-day window ≈ 168 hourly bars
- 52 rolling windows per year

## Example Workflow

```bash
# 1. Collect dataset for major tech stocks and crypto
python strategytraining/collect_position_sizing_dataset.py \
    --symbols AAPL MSFT GOOGL AMZN BTC-USD ETH-USD \
    --dataset-name tech_crypto_v1

# 2. Analyze the collected dataset
python strategytraining/analyze_dataset.py \
    strategytraining/datasets/tech_crypto_v1_20250131_120000

# 3. Export detailed report
python strategytraining/analyze_dataset.py \
    strategytraining/datasets/tech_crypto_v1_20250131_120000 \
    --export strategytraining/reports/tech_crypto_v1_report.json

# 4. Analyze specific symbol
python strategytraining/analyze_dataset.py \
    strategytraining/datasets/tech_crypto_v1_20250131_120000 \
    --symbol BTC-USD
```

## Output Structure

```
strategytraining/
├── collect_position_sizing_dataset.py  # Main collection script
├── analyze_dataset.py                   # Analysis utilities
├── README.md                            # This file
└── datasets/                            # Output datasets
    ├── position_sizing_dataset_20250131_120000_trades.parquet
    ├── position_sizing_dataset_20250131_120000_summaries.parquet
    ├── position_sizing_dataset_20250131_120000_positions.parquet
    └── position_sizing_dataset_20250131_120000_metadata.json
```

## Integration with marketsimulator

This system leverages the existing `marketsimulator/` framework:
- `marketsimulator.environment`: Simulation activation
- `marketsimulator.state`: Position and capital tracking
- `marketsimulator.runner`: Multi-day simulation runner
- `marketsimulator.data_feed`: Historical data loading

Currently uses a simplified simulation strategy for data collection. Future enhancements can integrate with:
- `backtest_test3_inline.py` for realistic forecasting
- `trade_stock_e2e.py` for portfolio management
- MaxDiff strategy implementation

## Dataset Statistics

After collection, you can expect:
- **~10-50 trades** per 7-day window (varies by volatility)
- **~52 windows** per symbol (1 year with weekly stride)
- **1000-2000 total trades** per symbol
- **File sizes**: 1-10 MB per 10 symbols (Parquet compression)

## Performance

Collection speed (approximate):
- **Single symbol**: 1-5 minutes (depends on data size)
- **10 symbols**: 10-30 minutes
- **100 symbols**: 2-4 hours

Tips for faster collection:
- Use `--max-symbols` for testing
- Increase `--stride-days` for coarser sampling
- Process symbols in batches

## Next Steps

After collecting the dataset:

1. **Feature Engineering**: Extract features from trades and positions
2. **Position Sizing Model**: Train ML model to predict optimal position sizes
3. **Risk Management**: Analyze drawdowns and develop risk controls
4. **Strategy Optimization**: Use dataset to optimize entry/exit logic
5. **Backtesting**: Validate learned position sizing on holdout data

## Troubleshooting

**No data collected:**
- Check that `trainingdata/train/` contains CSV files
- Verify CSV format matches expected columns
- Ensure symbols have sufficient data (2000+ points)

**Simulation errors:**
- Check marketsimulator installation
- Verify all dependencies are installed
- Review error logs for specific issues

**Memory issues:**
- Process fewer symbols at once
- Use `--max-symbols` to limit scope
- Close other applications

## Future Enhancements

Planned improvements:
- [ ] Integration with real forecasting models (Toto, Kronos)
- [ ] Multiple strategy variants (MaxDiff, CI Guard, etc.)
- [ ] Parallel processing for faster collection
- [ ] Real-time data streaming support
- [ ] Advanced slippage modeling
- [ ] Market regime detection and labeling
- [ ] Feature extraction pipeline
- [ ] Direct ML model training integration

---

# Multi-Strategy Backtesting on Top Stocks

## Overview

This system tests **7 trading strategies** across the **top 200 stocks from Alpaca**, generating comprehensive performance reports showing which strategies work best for which stocks.

### Available Strategies

1. **`maxdiff`** - Maximum directional edge (often best performer)
2. **`entry_takeprofit`** - Profit potential from predicted highs
3. **`highlow`** - Range-based trading (high-low spread)
4. **`simple_strategy`** - Close price change predictions
5. **`all_signals_strategy`** - Average of close/high/low predictions
6. **`ci_guard`** - Conservative blended approach
7. **`buy_hold`** - Buy and hold baseline

## Quick Start

### 1. Test with a small subset (recommended)

```bash
# Test with 10 stocks to verify everything works
python strategytraining/run_top_stocks_backtest.py --num-stocks 10
```

### 2. Run full backtest on top 200 stocks

```bash
# Fetch and test top 200 stocks from Alpaca
python strategytraining/run_top_stocks_backtest.py --num-stocks 200
```

### 3. Use a custom stock list

```bash
# Create CSV with 'symbol' column, then:
python strategytraining/run_top_stocks_backtest.py --symbols-file my_stocks.csv
```

## Command Line Options

```bash
python strategytraining/run_top_stocks_backtest.py \
    --num-stocks 200 \              # Number of top stocks to test
    --data-dir trainingdata/train \ # Where to store/find historical data
    --output-dir strategytraining/reports \ # Where to save reports
    --window-days 7 \               # Backtest window size (days)
    --stride-days 7 \               # Stride between windows (days)
    --skip-download                 # Skip data download (use existing)
```

## Generated Reports

All reports saved to `strategytraining/reports/` (gitignored):

### 1. Strategy Performance by Stock
**File**: `strategy_performance_by_stock_TIMESTAMP.csv`

Shows every stock-strategy combination, sorted by average PnL:
- Average PnL per window
- Total PnL
- Annualized PnL
- Sharpe ratio
- Win rate
- Total trades
- Number of windows tested

### 2. Best Strategy Per Stock
**File**: `best_strategy_per_stock_TIMESTAMP.csv`

The winning strategy for each stock. Use this to:
- See which strategy works best for each symbol
- Identify patterns (e.g., "maxdiff dominates tech stocks")
- Build a symbol-specific strategy mapping

### 3. Strategy Rankings
**File**: `strategy_rankings_TIMESTAMP.csv`

Overall strategy performance across all stocks:
- Average PnL per window
- Total PnL
- Avg Sharpe ratio
- Avg win rate
- Total trades

### 4. Top Stocks by PnL
**File**: `top_stocks_by_pnl_TIMESTAMP.csv`

Stocks ranked by profitability across all strategies:
- Best/worst performing symbols
- Most tradeable stocks
- Risk-adjusted returns

### 5. Window-Level Details
**File**: `window_level_details_TIMESTAMP.csv`

Detailed results for every window:
- Per-window PnL, returns, metrics
- Time-series view of strategy performance
- Useful for analyzing performance over different market regimes

### 6. All Trades
**File**: `all_trades_TIMESTAMP.csv`

Individual trade records:
- Entry/exit timestamps and prices
- Position size
- PnL and PnL %
- Duration
- Strategy signals

## Example Output

```
================================================================================
BACKTEST SUMMARY
================================================================================
Total Symbols: 200
Total Strategies: 7
Total Strategy-Window Combinations: 72,800
Total Trades: 145,600

TOP 10 STOCK-STRATEGY COMBINATIONS (by avg PnL per window):
--------------------------------------------------------------------------------
AAPL     / maxdiff            :  $ 127.45 (Sharpe:   2.13, Win Rate:  68.5%)
MSFT     / maxdiff            :  $ 115.32 (Sharpe:   1.98, Win Rate:  65.2%)
NVDA     / entry_takeprofit   :  $ 210.87 (Sharpe:   2.45, Win Rate:  71.3%)
...

STRATEGY RANKINGS (by avg PnL across all stocks):
--------------------------------------------------------------------------------
maxdiff             :  $  45.23 avg ($ 3,287,142 total, Sharpe:   1.45)
entry_takeprofit    :  $  38.67 avg ($ 2,813,446 total, Sharpe:   1.32)
highlow             :  $  31.45 avg ($ 2,287,940 total, Sharpe:   1.18)
...

TOP 10 STOCKS (by avg PnL across all strategies):
--------------------------------------------------------------------------------
NVDA    :  $  89.23 avg (Sharpe:   2.01, Win Rate:  69.4%)
AAPL    :  $  76.45 avg (Sharpe:   1.87, Win Rate:  66.8%)
...
```

## Use Cases

### 1. Strategy Selection
Find the best strategy for each stock you trade:
```bash
# Run backtest
python strategytraining/run_top_stocks_backtest.py --num-stocks 50

# Check best_strategy_per_stock_*.csv
# Use this mapping in your trading system
```

### 2. Stock Selection
Identify the most profitable stocks:
```bash
# Look at top_stocks_by_pnl_*.csv
# Focus trading on top performers
# Avoid or short bottom performers
```

### 3. Strategy Development
Analyze what makes strategies successful:
```bash
# Review strategy_rankings_*.csv
# Study window_level_details_*.csv
# Improve underperforming strategies
```

## Data Requirements

- Historical OHLCV data (automatically downloaded from Alpaca)
- Minimum 2000 data points per symbol
- ~4 years of history recommended
- Stocks use market hours (7 bars/day)
- Crypto uses 24/7 data (24 bars/day)

## Helper Scripts

### Fetch Top Stocks Only

```bash
python strategytraining/fetch_top_stocks.py --limit 200 --output top_200.csv
```

### Collect Strategy PnL Dataset

```bash
python strategytraining/collect_strategy_pnl_dataset.py \
    --symbols AAPL MSFT NVDA \
    --dataset-name tech_stocks
```

## Notes

- Reports directory (`strategytraining/reports/`) is gitignored
- Data is cached in `trainingdata/` to avoid re-downloads
- Window analysis captures different market regimes
- All metrics are annualized for comparison
- System uses existing marketsimulator infrastructure
