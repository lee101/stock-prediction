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
