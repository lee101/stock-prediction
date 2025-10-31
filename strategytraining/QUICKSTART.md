# Position Sizing Dataset Collection - Quickstart Guide

## 🚀 Quick Start (5 minutes)

### Step 1: Install Dependencies
```bash
cd /home/administrator/code/stock-prediction
uv pip install pandas numpy tqdm pyarrow
```

### Step 2: Run Quick Test
Test with 3 symbols (AAPL, MSFT, BTC-USD):
```bash
python strategytraining/quick_test.py
```

This will:
- Process 3 symbols with rolling 7-day windows
- Generate ~52 windows per symbol (1 year of data)
- Collect all trades, positions, and PnL
- Save to `strategytraining/datasets/test_dataset_*.parquet`

Expected output:
```
Processing AAPL... ✓
Processing MSFT... ✓
Processing BTC-USD... ✓
Collected 500+ trades across 150+ windows
Dataset saved!
```

### Step 3: Analyze Results
```bash
# Find your dataset (example path)
DATASET_PATH="strategytraining/datasets/test_dataset_20250131_120000"

# Run analysis
python strategytraining/analyze_dataset.py $DATASET_PATH
```

You'll see:
- Overall statistics (trades, windows, symbols)
- Trade performance (PnL, win rate, Sharpe ratio)
- Stocks vs Crypto comparison
- Top performing symbols

---

## 📊 Full Dataset Collection

### Option 1: Collect All Symbols
Process all 100+ symbols from `trainingdata/train/`:
```bash
python strategytraining/collect_position_sizing_dataset.py \
    --dataset-name full_dataset
```

⚠️ **Warning**: This will take 2-4 hours and generate ~100MB of data.

### Option 2: Specific Symbols
Process only symbols you care about:
```bash
python strategytraining/collect_position_sizing_dataset.py \
    --symbols AAPL MSFT GOOGL AMZN TSLA BTC-USD ETH-USD \
    --dataset-name tech_crypto
```

### Option 3: Batch Processing
Process symbols in batches to avoid memory issues:
```bash
bash strategytraining/batch_collect.sh
```

Then merge:
```bash
python strategytraining/merge_datasets.py
```

---

## 🔧 Configuration Options

### Window Settings

**Default (7-day windows, 7-day stride)**
```bash
python strategytraining/collect_position_sizing_dataset.py \
    --window-days 7 \
    --stride-days 7
```

**Longer overlapping windows (14-day windows, 7-day stride)**
```bash
python strategytraining/collect_position_sizing_dataset.py \
    --window-days 14 \
    --stride-days 7 \
    --dataset-name long_windows
```

**Non-overlapping windows (7-day windows, 7-day stride)**
```bash
python strategytraining/collect_position_sizing_dataset.py \
    --window-days 7 \
    --stride-days 7 \
    --dataset-name no_overlap
```

### Symbol Selection

**First N symbols only**
```bash
python strategytraining/collect_position_sizing_dataset.py \
    --max-symbols 10
```

**Specific categories**
```bash
# Tech stocks
python strategytraining/collect_position_sizing_dataset.py \
    --symbols AAPL MSFT GOOGL AMZN META NVDA \
    --dataset-name tech_stocks

# Cryptocurrencies
python strategytraining/collect_position_sizing_dataset.py \
    --symbols BTC-USD ETH-USD SOL-USD AVAX-USD \
    --dataset-name crypto

# ETFs
python strategytraining/collect_position_sizing_dataset.py \
    --symbols SPY QQQ VTI XLK \
    --dataset-name etfs
```

---

## 📈 Analysis Examples

### Basic Statistics
```bash
python strategytraining/analyze_dataset.py \
    strategytraining/datasets/full_dataset_20250131_120000
```

### Symbol-Specific Analysis
```bash
python strategytraining/analyze_dataset.py \
    strategytraining/datasets/full_dataset_20250131_120000 \
    --symbol AAPL
```

### Export Full Report
```bash
python strategytraining/analyze_dataset.py \
    strategytraining/datasets/full_dataset_20250131_120000 \
    --export analysis_report.json
```

---

## 🐍 Programmatic Usage

### Collect Dataset
```python
from strategytraining import DatasetCollector

collector = DatasetCollector(
    data_dir='trainingdata/train',
    output_dir='strategytraining/datasets',
    window_days=7,
    stride_days=7
)

# Collect data
symbols = ['AAPL', 'MSFT', 'BTC-USD']
collector.collect_all_symbols(symbols=symbols)

# Save
paths = collector.save_dataset(dataset_name='my_dataset')
print(f"Dataset saved to: {paths['trades_path']}")
```

### Analyze Dataset
```python
from strategytraining import DatasetAnalyzer

analyzer = DatasetAnalyzer('strategytraining/datasets/my_dataset_20250131_120000')

# Get statistics
stats = analyzer.get_summary_statistics()
print(f"Total trades: {stats['total_trades']}")
print(f"Win rate: {stats['trade_stats']['win_rate']:.2%}")

# Compare stocks vs crypto
comparison = analyzer.compare_stocks_vs_crypto()
print(f"Stocks avg return: {comparison['stocks']['avg_return']:.2%}")
print(f"Crypto avg return: {comparison['crypto']['avg_return']:.2%}")

# Analyze specific symbol
analysis = analyzer.analyze_by_symbol('AAPL')
print(f"AAPL total PnL: ${analysis['trade_stats']['total_pnl']:,.2f}")
```

### Direct DataFrame Access
```python
from strategytraining import DatasetAnalyzer

analyzer = DatasetAnalyzer('strategytraining/datasets/my_dataset_20250131_120000')

# Access raw data
trades_df = analyzer.trades_df
summaries_df = analyzer.summaries_df
positions_df = analyzer.positions_df

# Custom analysis
profitable = trades_df[trades_df['pnl'] > 0]
print(f"Profitable trades: {len(profitable)}")

# Group by symbol
pnl_by_symbol = trades_df.groupby('symbol')['pnl'].sum()
print(pnl_by_symbol.sort_values(ascending=False))
```

---

## 📁 Dataset Structure

After collection, you'll have 4 files per dataset:

```
strategytraining/datasets/
├── my_dataset_20250131_120000_trades.parquet       # Individual trades
├── my_dataset_20250131_120000_summaries.parquet    # Window summaries
├── my_dataset_20250131_120000_positions.parquet    # Position snapshots
└── my_dataset_20250131_120000_metadata.json        # Dataset metadata
```

### Trades DataFrame
Columns: `entry_timestamp`, `exit_timestamp`, `entry_price`, `exit_price`, `position_size`, `pnl`, `pnl_pct`, `duration_bars`, `symbol`, `window_num`, `is_crypto`

### Summaries DataFrame
Columns: `initial_capital`, `final_capital`, `total_return`, `num_trades`, `sharpe_ratio`, `max_drawdown`, `win_rate`, `avg_pnl`, `symbol`, `window_num`, `start_time`, `end_time`

### Positions DataFrame
Columns: `timestamp`, `action`, `position`, `price`, `capital`, `equity`, `symbol`, `window_num`

---

## 🎯 Next Steps

After collecting your dataset:

1. **Feature Engineering**: Extract features for ML models
   - Recent returns, volatility, volume
   - Position sizing history
   - Market regime indicators

2. **Train Position Sizing Model**: Use collected data to train
   - Random Forest / XGBoost for position size prediction
   - LSTM for sequential patterns
   - Reinforcement learning for optimal sizing

3. **Backtest**: Validate learned strategy on holdout data

4. **Deploy**: Integrate with live trading system

---

## 🐛 Troubleshooting

**No data collected**
- Check `trainingdata/train/` has CSV files
- Verify CSV format: `timestamp,Open,High,Low,Close,Volume`
- Ensure at least 500 data points per symbol

**Memory issues**
- Use `--max-symbols` to limit scope
- Run batch collection: `bash strategytraining/batch_collect.sh`
- Process symbols one at a time

**Slow performance**
- Normal: ~10-30 minutes for 10 symbols
- Use SSD storage if available
- Close other applications

**Import errors**
- Install dependencies: `uv pip install pandas numpy tqdm pyarrow`
- Ensure running from project root

---

## 📚 More Information

- Full documentation: `strategytraining/README.md`
- Examples: `strategytraining/example_usage.py`
- Source code: `strategytraining/collect_position_sizing_dataset.py`

---

## ✅ Verification

After running the quick test, verify everything works:

```bash
# 1. Check dataset was created
ls -lh strategytraining/datasets/

# 2. Verify file sizes (should be ~1-5 MB for test)
du -sh strategytraining/datasets/test_dataset_*

# 3. Run analysis
python strategytraining/analyze_dataset.py \
    strategytraining/datasets/test_dataset_20250131_120000

# 4. Check you see statistics output
```

If you see statistics and no errors, you're ready to go! 🎉
