## 🚀 COLLECT ALL STRATEGY PNL DATA

This guide will help you collect the complete strategy PnL dataset across all symbols and strategies.

### What You'll Get

A comprehensive dataset showing how each strategy performs on each symbol over time:

- **7 strategies** tested on each symbol
- **52 windows per symbol** (rolling 7-day windows across 1 year)
- **All trades, PnL, and performance metrics** per (symbol, strategy, window)
- **Perfect for training position sizing algorithms**

### Strategies Tracked

1. **simple_strategy** - Predicted close price changes
2. **all_signals_strategy** - Average of close/high/low predictions
3. **entry_takeprofit** - Profit potential from high prices
4. **highlow** - Range-based trading
5. **maxdiff** - Maximum directional edge (your current strategy!)
6. **ci_guard** - Conservative blended approach
7. **buy_hold** - Buy and hold baseline

---

## Quick Test (5 minutes)

Test with 2 symbols first:

```bash
python strategytraining/quick_test_strategies.py
```

This will:
- Process AAPL and BTC-USD
- Run all 7 strategies on each
- Generate ~100 strategy-window records
- Save to `test_strategy_dataset_*.parquet`

Then analyze:
```bash
# Find your dataset path
ls strategytraining/datasets/test_strategy_dataset_*_metadata.json

# Analyze (replace TIMESTAMP)
python strategytraining/analyze_strategy_dataset.py \
    strategytraining/datasets/test_strategy_dataset_TIMESTAMP
```

---

## Full Collection

### Option 1: All Symbols (Recommended)

Collect data for ALL symbols in trainingdata/train/:

```bash
python strategytraining/collect_strategy_pnl_dataset.py \
    --dataset-name full_strategy_dataset
```

**Time**: 2-4 hours (100+ symbols)
**Output**: ~500MB of data
**Records**: 30,000-50,000 (symbol, strategy, window) combinations

### Option 2: Specific Categories

**Major Tech Stocks**
```bash
python strategytraining/collect_strategy_pnl_dataset.py \
    --symbols AAPL MSFT GOOGL AMZN TSLA NVDA META \
    --dataset-name tech_stocks
```

**Cryptocurrencies**
```bash
python strategytraining/collect_strategy_pnl_dataset.py \
    --symbols BTC-USD ETH-USD SOL-USD AVAX-USD ATOM-USD ADA-USD DOGE-USD \
    --dataset-name crypto
```

**ETFs**
```bash
python strategytraining/collect_strategy_pnl_dataset.py \
    --symbols SPY QQQ VTI XLK XLE XLI EFA EEM \
    --dataset-name etfs
```

**Everything Important**
```bash
python strategytraining/collect_strategy_pnl_dataset.py \
    --symbols AAPL MSFT GOOGL AMZN TSLA NVDA META \
              BTC-USD ETH-USD SOL-USD \
              SPY QQQ VTI \
    --dataset-name core_assets
```

### Option 3: Batch Processing

For very large collections, process in batches:

```bash
# Process first 20 symbols
python strategytraining/collect_strategy_pnl_dataset.py \
    --max-symbols 20 \
    --dataset-name batch1

# Then next 20, etc...
```

---

## After Collection

### Analyze Results

```bash
# Basic analysis
python strategytraining/analyze_strategy_dataset.py \
    strategytraining/datasets/full_strategy_dataset_TIMESTAMP

# Export detailed report
python strategytraining/analyze_strategy_dataset.py \
    strategytraining/datasets/full_strategy_dataset_TIMESTAMP \
    --export strategy_analysis_report.json

# Analyze specific symbol
python strategytraining/analyze_strategy_dataset.py \
    strategytraining/datasets/full_strategy_dataset_TIMESTAMP \
    --symbol AAPL
```

### What You'll See

**Strategy Rankings**
- Which strategies work best overall
- Average returns, Sharpe ratios, win rates
- Total PnL per strategy

**Top (Symbol, Strategy) Pairs**
- Best 20 combinations by Sharpe ratio
- Shows which strategy works best for each symbol
- Consistency metrics (% profitable windows)

**Symbol-Strategy Matrix**
- Best strategy for each symbol
- Performance breakdown by strategy
- Easy to see patterns (e.g., "maxdiff works best for AAPL")

---

## Use Cases

### 1. Position Sizing Algorithm

Train ML model on this data:
- **Input**: Symbol, strategy, recent performance, market conditions
- **Output**: Optimal position size
- **Target**: Maximize risk-adjusted returns

### 2. Strategy Selection

Learn which strategy to use for each symbol:
- Some symbols work better with maxdiff
- Others with highlow or entry_takeprofit
- Adapt strategy selection dynamically

### 3. Risk Management

Understand drawdown patterns:
- Which (symbol, strategy) pairs are risky
- When to reduce position sizes
- Stop-loss levels per strategy

### 4. Portfolio Construction

Build diversified portfolio:
- Select best (symbol, strategy) pairs
- Balance across asset classes
- Optimize for Sharpe ratio

---

## Dataset Schema

### Strategy Performance (`*_strategy_performance.parquet`)

Main dataset with one row per (symbol, strategy, window):

| Column | Description |
|--------|-------------|
| symbol | Asset symbol (AAPL, BTC-USD, etc.) |
| strategy | Strategy name (maxdiff, highlow, etc.) |
| window_num | Window number (0-51) |
| start_time | Window start timestamp |
| end_time | Window end timestamp |
| initial_capital | Starting capital ($100,000) |
| final_capital | Ending capital |
| total_return | Return % for this window |
| total_pnl | Profit/loss in dollars |
| num_trades | Number of trades executed |
| sharpe_ratio | Risk-adjusted return |
| max_drawdown | Worst drawdown % |
| win_rate | % of profitable trades |
| avg_pnl | Average PnL per trade |
| avg_duration | Average trade duration (bars) |
| is_crypto | True for crypto symbols |

### Trades (`*_trades.parquet`)

Individual trades with one row per trade:

| Column | Description |
|--------|-------------|
| symbol | Asset symbol |
| strategy | Strategy used for this trade |
| window_num | Window number |
| entry_timestamp | Trade entry time |
| exit_timestamp | Trade exit time |
| entry_price | Entry price |
| exit_price | Exit price |
| position_size | Number of shares/units |
| pnl | Profit/loss |
| pnl_pct | Return % |
| duration_bars | How long position was held |
| signal_at_entry | Strategy signal when entered |
| signal_at_exit | Strategy signal when exited |

---

## Tips for Best Results

### 1. Start Small
- Test with 5-10 symbols first
- Verify data looks good
- Then run full collection

### 2. Check Data Quality
After collection, verify:
```python
import pandas as pd

# Load dataset
df = pd.read_parquet('strategytraining/datasets/full_strategy_dataset_TIMESTAMP_strategy_performance.parquet')

# Check coverage
print(f"Symbols: {df['symbol'].nunique()}")
print(f"Strategies: {df['strategy'].nunique()}")
print(f"Total records: {len(df)}")

# Check for missing data
print(df.isnull().sum())
```

### 3. Watch for Errors
If some symbols fail:
- Check data quality in trainingdata/train/
- Verify CSV format is correct
- Some symbols may have insufficient history

### 4. Save Raw Logs
```bash
python strategytraining/collect_strategy_pnl_dataset.py \
    --dataset-name full_dataset 2>&1 | tee collection.log
```

---

## Expected Output

For ~100 symbols:

```
COLLECTING STRATEGY PNL DATASET
================================================================================
Symbols: 100
Strategies: 7
Window: 7 days, Stride: 7 days
Output: strategytraining/datasets
Strategies: simple_strategy, all_signals_strategy, entry_takeprofit, highlow, maxdiff, ci_guard, buy_hold

Processing AAPL...
  ✓ 52 windows x 7 strategies = 364 records
  ✓ 1,243 trades

Processing MSFT...
  ✓ 52 windows x 7 strategies = 364 records
  ✓ 1,189 trades

... (98 more symbols)

SAVING DATASET
================================================================================
✓ Saved 35,000 strategy-window results
✓ Saved 120,000 trades

DATASET SUMMARY
================================================================================
Symbols: 100
Strategies: 7
Avg windows per symbol: 50
Total records: 35,000

Per Strategy Performance:
  maxdiff:
    Avg Return: 1.23%
    Avg Sharpe: 0.45
    Win Rate: 53%
    Total PnL: $123,456

  ... (6 more strategies)

SUCCESS! Dataset ready for position sizing training
```

---

## Next: Train Position Sizing Model

Once you have the dataset, you can:

1. **Feature Engineering**: Extract features from recent windows
2. **Train Model**: XGBoost, Random Forest, or Neural Network
3. **Predict Position Sizes**: For new (symbol, strategy) pairs
4. **Backtest**: Validate on holdout data
5. **Deploy**: Use in live trading

Example features:
- Recent returns (last 3 windows)
- Strategy Sharpe ratio trend
- Win rate trend
- Volatility
- Market regime
- Symbol characteristics (is_crypto, sector, etc.)

Target: Optimal position size (0-100% of capital)

---

## Questions?

- Check `strategytraining/README.md` for detailed documentation
- Run `python strategytraining/collect_strategy_pnl_dataset.py --help`
- Example usage in `strategytraining/example_usage.py`
