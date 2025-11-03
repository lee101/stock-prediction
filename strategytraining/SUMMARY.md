# Strategy Training - Position Sizing Dataset System

## 🎯 What We Built

A **comprehensive dataset collection system** that gathers PnL data for **multiple trading strategies** across **all symbols** over rolling time windows. This creates the perfect dataset for training **position sizing algorithms**.

### The Core Idea

Instead of collecting data for just one strategy, we now:
1. Run **7 different strategies** on each symbol
2. Track their **PnL, trades, and performance** over rolling 7-day windows
3. Create a dataset showing which **(symbol, strategy)** combinations work best
4. Use this to train ML models that **dynamically size positions** based on strategy performance

## 📊 Strategies Tracked

All strategies from `marketsimulator/backtest_test3_inline.py`:

1. **simple_strategy** - Predicted close price changes
2. **all_signals_strategy** - Average of close/high/low predictions
3. **entry_takeprofit** - Profit potential from high prices
4. **highlow** - Range-based trading
5. **maxdiff** - Maximum directional edge ⭐ (your current strategy!)
6. **ci_guard** - Conservative blended approach
7. **buy_hold** - Buy and hold baseline

## 📁 Files Created

### Core Collection Scripts
- **`collect_strategy_pnl_dataset.py`** (21KB) - Main collector for multi-strategy data
- **`collect_position_sizing_dataset.py`** (21KB) - Original single-strategy collector
- **`run_collection.sh`** - One-click full collection script

### Analysis Tools
- **`analyze_strategy_dataset.py`** (9.3KB) - Analyze multi-strategy data
- **`analyze_dataset.py`** (13KB) - Analyze single-strategy data

### Testing & Examples
- **`quick_test_strategies.py`** - Quick test with 2 symbols
- **`quick_test.py`** - Original single-strategy test
- **`example_usage.py`** - Code examples

### Utilities
- **`merge_datasets.py`** - Merge batch-collected datasets
- **`batch_collect.sh`** - Process symbols in batches
- **`__init__.py`** - Python package interface

### Documentation
- **`RUN_FULL_COLLECTION.md`** (8.7KB) - Complete collection guide
- **`QUICKSTART.md`** (7.9KB) - 5-minute getting started
- **`README.md`** (7.9KB) - Full documentation
- **`SUMMARY.md`** - This file

## 🚀 Quick Start

### 1. Install Dependencies
```bash
uv pip install pandas numpy tqdm pyarrow
```

### 2. Quick Test (5 minutes)
```bash
python strategytraining/quick_test_strategies.py
```

### 3. Collect Full Dataset (2-4 hours)
```bash
bash strategytraining/run_collection.sh
```

### 4. Analyze Results
```bash
python strategytraining/analyze_strategy_dataset.py \
    strategytraining/datasets/full_strategy_dataset_TIMESTAMP
```

## 📈 What You Get

### Dataset Structure

**1. Strategy Performance** (`*_strategy_performance.parquet`)
- One row per (symbol, strategy, window)
- ~35,000 rows for 100 symbols
- Columns: symbol, strategy, window_num, total_return, sharpe_ratio, win_rate, total_pnl, etc.

**2. Trades** (`*_trades.parquet`)
- Individual trades for each strategy
- ~120,000 trades for 100 symbols
- Columns: symbol, strategy, entry_price, exit_price, pnl, duration, signals, etc.

**3. Metadata** (`*_metadata.json`)
- Dataset info, symbols, strategies, timestamps

### Example Data

```python
import pandas as pd

# Load strategy performance
df = pd.read_parquet('strategytraining/datasets/full_strategy_dataset_20250131_120000_strategy_performance.parquet')

# See which strategies work best for AAPL
aapl = df[df['symbol'] == 'AAPL']
aapl.groupby('strategy')['avg_sharpe'].mean().sort_values(ascending=False)

# Output:
# strategy
# maxdiff                 0.45
# entry_takeprofit        0.38
# highlow                 0.32
# all_signals_strategy    0.28
# ci_guard                0.25
# simple_strategy         0.22
# buy_hold                0.15
```

## 🎓 Use Cases

### 1. Position Sizing Algorithm (Primary Goal!)

Train ML model to predict optimal position sizes:

```python
# Features: symbol, strategy, recent_returns, volatility, etc.
# Target: position_size (0-100% of capital)

from sklearn.ensemble import RandomForestRegressor

X = features  # Recent performance, market conditions
y = optimal_position_sizes  # Learned from dataset

model = RandomForestRegressor()
model.fit(X, y)

# Predict position size for new trades
position_size = model.predict([[symbol, strategy, recent_perf, ...]])
```

### 2. Strategy Selection

Learn which strategy to use for each symbol:
- **AAPL** → maxdiff (Sharpe 0.45)
- **BTC-USD** → highlow (Sharpe 0.52)
- **SPY** → entry_takeprofit (Sharpe 0.38)

### 3. Risk Management

Understand when strategies fail:
- Identify high-drawdown periods
- Reduce positions during risky conditions
- Diversify across strategies

### 4. Portfolio Optimization

Build optimal portfolio:
- Select top (symbol, strategy) pairs
- Weight by Sharpe ratio
- Balance across asset classes

## 📊 Expected Results

For 100 symbols:

```
COLLECTING STRATEGY PNL DATASET
================================================================================
Symbols: 100
Strategies: 7 (simple_strategy, all_signals_strategy, entry_takeprofit, highlow, maxdiff, ci_guard, buy_hold)
Window: 7 days, Stride: 7 days
================================================================================

Processing AAPL... ✓ 52 windows x 7 strategies = 364 records, 1,243 trades
Processing MSFT... ✓ 52 windows x 7 strategies = 364 records, 1,189 trades
Processing BTC-USD... ✓ 52 windows x 7 strategies = 364 records, 2,156 trades
... (97 more)

DATASET SUMMARY
================================================================================
Total (symbol, strategy, window) records: 35,000
Total trades: 120,000
Unique symbols: 100
Unique strategies: 7

Per Strategy Performance:
  maxdiff:
    Avg Return: 1.23%
    Avg Sharpe: 0.45
    Win Rate: 53%
    Total PnL: $123,456

  entry_takeprofit:
    Avg Return: 0.98%
    Avg Sharpe: 0.38
    Win Rate: 51%
    Total PnL: $98,765

  ... (5 more)
```

## 🔍 Analysis Examples

### Strategy Rankings
```bash
python strategytraining/analyze_strategy_dataset.py \
    strategytraining/datasets/full_strategy_dataset_TIMESTAMP
```

Output:
```
STRATEGY RANKINGS
================================================================================
        strategy  avg_return  avg_sharpe  win_rate  total_pnl
         maxdiff       0.0123        0.45      0.53   123456.78
entry_takeprofit       0.0098        0.38      0.51    98765.43
         highlow       0.0087        0.32      0.49    87654.32
...
```

### Top (Symbol, Strategy) Pairs
```
TOP 10 (SYMBOL, STRATEGY) PAIRS
================================================================================
  symbol           strategy  avg_return  avg_sharpe  win_rate   total_pnl
 BTC-USD            highlow       0.0245        0.52      0.61   15678.90
    NVDA            maxdiff       0.0198        0.48      0.58   12345.67
    AAPL            maxdiff       0.0176        0.45      0.55   10234.56
...
```

### Best Strategy per Symbol
```
SYMBOL-STRATEGY MATRIX
================================================================================
   symbol    best_strategy  best_sharpe
     AAPL          maxdiff         0.45
     MSFT          maxdiff         0.42
  BTC-USD          highlow         0.52
  ETH-USD          highlow         0.48
      SPY  entry_takeprofit        0.38
...
```

## 🔧 Key Design Features

### 1. Multiple Strategies
- Runs **all 7 strategies** on each window
- Compares performance side-by-side
- Shows which strategies work best for each symbol

### 2. Rolling Windows
- **52 windows per year** (7-day windows, 7-day stride)
- Captures full year of strategy behavior
- Shows performance across different market conditions

### 3. Real Market Simulation
- Uses `marketsimulator` infrastructure
- Realistic execution with slippage and fees
- Based on actual forecast signals from `backtest_test3_inline.py`

### 4. Comprehensive Metrics
- Returns, Sharpe ratios, drawdowns
- Trade-level data (entry, exit, PnL, duration)
- Strategy signals at entry/exit

### 5. Smart Calendar Handling
- **Stocks**: 252 trading days/year
- **Crypto**: 365 days/year
- Automatic detection based on symbol

## 🎯 Next Steps

After collecting the dataset:

### 1. Feature Engineering
Extract features for ML:
- Recent returns (last 3 windows)
- Sharpe ratio trend
- Win rate trend
- Volatility
- Market regime indicators
- Symbol characteristics

### 2. Train Position Sizing Model
```python
from xgboost import XGBRegressor

# Features
features = [
    'strategy',
    'recent_return_1', 'recent_return_2', 'recent_return_3',
    'sharpe_trend',
    'win_rate_trend',
    'volatility',
    'is_crypto',
]

# Target: optimal position size (as % of capital)
target = 'optimal_position_size'

model = XGBRegressor()
model.fit(X_train, y_train)
```

### 3. Backtest
Validate learned position sizing on holdout data

### 4. Deploy
Integrate with live trading:
```python
# Get recent performance for (symbol, strategy)
recent_perf = get_recent_performance(symbol, strategy)

# Predict position size
position_size = model.predict(recent_perf)

# Execute trade with learned size
execute_trade(symbol, strategy, position_size)
```

## 📚 File Reference

| File | Purpose | Size |
|------|---------|------|
| `collect_strategy_pnl_dataset.py` | Multi-strategy collector | 21KB |
| `analyze_strategy_dataset.py` | Multi-strategy analysis | 9.3KB |
| `run_collection.sh` | One-click collection | 1.7KB |
| `quick_test_strategies.py` | Quick test script | 1.8KB |
| `RUN_FULL_COLLECTION.md` | Complete guide | 8.7KB |
| `QUICKSTART.md` | 5-minute guide | 7.9KB |
| `README.md` | Full documentation | 7.9KB |

## ✅ Ready to Go!

Everything is set up and ready to collect data:

```bash
# Quick test first
python strategytraining/quick_test_strategies.py

# Then full collection
bash strategytraining/run_collection.sh

# Analyze results
python strategytraining/analyze_strategy_dataset.py \
    strategytraining/datasets/full_strategy_dataset_TIMESTAMP
```

## 🎉 What Makes This Special

1. **Comprehensive**: Tracks ALL strategies, not just one
2. **Time-aware**: Rolling windows show performance over time
3. **Realistic**: Uses real market simulator with fees/slippage
4. **Ready for ML**: Perfect format for training position sizing models
5. **Well-documented**: Multiple guides and examples
6. **Production-ready**: Handles stocks, crypto, batch processing, etc.

This dataset will let you train a position sizing algorithm that learns:
- Which strategies work best for each symbol
- When to increase/decrease position sizes
- How to adapt to changing market conditions
- Optimal capital allocation across strategies

**Let's collect that data!** 🚀
