# Portfolio Selection: Strategy-Aware Fix

## Problem

The `build_portfolio()` function was filtering out profitable trades because it used **blanket strategy filters** instead of checking the **selected strategy's returns**.

### Example: BTCUSD

**Backtest Results:**
```
Strategy          Return   Sharpe   AnnualRet
----------------- -------- -------- ---------
Simple            -0.0056  -0.6070  -0.3336   ❌
MaxDiffAlwaysOn    0.0953  29.2080   5.7986   ✅✅✅ Excellent!
```

**Old Logic:**
```python
# Required ALL strategies to be profitable
if (
    data.get("avg_return", 0) > 0 and
    data.get("unprofit_shutdown_return", 0) > 0 and
    data.get("simple_return", 0) > 0  # ❌ BTCUSD has -0.0056
):
    picks[symbol] = data
```

**Result:** BTCUSD excluded from portfolio despite 5.8% annual MaxDiffAlwaysOn return!

## Solution

Changed to **strategy-aware filtering** - only check if the **selected strategy** is profitable:

```python
# Check if the SELECTED strategy has positive returns
strategy = data.get("strategy", "simple")
strategy_return_key = f"{strategy}_return"
strategy_return = data.get(strategy_return_key, 0)

# Include if selected strategy is profitable
if data.get("avg_return", 0) > 0 and strategy_return > 0:
    picks[symbol] = data
```

## Changes Made

### 1. Core Picks (trade_stock_e2e.py:2192-2199)
Changed from requiring `simple_return > 0` to checking `{strategy}_return > 0`

### 2. Fallback Picks (trade_stock_e2e.py:2209-2216)
Changed from `simple_return > 0` to `{strategy}_return > 0`

### 3. Simplified Mode (trade_stock_e2e.py:2164-2171)
Changed from `avg_return > 0` to `{strategy}_return > 0`

## Test Results

```
Test Results:
  BTCUSD: strategy=maxdiffalwayson return=0.0953 (simple_return=-0.0056)
  ETHUSD: strategy=maxdiffalwayson return=0.1440 (simple_return=-0.0177)
  GOOG: strategy=simple return=0.0030 (simple_return=0.0030)

Portfolio selected 3 symbols:
  ✅ ETHUSD: strategy=maxdiffalwayson return=0.1440
  ✅ BTCUSD: strategy=maxdiffalwayson return=0.0953
  ✅ GOOG: strategy=simple return=0.0030

✅ SUCCESS: BTCUSD included despite negative simple_return!
```

## Impact

- **Before:** MaxDiffAlwaysOn trades blocked by portfolio selection
- **After:** Best annual return strategy gets selected
- **Fallback:** If strategy can't execute (EV=0, e.g., can't short crypto), next best strategy selected

## Edge Case Handling

The strategy selection already handles cases where a strategy can't execute:
- If a strategy is blocked/disabled, it gets `strategy_return = 0` or negative
- Portfolio builder skips symbols where `strategy_return <= 0`
- Falls back to next viable symbol in ranking

This ensures we only trade strategies that can actually execute and are profitable.
