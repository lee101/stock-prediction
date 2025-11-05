# Close Policy Evaluation Bug

## Problem

The close policy comparison produces **mathematically inconsistent results** with the main backtest.

### Example: UNIUSD

**Main Backtest (70 simulations):**
```
Strategy          Return  Sharpe   AnnualRet
MaxDiff           0.1199  14.68    7.2957    (729%)
MaxDiffAlwaysOn   0.1900  26.52    11.5599   (1155%) ✅ Excellent!
```

**Close Policy Evaluation (10 simulations):**
```
Policy          Gross Ret    Net Return
instant_close      -3.79%       -3.79%   ❌ Negative
keep_open         -10.64%      -10.64%   ❌ Both negative!
```

The main backtest shows +19% return, but the close policy shows -3.79% to -10.64%. These can't both be correct.

## Root Cause

The close policy evaluation **does not use the actual MaxDiffAlwaysOn strategy**. It uses completely different logic.

### Main Backtest: `evaluate_maxdiff_always_on_strategy()`

**File:** `backtest_test3_inline.py:558-738`

**Key logic (lines 655-692):**
```python
# Grid search over high/low multipliers
multiplier_candidates = np.linspace(-0.03, 0.03, 81)  # 81 steps

for high_multiplier in multiplier_candidates:
    for low_multiplier in multiplier_candidates:
        # Test 81 × 81 = 6,561 combinations!
        adjusted_high_pred = high_pred + float(high_multiplier)
        adjusted_low_pred = low_pred + float(low_multiplier)

        buy_returns = calculate_profit_torch_with_entry_buysell_profit_values(
            close_actual, high_actual, adjusted_high_pred,
            low_actual, adjusted_low_pred, buy_indicator
        )

        # Find combination that maximizes total profit
        if total_profit > best_total_profit:
            best_total_profit = total_profit
            best_state = (high_multiplier, low_multiplier, ...)
```

**Result:** Optimizes entry/exit prices by testing 6,561 combinations → **+19% return**

### Close Policy: `evaluate_strategy_with_close_policy()`

**File:** `test_backtest4_instantclose_inline.py:195-336`

**Key logic (lines 261-266):**
```python
# Just use raw predictions, no optimization
buy_indicator = torch.ones_like(close_actual)  # Always buy
buy_returns, buy_unfilled, buy_hold = calculate_profit_with_limit_orders(
    close_actual, high_actual, high_pred, low_actual, low_pred, buy_indicator,
    close_at_eod=close_at_eod, is_crypto=is_crypto
)
```

**Result:** Uses unoptimized predictions → **-3.79% return**

## Technical Details

### Different Profit Calculation Functions

1. **Main backtest uses:**
   - Function: `calculate_profit_torch_with_entry_buysell_profit_values()` (loss_utils.py:283)
   - Parameters: NO `close_at_eod` parameter
   - Logic: Optimized with grid search over multipliers

2. **Close policy uses:**
   - Function: `calculate_profit_with_limit_orders()` (test_backtest4_instantclose_inline.py)
   - Parameters: HAS `close_at_eod` parameter
   - Logic: No optimization, uses raw predictions

These are **completely different trading simulations**.

### Why MaxDiffAlwaysOn Works

The strategy optimizes two parameters:
- `high_multiplier`: Adjusts take-profit level (±3% range)
- `low_multiplier`: Adjusts stop-loss level (±3% range)

Example from BTCUSD backtest:
```
maxdiffalwayson_high_multiplier: 0.0175   (add 1.75% to predicted high)
maxdiffalwayson_low_multiplier: 0.0088    (add 0.88% to predicted low)
```

These optimal multipliers make the difference between -3.79% and +19% return!

## Why This Is Misleading

The close policy comparison is supposed to answer:
> "Should we close positions at end-of-day (instant_close) or hold overnight (keep_open)?"

But instead it's comparing:
> "Unoptimized strategy with instant close vs unoptimized strategy with keep open"

Since BOTH use the wrong strategy, the comparison is meaningless for actual trading decisions.

## Impact

### Current State
- ❌ Close policy recommendations are **unreliable**
- ❌ Both policies show negative returns even though strategy is profitable
- ❌ Users may disable profitable strategies based on misleading data
- ❌ Wasted compute time running broken comparisons

### Example Saved Policy
```json
{
  "policy": "INSTANT_CLOSE",
  "advantage": "16.12%",
  "instant_close_return": -3.79%,
  "keep_open_return": -10.64%
}
```

This says "INSTANT_CLOSE is better" but both are negative! The actual strategy gets +19%.

## Proposed Solutions

### Option 1: Disable Close Policy Evaluation (Quick Fix)

Comment out the close policy call in `backtest_test3_inline.py:3015`:

```python
def evaluate_and_save_close_policy(symbol: str, num_comparisons: int = 10) -> None:
    logger.warning(
        f"Close policy evaluation disabled due to bug (see docs/CLOSE_POLICY_BUG.md). "
        f"Using default policy for {symbol}."
    )
    return  # Skip broken evaluation

    # from test_backtest4_instantclose_inline import compare_close_policies
    # result = compare_close_policies(symbol, num_simulations=num_comparisons)
    # ...
```

**Pros:** Immediate fix, stops misleading results
**Cons:** Loses close policy optimization

### Option 2: Add `close_at_eod` to Main Strategy Functions (Proper Fix)

Modify `calculate_profit_torch_with_entry_buysell_profit_values` to support `close_at_eod`:

```python
def calculate_profit_torch_with_entry_buysell_profit_values(
    y_test, y_test_high, y_test_high_pred,
    y_test_low, y_test_low_pred, y_test_pred,
    *,
    close_at_eod: bool = False  # ← Add parameter
):
    # ... existing logic ...

    if close_at_eod:
        # Force close all positions at end of day
        # Apply trading fees for closing
        profit_values = apply_instant_close_fees(profit_values)

    return profit_values
```

Then update `evaluate_maxdiff_always_on_strategy` to accept and pass through `close_at_eod`.

**Pros:** Proper fix, maintains optimization
**Cons:** Requires refactoring multiple functions

### Option 3: Make Close Policy Call Real Strategy

Modify `evaluate_strategy_with_close_policy` to call the actual strategy:

```python
def evaluate_strategy_with_close_policy(
    last_preds, simulation_data,
    close_at_eod: bool, is_crypto: bool, strategy_name: str
) -> PositionCloseMetrics:

    if strategy_name == "maxdiffalwayson":
        # Call the REAL strategy function
        evaluation, returns, metadata = evaluate_maxdiff_always_on_strategy(
            last_preds, simulation_data,
            trading_fee=CRYPTO_TRADING_FEE if is_crypto else TRADING_FEE,
            trading_days_per_year=365 if is_crypto else 252,
            is_crypto=is_crypto
        )

        # Then adjust for close_at_eod if needed
        if close_at_eod:
            returns = apply_instant_close_adjustment(returns)

        return convert_to_metrics(evaluation, returns)
```

**Pros:** Reuses existing code
**Cons:** Still need to figure out how to apply close_at_eod adjustment

## Recommended Action

**Option 1 (disable evaluation)** for immediate fix, then implement **Option 2 (proper fix)** when time allows.

The current close policy evaluation is worse than useless - it actively misleads by showing negative returns for a profitable strategy.

## Files Affected

- `backtest_test3_inline.py:2995-3030` - Close policy integration
- `test_backtest4_instantclose_inline.py:195-336` - Broken evaluation logic
- `test_backtest4_instantclose_inline.py:341-490` - Comparison runner
- `loss_utils.py:283` - Core profit calculation (needs `close_at_eod` param)
- `hyperparams/close_policy/*.json` - Misleading saved policies

## Detection

If you see close policy results where:
- Main backtest shows positive returns (>5%)
- Close policy shows negative returns for BOTH policies
- The gap is >15 percentage points

Then this bug is affecting your results.
