# MaxDiff Live Trading Alignment

## Problem
MaxDiff strategy showed **11.7-15.7 Sharpe** in backtests but wasn't trading live due to mismatched gates and filters.

## Root Cause
The backtest (backtest_test3_inline.py:307-522) has **simple logic**:
- Decide direction: buy if `high_pred > low_pred`, else sell
- Optimize entry/exit ±3%
- Trade every day with full position size
- **NO gates, NO Kelly, NO consensus checks**

But live trading (trade_stock_e2e.py) had 7+ blockers not in backtest:
1. Consensus checks (Toto+Kronos agreement required)
2. Kelly fraction <= 0 blocking
3. Edge gates (minimum movement threshold)
4. Cooldowns after losses
5. Recent returns filters
6. Walk-forward Sharpe cutoffs
7. Sign flip checks (calibrated vs predicted direction)

## Fixes Applied

### 1. Skip calibration for MaxDiff (line 1676)
```python
if calibrated_move_pct is not None and selected_strategy != "maxdiff":
    expected_move_pct = calibrated_move_pct
```
**Why**: MaxDiff uses its own high/low target prices, not calibrated close predictions

### 2. Remove sign flip checks entirely (lines 1687-1689)
```python
# Strategy already determined position_side based on its own logic.
# Don't second-guess with calibrated close prediction - strategies may use
# mean reversion, high/low targets, or other logic that differs from close prediction.
```
**Why**: Each strategy determines position_side using its own logic. Second-guessing with calibrated close prediction blocks legitimate strategies like mean reversion and high/low trading. The strategy already made its directional decision - respect it.

### 3. Bypass consensus checks for MaxDiff (line 1893)
```python
if consensus_reason and not is_maxdiff:
    gating_reasons.append(consensus_reason)
```
**Why**: Backtest doesn't require Toto+Kronos agreement - MaxDiff has own predictions

### 4. Bypass Kelly fraction gate for MaxDiff (line 1898)
```python
if kelly_fraction <= 0 and not is_maxdiff:
    gating_reasons.append("Kelly fraction <= 0")
```
**Why**: Backtest uses full position size, not Kelly

### 5. Bypass cooldown for MaxDiff (line 1895)
```python
if not cooldown_ok and not kronos_only_mode and not is_maxdiff:
    gating_reasons.append("Cooldown active after recent loss")
```
**Why**: Backtest trades every day regardless of previous losses

### 6. Bypass recent returns filter for MaxDiff (line 1902)
```python
if recent_sum is not None and recent_sum <= 0 and not is_maxdiff:
    gating_reasons.append(f"Recent {best_strategy} returns sum {recent_sum:.4f} <= 0")
```
**Why**: Backtest doesn't filter based on recent performance

### 7. Bypass edge gate for MaxDiff (line 1884)
```python
if not edge_ok and not is_maxdiff:
    gating_reasons.append(edge_reason)
```
**Why**: Backtest trades whenever high_pred != low_pred, no minimum movement

### 8. Bypass walk-forward Sharpe cutoff for MaxDiff (line 1856)
```python
if not SIMPLIFIED_MODE and not is_maxdiff:
    # apply walk-forward filters
```
**Why**: Backtest doesn't have walk-forward validation gates

### 9. Use full position sizing for MaxDiff (line 2529)
```python
if data.get("strategy") == "maxdiff":
    kelly_value = 1.0
    logger.info(f"{symbol}: MaxDiff using full position size (no Kelly scaling)")
```
**Why**: Backtest uses full size, not Kelly-scaled positions

## Expected Result
MaxDiff should now:
- ✅ Trade when selected (no sign flip blocks)
- ✅ Use full position size (matching backtest)
- ✅ Trade both long and short
- ✅ Ignore consensus requirements
- ✅ Ignore Kelly, cooldown, edge, and Sharpe filters
- ✅ Match backtest Sharpe 11-15+ in live trading

## Still Applied
- Spread checks (realistic trading costs)
- Symbol-specific side restrictions (MARKETSIM_SYMBOL_SIDE_MAP)
- Position exists/rebalance logic

## Test
Run trading system and verify:
1. No "calibrated move flipped sign" skips for MaxDiff
2. "MaxDiff using full position size" log messages
3. Actual trades executed when MaxDiff selected
4. Performance matching backtest expectations
