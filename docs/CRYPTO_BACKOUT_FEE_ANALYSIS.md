# Crypto Backout Fee Analysis

## Problem
The current `backout_near_market()` function uses market orders for crypto after a timeout or near market close, which incurs **0.25% taker fees** on Alpaca.

## Evidence from Logs
```
2025-11-03 22:12:34 | trade_stock_e2e.py:manage_market_close:3228 INFO | Closing position for ETHUSD due to maxdiff strategy underperforming (avg return -0.0074)
2025-11-03 22:12:34 | src.process_utils:backout_near_market:392 - Running command python scripts/alpaca_cli.py backout_near_market ETHUSD
```

## Current Behavior (scripts/alpaca_cli.py:317-406)
1. **Initial phase**: Uses limit orders with progressive crossing
2. **Timeout/Market Close**: Switches to market order after `market_after_minutes` or near market close
3. **Spread check**: Only uses market order if spread < 1% (`BACKOUT_MARKET_MAX_SPREAD_PCT`)

## Fee Impact

### Taker Fees (Market Orders)
- **Crypto**: 0.25% per trade
- **Stocks**: $0 (but SEC fees ~0.00278% on sells)

### Example Calculation
- Position size: $10,000 ETHUSD
- Market order close: **$25 fee** (0.25%)
- vs Limit order close: **$0 fee**

For a strategy with avg return of -0.74%, paying 0.25% in fees worsens it to -0.99%.

## Recommendations

### Option 1: Disable Market Orders for Crypto ✅ RECOMMENDED
```python
# In backout_near_market()
if minutes_since_start >= effective_market_after or force_market_due_to_time:
    # NEW: Skip market orders for crypto
    if pair in crypto_symbols:
        logger.warning(
            "Crypto symbol %s - using aggressive limit order instead of market to avoid taker fees",
            pair
        )
        # Use very aggressive limit (cross spread significantly)
        pct_above_market = -0.05 if is_long else 0.05  # 5% cross
    else:
        # Stocks: proceed with market order check
        spread_pct = _current_spread_pct(pair)
        ...
```

### Option 2: More Aggressive Limit Orders for Crypto
- Start limit ramp earlier
- Use tighter initial offsets
- Cross spread more aggressively (1-2% instead of 0.5%)

### Option 3: Extended Timeout for Crypto
- Increase `market_after_minutes` to 120 for crypto (vs 60 for stocks)
- Gives more time for limit orders to fill

## MaxDiffAlwaysOn Strategy Behavior

### Current Implementation
From `backtest_test3_inline.py:525-704`:
- Calculates profit assuming **both entry and exit fill**
- No explicit handling of unfilled positions
- No end-of-day close logic

### Question: Should maxdiffalwayson close at EOD?

**Option A: Close at EOD (Instant Close)**
- ✅ Frees capital for next day
- ✅ No overnight risk
- ❌ Pays taker fees (0.25% for crypto)
- ❌ May close winning positions prematurely

**Option B: Keep Open (Until Fill)**
- ✅ No close fees
- ✅ Position can fill on subsequent days
- ❌ Capital tied up
- ❌ Opportunity cost (~5% annual = 0.014% per day)

### Fee Comparison

For a typical maxdiffalwayson trade:
```
Scenario: $10,000 position, 30% don't fill on day 1

Instant Close:
- Close 30 unfilled positions at market
- Fee: 30 * $10,000 * 0.0025 = $75
- Opportunity cost: $0

Keep Open (avg 3 days to fill):
- Close fee: $0
- Opportunity cost: $10,000 * 0.30 * (0.05/365) * 3 = $12.33

Winner: KEEP OPEN saves $62.67
```

## Test Results Needed

Run `test_backtest4_instantclose_inline.py` to compare:
1. Gross returns (should be same)
2. Close fees (instant close pays more)
3. Opportunity cost (keep open pays more)
4. Net returns (which is higher?)

### Expected Outcome
**Hypothesis**: For crypto, KEEP_OPEN strategy wins because:
- Taker fees (0.25%) >> Opportunity cost (~0.014%/day)
- Even if position takes 5 days to fill: 0.014% * 5 = 0.07% < 0.25%

For stocks, closer race because:
- No taker fees (just SEC fees ~0.00278%)
- Opportunity cost becomes main factor

## Implementation Roadmap

1. **Immediate**: Document current behavior ✅
2. **Short-term**: Add crypto market order protection to backout_near_market()
3. **Medium-term**: Integrate actual backtest data into test_backtest4_instantclose_inline.py
4. **Long-term**: Implement adaptive close policy based on backtested results

## Related Files
- `scripts/alpaca_cli.py:317-406` - backout_near_market()
- `trade_stock_e2e.py:manage_market_close()` - Calls backout
- `backtest_test3_inline.py:525` - maxdiffalwayson strategy
- `test_backtest4_instantclose_inline.py` - New comparison test
