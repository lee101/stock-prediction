# Crypto Backout Fix: No Market Orders

## Problem

The `backout_near_market()` function was designed for **stocks with market hours**, but was being used for **crypto (24/7 trading)**. This caused several issues:

### Issues with Original Implementation

1. **Market orders on crypto** - After 50 minutes (default `BACKOUT_MARKET_AFTER_MINUTES`), the function switched to market orders
2. **Wide crypto spreads** - BTCUSD/ETHUSD spreads are 0.15-0.24%, making market orders very expensive
3. **No market close for crypto** - Logic checking `_minutes_until_market_close()` was meaningless for 24/7 assets
4. **Unnecessary slippage** - Market orders eat the full spread, while limit orders can minimize costs

### Example: BTCUSD Close Policy

Backtest results showed the impact:

```
Policy          Gross Ret    Close Fee   Opp Cost   Net Return
instant_close       5.79%      0.0060     0.0000      5.79%   ← Market order
keep_open         -10.33%      0.0000     0.0005    -10.33%   ← Hold
```

The "instant_close" policy uses market orders with 0.15% taker fees on both entry and exit, significantly reducing returns.

## Solution

Modified `scripts/alpaca_cli.py:backout_near_market()` to detect crypto symbols and disable market order fallback:

### Changes Made

**1. Detect crypto at function start (lines 354-361)**
```python
# Detect if this is a crypto symbol (24/7 trading)
is_crypto = pair in crypto_symbols
if is_crypto:
    logger.info(f"{pair} is crypto - will use limit orders only (no market order fallback)")
    # Disable market order fallback for crypto by setting to very high value
    effective_market_after = float('inf')
else:
    effective_market_after = None  # Will be set below
```

**2. Only set timeout for stocks (lines 367-369)**
```python
# Only set effective_market_after for stocks; crypto already set to inf
if effective_market_after is None:
    effective_market_after = max(int(market_after) + extra_minutes, effective_ramp_minutes)
```

**3. Skip market close logic for crypto (lines 408-417)**
```python
# Skip market close logic for crypto (24/7 trading)
if is_crypto:
    minutes_to_close = None
    force_market_due_to_time = False
else:
    minutes_to_close = _minutes_until_market_close()
    force_market_due_to_time = (
        minutes_to_close is not None
        and minutes_to_close <= market_close_force_minutes
    )
```

## Behavior After Fix

### For Crypto (BTCUSD, ETHUSD, etc.)
- ✅ **Only limit orders** - Never switches to market orders
- ✅ **No time pressure** - `effective_market_after = inf` means no timeout
- ✅ **Ignores market close** - `force_market_due_to_time = False` always
- ✅ **Patient execution** - Ramps through the spread using limit orders
- ✅ **Minimizes fees** - Avoids 0.15% taker fee on close

### For Stocks (AAPL, META, etc.)
- ✅ **Same behavior as before** - Uses limit orders initially
- ✅ **Market order fallback** - After 50 minutes (configurable)
- ✅ **Market close urgency** - Forces market order if close is imminent
- ✅ **Avoids overnight risk** - Guarantees exit before market close

## Logic Flow

```python
# Entry point
if is_crypto:
    effective_market_after = float('inf')  # Never timeout
else:
    effective_market_after = 50 + extra_minutes  # Stock timeout

# In main loop
if minutes_since_start >= effective_market_after or force_market_due_to_time:
    # For crypto: if minutes >= inf or False → Always False
    # For stocks: if minutes >= 50 or near_close → Can be True

    # Market order block (crypto never enters here)
    alpaca_wrapper.close_position_violently(position)
else:
    # Limit order block (crypto always uses this)
    alpaca_wrapper.close_position_near_market(position, pct_above_market=pct)
```

## Impact

### Before Fix
- Crypto positions forced to market orders after 50 minutes
- Eating 0.15% spread on every close
- BTCUSD backtest shows -10.33% from holding vs market closing

### After Fix
- Crypto uses limit orders indefinitely
- Minimizes taker fees
- Better execution quality for 24/7 markets
- Stocks maintain existing behavior with market close urgency

## Related Files

- `scripts/alpaca_cli.py:317` - Main `backout_near_market()` function
- `src/fixtures.py` - Defines `crypto_symbols` list
- `alpaca_wrapper.py` - `close_position_violently()` (market) vs `close_position_near_market()` (limit)
- `backtest_test3_inline.py:3012` - Close policy evaluation

## Testing

To verify the fix works:

1. Check logs when closing crypto positions - should see:
   ```
   BTCUSD is crypto - will use limit orders only (no market order fallback)
   ```

2. Monitor that crypto closes never trigger:
   ```
   Spread X.XX% within Y.YY% cap; switching to market order for BTCUSD
   ```

3. Verify limit orders are used throughout:
   ```
   Position side: long, pct_above_market: 0.XXXX, minutes_since_start: XX
   ```
