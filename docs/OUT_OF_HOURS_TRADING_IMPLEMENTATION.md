# Out-of-Hours Trading Implementation

## Summary

Implemented support for out-of-hours stock trading with strict safety restrictions on market orders to prevent losses from low liquidity and high spreads.

## Key Features

### 1. Market Order Restrictions

**Market orders are NEVER allowed during:**
- Pre-market hours (before 9:30 AM ET)
- After-hours trading (after 4:00 PM ET)

**Market orders are also blocked when:**
- Spread exceeds 1% (configurable via `MARKET_ORDER_MAX_SPREAD_PCT`) when closing positions
- This prevents getting hit by wide spreads during low-liquidity periods

### 2. Limit Orders Work Anytime

- Limit orders work during regular hours, pre-market, and after-hours
- No restrictions on limit order placement based on market hours
- This allows full out-of-hours trading capability with proper price control

### 3. Backwards Compatibility

- All existing functionality preserved
- Existing tests continue to pass
- Only adds additional safety checks, doesn't break existing behavior

## Implementation Details

### Files Modified

1. **alpaca_wrapper.py**
   - Added `MARKET_ORDER_MAX_SPREAD_PCT` env var (default: 0.01 = 1%)
   - Added `_calculate_spread_pct()` helper function
   - Added `_can_use_market_order()` validation function
   - Updated `open_market_order_violently()` to check market hours
   - Updated `close_position_violently()` to check market hours and spread

2. **scripts/alpaca_cli.py**
   - Updated documentation to clarify market order restrictions
   - Added warnings to `violently_close_all_positions()` function
   - Existing `backout_near_market` already had spread checking (now consistent)

3. **tests/prod/brokers/test_alpaca_wrapper.py**
   - Added 6 new integration tests covering:
     - Market orders blocked when market closed
     - Market orders allowed when market open
     - Market orders blocked when spread too high
     - Market orders allowed when spread acceptable
     - Limit orders work when market closed
     - force_open_the_clock flag for out-of-hours

4. **test_out_of_hours_integration.py**
   - New integration test script for PAPER=1 testing
   - Tests all safety restrictions with real API calls

## Configuration

### Environment Variables

```bash
# Maximum spread percentage for market orders when closing positions
# Default: 0.01 (1%)
export MARKET_ORDER_MAX_SPREAD_PCT=0.01

# Same setting for backout operations (already existed)
export BACKOUT_MARKET_MAX_SPREAD_PCT=0.01
```

## Usage Examples

### Out-of-Hours Trading (Recommended)

```bash
# Ramp into a position - works during and after market hours
# Uses limit orders which are safe with controlled pricing
PAPER=1 python scripts/alpaca_cli.py ramp_into_position AAPL buy
```

### Market Hours Only Operations

```bash
# Market orders only work during regular market hours
# Will fail with error during pre-market/after-hours
PAPER=1 python scripts/alpaca_cli.py close_position_violently AAPL
```

### Gradual Exit with Spread Protection

```bash
# Backout uses limit orders by default
# Falls back to market orders only if:
# 1. Market is open AND
# 2. Spread <= 1% AND
# 3. Time deadline reached
PAPER=1 python scripts/alpaca_cli.py backout_near_market AAPL
```

## Testing

### Run Unit Tests

```bash
python -m pytest tests/prod/brokers/test_alpaca_wrapper.py -v
python -m pytest tests/prod/scripts/test_alpaca_cli.py -v
```

### Run Integration Tests with Paper Account

```bash
PAPER=1 python test_out_of_hours_integration.py
```

## Safety Guarantees

1. **No market orders outside regular hours** - Prevents losses from wide spreads in pre-market/after-hours
2. **No market orders when spread > 1%** - Protects against high slippage when exiting positions
3. **Limit orders always available** - Full trading capability maintained with proper price control
4. **Backwards compatible** - Existing workflows continue to work

## Technical Details

### Spread Calculation

```python
spread_pct = (ask_price - bid_price) / mid_price
```

Where `mid_price = (ask_price + bid_price) / 2`

### Market Hours Detection

Uses Alpaca's `get_clock()` API to determine if market is open:
- Regular hours: 9:30 AM - 4:00 PM ET
- Pre-market: Before 9:30 AM ET (market orders blocked)
- After-hours: After 4:00 PM ET (market orders blocked)

### Validation Flow

```
Market Order Attempt
    ↓
Is market open?
    ├─ No → BLOCKED (use limit orders)
    └─ Yes → Continue
         ↓
Is closing position?
    ├─ No → ALLOWED
    └─ Yes → Check spread
         ↓
Is spread <= 1%?
    ├─ No → BLOCKED (use limit orders)
    └─ Yes → ALLOWED
```

## Migration Notes

### For Existing Code

No changes required! All existing code continues to work. The new restrictions only add safety checks.

### For New Code

**Recommended approach for out-of-hours trading:**

```python
# Use limit orders for out-of-hours safety
from alpaca_wrapper import open_order_at_price_or_all

# This works during and outside market hours
result = open_order_at_price_or_all(
    symbol="AAPL",
    qty=10,
    side="buy",
    price=150.50  # Your limit price
)
```

**Avoid for out-of-hours:**

```python
# This will fail outside market hours
from alpaca_wrapper import open_market_order_violently

result = open_market_order_violently(
    symbol="AAPL",
    qty=10,
    side="buy"
)  # Returns None if market closed
```

## Future Enhancements

Potential improvements for consideration:

1. Add configurable spread thresholds per symbol class (stocks vs crypto)
2. Add time-weighted spread checking (average over N minutes)
3. Add position size considerations to spread checking
4. Add alerts/notifications when market orders are blocked

## References

- [Alpaca Markets API Documentation](https://alpaca.markets/docs/)
- [Market Hours Information](https://www.nyse.com/markets/hours-calendars)
- Test file: `tests/prod/brokers/test_alpaca_wrapper.py`
- Integration test: `test_out_of_hours_integration.py`
