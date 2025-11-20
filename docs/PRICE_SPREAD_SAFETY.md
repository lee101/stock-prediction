# Price Spread Safety Protections

## Critical Bug History

### The $2,791 Bug (2025-11-20)

**What happened:**
A catastrophic bug in `hourlycrypto/trade_stock_crypto_hourly.py` line 524 caused the bot to pass `"buy"` instead of `"sell"` when spawning sell watchers. This caused the system to:
- Buy at the **sell price** (higher price)
- Sell at the **buy price** (lower price)
- Lose money on every round trip

**Financial impact:**
- Starting balance: $70,729.75
- Ending balance: $67,938.53
- **Loss: $2,791.22 (3.9%)**

**Root cause:**
```python
# WRONG (before fix):
spawn_close_position_at_maxdiff_takeprofit(
    symbol,
    "buy",  # ← Should be "sell"!
    plan.sell_price,
    ...
)

# CORRECT (after fix):
spawn_close_position_at_maxdiff_takeprofit(
    symbol,
    "sell",  # ✓ Correct
    plan.sell_price,
    ...
)
```

## Protections Implemented

### 1. Price Spread Validation (hourlycrypto/trade_stock_crypto_hourly.py:478-494)

**Enforces minimum 3 basis point (0.03%) spread:**
```python
MIN_SPREAD_PCT = 0.0003  # 3 basis points = 0.03%
required_min_sell = plan.buy_price * (1.0 + MIN_SPREAD_PCT)

if plan.sell_price <= plan.buy_price or plan.sell_price < required_min_sell:
    logger.error("INVALID PRICE SPREAD - BLOCKING ALL TRADES!")
    return  # Exit immediately - no orders placed
```

**Protection against:**
- Inverted prices (sell < buy)
- Insufficient spread (< 3bp)
- Model errors generating invalid prices

### 2. Comprehensive Test Suite (tests/test_price_spread_safety.py)

**14 tests covering:**
- ✓ Inverted price detection
- ✓ Minimum spread enforcement (3bp)
- ✓ Buy/sell side correctness
- ✓ Edge cases and boundary conditions
- ✓ Regression test for the $2,791 bug

**Critical regression test:**
```python
def test_regression_sell_side_bug(self, monkeypatch):
    """
    REGRESSION TEST: Prevent the catastrophic bug where 'buy' was
    hardcoded when spawning sell watchers, causing $2,791 loss.
    """
    # ... test code that would have caught the bug ...

    assert call["side"] == "sell", (
        f"CRITICAL BUG: When selling, side must be 'sell', not '{call['side']}'! "
        f"This bug caused $2,791 loss..."
    )
```

## Code Review Checklist

Before any changes to trading logic, verify:

### For Price Validation:
- [ ] Buy price < Sell price (always!)
- [ ] Spread >= 3 basis points (0.03%)
- [ ] Prices are validated BEFORE spawning watchers
- [ ] Invalid prices block ALL trades (not just log warnings)

### For Order Spawning:
- [ ] When BUYING: `spawn_open_position_at_maxdiff_takeprofit(symbol, "buy", buy_price, ...)`
- [ ] When SELLING: `spawn_close_position_at_maxdiff_takeprofit(symbol, "sell", sell_price, ...)`
- [ ] Side parameter matches the action (buy→"buy", sell→"sell")
- [ ] Price parameter matches the side (buy→buy_price, sell→sell_price)

### For Testing:
- [ ] Test covers inverted prices
- [ ] Test covers insufficient spread
- [ ] Test covers correct side parameter
- [ ] Test covers edge cases (exactly 3bp, etc.)

## Running Safety Tests

```bash
# Run all price spread safety tests
pytest tests/test_price_spread_safety.py -v

# Run with coverage
pytest tests/test_price_spread_safety.py --cov=hourlycrypto --cov-report=term-missing
```

## Architecture Notes

### Function Signatures

```python
# Opening positions (buying or shorting)
spawn_open_position_at_maxdiff_takeprofit(
    symbol: str,
    side: str,        # "buy" or "sell" (for shorting)
    limit_price: float,
    target_qty: float,
    ...
)

# Closing positions (exiting)
spawn_close_position_at_maxdiff_takeprofit(
    symbol: str,
    side: str,        # "sell" (to close long) or "buy" (to close short)
    takeprofit_price: float,
    target_qty: Optional[float],
    ...
)
```

### Typical Usage Pattern

```python
# For long positions (buy to open, sell to close):
if buy_qty > 0:
    spawn_open_position_at_maxdiff_takeprofit(
        symbol, "buy", buy_price, buy_qty, ...
    )

if sell_qty > 0:
    spawn_close_position_at_maxdiff_takeprofit(
        symbol, "sell", sell_price, sell_qty, ...
    )
```

## Related Files

- **Implementation:** `hourlycrypto/trade_stock_crypto_hourly.py`
- **Tests:** `tests/test_price_spread_safety.py`
- **Watcher logic:** `src/process_utils.py`
- **Model logic:** `hourlycryptotraining/model.py` (decode_actions)

## Lessons Learned

1. **Always validate prices before trading** - A simple check would have prevented the $2,791 loss
2. **Test critical paths thoroughly** - Buy/sell logic is too important to leave untested
3. **Use explicit parameters** - Don't hardcode "buy" or "sell", use variables that match intent
4. **Add regression tests immediately** - When a costly bug occurs, add tests to prevent recurrence
5. **Fail fast and loud** - Invalid prices should BLOCK trades, not just log warnings

## Emergency Response

If you suspect a price spread bug:

1. **STOP TRADING IMMEDIATELY**
   ```bash
   pkill -f "watcher"
   pkill -f "trade_stock"
   ```

2. **Check recent trades**
   ```bash
   tail -100 alpaca_cli.log
   tail -100 /tmp/btcusd_hourly_fixed.log
   ```

3. **Look for inverted trades** - Selling below buy price
   ```bash
   grep -E "buy.*sell|INVALID PRICE" *.log
   ```

4. **Verify account balance**
   ```bash
   python -c "import alpaca_wrapper; print(alpaca_wrapper.get_account())"
   ```

5. **Run safety tests**
   ```bash
   pytest tests/test_price_spread_safety.py -v
   ```

## Future Improvements

- [ ] Add price spread validation to neural_daily trading
- [ ] Add spread monitoring dashboard
- [ ] Alert on spread < 5bp (warning threshold)
- [ ] Add daily PnL sanity checks
- [ ] Consider adding pre-commit hooks for trading code
