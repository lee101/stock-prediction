# Watcher Fixes - November 11, 2025

## Issues Found and Fixed

### 1. **Critical: Missing Imports in scripts/maxdiff_cli.py**
**Problem:** Watchers were stuck in endless loop with error:
```
Failed work stealing for UNIUSD: name 'load_latest_forecast_snapshot' is not defined
```

**Root Cause:** Work stealing code tried to call `load_latest_forecast_snapshot()` and `extract_forecasted_pnl()` but they weren't imported.

**Fix:** Added imports:
```python
from src.forecast_utils import extract_forecasted_pnl, load_latest_forecast_snapshot
```

**Impact:** Watchers can now check if they have cash, and if not, attempt work stealing to free up buying power.

---

### 2. **Bug: Old Orders Never Canceled**
**Problem:** Orders 3-5 days old (from Nov 5-7) were blocking new orders from being placed.

**Root Cause:** Watchers detected existing orders but had no logic to cancel stale orders.

**Fix:** Added `_cancel_old_orders()` function that:
- Checks order age via `submitted_at` timestamp
- Cancels any orders older than 24 hours
- Logs cancellations and refreshes order list

**Code Location:** scripts/maxdiff_cli.py:145-177

---

### 3. **Bug: Out-of-Hours Order Cancellation Failed**
**Problem:** Function tried to cancel crypto orders during stock market hours but failed:
```
Failed to cancel BTC/USD order...: 'UUID' object has no attribute 'id'
```

**Root Cause:** Code passed `order_id` (UUID) to `cancel_order()` instead of `order` object.

**Fix:** Changed line 2369 in trade_stock_e2e.py:
```python
# Before:
alpaca_wrapper.cancel_order(order_id)

# After:
alpaca_wrapper.cancel_order(order)
```

---

### 4. **Enhancement: Individual Watcher Log Files**
**Problem:** All watchers logged to single `maxdiff_cli.log` making debugging difficult.

**Solution:** Created per-watcher log files:
- `logs/btcusd_buy_entry_watcher.log`
- `logs/btcusd_buy_exit_watcher.log`
- `logs/ethusd_buy_entry_watcher.log`
- etc.

**Implementation:**
1. Created `_get_watcher_logger(symbol, side, mode)` function
2. Updated `open_position_at_maxdiff_takeprofit()` to use symbol-specific logger
3. Updated `close_position_at_maxdiff_takeprofit()` to use symbol-specific logger
4. `logs/` directory created and already in .gitignore

**Code Location:** scripts/maxdiff_cli.py:24-43

---

## Manual Cleanup Performed

**Canceled 3 stale crypto orders:**
- BTC/USD @ $100,319.31 (3 days old)
- ETH/USD @ $3,158.32 (3 days old)
- UNI/USD @ $4.67 (5 days old)

These were preventing new orders from being placed.

---

## Expected Behavior After Fixes

1. **Entry Watchers:**
   - Automatically cancel orders >24 hours old
   - Check for sufficient buying power
   - Use work stealing if needed to free capital
   - Place limit orders when price tolerance met

2. **Exit Watchers:**
   - Automatically cancel stale take-profit orders >24 hours
   - Re-arm fresh take-profit orders
   - Monitor position and adjust as needed

3. **Logging:**
   - Each watcher logs to its own file in `logs/`
   - Easier to debug individual symbol issues
   - Global operations still log to `maxdiff_cli.log`

---

## Testing Recommendations

1. Monitor `logs/` directory for new watcher log files
2. Check that orders are being placed within ~12 seconds of watcher spawn
3. Verify old orders (>24h) are automatically canceled
4. Confirm work stealing succeeds when buying power is low

---

## Related Files Changed

- `scripts/maxdiff_cli.py` - Added imports, old order cancellation, per-watcher logging
- `trade_stock_e2e.py` - Fixed order cancellation bug (line 2369)
- `logs/` - New directory for watcher-specific logs
