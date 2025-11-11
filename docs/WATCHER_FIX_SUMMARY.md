# Watcher Fix Summary - 2025-11-11

## Issue
UNIUSD position (BUY 5184 qty, strategy=maxdiffalwayson) had no watchers running for 24/7 trading.

## Root Causes
1. Missing `UNIUSD|buy` entry in `active_trades.json` (only had stale `UNIUSD|sell`)
2. Exit watcher refresh used wrong price field (`maxdiffprofit_high` instead of `maxdiffalwayson_high`)

## Fixes Applied

### Code Changes

**trade_stock_e2e.py:2566-2581** - Auto-repair missing active_trade entries
```python
# Create active_trade entry if missing but position exists with matching forecast
if not active_trade and entry_strategy in MAXDIFF_LIMIT_STRATEGIES:
    _update_active_trade(symbol, normalized_side, mode="normal", qty=position_qty, strategy=entry_strategy)
```

**trade_stock_e2e.py:2675-2678** - Correct exit watcher prices by strategy
```python
# Determine new forecast takeprofit price
if entry_strategy == "maxdiffalwayson":
    new_takeprofit_price = pick_data.get("maxdiffalwayson_high_price" if is_buy else "maxdiffalwayson_low_price")
else:
    new_takeprofit_price = pick_data.get("maxdiffprofit_high_price" if is_buy else "maxdiffprofit_low_price")
```

**stock_cli.py:25, 528-534** - Color-coded status output
- Yellow: Crypto symbols
- Green: Symbols with open positions

### Manual Fix
Updated `strategy_state/active_trades.json`:
- Added `UNIUSD|buy` entry
- Removed stale `UNIUSD|sell` entry

### Unit Tests
Created `tests/prod/trading/test_watcher_refresh.py` with 10 tests:
- 5 test classes covering all aspects of watcher logic
- Specific UNIUSD regression tests
- Prevents future occurrences

Run: `python -m pytest tests/prod/trading/test_watcher_refresh.py -v`

## Expected Behavior

For UNIUSD with maxdiffalwayson strategy:
- **Entry watcher**: BUY @ 8.9696 (maxdiffalwayson_low)
- **Exit watcher**: SELL @ 9.5698 (maxdiffalwayson_high)

Both run 24/7 with persistent limit orders → multiple round trips per day.

## Documentation
- `docs/MAXDIFFALWAYSON_WATCHER_FIX.md` - Full fix details
- `docs/UNIUSD_WATCHER_FIX.md` - Initial root cause analysis
- `tests/prod/trading/test_watcher_refresh.py` - Unit tests

## Verification
```bash
# Status with colors
python stock_cli.py status

# Run tests
python -m pytest tests/prod/trading/test_watcher_refresh.py -v

# Check watchers after next trade_stock_e2e.py run
python stock_cli.py status | grep -A3 UNIUSD
```
