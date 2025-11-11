# MaxdiffAlwaysOn Watcher Fix - Always-On Trading

## Issue
UNIUSD position (BUY 5184 qty, strategy=maxdiffalwayson) had no exit watcher running. Exit watchers should place persistent limit orders at the high price for 24/7 trading with multiple round trips per day.

## Root Causes

### 1. Missing active_trade entry
- Position existed but `active_trades.json` had no `UNIUSD|buy` entry
- Watcher refresh skipped position without active_trade

### 2. Wrong price field for exit watchers
- Watcher refresh used `maxdiffprofit_high_price` for ALL strategies
- Should use `maxdiffalwayson_high_price` for maxdiffalwayson strategy

## Fixes Applied

### 1. Auto-create missing active_trade entries (trade_stock_e2e.py:2566-2581)
```python
# Create active_trade entry if missing but position exists with matching forecast
if not active_trade and entry_strategy in MAXDIFF_LIMIT_STRATEGIES:
    position_qty = abs(float(getattr(position, "qty", 0.0)))
    logger.info(f"Creating missing active_trade entry for {symbol} {normalized_side}")
    _update_active_trade(symbol, normalized_side, mode="normal", qty=position_qty, strategy=entry_strategy)
```

### 2. Use correct price for maxdiffalwayson exit watchers (trade_stock_e2e.py:2675-2678)
```python
# Determine new forecast takeprofit price
if entry_strategy == "maxdiffalwayson":
    new_takeprofit_price = pick_data.get("maxdiffalwayson_high_price" if is_buy else "maxdiffalwayson_low_price")
else:
    new_takeprofit_price = pick_data.get("maxdiffprofit_high_price" if is_buy else "maxdiffprofit_low_price")
```

### 3. Color coding in stock_cli.py status
- Yellow: Crypto symbols
- Green: Non-crypto with open positions

## Expected Behavior - Next trade_stock_e2e.py Run

For UNIUSD (current forecast from your log):
- `maxdiffalwayson_low_price`: 8.9696
- `maxdiffalwayson_high_price`: 9.5698

Watchers will spawn:

### Entry Watcher (buy)
- Places BUY limit order @ 8.9696
- Runs 24/7, polls every 12s
- If position doesn't exist, executes when price hits low
- Can re-enter after exit watcher sells

### Exit Watcher (sell)
- Places SELL limit order @ 9.5698
- Runs 24/7, polls every 12s
- When position exists + price hits high, executes sell
- Closes position at target price

### Multiple Round Trips
1. Entry watcher buys at 8.9696
2. Exit watcher sells at 9.5698 (profit: ~6.5%)
3. Entry watcher re-buys at 8.9696
4. Repeat all day

## Log Files to Check
- `logs/UNIUSD_buy_entry_watcher.log` - Entry watcher activity
- `logs/UNIUSD_buy_exit_watcher.log` - Exit watcher activity
- `trade_stock_e2e.log` - Position management, watcher spawning
- `alpaca_cli.log` - Order submissions/fills

## Verification Commands
```bash
# Check watchers running
python stock_cli.py status | grep -A3 UNIUSD

# Check watcher processes
ps aux | grep "maxdiff_cli.py.*UNIUSD"

# Check watcher logs
tail -f logs/UNIUSD_buy_entry_watcher.log
tail -f logs/UNIUSD_buy_exit_watcher.log
```
