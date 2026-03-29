# UNIUSD Watcher Issue - Root Cause & Fix

**See also**: [MAXDIFFALWAYSON_WATCHER_FIX.md](MAXDIFFALWAYSON_WATCHER_FIX.md) for the proper fix enabling 24/7 trading with multiple round trips.

## Issue
UNIUSD position existed (5184.2 qty, BUY/LONG) but no maxdiff watchers were running.

## Root Cause
Data inconsistency between actual positions and `active_trades.json`:
- **Actual position**: UNIUSD BUY/LONG @ 5184.201609 qty
- **active_trades.json**: Only had `UNIUSD|sell` entry (stale)
- **Missing**: `UNIUSD|buy` entry

When watcher refresh code ran (trade_stock_e2e.py:2543-2677):
1. Found UNIUSD position with side=LONG (buy)
2. Called `_get_active_trade("UNIUSD", "buy")` → returned None
3. Skipped watcher refresh (line 2550-2551)
4. No watchers spawned

## Immediate Fixes Applied

### 1. Manual active_trades.json Update
- Added `UNIUSD|buy` entry with strategy=maxdiffalwayson
- Removed stale `UNIUSD|sell` entry
- Next run will spawn watchers correctly

### 2. Color Coding in stock_cli.py status
Added visual indicators in trading plan output:
- **Yellow**: Crypto symbols (BTCUSD, ETHUSD, UNIUSD)
- **Green**: Non-crypto symbols with open positions

Changes:
- Added `from src.symbol_utils import is_crypto_symbol`
- Build `position_symbols` set before printing plan
- Use `typer.secho(line, fg=typer.colors.YELLOW)` for crypto
- Use `typer.secho(line, fg=typer.colors.GREEN)` for positions

### 3. Code Fix in trade_stock_e2e.py:2543-2593
Modified watcher refresh logic to auto-create missing active_trade entries:

**Before**: Skipped positions without active_trade entries
**After**: Creates entry if position exists with matching forecast

```python
# Now checks pick_data first, then creates active_trade if missing
if not active_trade and entry_strategy in MAXDIFF_LIMIT_STRATEGIES:
    logger.info(f"Creating missing active_trade entry for {symbol} {normalized_side}")
    _update_active_trade(symbol, normalized_side, mode="normal", qty=position_qty, strategy=entry_strategy)
```

## Verification
Run `python stock_cli.py status` to see:
- UNIUSD [buy] in yellow (crypto)
- UNIUSD [buy] with strategy=maxdiffalwayson

Next `trade_stock_e2e.py` run will spawn:
- Entry watcher for UNIUSD buy @ maxdiffalwayson_low_price
- Exit watcher for UNIUSD buy @ maxdiffalwayson_high_price

## Prevention
The code fix ensures this won't happen again - positions with matching forecasts but missing active_trade entries will be auto-repaired during watcher refresh.
