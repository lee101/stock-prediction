# Active Trades State Redesign

## Problem

`active_trades.json` caches position quantities (`qty`) which become stale when:
- Positions are manually adjusted
- Orders partially fill
- System crashes/restarts
- Multiple processes modify positions

This causes watchers not to spawn because the system thinks positions exist at wrong sizes.

## Solution: Separate Source of Truth from Metadata

### Positions (Source of Truth): Alpaca API
Query `alpaca_wrapper.get_all_positions()` for:
- `symbol`
- `side`
- `qty` (actual position size)
- `avg_entry_price`
- `market_value`

### Metadata (Local Cache): `active_trades.json`
Store **only** metadata that Alpaca doesn't provide:
```json
{
  "BTCUSD|buy": {
    "entry_strategy": "maxdiff",
    "mode": "normal",
    "opened_at": "2025-11-04T03:47:51+00:00",
    "opened_at_sim": "2024-01-01T00:00:00+00:00"
  }
}
```

**Remove**: `qty` field entirely

### Reconciliation Logic

On every access to active trade state:

```python
def get_active_trade_with_position(symbol: str, side: str) -> Dict:
    """Get active trade metadata enriched with live position data."""

    # 1. Get metadata (strategy, mode, timestamps)
    metadata = load_from_json(symbol, side)

    # 2. Get actual position from Alpaca (qty, price, value)
    position = alpaca_wrapper.get_position(symbol, side)

    # 3. Reconcile
    if position and not metadata:
        # Position exists but no metadata - create default metadata
        logger.warning(f"Position {symbol} {side} exists but no metadata - creating default")
        metadata = {
            "entry_strategy": "unknown",
            "mode": "normal",
            "opened_at": datetime.now().isoformat()
        }
        save_metadata(symbol, side, metadata)

    elif metadata and not position:
        # Metadata exists but no position - clean up stale metadata
        logger.warning(f"Stale metadata for {symbol} {side} - removing")
        delete_metadata(symbol, side)
        return {}

    # 4. Merge live position data with metadata
    if position and metadata:
        return {
            **metadata,
            "qty": position.qty,  # Always from Alpaca
            "avg_entry_price": position.avg_entry_price,
            "market_value": position.market_value
        }

    return {}
```

### Benefits

1. **No Desync**: Position size always from Alpaca
2. **Automatic Cleanup**: Stale metadata detected and removed
3. **Resilient**: Manual trades or crashes don't break state
4. **Simple**: Metadata file is smaller and clearer

### Migration Path

1. Create `get_active_trade_with_position()` helper
2. Replace all `_get_active_trade()` calls
3. Update `_update_active_trade()` to not store `qty`
4. Add reconciliation on startup
5. Optionally add TTL-based cache to avoid hitting Alpaca API every time

### Implementation Notes

- Cache Alpaca positions for ~30 seconds to reduce API calls
- Run reconciliation on every `trade_stock_e2e.py` cycle
- Log warnings for any desync found
- Consider moving to a proper database (SQLite) for ACID properties
