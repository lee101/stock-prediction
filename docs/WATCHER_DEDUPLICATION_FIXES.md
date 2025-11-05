# MaxDiff Watcher Deduplication and Process Cleanup Fixes

## Problem Summary

The trading system had two critical issues:

1. **Multiple exit watchers fighting each other**: Unlike entry watchers which had conflict resolution, exit watchers could create duplicates for the same symbol/side/strategy with different take-profit prices, leading to conflicting orders.

2. **Orphaned processes on shutdown**: When `trade_stock_e2e.py` was terminated with Ctrl+C or kill, spawned watcher processes continued running because:
   - `start_new_session=True` isolated child processes from parent signals
   - No signal handlers were registered in the main process
   - No cleanup mechanism existed to terminate spawned processes

## Fixes Implemented

### 1. Signal Handler for Process Cleanup (`trade_stock_e2e.py`)

**Added functions (lines 3355-3445):**

- `cleanup_spawned_processes()`: Reads all watcher config files from `strategy_state/maxdiff_watchers*/` and:
  1. Sends SIGTERM to all active PIDs (graceful shutdown)
  2. Waits 2 seconds for processes to exit
  3. Force kills survivors with SIGKILL
  4. Logs cleanup activity

- `signal_handler()`: Registered for SIGINT (Ctrl+C) and SIGTERM (kill command)
  1. Calls `cleanup_spawned_processes()`
  2. Releases model resources
  3. Exits cleanly

**Registration (lines 3475-3478):**
```python
signal.signal(signal.SIGINT, signal_handler)  # Ctrl+C
signal.signal(signal.SIGTERM, signal_handler)  # kill command
```

### 2. Exit Watcher Conflict Resolution (`src/process_utils.py`)

**Added function `_stop_conflicting_exit_watchers()` (lines 327-381):**

Similar to `_stop_conflicting_entry_watchers()`, this function:
1. Finds all exit watchers for the same symbol/side
2. Filters to same `entry_strategy` (maxdiff, maxdiffalwayson, highlow)
3. Compares `takeprofit_price` - if different by more than 1e-6, terminates the old watcher
4. Sends SIGTERM to old process PID
5. Updates old config file with `state="superseded_exit_watcher"` and `active=False`

**Integration (lines 716-723):**
Called in `spawn_close_position_at_maxdiff_takeprofit()` before spawning new exit watcher, ensuring only one exit watcher per symbol/side/strategy with the latest take-profit price.

## How It Works

### Graceful Shutdown Flow

```
User presses Ctrl+C
    ↓
SIGINT received by main process
    ↓
signal_handler() invoked
    ↓
cleanup_spawned_processes() reads all watcher configs
    ↓
Sends SIGTERM to all active PIDs
    ↓
Waits 2 seconds
    ↓
Force kills survivors with SIGKILL
    ↓
release_model_resources()
    ↓
sys.exit(0)
```

### Exit Watcher Deduplication Flow

```
spawn_close_position_at_maxdiff_takeprofit() called with new takeprofit_price
    ↓
_stop_conflicting_exit_watchers() scans for existing exit watchers
    ↓
Finds watchers for same symbol/side/strategy with different prices
    ↓
Sends SIGTERM to old watchers
    ↓
Marks old watchers as "superseded_exit_watcher"
    ↓
Spawns new watcher with latest takeprofit_price
```

## Testing

### Manual Test for Signal Handling

1. Start the main trading loop in paper mode:
   ```bash
   PAPER=1 python trade_stock_e2e.py
   ```

2. Wait for "Signal handlers registered for graceful shutdown" log message

3. Press Ctrl+C and verify:
   - "Received SIGINT, shutting down gracefully..." message
   - "Cleaning up spawned watcher processes..." message
   - PIDs being terminated
   - "Shutdown complete" message

4. Verify no orphaned processes remain:
   ```bash
   ps aux | grep maxdiff_cli.py
   ```
   Should return no results (except the grep itself)

### Manual Test for Exit Watcher Deduplication

1. Run trading system and wait for position to open with exit watcher

2. Trigger another exit watcher spawn for same symbol/side/strategy with different take-profit price

3. Check logs for:
   ```
   Terminating conflicting BTCUSD buy exit watcher at BTCUSD_buy_exit_*.json (takeprofit X.XX) in favor of Y.YY
   ```

4. Verify only one exit watcher remains active for each symbol/side/strategy:
   ```bash
   PAPER=1 python stock_cli.py status
   ```

### Automated Testing

Run existing tests to ensure no regressions:
```bash
pytest tests/ -v
```

## Files Modified

1. **trade_stock_e2e.py**:
   - Added imports: `signal`, `sys`
   - Added import: `MAXDIFF_WATCHERS_DIR` from `process_utils`
   - Added `cleanup_spawned_processes()` function
   - Added `signal_handler()` function
   - Registered signal handlers in `main()`

2. **src/process_utils.py**:
   - Added `_stop_conflicting_exit_watchers()` function
   - Integrated into `spawn_close_position_at_maxdiff_takeprofit()`

## Behavior Changes

### Before

- **Exit watchers**: Multiple exit watchers could coexist for same symbol/side with different take-profit prices, potentially creating conflicting orders
- **Shutdown**: Orphaned watcher processes continued running indefinitely after main process was killed, requiring manual cleanup via `kill_all_watchers.py`

### After

- **Exit watchers**: Only one exit watcher per symbol/side/strategy exists at a time; old ones are automatically terminated when new forecast arrives
- **Shutdown**: All spawned watcher processes are automatically cleaned up when main process receives SIGINT or SIGTERM

## Remaining Considerations

1. **Debounce mechanism**: Still in-memory only, so race conditions can occur if multiple instances of `trade_stock_e2e.py` run simultaneously. Consider file-based locking for multi-process safety.

2. **Entry watcher deduplication**: Already implemented, works correctly.

3. **cancel_multi_orders.py**: Still runs independently to clean up duplicate orders at the broker level. This is a safety net and should continue running.

4. **Manual cleanup**: `kill_all_watchers.py` remains useful for emergency cleanup or when processes become truly orphaned (e.g., system crash).

## Related Files

- `/nvme0n1-disk/code/stock-prediction/scripts/kill_all_watchers.py` - Manual cleanup utility
- `/nvme0n1-disk/code/stock-prediction/scripts/cancel_multi_orders.py` - Order deduplication script
- `/nvme0n1-disk/code/stock-prediction/scripts/maxdiff_cli.py` - Watcher process implementation
