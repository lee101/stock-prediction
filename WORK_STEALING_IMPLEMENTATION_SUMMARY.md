# Work Stealing Implementation Summary

## Overview
Implemented intelligent order management system that allows maxdiff watchers to compete for limited capacity (2x leverage cap) by stealing work from lower-priority orders.

## Key Features Implemented

### 1. Crypto Out-of-Hours Aggressive Entry
**File**: `src/work_stealing_config.py`

**Behavior**:
- **During stock hours**: All cryptos use 0.66% standard tolerance
- **Out of hours** (weekends, after NYSE close):
  - Top crypto (rank 1): `force_immediate=True` (no tolerance, enter immediately)
  - Other cryptos: 1.6% tolerance (2.4x more aggressive than normal)

**Config**:
```bash
CRYPTO_OUT_OF_HOURS_FORCE_COUNT=1    # How many top cryptos get force_immediate
CRYPTO_OUT_OF_HOURS_TOLERANCE=0.016  # 1.6% for non-top cryptos
```

### 2. Work Stealing Coordinator
**File**: `src/work_stealing_coordinator.py`

**Core Algorithm**:
1. **Capacity Check**: Can new order fit within 2x leverage?
2. **Find Candidates**: If no capacity, find orders that can be stolen:
   - Exclude probe trades (always protected)
   - Exclude orders within 0.4% of limit (about to fill)
   - Exclude recently stolen (5min cooldown)
   - Rank by forecasted PnL (worst first)
3. **Validate Steal**:
   - New order must be within 0.3% of limit price
   - New order must have 10% better PnL than victim
   - Check for fighting (A↔B oscillation)
4. **Execute**: Cancel worst order, record steal event

**Tolerances**:
```bash
WORK_STEALING_TOLERANCE=0.003        # 0.3% - can attempt steal
WORK_STEALING_PROTECTION=0.004       # 0.4% - can't be stolen
WORK_STEALING_COOLDOWN=300           # 5min per symbol
WORK_STEALING_MIN_PNL_IMPROVEMENT=1.1  # 10% better required
```

### 3. Fighting Prevention
**Prevents oscillation** between two symbols stealing from each other:

- Tracks steal history per symbol pair
- If 3+ steals in 30min window → extended 30min cooldown
- Blocks new steals that would cause fighting

**Config**:
```bash
WORK_STEALING_FIGHT_THRESHOLD=3      # Steals = fighting
WORK_STEALING_FIGHT_WINDOW=1800      # 30min window
WORK_STEALING_FIGHT_COOLDOWN=1800    # 30min extended cooldown
```

### 4. Entry Watcher Integration
**File**: `scripts/maxdiff_cli.py`

Modified entry watcher logic:
```python
# Check cash/capacity
has_cash = _entry_requires_cash(side, limit_price, target_qty)

if not has_cash:
    # Try work stealing
    coordinator = get_coordinator()
    stolen_symbol = coordinator.attempt_steal(...)
    
    if stolen_symbol:
        logger.info(f"{symbol}: Stole capacity from {stolen_symbol}")
        # Continue to order submission
    else:
        # Still blocked, wait
        continue
```

### 5. Process Utils Integration
**File**: `src/process_utils.py`

Added crypto rank parameter to `spawn_open_position_at_maxdiff_takeprofit()`:
- Auto-calculates tolerance based on rank and market hours
- Auto-enables force_immediate for top crypto out-of-hours
- Passes through to watcher metadata

**File**: `trade_stock_e2e.py`

Calculate crypto ranks in `manage_positions()`:
```python
crypto_ranks = {symbol: rank for rank, (symbol, pnl) in enumerate(crypto_candidates)}
```

Pass rank to spawner:
```python
spawn_open_position_at_maxdiff_takeprofit(
    symbol,
    side,
    limit_price,
    qty,
    crypto_rank=crypto_ranks.get(symbol),
    ...
)
```

## Protection Rules

**Cannot Steal From:**
1. Probe trades (mode='probe')
2. Orders within 0.4% of limit (about to execute)
3. Orders stolen in last 5 minutes
4. Orders involved in fighting (3+ steals in 30min)

**Priority Tiers:**
1. Top crypto out-of-hours: force_immediate, no tolerance
2. Other crypto out-of-hours: 1.6% tolerance
3. Normal orders: 0.66% tolerance
4. Can steal at 0.3% if capacity needed

## Test Coverage

### Unit Tests (`tests/unit/`)

**test_work_stealing_config.py** (20+ tests):
- NYSE hours detection (weekdays, weekends, before/after market)
- Crypto out-of-hours detection
- Tolerance calculation (stocks, crypto, top crypto)
- Force immediate logic
- Edge cases (unknown symbols, configurable limits)

**test_work_stealing_coordinator.py** (25+ tests):
- Capacity checking
- Order protection logic
- Steal attempt validation
- PnL comparison logic
- Cooldown enforcement
- Fighting detection
- Candidate selection
- Dry run mode

### Integration Tests (`tests/integration/`)

**test_work_stealing_integration.py** (15+ tests):
- Crypto out-of-hours full flow
- 3 cryptos competing for 2 slots
- All orders protected scenario
- Fighting oscillation detection
- Probe trade immunity
- Entry watcher integration
- Edge cases (zero PnL, negative PnL)

## Usage Examples

### Enable Work Stealing
```bash
WORK_STEALING_ENABLED=1 python trade_stock_e2e.py
```

### Dry Run (log but don't cancel)
```bash
WORK_STEALING_DRY_RUN=1 python trade_stock_e2e.py
```

### Aggressive Crypto Out-of-Hours
```bash
CRYPTO_OUT_OF_HOURS_FORCE_COUNT=2    # Top 2 cryptos force immediate
CRYPTO_OUT_OF_HOURS_TOLERANCE=0.025  # 2.5% tolerance for others
```

### Tune Stealing Aggressiveness
```bash
WORK_STEALING_TOLERANCE=0.005        # 0.5% (more conservative)
WORK_STEALING_MIN_PNL_IMPROVEMENT=1.2  # 20% better required
```

## Monitoring

Track work stealing events via logs:
```
[INFO] Work steal: UNIUSD (PnL=2.50) stealing from ETHUSD (PnL=1.00)
[INFO] BTCUSD: Auto-enabled force_immediate (crypto rank 1 during out-of-hours)
[WARNING] AAPL: Would cause fighting with MSFT, aborting steal
```

## Files Created/Modified

### New Files:
- `src/work_stealing_config.py` - Configuration and market hours
- `src/work_stealing_coordinator.py` - Core coordinator logic
- `tests/unit/test_work_stealing_config.py` - Config tests
- `tests/unit/test_work_stealing_coordinator.py` - Coordinator tests
- `tests/integration/test_work_stealing_integration.py` - Integration tests
- `docs/WORK_STEALING_DESIGN.md` - Design document
- `docs/WORK_STEALING_IMPLEMENTATION_SUMMARY.md` - This file

### Modified Files:
- `scripts/maxdiff_cli.py` - Added work stealing to entry watcher
- `src/process_utils.py` - Added crypto_rank parameter, auto-tolerance
- `trade_stock_e2e.py` - Calculate crypto ranks, pass to spawner

## Testing

Run all work stealing tests:
```bash
# Unit tests
pytest tests/unit/test_work_stealing_config.py -v
pytest tests/unit/test_work_stealing_coordinator.py -v

# Integration tests
pytest tests/integration/test_work_stealing_integration.py -v
```

## Performance Characteristics

- **Capacity calculation**: O(N) where N = open orders
- **Candidate selection**: O(N log N) for sorting by PnL
- **Fighting detection**: O(F) where F = fight pairs (typically < 10)
- **Memory**: O(H) where H = steal history (auto-cleanup after 1 hour)

## Safety Features

1. **Dry run mode**: Test without canceling orders
2. **Cooldown**: Prevents rapid oscillation
3. **Fighting detection**: Stops A↔B battles
4. **Protection**: Never steals from near-execution or probe orders
5. **PnL threshold**: Only steals if significantly better

## Future Enhancements

Potential improvements:
1. **Machine learning**: Learn optimal stealing thresholds
2. **Multi-tier capacity**: Different limits for crypto vs stocks
3. **Time-weighted PnL**: Consider how long order has been waiting
4. **Partial steals**: Reduce order size instead of full cancel
5. **Priority groups**: VIP symbols that can't be stolen from
