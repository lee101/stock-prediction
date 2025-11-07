# MaxDiff Order Work Stealing Design

## Problem
Multiple maxdiff watchers compete for limited buying power (2x leverage cap). Need intelligent order management to ensure best opportunities always have open orders.

## Crypto Out-of-Hours Behavior

### Market Detection
- **Out of hours**: NYSE closed (weekends, after 4pm ET, before 9:30am ET)
- **Crypto active**: BTC, ETH, UNI (24/7 trading)

### Tolerance Strategy
```python
CRYPTO_OUT_OF_HOURS_FORCE_IMMEDIATE_COUNT = 1  # Top 1 crypto gets immediate entry
CRYPTO_OUT_OF_HOURS_TOLERANCE_PCT = 0.016      # 1.6% for others (vs 0.66% normal)
CRYPTO_NORMAL_TOLERANCE_PCT = 0.0066           # 0.66% during stock hours
```

**Behavior:**
- Top crypto by forecasted PnL: `force_immediate=True` (ignores price, enters immediately)
- Next 2 cryptos: 1.6% tolerance (more aggressive than stocks)
- During stock hours: all use 0.66% standard tolerance

## Work Stealing Algorithm

### Core Concept
When watcher can't open order due to capacity, steal from worst open order if justified.

### Tolerances
```python
WORK_STEALING_ENTRY_TOLERANCE_PCT = 0.003      # 0.3% - can attempt steal
WORK_STEALING_PROTECTION_PCT = 0.004           # 0.4% - can't be stolen (close to fill)
WORK_STEALING_COOLDOWN_SECONDS = 300           # 5min cooldown per symbol
BEST_ORDERS_TIGHT_TOLERANCE_PCT = 0.005        # 0.5% for top N orders
```

### Work Stealing Flow

1. **Check Capacity**
   ```python
   current_exposure = sum(abs(order.value) for order in open_orders)
   available_capacity = (buying_power * 2) - current_exposure
   needed_capacity = target_qty * limit_price
   ```

2. **Find Steal Candidates** (if insufficient capacity)
   - Filter: not probe trades (`mode != 'probe'`)
   - Filter: not protected (price > protection_tolerance away)
   - Filter: not recently stolen (cooldown check)
   - Rank by: forecasted PnL (ascending - worst first)

3. **Evaluate Steal**
   ```python
   price_distance = abs(current_price - limit_price) / limit_price
   can_steal = price_distance <= WORK_STEALING_ENTRY_TOLERANCE_PCT
   
   if can_steal:
       worst_order_pnl = candidate_orders[0].forecasted_pnl
       my_forecasted_pnl = get_forecasted_pnl(symbol, strategy)
       steal_justified = my_forecasted_pnl > worst_order_pnl * 1.1  # 10% better
   ```

4. **Execute Steal**
   - Cancel worst order
   - Record steal timestamp
   - Submit new order
   - Update cooldown tracker

5. **Fighting Prevention**
   - Track last steal time per symbol: `{symbol: timestamp}`
   - Cooldown: 5 minutes
   - If A steals from B, B can't steal back for cooldown period
   - Oscillation detection: if same pair fights >3 times in 30min, increase cooldown to 30min

### Protection Rules

**Can't Steal From:**
- Probe trades (small positions, not helpful)
- Orders within 0.4% of limit (about to fill)
- Orders stolen in last 5 minutes
- Orders with `force_immediate=True` flag

**Priority Tiers:**
1. Top N by capacity: Always maintain orders at 0.5% tolerance
2. Next tier: Can open orders at 0.66% tolerance if capacity available
3. Overflow tier: Can only steal at 0.3% tolerance

### Capacity Calculation
```python
def get_max_concurrent_orders(buying_power: float) -> int:
    # Assume average order size = buying_power / 2
    # With 2x leverage = 2 * buying_power total
    # Conservative: allow N orders where N * avg_size <= 2 * buying_power
    return 4  # Conservative for crypto (3 cryptos + 1 buffer)
```

## Implementation Plan

### 1. Configuration Module (`src/work_stealing_config.py`)
- All tolerance constants
- Market hours detection
- Crypto symbol lists

### 2. Work Stealing Coordinator (`src/work_stealing_coordinator.py`)
- `can_open_order(symbol, strategy, limit_price, qty) -> bool`
- `attempt_steal(symbol, strategy, limit_price, qty) -> Optional[str]`  # Returns canceled order symbol
- `get_steal_candidates() -> List[StealCandidate]`
- `is_protected(order) -> bool`
- `record_steal(from_symbol, to_symbol)`
- Cooldown tracking
- Fighting detection

### 3. Entry Watcher Integration (`scripts/maxdiff_cli.py`)
- Check work stealing before submitting order
- Different tolerance based on tier/market hours
- Call coordinator for steal attempts

### 4. Process Utils Integration (`src/process_utils.py`)
- Pass work stealing context to spawn functions
- Determine tier (force_immediate, tight_tolerance, normal, steal_only)

## Test Scenarios

### Crypto Out-of-Hours Tests
1. **Top crypto force immediate**: Verify force_immediate=True set
2. **Second crypto aggressive**: Verify 1.6% tolerance used
3. **During stock hours**: Verify 0.66% tolerance used
4. **Market hours detection**: Test weekends, after-hours, pre-market

### Work Stealing Tests
1. **Capacity available**: Normal order placement
2. **Capacity exceeded**: Steal from worst order
3. **Protection works**: Can't steal from near-fill orders
4. **Probe protection**: Can't steal from probe trades
5. **Better PnL required**: Don't steal unless significantly better
6. **Cooldown enforced**: Can't steal same symbol twice in 5min
7. **Fighting prevention**: Detect A ↔ B oscillation
8. **Multiple candidates**: Choose worst PnL to cancel
9. **Tie-breaking**: When PnLs equal, use oldest order
10. **Edge case**: All orders protected → can't steal

### Integration Tests
1. **3 cryptos, 2 capacity**: Top 2 get orders, 3rd steals when close
2. **Stock + crypto mix**: Verify separate tolerances
3. **Probe + normal mix**: Verify probes never canceled
4. **Rapid price moves**: Multiple steal attempts handled
5. **Recovery**: After steal, old symbol can re-enter if still good

## Monitoring & Metrics

Track:
- `work_steal_attempts_total` (counter)
- `work_steal_success_total` (counter)
- `work_steal_fights_detected` (counter)
- `orders_protected_total` (counter)
- Current tier per symbol/strategy

## Configuration Override

```bash
# Crypto aggressive
CRYPTO_OUT_OF_HOURS_FORCE_COUNT=1
CRYPTO_OUT_OF_HOURS_TOLERANCE=0.016

# Work stealing
WORK_STEALING_ENABLED=1
WORK_STEALING_TOLERANCE=0.003
WORK_STEALING_PROTECTION=0.004
WORK_STEALING_COOLDOWN=300

# Testing
WORK_STEALING_DRY_RUN=1  # Log but don't cancel
```
