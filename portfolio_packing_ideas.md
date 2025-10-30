# Portfolio Packing Ideas

## Core Insight: Parallel Trading Strategies

Maxdiff trades are **independent from directional trades** - they should execute in parallel!

### Problem Example
```
9:30 AM - Entry
  - takeprofit strategy: Buy AAPL, target $180 → $185
  - Uses $10k of leverage budget

10:00 AM - Exit
  - AAPL hits takeprofit at $185 (+$277 profit)
  - Position closed, $10k leverage freed

10:00 AM - 4:00 PM
  - That $10k sits idle for 6 hours!
  - Could have been used for maxdiff trades
```

### Solution: Parallel Maxdiff Layer

**Two independent trading layers:**

1. **Directional Layer** (primary)
   - takeprofit, highlow, simple strategies
   - Enters/exits based on predictions
   - Can be quick (minutes to hours)

2. **Maxdiff Layer** (parallel, opportunistic)
   - Listens for price extremes
   - Places limit orders at high/low targets
   - Can execute while directional trades are active
   - Uses "spare" leverage capacity

## Implementation Ideas

### 1. Separate Leverage Buckets

```python
DIRECTIONAL_MAX_LEVERAGE = 1.3x  # Primary strategies get first priority
MAXDIFF_MAX_LEVERAGE = 1.5x      # Can use extra 0.2x if available
```

- Directional trades respect 1.3x limit
- Maxdiff trades can fill the gap between 1.3x and 1.5x
- Ensures directional strategies always have room

### 2. Dynamic Maxdiff Sizing

Current: Fixed qty based on Kelly/get_qty at spawn time

Better: Recalculate qty continuously based on:
```python
available_maxdiff_room = (equity × MAXDIFF_MAX_LEVERAGE) - current_total_exposure
maxdiff_qty = available_maxdiff_room / price / num_active_maxdiff_trades
```

Benefits:
- Uses all available leverage
- Automatically shrinks when directional trades enter
- Automatically grows when directional trades exit

### 3. Persistent Maxdiff Plans

Store all maxdiff opportunities for the day:
```python
{
  "AAPL": {
    "high_target": 185.50,
    "low_target": 178.20,
    "side": "buy",  # or "sell" or "both"
    "avg_return": 0.0023,
    "status": "listening",
    "spawned_at": "2025-10-30T09:30:00Z"
  },
  "NVDA": {...},
  ...
}
```

Throughout the day:
- Check if any maxdiff plans can execute given current leverage
- Prioritize by avg_return
- Can spawn/respawn as leverage becomes available

### 4. Maxdiff Rotation

If 20 maxdiff opportunities but only room for 5:
- Start with top 5 by avg_return
- When one fills or expires, rotate in #6
- Keep rotating to give all opportunities a chance

### 5. Time-Based Priority Adjustment

Early day (9:30-11:00):
- Prioritize directional trades (more volatile)
- Maxdiff gets 20% of budget

Mid day (11:00-2:00):
- Equal priority
- Maxdiff gets 40% of budget

Late day (2:00-4:00):
- Prioritize maxdiff (directional positions closing)
- Maxdiff gets 60% of budget

### 6. Cross-Strategy Position Limits

Prevent over-concentration:
```python
MAX_POSITION_PER_SYMBOL = 2  # Can have directional + maxdiff on same symbol

# Example:
AAPL:
  - Position 1: takeprofit buy (100 shares)
  - Position 2: maxdiff limit sell @ high (50 shares)
```

Both can execute independently!

### 7. Smart Order Pricing for Maxdiff

Instead of fixed limit at exact high/low:
```python
# For buy side (targeting low)
initial_limit = predicted_low * 1.001  # Just above low
final_limit = predicted_low * 1.005    # Ramp up if not filling

# For sell side (targeting high)
initial_limit = predicted_high * 0.999  # Just below high
final_limit = predicted_high * 0.995    # Ramp down if not filling
```

Increases fill rate while maintaining edge.

### 8. Maxdiff Consolidation

If multiple maxdiff trades on same symbol, same side:
- Consolidate into single order
- Reduces order management overhead
- Better execution (one larger order vs many small)

Example:
```
Before:
  - AAPL maxdiff buy @ 178.20 (25 shares)
  - AAPL maxdiff buy @ 178.25 (30 shares)

After:
  - AAPL maxdiff buy @ 178.20 (55 shares)
```

### 9. Kelly Sizing with Leverage Awareness

Current Kelly doesn't account for leverage:
```python
kelly_fraction = edge / variance
qty = (equity × kelly_fraction) / price
```

Better:
```python
kelly_fraction = edge / variance
available_leverage = max_leverage - current_leverage
leveraged_equity = equity × available_leverage
qty = (leveraged_equity × kelly_fraction) / price
```

### 10. Partial Fill Tracking

Track when maxdiff orders partially fill:
```python
{
  "AAPL_maxdiff_buy": {
    "target_qty": 100,
    "filled_qty": 35,
    "remaining_qty": 65,
    "avg_fill_price": 178.23,
    "status": "partially_filled"
  }
}
```

Can adjust remaining qty based on leverage changes.

## Monitoring & Visibility

### Stock CLI Status Enhancement

Show both layers:
```
=== DIRECTIONAL POSITIONS ===
AAPL: +100 shares @ $182.50 (takeprofit, +$250 unrealized)
NVDA: +50 shares @ $145.00 (simple, +$125 unrealized)

=== MAXDIFF LISTENING (5/15 active) ===
TSLA: buy @ $242.50, sell @ $251.30 (avg_return=0.0031) [listening]
AAPL: sell @ $185.50 (avg_return=0.0023) [listening]
MSFT: buy @ $415.20 (avg_return=0.0019) [listening]

=== MAXDIFF QUEUED (10 waiting for leverage) ===
META: buy @ $520.10 (rank=6, needs $2.5k room)
GOOGL: sell @ $165.80 (rank=7, needs $1.8k room)

=== LEVERAGE USAGE ===
Current: 1.25x ($62.5k / $50k equity)
Directional: 1.15x ($57.5k)
Maxdiff: 0.10x ($5k)
Available for Maxdiff: 0.25x ($12.5k)
```

## Implementation Priority

1. ✅ Parallel maxdiff execution (independent from directional)
2. ✅ Dynamic sizing based on available leverage
3. ✅ Persistent maxdiff tracking
4. Status command showing both layers
5. Cross-strategy position limits
6. Time-based priority adjustment
7. Kelly sizing with leverage awareness
8. Maxdiff consolidation
9. Smart order pricing with ramp
10. Partial fill tracking

## Mathematical Optimization (Future)

Portfolio packing is essentially a **knapsack problem**:
- Items: Trading opportunities (with expected return & risk)
- Knapsack: Leverage capacity
- Goal: Maximize expected return subject to leverage constraint

Could use:
- Linear programming (scipy.optimize)
- Dynamic programming
- Genetic algorithms

But for now, greedy approach (sort by avg_return) works well!
