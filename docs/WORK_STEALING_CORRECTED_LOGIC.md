# Work Stealing: Corrected Distance-Based Logic

## The Core Insight

**Work stealing is about execution likelihood, NOT forecasted PnL.**

### Wrong Thinking (Initial Implementation)
❌ "Steal from worst PnL order"
- Problem: Order with great PnL but 10% from limit will likely never execute
- Result: Keep dead orders, block live ones

### Correct Thinking (Current Implementation)
✅ "Steal from furthest order"
- Reasoning: Order 10% from limit might have amazing PnL but will probably never fill
- Result: Prioritize orders that will actually execute

## The Distance-Based Algorithm

### 1. Sort Candidates by Distance (Furthest First)
```python
candidates.sort(key=lambda c: (-c.distance_pct, c.forecasted_pnl))
```

**Primary**: Distance from limit (descending)
- Order at 95 with limit 100 (5% away) = furthest
- Order at 99.5 with limit 100 (0.5% away) = closest

**Tiebreaker**: PnL (ascending) for same distance
- If two orders both 5% away, steal from worse PnL

### 2. No PnL Improvement Requirement

**Old logic** (WRONG):
```python
if new_pnl <= victim_pnl * 1.1:  # Need 10% better
    return None  # Don't steal
```

**New logic** (CORRECT):
```python
# No PnL check - distance is what matters
# If they're far from limit, steal regardless of PnL
```

**Why**: Even "bad" PnL order that's 0.5% from limit is better than "great" PnL order 5% away.

## Fighting Resolution with PnL

Fighting is when A and B keep stealing from each other. But now it's based on execution likelihood, not PnL battles.

### Scenario: A ↔ B Oscillation Detected

**Rule**: Use PnL to break the tie

```python
if would_cause_fight(symbol_a, symbol_b):
    if my_pnl > their_pnl:
        allow_steal()  # Better PnL wins
    else:
        block_steal()  # Keep current order
```

### Example

- AAPL: 5% from limit, PnL=2.0
- MSFT: 5% from limit, PnL=4.0
- Fighting detected (3+ steals in 30min)

**Resolution**: MSFT wins (better PnL), steal allowed

**Rationale**: When both equally likely to execute (same distance), better PnL should win.

## Complete Steal Decision Tree

```
Can I steal from this order?

1. Is it a probe trade?
   YES → Protected, skip
   NO → Continue

2. Is it within 0.4% of limit?
   YES → Protected (about to execute), skip
   NO → Continue

3. Was it stolen in last 5 minutes?
   YES → Protected (cooldown), skip
   NO → Continue

4. Add to candidates

---

After collecting candidates:

5. Sort by distance (furthest first), PnL tiebreaker
   
6. Take furthest candidate

7. Am I within 0.3% of MY limit?
   NO → Can't steal (not ready to execute)
   YES → Continue

8. Would this cause fighting?
   NO → Steal!
   YES → Check PnL:
     My PnL > their PnL? → Steal (better PnL wins)
     My PnL ≤ their PnL? → Block (they keep it)
```

## Key Configuration

```bash
# Entry tolerance - how close to be ready to steal
WORK_STEALING_TOLERANCE=0.003  # 0.3%

# Protection - how close to be immune from stealing
WORK_STEALING_PROTECTION=0.004  # 0.4%

# No PnL requirement anymore!
# WORK_STEALING_MIN_PNL_IMPROVEMENT removed
```

## Real-World Examples

### Example 1: Distance Beats PnL

**Orders**:
- AAPL: limit=100, current=90, distance=10%, PnL=10.0 (amazing!)
- MSFT: limit=100, current=99.5, distance=0.5%, PnL=1.0 (meh)

**New Entry**:
- NVDA: limit=100, current=100.1, distance=0.1%

**Decision**: Steal from AAPL
**Reason**: AAPL is 10% away (unlikely to execute), MSFT is 0.5% away (likely to execute)

### Example 2: Same Distance, PnL Tiebreaker

**Orders**:
- AAPL: limit=100, current=95, distance=5%, PnL=3.0
- MSFT: limit=100, current=95, distance=5%, PnL=1.0

**New Entry**:
- NVDA: limit=100, current=100.1

**Decision**: Steal from MSFT
**Reason**: Same distance (5%), so use PnL tiebreaker (MSFT has worse PnL)

### Example 3: Fighting Resolution

**History**: AAPL and MSFT have stolen from each other 3 times in 30min

**Current**:
- AAPL: limit=100, current=92, distance=8%, PnL=2.0 (furthest)
- MSFT wants to steal: limit=200, current=200.2, PnL=5.0

**Decision**: Allow steal (MSFT has better PnL)
**Reason**: Fighting detected, but PnL=5.0 > PnL=2.0, so better PnL wins

### Example 4: Protection Zones

**Orders**:
- AAPL: limit=100, current=100.2, distance=0.2% ← PROTECTED (within 0.4%)
- MSFT: limit=100, current=95, distance=5% ← STEALABLE

**New Entry**:
- NVDA: limit=100, current=100.1

**Decision**: Steal from MSFT
**Reason**: AAPL is protected (too close to execution), MSFT is only option

## Why This Makes Sense

### Market Reality

1. **Limit orders don't always fill**: Order 5% from limit might sit forever
2. **Close orders execute**: Order 0.5% from limit will likely fill soon
3. **Capacity is limited**: With 2x leverage, can only have ~15 concurrent orders
4. **Maximize execution**: Better to have 15 orders that execute than 15 orders that sit idle

### Trading Psychology

**Bad**: Keep orders with great PnL that never execute
- "AAPL has 10% expected return!" 
- But price is 10% away and trending opposite direction
- Order sits for days, tying up capacity

**Good**: Prioritize orders likely to execute
- "MSFT is only 0.5% from limit"
- Even if PnL is lower, it will actually execute
- Better to get IN the market than wait forever

## Testing Summary

### Tests Created

**Unit Tests** (`test_work_stealing_distance_logic.py`):
- ✅ Steals from furthest, not worst PnL
- ✅ PnL used as tiebreaker for equal distance
- ✅ No PnL requirement allows any steal
- ✅ Fighting resolved by better PnL
- ✅ Candidates sorted correctly
- ✅ Edge cases (zero distance, negative PnL, etc.)

**Integration Tests** (`test_work_stealing_scenarios.py`):
- ✅ Rapid price movements
- ✅ Mixed crypto/stock competition
- ✅ Capacity dynamics
- ✅ Protection scenarios
- ✅ Extreme cases (all protected, single far order, etc.)

### Run Tests
```bash
pytest tests/unit/test_work_stealing_distance_logic.py -v
pytest tests/integration/test_work_stealing_scenarios.py -v
```

## Configuration Tuning

### More Aggressive Stealing
```bash
WORK_STEALING_TOLERANCE=0.005  # 0.5% - wider net
WORK_STEALING_PROTECTION=0.003  # 0.3% - tighter protection
```

### More Conservative
```bash
WORK_STEALING_TOLERANCE=0.002  # 0.2% - must be very close
WORK_STEALING_PROTECTION=0.006  # 0.6% - wider protection
```

### Disable Fighting Checks (if desired)
```bash
WORK_STEALING_FIGHT_THRESHOLD=99999  # Effectively disabled
```

## Monitoring

Watch for these patterns in logs:

**Good**:
```
Work steal: MSFT (dist=0.001, PnL=3.0) stealing from AAPL (dist=0.05, PnL=5.0)
```
↑ Stealing from furthest order (5% vs 0.1%)

**Fighting with resolution**:
```
MSFT: Fighting with AAPL but PnL better (4.0 > 2.0), allowing steal
```
↑ PnL resolving the fight

**Protected**:
```
AAPL: Position notional $4500 exceeds probe limit $300; promoting to normal
```
↑ Order protected by proximity to execution
