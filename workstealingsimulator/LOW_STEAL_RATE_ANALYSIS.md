# Why Are Steals So Low?

## Observed Behavior

Across all optimization runs:
- **Blocks: 5,846-8,558** (thousands of rejected orders)
- **Steals: 0-2** (almost zero successful steals)
- **Ratio: ~0.02%** steal success rate

This means 99.98% of blocked orders fail to steal capacity, despite clear need.

## Root Cause Analysis

### 1. Protection Too Wide (0.4%)
Current protection threshold: `0.4%` from limit price

For a $30,000 BTC order:
- Protected if within $120 of limit
- With crypto volatility, many orders stay within this band
- Most active orders are protected, can't be stolen

### 2. Entry Tolerance vs Stealing Logic Mismatch

**Entry tolerance: 0.66%** - Orders only attempt entry when within 0.66% of limit
**Protection: 0.4%** - Orders protected when within 0.4% of limit

Window for stealing: **0.26% band** (0.66% - 0.4%)

For $30k BTC: Only $78 price window where:
- Order is close enough to attempt entry (< 0.66%)
- Target is far enough to steal (> 0.4%)

This tiny window explains the low steal rate.

### 3. Candidate Selection Issue

Current logic:
```python
candidates.sort(key=lambda c: (-c.distance_pct, c.forecasted_pnl))
```

This sorts by furthest first, but if ALL candidates are protected (< 0.4%), the list is empty.

### 4. Fighting Detection Too Sensitive

With only 3 cryptos and high churn:
- Threshold: 5 steals between same pair
- Cooldown: 10 minutes
- With only 3 symbols, fighting likely triggered quickly
- Blocks future steals even when needed

## Proposed Fixes

### Option 1: Relax Protection (Immediate)
```python
WORK_STEALING_PROTECTION_PCT = 0.001  # 0.1% instead of 0.4%
```

For $30k BTC: $30 window instead of $120
- Protects imminent fills (< 30 seconds away)
- Allows more steals for orders 0.1-0.66% away

### Option 2: Widen Stealing Window
```python
WORK_STEALING_ENTRY_TOLERANCE_PCT = 0.015  # 1.5% instead of 0.66%
```

Attempt steals earlier, before hitting entry tolerance.

### Option 3: Protection Based on Time, Not Distance
```python
# Protect orders created in last 60 seconds
# Regardless of distance
```

Prevents immediate steal-back oscillation without blocking distant orders.

### Option 4: Remove Fighting for Cryptos
With only 3 crypto symbols:
```python
# Don't track fighting for crypto-crypto steals
# Only track for stock-stock steals
```

Cryptos NEED work stealing due to 1x leverage constraint.

## Recommendation

**Immediate**: Change protection to 0.1% (10x tighter)
```python
WORK_STEALING_PROTECTION_PCT = 0.001
```

**If still low**: Also widen entry tolerance to 1.5%
```python
WORK_STEALING_ENTRY_TOLERANCE_PCT = 0.015
```

**If fighting detected**: Disable fighting for crypto pairs
```python
def would_cause_fight(self, new_symbol, steal_symbol, ts):
    # Don't fight for cryptos - only 3 pairs, need stealing
    if new_symbol.endswith("USD") and steal_symbol.endswith("USD"):
        return False
    # ... rest of logic
```

This should increase steals from <0.1% to 10-30% of blocks.
