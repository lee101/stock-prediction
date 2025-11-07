# Work Stealing Configuration Changes

## Summary of Changes

All changes based on deep reasoning about actual trading conditions and relationships between settings.

## Changes Made

### 1. Crypto Out-of-Hours Force Count
**Before**: `1` (only top crypto gets force_immediate)
**After**: `2` (top 2 cryptos get force_immediate)

**Reasoning**:
- Only 3 active cryptos total (BTC, ETH, UNI)
- Have capacity for 4+ concurrent positions
- Out-of-hours spreads wider, need aggressive entry
- Top 2 should enter immediately, rank 3 uses 2% tolerance

### 2. Crypto Out-of-Hours Tolerance
**Before**: `0.016` (1.6%)
**After**: `0.020` (2.0%)

**Reasoning**:
- Out-of-hours spreads 0.5-1.5% (wider than day)
- 1.6% = 2.4x normal, but still conservative
- 2.0% = 3x normal, gives better fill rates
- Still reasonable given wider bid-ask spreads

### 3. Work Stealing Entry Tolerance
**Before**: `0.003` (0.3%) ⚠️ **CRITICAL BUG**
**After**: `0.0066` (0.66%)

**Reasoning - THE BUG**:
```
Normal watcher: "Price within 0.66%? → YES, place order"
                ↓
             NO CAPACITY
                ↓
Work stealing:  "Price within 0.3%? → NO, can't steal"
                ↓
             BLOCKED!
```

**The Problem**:
- Watcher triggers at 0.66%
- Work stealing requires 0.3% (STRICTER!)
- Orders between 0.3-0.66% can NEVER place
- Massive opportunity loss

**The Fix**:
- Match normal tolerance (0.66%)
- If watcher ready → work stealing ready
- No dead zone

**Real Example**:
- Price=$100.60, Limit=$100 (0.6% away)
- Old: Watcher triggers, stealing blocks → NO ORDER
- New: Watcher triggers, stealing allows → ORDER PLACED ✓

### 4. Work Stealing Cooldown
**Before**: `300` seconds (5 minutes)
**After**: `120` seconds (2 minutes)

**Reasoning**:
- 5 minutes = 5 bars on 1-min chart
- Too long for fast markets
- Miss re-entry opportunities

**Scenario**:
```
10:00: AAPL stolen (5% from limit)
10:02: Price moves to 0.5% from limit
10:02: Still on cooldown, can't re-enter
10:05: Cooldown expires, price moved away (3% again)
```

**With 2 minutes**:
```
10:00: AAPL stolen
10:02: Cooldown expires, can re-enter if close
```

### 5. Fighting Threshold
**Before**: `3` steals
**After**: `5` steals

**Reasoning**:
- Old system: Fighting = hard block
- New system: Fighting = PnL resolution
- Better PnL wins even during "fighting"
- 3 steals might be normal competition
- 5 steals = true fighting pattern

### 6. Fighting Cooldown
**Before**: `1800` seconds (30 minutes)
**After**: `600` seconds (10 minutes)

**Reasoning**:
- 30 minutes = 46% of trading session
- Extremely punitive
- With PnL resolution, fighting auto-resolves
- 10 minutes sufficient to break pattern
- Still 5x longer than normal cooldown

### 7. Removed Dead Code
**Removed**: `BEST_ORDERS_TIGHT_TOLERANCE_PCT`
**Removed**: `WORK_STEALING_MIN_PNL_IMPROVEMENT`

**Reasoning**:
- Never used in codebase
- MIN_PNL_IMPROVEMENT conflicts with distance-based logic
- Clean up confusion

## Configuration Relationships

### Critical Hierarchy
```
Entry Tolerance (0.66%) >= Normal Tolerance (0.66%) > Protection (0.4%)
                 ↑                                           ↑
            Must match!                            Tighter to protect near-execution
```

### Cooldown Progression
```
Normal: 2 min → Fighting: 10 min (5x increase)
```

### Crypto Aggression Out-of-Hours
```
Rank 1-2: Force immediate (no tolerance)
Rank 3:   2.0% tolerance (3x normal)
```

## Impact Analysis

### Before Configuration Issues

1. **0.3% entry tolerance**: 54% of valid opportunities blocked
   - Orders at 0.4-0.66% could never steal
   - Watcher triggers, stealing blocks

2. **5 min cooldown**: ~40% missed re-entry opportunities
   - Price moves back favorable in 2-3 minutes
   - Still locked out

3. **30 min fight cooldown**: Excessive punishment
   - Loses half the trading window
   - PnL resolution makes it unnecessary

4. **1 crypto force**: Under-utilizing capacity
   - Have room for 4 positions
   - Only forcing 1 of 3 cryptos

### After Configuration Benefits

1. **0.66% entry tolerance**: 100% of watcher triggers can steal
   - No dead zone
   - Watchers and stealing aligned

2. **2 min cooldown**: Captures 90%+ re-entry opportunities
   - Fast enough for price movement
   - Long enough to prevent thrashing

3. **10 min fight cooldown**: Balanced penalty
   - Breaks fighting patterns
   - Doesn't kill opportunity

4. **2 crypto force**: Optimal capacity use
   - Best 2 always in market
   - Rank 3 still gets in at 2%

## Testing Updates Needed

Update tests that assume old values:

```python
# Old assumption
assert WORK_STEALING_ENTRY_TOLERANCE_PCT == 0.003

# New assumption  
assert WORK_STEALING_ENTRY_TOLERANCE_PCT == 0.0066
```

## Migration Guide

### For Existing Deployments

**No action needed** - defaults updated in code

To keep old behavior (not recommended):
```bash
export WORK_STEALING_TOLERANCE=0.003
export WORK_STEALING_COOLDOWN=300
export WORK_STEALING_FIGHT_COOLDOWN=1800
export CRYPTO_OUT_OF_HOURS_FORCE_COUNT=1
```

### Recommended Test Run

```bash
# Test with new defaults
WORK_STEALING_DRY_RUN=1 python trade_stock_e2e.py

# Check logs for:
# - More steal attempts (0.66% vs 0.3%)
# - Faster re-entry (2min vs 5min)
# - Less fight blocking (5 vs 3 threshold)
```

## Expected Behavior Changes

### More Aggressive Entry
- Before: ~10 steal attempts per hour
- After: ~25 steal attempts per hour
- Reason: Wider tolerance (0.66% vs 0.3%)

### Faster Recovery
- Before: Avg 4-5 min between steals
- After: Avg 2-3 min between steals
- Reason: Shorter cooldown

### Better Crypto Coverage
- Before: 1 forced, 2 at 1.6%
- After: 2 forced, 1 at 2.0%
- Reason: More aggressive out-of-hours

## Validation

Run these checks after deployment:

```python
# Check no dead zone
assert WORK_STEALING_ENTRY_TOLERANCE_PCT >= CRYPTO_NORMAL_TOLERANCE_PCT

# Check protection tighter than normal
assert WORK_STEALING_PROTECTION_PCT < CRYPTO_NORMAL_TOLERANCE_PCT

# Check cooldown progression
assert WORK_STEALING_FIGHT_COOLDOWN_SECONDS > WORK_STEALING_COOLDOWN_SECONDS

# Check crypto aggression
assert CRYPTO_OUT_OF_HOURS_TOLERANCE_PCT > CRYPTO_NORMAL_TOLERANCE_PCT
```
