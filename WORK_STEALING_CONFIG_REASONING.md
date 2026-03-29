# Work Stealing Configuration: Deep Reasoning

## Settings Analysis

### 1. CRYPTO_OUT_OF_HOURS_FORCE_IMMEDIATE_COUNT = 1

**Current**: Top 1 crypto gets force_immediate (ignores tolerance)

**Analysis**:
- Out of hours: Only 3 active cryptos (BTC, ETH, UNI)
- Capacity: 2x leverage on ~$10k = $20k total
- Avg crypto position: ~$5k each
- Can hold 4 positions simultaneously

**Issue**: With only 3 cryptos total and capacity for 4, why force only 1?

**Better Logic**:
- Top 2 cryptos should force immediate (we have capacity)
- Rank 3 uses 2.0% tolerance (still gets in easily)
- This ensures we're always in the market with best 2 cryptos

**Recommendation**: Change to **2**

---

### 2. CRYPTO_OUT_OF_HOURS_TOLERANCE_PCT = 0.016 (1.6%)

**Current**: 1.6% for non-forced cryptos out-of-hours

**Analysis**:
- Normal tolerance: 0.66%
- Multiplier: 2.4x more aggressive
- Out-of-hours spreads: Typically 0.5-1.0% on crypto
- Competition: None (no stocks trading)

**Considerations**:
- Wider spreads at night/weekends
- Less liquidity = wider bid-ask
- Want to get filled but not overpay

**Issue**: 1.6% might still be conservative

**Recommendation**: Increase to **0.020 (2.0%)** = 3x normal
- Gives ~2% room above limit to catch fills
- Still reasonable given wider spreads

---

### 3. CRYPTO_NORMAL_TOLERANCE_PCT = 0.0066 (0.66%)

**Current**: 0.66% for all assets during market hours

**Analysis**:
- Proven default from original maxdiff implementation
- During market hours: tight spreads, good liquidity
- 0.66% ≈ typical bid-ask spread

**Recommendation**: **Keep 0.0066** (battle-tested)

---

### 4. WORK_STEALING_ENTRY_TOLERANCE_PCT = 0.003 (0.3%)

**Current**: Must be within 0.3% to attempt steal

**Analysis - Flow**:
1. Watcher checks: "within 0.66% of limit?" → YES
2. Watcher tries to place order → NO CAPACITY
3. Work stealing: "am I within 0.3%?" → Maybe NO
4. Can't steal, blocked

**Problem**: This is STRICTER than watcher tolerance (0.66%)!

**Better Logic**:
- If watcher is ready to place (within 0.66%), should be able to steal
- Stealing tolerance should MATCH or be WIDER than normal tolerance
- 0.3% means many valid steal opportunities are blocked

**Real Example**:
- Price at $100.60, limit $100 → 0.6% away
- Watcher triggers (within 0.66%)
- Work stealing blocks (not within 0.3%)
- Result: Order never places, opportunity missed

**Recommendation**: Change to **0.0066 (0.66%)** to match normal tolerance
- Or **0.008 (0.8%)** to be slightly more permissive

---

### 5. WORK_STEALING_PROTECTION_PCT = 0.004 (0.4%)

**Current**: Orders within 0.4% can't be stolen

**Analysis**:
- Normal tolerance: 0.66%
- Protection: 0.4% (tighter)
- Relationship: Protection < Normal makes sense

**Logic Check**:
- Order at 0.3% from limit → Protected ✓
- Order at 0.5% from limit → Stealable ✓
- Order at 1.0% from limit → Stealable ✓

**Edge Case**:
- With normal=0.66%, protection=0.4%
- Order placed at 0.66%, moves to 0.5%
- Now stealable even though recently placed

**Recommendation**: **Keep 0.004** but ensure it's < normal tolerance
- If normal becomes 0.8%, protection should stay 0.004

---

### 6. WORK_STEALING_COOLDOWN_SECONDS = 300 (5 minutes)

**Current**: 5 minute cooldown after being stolen from

**Analysis - Trading Timeframes**:
- 1-minute bars: 5 bars pass
- 5-minute bars: 1 bar passes
- Price can move significantly in 5 minutes

**Problem**: Too long for fast-moving markets

**Scenario**:
- 10:00: AAPL stolen from (far from limit)
- 10:02: Price moves favorably, now 0.5% from limit
- 10:02: Can't re-enter (still on cooldown)
- 10:05: Cooldown expires, price moved away again

**Recommendation**: Reduce to **120 seconds (2 minutes)**
- 2 bars on 1-min chart
- Enough to prevent rapid oscillation
- Fast enough to catch favorable price moves

---

### 7. WORK_STEALING_FIGHT_THRESHOLD = 3 (steals)

**Current**: 3 steals in window = fighting

**Analysis - With PnL Resolution**:
- Old: Fighting = deadlock, block all steals
- New: Fighting = use PnL to resolve
- Better PnL wins even during fighting

**Implications**:
- Fighting is now RESOLVED, not BLOCKED
- Threshold less critical
- Higher threshold = more tolerance for competition

**Recommendation**: Increase to **5**
- Allows more competitive stealing
- PnL will resolve true fights
- 3 steals might be normal competition, not fighting

---

### 8. WORK_STEALING_FIGHT_WINDOW_SECONDS = 1800 (30 minutes)

**Current**: Look back 30 min for fighting pattern

**Analysis**:
- Trading session: 6.5 hours (NYSE)
- 30 minutes = 7.7% of session
- Enough to detect patterns
- Not so long that stale

**Recommendation**: **Keep 1800** (well-balanced)

---

### 9. WORK_STEALING_FIGHT_COOLDOWN_SECONDS = 1800 (30 minutes)

**Current**: 30 minute extended cooldown when fighting detected

**Analysis - Severity**:
- Normal cooldown: 5 minutes (or proposed 2 minutes)
- Fight cooldown: 30 minutes (15x or 900x longer!)
- Very punitive

**With PnL Resolution**:
- Fighting resolves via PnL
- Better PnL wins automatically
- Extended cooldown less necessary

**Recommendation**: Reduce to **600 seconds (10 minutes)**
- Still longer than normal (5-10x)
- Not punitive (30 min = half the opportunity window)
- Gives time for conditions to change

---

### 10. BEST_ORDERS_TIGHT_TOLERANCE_PCT = 0.005 (0.5%)

**Current**: Unused in codebase

**Search Results**: Not referenced anywhere in stealing logic

**Recommendation**: **REMOVE** - Dead code, confusing

---

### 11. WORK_STEALING_MIN_PNL_IMPROVEMENT = 1.1 (10% better)

**Current**: Defined but NO LONGER USED (removed in distance-based refactor)

**Recommendation**: **REMOVE** - Obsolete, conflicts with new logic

---

## Critical Relationships

### Tolerance Hierarchy
```
Entry Stealing (0.66%) >= Normal Watcher (0.66%) > Protection (0.4%)
```

**Why**: 
- Can steal when watcher would normally place
- Protection is tighter to save near-execution orders

### Cooldown Hierarchy
```
Fight Cooldown (10 min) > Normal Cooldown (2 min)
```

**Why**:
- Extended penalty for fighting
- But not so long it kills opportunity

### Crypto Out-of-Hours Aggression
```
Force Immediate (top 2) > High Tolerance (2.0%) > Normal (0.66%)
```

**Why**:
- Maximize market exposure when spreads wide
- Only 3 cryptos, can afford to be aggressive
