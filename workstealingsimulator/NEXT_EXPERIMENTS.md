# Next Experiments for Work Stealing Optimization

## Current State

### What We Know
From optimization runs with protection range 0.05-0.3%:

**Winning Configs (11 steals, 0 blocks):**
- PnL: $17-19M
- Sharpe: 0.43-0.44
- Win rate: 53.5-54.1%
- Trades: 14,700-16,800
- Score: 7.2M-8.8M

**Losing Configs (2 steals, 8k blocks):**
- PnL: $720k-880k
- Sharpe: 2.06-2.31
- Win rate: 53.6-54.4%
- Trades: 7,500-7,900
- Score: 280k-366k

**Key Finding:** Protection threshold around 0.15-0.2% enables perfect stealing.

### Parameter Ranges Tested
```python
crypto_ooh_force_count: 1-3
crypto_ooh_tolerance_pct: 1-5%
crypto_normal_tolerance_pct: 0.3-1.5%
entry_tolerance_pct: 0.3-1.5%
protection_pct: 0.05-0.3%  # ← Key parameter!
cooldown_seconds: 60-300s
fight_threshold: 3-10
fight_cooldown_seconds: 300-1800s
```

## 🧪 Experiment Ideas

### Experiment 1: Narrow Protection Range
**Goal:** Find optimal protection within winning range

**Hypothesis:** Optimal is 0.15-0.18% based on current results

**Setup:**
```python
bounds = [
    (2, 3),           # crypto_ooh_force_count - focus on 2-3
    (0.02, 0.04),     # crypto_ooh_tolerance_pct - narrow range
    (0.008, 0.013),   # crypto_normal_tolerance_pct
    (0.005, 0.013),   # entry_tolerance_pct
    (0.0012, 0.0022), # protection_pct - TIGHT RANGE around 0.15-0.2%
    (120, 200),       # cooldown_seconds - focus on 2-3 min
    (6, 10),          # fight_threshold - higher end
    (500, 1000),      # fight_cooldown_seconds
]
```

**Expected:** Find exact optimal protection, may hit $20M+ PnL

---

### Experiment 2: Add Stock Positions
**Goal:** Test work stealing with mixed crypto/stock portfolio

**Hypothesis:** Stocks with 4x leverage have different optimal parameters

**Setup:**
1. Modify simulator to include stock symbols (AAPL, TSLA, etc.)
2. Test position sizes: $2k crypto, $8k stocks
3. Apply leverage fees on stock positions
4. Measure steal patterns: crypto→crypto, stock→stock, crypto↔stock

**New metrics to track:**
- Crypto steals vs stock steals
- Leverage fees paid
- Cross-asset stealing behavior
- EOD deleveraging triggers

---

### Experiment 3: Time-Based Protection
**Goal:** Replace distance-based protection with time-based

**Hypothesis:** "Protect orders created in last 60s" may work better than distance

**Setup:**
```python
def is_protected(order, timestamp):
    # Protect recently created orders regardless of distance
    age_seconds = (timestamp - order.created_at).total_seconds()
    return age_seconds < PROTECTION_TIME_SECONDS

bounds = [
    # ... other params ...
    (30, 300),  # protection_time_seconds - 30s to 5min
]
```

**Advantages:**
- Prevents immediate steal-back oscillation
- Allows stealing distant orders that are "stuck"
- Simpler logic, less brittle

---

### Experiment 4: Disable Fighting for Cryptos
**Goal:** Test if fighting detection hurts more than helps

**Hypothesis:** With only 3 cryptos, fighting is necessary behavior, not a bug

**Setup:**
```python
def would_cause_fight(new_sym, steal_sym):
    # Never fight for crypto-crypto steals
    if is_crypto(new_sym) and is_crypto(steal_sym):
        return False
    # Only track fighting for stocks
    return normal_fighting_logic()
```

**Expected:** More steals, possibly better PnL, test if stability maintained

---

### Experiment 5: Position Size Sensitivity
**Goal:** How does position sizing affect optimal parameters?

**Current:** $3,500 per crypto order (3 orders = 105% of capital)

**Test scenarios:**
- Small: $2,000 per order (5 orders fit)
- Medium: $3,500 per order (current)
- Large: $5,000 per order (2 orders only)

**Hypothesis:** Larger positions need tighter protection (less room for error)

---

### Experiment 6: Market Regime Testing
**Goal:** Do parameters work across different market conditions?

**Setup:**
1. Split data into regimes:
   - Bull market (2021 crypto run)
   - Bear market (2022 crypto crash)
   - Sideways (2023-2024)
   
2. Run optimization on each regime separately
3. Test if parameters generalize or need regime-specific tuning

**Key question:** Can one config rule them all?

---

### Experiment 7: Cooldown Optimization
**Goal:** Is 2-3 min cooldown optimal or just good enough?

**Current winning configs:** 120-200s cooldown

**Test:**
- Very short: 30-90s (aggressive restealing)
- Short: 90-150s
- Medium: 150-240s (current sweet spot)
- Long: 240-600s (conservative)

**Trade-off:** Shorter allows more steals but risks thrashing

---

### Experiment 8: Multi-Objective Optimization
**Goal:** Optimize for PnL AND Sharpe simultaneously

**Current score:** `0.5*pnl + 1000*sharpe + 2000*win_rate - 10*blocks`

**Problem:** High PnL configs have low Sharpe (0.43 vs 2.3)

**Alternative scores:**
1. Pareto optimization (find frontier)
2. Risk-adjusted: `pnl / max_drawdown`
3. Kelly criterion: `pnl * win_rate / avg_loss`

**Test:** Can we get $15M+ PnL with Sharpe > 1.0?

---

### Experiment 9: Adaptive Protection
**Goal:** Dynamic protection based on market volatility

**Hypothesis:** Tight protection (0.15%) in calm markets, wider (0.3%) in volatile

**Setup:**
```python
def get_protection(symbol, volatility):
    base = 0.0015  # 0.15%
    vol_multiplier = min(volatility / avg_volatility, 2.0)
    return base * vol_multiplier
```

**Measure:** Rolling 1-hour volatility from price data

---

### Experiment 10: Ensemble Testing
**Goal:** Run multiple configs simultaneously and blend

**Setup:**
1. Identify top 5 configs from optimization
2. Run each on same data
3. Blend signals: steal when 3+ configs agree

**Expected:** More robust, less sensitive to parameter choice

---

## 🎯 Recommended Priority

### Phase 1: Refinement (Next 1-2 days)
1. **Experiment 1** - Narrow protection range (quick win)
2. **Experiment 7** - Cooldown optimization
3. **Experiment 4** - Disable crypto fighting

### Phase 2: Reality Check (Next week)
4. **Experiment 2** - Add stocks (critical for production)
5. **Experiment 5** - Position size sensitivity
6. **Experiment 6** - Market regime testing

### Phase 3: Advanced (Future)
7. **Experiment 3** - Time-based protection (architectural change)
8. **Experiment 8** - Multi-objective optimization
9. **Experiment 9** - Adaptive protection
10. **Experiment 10** - Ensemble approach

## 📊 Expected Outcomes

### Experiment 1 (Narrow Protection)
- Time: 1-2 hours
- Confidence: High
- Expected improvement: 5-10% (reach $20M PnL)

### Experiment 2 (Add Stocks)
- Time: 4-6 hours (more complex)
- Confidence: Medium
- Expected: Understand stock vs crypto stealing dynamics

### Experiment 4 (No Crypto Fighting)
- Time: 30 minutes
- Confidence: Medium
- Expected: 10-20% more steals, test stability

## 🚧 Implementation Notes

### For Each Experiment:
1. Create branch: `experiment-{name}`
2. Modify simulator.py with changes
3. Run optimization: `python3 simulator.py > exp_{name}.log`
4. Document results in `EXPERIMENT_{NAME}_RESULTS.md`
5. Compare against baseline ($19.1M, 12 steals, 0 blocks)
6. Merge if better, discard if worse

### Baseline Config (to beat):
```python
Config: [2.85, 0.047, 0.013, 0.0057, 0.00153, 166, 8.0, 645]
PnL: $19.1M, Steals: 12, Blocks: 0, Score: 9.5M
```

## 💡 Key Questions to Answer

1. What's the absolute optimal protection? (Exp 1)
2. Do stocks need different parameters? (Exp 2)
3. Is time-based protection better? (Exp 3)
4. Is fighting detection helping or hurting? (Exp 4)
5. Do parameters generalize across market regimes? (Exp 6)
6. Can we get high PnL with high Sharpe? (Exp 8)
