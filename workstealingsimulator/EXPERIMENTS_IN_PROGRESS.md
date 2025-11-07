# Steal Hyperparameter Optimization - Experiments In Progress

## Current Status

**Baseline optimization:** Still running (1750/1756 complete)
- Best so far: $19.1M PnL, 12 steals, 0 blocks, Score: 9.5M
- Protection range tested: 0.05-0.3%
- Found sweet spot: 0.15-0.20%

**Queued experiments:** 3 experiments waiting to run sequentially

## Experiment Queue

### Experiment 1: Narrow Protection Range
**File:** `experiment1_narrow_protection.py`
**Log:** `exp1_narrow_protection.log`

**Goal:** Fine-tune optimal protection threshold

**Setup:**
- Protection: **0.12-0.22%** (narrow range around winners)
- Crypto OOH tolerance: 2.5-4.5%
- Normal tolerance: 0.9-1.2%
- Entry tolerance: 0.6-1.2%
- Cooldown: 140-200s
- Iterations: 15, Population: 5

**Expected:**
- Find exact optimal protection (likely ~0.16%)
- Reach $20M+ PnL
- Maintain 0 blocks, 10-12 steals

**Runtime:** ~1-2 hours

---

### Experiment 2: Disable Crypto Fighting
**File:** `experiment2_no_fighting.py`
**Log:** `exp2_no_fighting.log`

**Goal:** Test if fighting detection helps or hurts with only 3 cryptos

**Setup:**
- Modified simulator: `would_cause_fight()` always returns False for crypto pairs
- Same parameter ranges as baseline
- Iterations: 12, Population: 4

**Hypothesis:**
- With only 3 crypto symbols, "fighting" is actually necessary behavior
- Disabling fighting may increase steals by 20-30%
- May see 15-18 steals instead of 10-12

**Expected:**
- Higher steal count
- Possibly higher PnL if more aggressive stealing helps
- Test if stability maintained

**Runtime:** ~1-1.5 hours

---

### Experiment 3: Narrow Cooldown Range
**File:** `experiment3_narrow_cooldown.py`
**Log:** `exp3_narrow_cooldown.log`

**Goal:** Optimize cooldown within winning range

**Setup:**
- Protection: 0.14-0.19% (fixed around optimal)
- Cooldown: **120-200s** (focus parameter)
- Other params: narrow ranges around winners
- Iterations: 12, Population: 4

**Hypothesis:**
- Cooldown around 150-170s is optimal
- Too short (<120s) causes thrashing
- Too long (>200s) misses opportunities

**Expected:**
- Confirm 2-3 minute cooldown is best
- Minor improvement over baseline (5-10%)

**Runtime:** ~1-1.5 hours

---

## Total Expected Runtime

- Current optimization: ~30 min remaining
- Experiment 1: 1-2 hours
- Experiment 2: 1-1.5 hours  
- Experiment 3: 1-1.5 hours

**Total: ~4-5 hours**

## Success Criteria

### Experiment 1 Success:
- PnL > $20M
- Steals: 10-14
- Blocks: 0
- Protection: 0.14-0.18%

### Experiment 2 Success:
- Steals > 15 (increase from 12)
- Blocks: 0
- PnL >= $19M (maintain or improve)
- No instability/oscillations

### Experiment 3 Success:
- Cooldown: 140-180s (confirm range)
- PnL >= $19M
- Steals: 10-12
- Blocks: 0

## Monitoring

Check progress:
```bash
# Overall progress
tail -f workstealingsimulator/experiments_master.log

# Individual experiments
tail -f workstealingsimulator/exp1_narrow_protection.log
tail -f workstealingsimulator/exp2_no_fighting.log  
tail -f workstealingsimulator/exp3_narrow_cooldown.log
```

Extract results:
```bash
# Get best configs from each
grep "OPTIMAL CONFIGURATION" workstealingsimulator/exp*.log -A 15

# Get performance metrics
grep "FINAL PERFORMANCE" workstealingsimulator/exp*.log -A 10

# Quick comparison
grep "steals" workstealingsimulator/exp*.log | grep "blocks"
```

## What We'll Learn

1. **Exact optimal protection** (0.12% vs 0.15% vs 0.18% vs 0.22%)
2. **Fighting necessity** (helps or hurts with 3 cryptos?)
3. **Cooldown sweet spot** (2min vs 2.5min vs 3min?)
4. **Steal count limits** (can we get >15 steals?)
5. **Parameter interactions** (how do these interact?)

## Next Steps After Experiments

1. **Update production config** with best parameters found
2. **Document final recommendations** with evidence
3. **Consider Experiment 4+** if more optimization needed:
   - Add stock positions
   - Time-based protection
   - Position size sensitivity
   - Market regime testing

## Files Created

- `experiment1_narrow_protection.py` (95 lines)
- `experiment2_no_fighting.py` (163 lines)
- `experiment3_narrow_cooldown.py` (88 lines)
- `run_all_experiments.sh` (43 lines)
- `EXPERIMENTS_IN_PROGRESS.md` (this file)

Total: ~389 lines of experiment code
