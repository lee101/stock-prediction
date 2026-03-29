# Quick Start: Production Bot Optimizations

## TL;DR

**3 optimizations implemented, ~30-40% speedup expected**

Enable all optimizations:
```bash
export KRONOS_COMPILE=1
export KRONOS_PREDICTION_CACHE_ENABLED=1
PAPER=1 python trade_stock_e2e.py
```

Check it's working:
```bash
# Should see in logs:
# "Kronos Prediction Cache Stats: hits=X misses=Y hit_rate=Z%"
# "Applying torch.compile to Kronos decode methods"
```

---

## What Was Optimized

### 1. Disk-Based Prediction Cache (15-25% speedup)
- **What**: Caches Kronos ML predictions to disk
- **Why**: Avoids re-computing same predictions across analysis cycles
- **Safe**: 5-minute TTL, comprehensive cache keys include all settings
- **File**: `src/kronos_prediction_cache.py`

### 2. Non-Blocking GPU Transfers (5-10% speedup)
- **What**: Defers GPU→CPU data movement, uses async transfers
- **Why**: Was spending 34s (1.6%) on sync transfers
- **Safe**: Only changes when data is moved, not what is computed
- **Files**: `backtest_test3_inline.py` lines 132-218

### 3. PyTorch Compilation (10-15% speedup)
- **What**: Compiles Kronos decode methods using torch.compile
- **Why**: decode_s1/s2 are hot paths (109s+, 5.3%)
- **Safe**: Compilation already implemented, just needs env var
- **File**: Already in `src/models/kronos_wrapper.py`

---

## Expected Results

**Before**: 2047s (~34 min) per analysis cycle
**After**: 1226-1536s (~20-26 min) per analysis cycle
**Speedup**: **25-40%**

---

## How To Use

### Enable All Optimizations (Recommended)
```bash
# Set environment variables
export KRONOS_COMPILE=1
export KRONOS_PREDICTION_CACHE_ENABLED=1

# Run paper trading
PAPER=1 python trade_stock_e2e.py
```

### Fine-Tune Cache Settings
```bash
# Increase cache TTL (if symbols analyzed frequently)
export KRONOS_PREDICTION_CACHE_TTL=600  # 10 minutes

# Increase max cache size
export KRONOS_PREDICTION_CACHE_MAX_SIZE_MB=2000  # 2GB

# Disable cache (testing)
export KRONOS_PREDICTION_CACHE_ENABLED=0
```

### Torch Compile Settings
```bash
# Default (recommended)
export KRONOS_COMPILE=1
export KRONOS_COMPILE_MODE=max-autotune

# Faster compilation, slightly less optimization
export KRONOS_COMPILE_MODE=reduce-overhead

# Disable if issues
export KRONOS_COMPILE=0
```

---

## Monitoring

### Check Cache Performance
Look for this in logs:
```
INFO: Kronos Prediction Cache Stats: hits=45 misses=12 hit_rate=78.9% entries=8 size=12.3MB
```

**Good**: Hit rate >50%
**Excellent**: Hit rate >70%
**Investigate**: Hit rate <30% (may need to tune TTL)

### Check Torch Compile
Look for this on startup:
```
INFO: Applying torch.compile to Kronos decode methods (mode=max-autotune, backend=inductor)
INFO: Kronos torch.compile applied successfully
```

### Performance Comparison
```bash
# Baseline (no optimizations)
time KRONOS_COMPILE=0 KRONOS_PREDICTION_CACHE_ENABLED=0 PAPER=1 python trade_stock_e2e.py

# Optimized
time KRONOS_COMPILE=1 KRONOS_PREDICTION_CACHE_ENABLED=1 PAPER=1 python trade_stock_e2e.py
```

---

## Validation (Before Production)

Run paper trading for 1 week and compare:

1. **Runtime**: Should be 25-40% faster
2. **Sharpe Ratio**: Should be unchanged (±5%)
3. **Win Rate**: Should be unchanged (±3%)
4. **No errors**: Check logs for OOM, cache errors

**If any issues**: Disable optimizations and investigate.

---

## Troubleshooting

### "ModuleNotFoundError: No module named 'src.kronos_prediction_cache'"
**Fix**: Ensure running from project root, or:
```bash
export PYTHONPATH=/nvme0n1-disk/code/stock-prediction:$PYTHONPATH
```

### Cache hit rate is very low (<20%)
**Possible causes**:
- TTL too short for analysis frequency
- Symbols not being re-analyzed within TTL window
- Data changing frequently (expected)

**Try**: Increase TTL to 10 minutes
```bash
export KRONOS_PREDICTION_CACHE_TTL=600
```

### GPU OOM errors
**Fix**: Reduce sample count or disable lazy GPU transfers
```bash
# Quick fix: revert lazy transfers
git checkout backtest_test3_inline.py
```

### Torch compile errors
**Fix**: Disable compilation
```bash
export KRONOS_COMPILE=0
```

---

## Files Changed

All changes are backward compatible - optimizations can be disabled via env vars.

**New files**:
- `src/kronos_prediction_cache.py` - Disk cache implementation
- `scripts/analyze_profile.py` - Profile analyzer
- `docs/OPTIMIZATION_PLAN.md` - Full optimization strategy
- `docs/OPTIMIZATIONS_IMPLEMENTED.md` - Detailed implementation docs

**Modified files**:
- `backtest_test3_inline.py` - Lazy GPU transfers, cache integration

**Existing features used**:
- `src/models/kronos_wrapper.py` - torch.compile (already implemented!)

---

## Rollback

If needed, disable all optimizations:
```bash
export KRONOS_COMPILE=0
export KRONOS_PREDICTION_CACHE_ENABLED=0
```

Full code rollback:
```bash
git checkout backtest_test3_inline.py
rm src/kronos_prediction_cache.py
```

---

## Next Steps

1. **Test in paper mode**: Run for a few hours, check logs
2. **Monitor cache hit rate**: Should be >40% after warmup
3. **Validate speedup**: Compare with baseline
4. **Run for 1 week**: Ensure no degradation in trading metrics
5. **Profile again**: `python -m cProfile -o new.prof trade_stock_e2e.py`

See `docs/OPTIMIZATIONS_IMPLEMENTED.md` for full details.

---

## Questions?

- Full docs: `docs/OPTIMIZATIONS_IMPLEMENTED.md`
- Optimization plan: `docs/OPTIMIZATION_PLAN.md`
- Profile analysis: `docs/trade_stock_e2e_paper_hotpaths.md`
