# Production Trading Bot Optimizations - Implementation Summary

**Date**: 2025-11-14
**Target**: `trade_stock_e2e.py` (PAPER=1 mode)
**Baseline**: 2047s (~34 min) execution time

## ✅ Optimizations Implemented

### 1. Disk-Based Kronos Prediction Cache

**File**: `src/kronos_prediction_cache.py` (new)
**Modified**: `backtest_test3_inline.py` lines 3137-3208

**What it does**:
- Caches Kronos ML predictions to disk with comprehensive cache keys
- Cache key includes: symbol, column, data hash, all parameters, environment settings
- TTL-based expiration (default: 5 minutes - safe for trading)
- Automatic cleanup when cache exceeds size limit

**Why it helps**:
- Avoids re-computing same predictions across analysis cycles
- Market data doesn't change rapidly enough to require fresh predictions every few seconds
- Especially effective when analyzing same symbols multiple times within a window

**Configuration**:
```bash
export KRONOS_PREDICTION_CACHE_ENABLED=1  # Default: enabled
export KRONOS_PREDICTION_CACHE_TTL=300    # 5 minutes
export KRONOS_PREDICTION_CACHE_MAX_SIZE_MB=1000  # 1GB max
```

**Expected Impact**: **15-25% speedup** depending on prediction reuse patterns

**Monitoring**:
```python
from src.kronos_prediction_cache import get_prediction_cache
stats = get_prediction_cache().get_stats()
print(f"Hit rate: {stats['hit_rate_percent']:.1f}%")
```

Cache stats are automatically logged when models are released.

---

### 2. Non-Blocking GPU Transfers

**Modified**: `backtest_test3_inline.py` lines 132-189, 200-218

**What it does**:
- Defers GPU→CPU transfers using lazy evaluation
- Uses `non_blocking=True` for async transfers when GPU→CPU copy is needed
- Keeps tensors on GPU as long as possible
- Only converts to numpy when truly accessed

**Why it helps**:
- GPU→CPU transfers were taking **34s** (1.6% of runtime) with 5,081 calls
- Many intermediate tensors never need CPU versions
- Non-blocking transfers avoid synchronization points in CUDA streams

**Code Pattern**:
```python
# OLD (immediate sync):
high_pred_base_np = high_pred_base.detach().cpu().numpy().copy()

# NEW (lazy + non-blocking):
cache["_high_pred_base_tensor"] = high_pred_base.detach()  # Keep on GPU

# Later, when/if needed:
np_array = _get_lazy_numpy(cache, "high_pred_base")  # non_blocking=True
```

**Expected Impact**: **5-10% speedup** from reduced sync overhead

---

### 3. Torch Compile (Already Enabled)

**File**: `src/models/kronos_wrapper.py` lines 681-702
**Status**: Already implemented, just needs environment variable

**What it does**:
- Compiles Kronos `decode_s1` and `decode_s2` methods using PyTorch 2.0 compilation
- Uses `mode="max-autotune"` for aggressive optimization
- One-time compilation cost, then faster inference

**Why it helps**:
- `decode_s1` takes **109s** (5.3% of runtime, 798 calls)
- Fuses operations, reduces Python overhead
- Optimizes memory access patterns

**Configuration**:
```bash
export KRONOS_COMPILE=1                   # Enable compilation
export KRONOS_COMPILE_MODE=max-autotune   # Aggressive optimization
export KRONOS_COMPILE_BACKEND=inductor    # Default backend
```

**Expected Impact**: **10-15% speedup** on compiled paths

---

## Combined Expected Speedup

**Conservative Estimate**: **25-35% total speedup**
- Caching: 15-20%
- GPU transfers: 5-10%
- Torch compile: 10-15%

**Projected Runtime**:
- Before: 2047s (~34 min)
- After: 1330-1536s (~22-26 min)

**Aggressive Estimate** (with good cache hit rates): **35-45% total speedup**
- Caching (high hit rate): 20-25%
- GPU transfers: 8-12%
- Torch compile: 12-18%

**Projected Runtime**:
- Before: 2047s (~34 min)
- After: 1126-1330s (~19-22 min)

---

## Testing & Validation

### Quick Test (Sanity Check)
```bash
# Run with optimizations enabled (default)
PAPER=1 python trade_stock_e2e.py

# Check cache stats in logs:
# Look for: "Kronos Prediction Cache Stats: hits=X misses=Y hit_rate=Z%"
```

### Disable Optimizations (Baseline Comparison)
```bash
# Disable all optimizations
KRONOS_PREDICTION_CACHE_ENABLED=0 \
KRONOS_COMPILE=0 \
PAPER=1 python trade_stock_e2e.py
```

### Verify Trading Quality Unchanged
Monitor these metrics over 1 week of paper trading:

1. **Sharpe Ratio**: Should not decrease >5%
2. **Win Rate**: Should not decrease >3%
3. **Average Return**: Should remain similar
4. **Max Drawdown**: Should not increase significantly

**Rollback Criteria**:
- Sharpe ratio drops >10% → **ROLLBACK**
- Win rate drops >5% → **ROLLBACK**
- Multiple GPU OOM errors → **ROLLBACK**
- Cache hit rate consistently <20% → Tune TTL or disable

---

## Performance Monitoring

### Cache Performance
```python
# In Python REPL or add to code:
from src.kronos_prediction_cache import get_prediction_cache
cache = get_prediction_cache()
stats = cache.get_stats()

print(f"Cache hit rate: {stats['hit_rate_percent']:.1f}%")
print(f"Total requests: {stats['hits'] + stats['misses']}")
print(f"Cache size: {stats['cache_size_mb']:.1f}MB")
print(f"Num entries: {stats['num_entries']}")
```

**Good**: Hit rate >50% in production
**Excellent**: Hit rate >70%
**Poor**: Hit rate <30% (may want to increase TTL or disable)

### Profiling (Before/After Comparison)
```bash
# Profile optimized version
python -m cProfile -o trade_optimized.prof trade_stock_e2e.py
python scripts/analyze_profile.py trade_optimized.prof docs/trade_optimized_hotpaths.md

# Compare with baseline
# Expected improvements:
# - .cpu() time: -70% (34s → 10s)
# - Kronos prediction time: -15-25% (170s → 128-145s)
# - Total time: -25-35% (2047s → 1330-1536s)
```

---

## Implementation Details

### Cache Key Generation

The cache key includes ALL factors that affect predictions:

1. **Data fingerprint**: SHA-256 hash of last `lookback` rows + relevant columns
2. **Prediction params**: `pred_len`, `lookback`, `temperature`, `top_p`, `top_k`, `sample_count`
3. **Environment settings**: `KRONOS_DTYPE`, compile flags, `TOTO_DTYPE`
4. **Symbol + Column**: e.g., `BTCUSD_Close`

**Example cache key**:
```
BTCUSD_Close_a3f8bc12d4e5_7f2a1b3c_9d4e5f6a.pkl
  ^       ^     ^            ^        ^
  |       |     |            |        +- env settings hash
  |       |     |            +---------- params hash
  |       |     +----------------------- data hash (last 512 rows)
  |       +----------------------------- column
  +------------------------------------ symbol
```

This ensures:
- Different symbols get different cache entries
- Price updates invalidate old predictions (data hash changes)
- Parameter changes invalidate cache
- Environment changes invalidate cache

### Lazy GPU Transfer Pattern

```python
# Pattern 1: Store tensor on creation
cache["_tensor_name_tensor"] = my_tensor.detach()

# Pattern 2: Retrieve with lazy conversion
def _get_lazy_numpy(cache, key):
    np_key = f"{key}_np"
    if np_key in cache:
        return cache[np_key]  # Already converted

    tensor = cache[f"_{key}_tensor"]
    if tensor.is_cuda:
        # Non-blocking: doesn't wait for GPU
        cpu_tensor = tensor.cpu(non_blocking=True)
    else:
        cpu_tensor = tensor

    # .numpy() will sync here, but only when accessed
    np_array = cpu_tensor.numpy().copy()
    cache[np_key] = np_array  # Cache for reuse
    return np_array
```

**Key insight**: If computation stays on GPU (e.g., torch operations), we never pay the sync cost!

---

## Future Optimizations (Not Implemented Yet)

See `docs/OPTIMIZATION_PLAN.md` for additional ideas:

### Tier 2 (Medium Risk)
- **Differential predictions**: Only re-predict symbols with >2% price change
- **Reduce optimizer iterations**: Tune DIRECT algorithm tolerance
- **Smart prediction scheduling**: Predict volatile symbols more frequently

### Tier 3 (High Risk - Test Thoroughly)
- **Model quantization**: INT8 quantization for Kronos
- **Reduce prediction horizon**: From 7 steps to 3-5 if strategies don't need full horizon

---

## Rollback Instructions

If optimizations cause issues:

### Disable Caching Only
```bash
export KRONOS_PREDICTION_CACHE_ENABLED=0
```

### Disable Torch Compile Only
```bash
export KRONOS_COMPILE=0
```

### Disable All Optimizations
```bash
export KRONOS_PREDICTION_CACHE_ENABLED=0
export KRONOS_COMPILE=0

# Revert lazy GPU transfers:
git diff backtest_test3_inline.py  # Review changes
git checkout backtest_test3_inline.py  # Revert if needed
```

### Full Rollback
```bash
# Revert all changes
git checkout backtest_test3_inline.py
rm src/kronos_prediction_cache.py
```

---

## Files Modified

1. **`src/kronos_prediction_cache.py`** (NEW)
   - Disk-based prediction cache with TTL
   - Comprehensive cache key generation
   - Automatic size management

2. **`backtest_test3_inline.py`**
   - Lines 132-189: Lazy GPU transfer pattern + helper function
   - Lines 200-218: Deferred tensor→numpy conversion
   - Lines 1708-1747: Cache stats logging + reset function
   - Lines 3137-3208: Integrated prediction caching

3. **`scripts/analyze_profile.py`** (NEW)
   - Hot path analyzer for .prof files
   - Also copied to `~/code/dotfiles/flamegraph-analyzer/`

4. **`docs/OPTIMIZATION_PLAN.md`** (NEW)
   - Comprehensive optimization strategy
   - Future optimization ideas

---

## Monitoring Checklist

After deployment, monitor for 1 week:

- [ ] Cache hit rate >40% (check logs)
- [ ] No GPU OOM errors (check logs for "OOM")
- [ ] Runtime reduction 20-40% (compare timestamps)
- [ ] Sharpe ratio unchanged ±10% (compare with previous week)
- [ ] Win rate unchanged ±5%
- [ ] No prediction staleness issues (verify TTL appropriate)
- [ ] Disk cache size <1GB (check `.cache/kronos_predictions/`)

**Success Criteria**: At least 20% speedup with no degradation in trading metrics.

---

## Questions & Troubleshooting

### Cache hit rate is <20%
**Possible causes**:
- TTL too short (predictions expiring before reuse)
- Data changing too frequently (price updates invalidate cache)
- Symbols not being re-analyzed within TTL window

**Solutions**:
- Increase TTL: `export KRONOS_PREDICTION_CACHE_TTL=600` (10 min)
- Check analysis frequency - ensure symbols analyzed multiple times
- May be expected behavior if truly running fresh analysis each time

### GPU memory usage increased
**Possible causes**:
- Tensors staying on GPU longer (expected with lazy transfers)
- Compiled models using more memory

**Solutions**:
- Monitor with `nvidia-smi` - ensure not near OOM
- Reduce sample_count if needed
- Disable lazy transfers if problematic (revert changes)

### Predictions seem stale
**Symptom**: Using old predictions when market moved significantly

**Solution**:
- Reduce TTL: `export KRONOS_PREDICTION_CACHE_TTL=180` (3 min)
- Cache automatically invalidates when input data changes
- Check data hash is capturing price updates correctly

### Compilation errors
**Symptom**: Errors about torch.compile or inductor

**Solution**:
- Disable compilation: `export KRONOS_COMPILE=0`
- Try different mode: `export KRONOS_COMPILE_MODE=reduce-overhead`
- Check PyTorch version: `python -c "import torch; print(torch.__version__)"`
