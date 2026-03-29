# Production Trading Bot Optimization Plan

**Context**: Optimizing `trade_stock_e2e.py` (PAPER=1 mode) - 2047s execution time across 27,874 functions.

## Executive Summary

**Total Runtime**: 2047 seconds (~34 minutes)
**Top Bottleneck**: Kronos ML predictions (28.7%, 170s)
**Quick Wins Available**: 10-15% speedup with low risk
**Aggressive Optimizations**: Up to 30-40% speedup with moderate risk

## Hot Path Analysis (Top 5 Bottlenecks)

### 1. Kronos Model Predictions (28.7% total - 576s cumulative)
- `predict`: 170.308s (114 calls, 1.49s/call)
- `generate`: 162.843s (114 calls, 1.43s/call)
- `auto_regressive_inference`: 153.404s (114 calls, 1.35s/call)
- `decode_s1`: 109.202s (798 calls, 0.14s/call)

**Analysis**: 114 prediction calls suggests one per symbol per analysis cycle. With ~28 symbols, this is likely 4-5 analysis runs.

### 2. GPU→CPU Data Transfers (1.6% - 33.7s)
- `.cpu()` calls: 33.730s (5,081 calls, 0.007s/call)
- High frequency indicates unnecessary data movement

### 3. PyTorch Operations (14.5% combined)
- `torch._C._nn.linear`: 37.083s (77,850 calls)
- `scaled_dot_product_attention`: 27.316s (11,053 calls)
- `forward` passes: Multiple instances totaling ~200s

### 4. Strategy Optimization (0.6%)
- `_multi_stage_grid_search`: 5.840s (472 calls)
- `scipy.optimize.direct`: 4.880s (107 calls)
- `_objective`: 4.482s (74,711 calls)

### 5. File I/O (0.8%)
- `write` operations: 15.520s + 11.851s (8,601 + 8,607 calls)

---

## Optimization Strategy Tiers

## 🟢 TIER 1: Low-Risk, High-Impact (Implement First)

### 1.1 Cache Kronos Predictions (Estimated: 15-20% speedup)
**Risk**: LOW | **Impact**: HIGH | **Effort**: LOW

**Current State**: 114 prediction calls, likely re-predicting same symbols multiple times

**Implementation**:
```python
# In backtest_test3_inline.py or kronos_wrapper.py
from functools import lru_cache
import hashlib

# Time-based cache (expires after N minutes)
_prediction_cache = {}
_cache_ttl_seconds = 300  # 5 minutes

def get_prediction_cache_key(symbol, context_hash, prediction_params):
    """Generate cache key for predictions."""
    params_str = f"{symbol}:{context_hash}:{prediction_params}"
    return hashlib.md5(params_str.encode()).hexdigest()

def get_cached_kronos_prediction(symbol, context, params):
    """Check if recent prediction exists."""
    import time
    cache_key = get_prediction_cache_key(symbol, hash(context.tobytes()), params)

    if cache_key in _prediction_cache:
        result, timestamp = _prediction_cache[cache_key]
        if time.time() - timestamp < _cache_ttl_seconds:
            return result
    return None

def cache_kronos_prediction(symbol, context, params, result):
    """Store prediction in cache."""
    import time
    cache_key = get_prediction_cache_key(symbol, hash(context.tobytes()), params)
    _prediction_cache[cache_key] = (result, time.time())
```

**Benefits**:
- Avoid re-computing same predictions across analysis runs
- Especially valuable if symbols analyzed multiple times within short window
- Market data doesn't change that fast (5-min cache is safe)

**Monitoring**: Track cache hit rate, log when using cached vs fresh predictions

---

### 1.2 Reduce GPU→CPU Transfers (Estimated: 5-8% speedup)
**Risk**: LOW | **Impact**: MEDIUM | **Effort**: LOW

**Current State**: 5,081 .cpu() calls taking 34s - likely in prediction output processing

**Implementation**:
```python
# In backtest_test3_inline.py around lines 132-133, 182

# BEFORE (current):
high_pred_base_np = high_pred_base.detach().cpu().numpy().copy()
low_pred_base_np = low_pred_base.detach().cpu().numpy().copy()

# AFTER (batched):
# Keep tensors on GPU as long as possible, only convert when truly needed
cache['high_pred_base_tensor'] = high_pred_base.detach()  # Keep on GPU
cache['low_pred_base_tensor'] = low_pred_base.detach()   # Keep on GPU

# Convert to numpy only when absolutely required for non-GPU operations
# Use lazy conversion: only convert when accessed
def get_numpy_prediction(tensor_name):
    if f"{tensor_name}_np" not in cache:
        cache[f"{tensor_name}_np"] = cache[tensor_name].cpu().numpy().copy()
    return cache[f"{tensor_name}_np"]
```

**Benefits**:
- Defer expensive GPU→CPU transfers until actually needed
- Many predictions might never need CPU version
- Batch transfers when they do happen

**Safety**: Ensure no GPU memory leaks, test memory usage

---

### 1.3 Enable PyTorch Compilation (Estimated: 10-15% speedup)
**Risk**: LOW | **Impact**: HIGH | **Effort**: LOW

**Current State**: Using standard PyTorch eager mode

**Implementation**:
```python
# In kronos_wrapper.py _ensure_predictor() method
# After model loading, before returning:

if hasattr(torch, 'compile') and os.getenv('KRONOS_USE_TORCH_COMPILE', '1') == '1':
    try:
        # Compile with reduce-overhead mode for inference
        predictor.model = torch.compile(
            predictor.model,
            mode='reduce-overhead',  # Good for repeated inference
            fullgraph=False,  # More flexible
        )
        logger.info("Kronos model compiled with torch.compile")
    except Exception as e:
        logger.warning(f"torch.compile failed: {e}, continuing without compilation")
```

**Benefits**:
- Fuses operations, reduces Python overhead
- Particularly effective for repeated inference patterns
- `reduce-overhead` mode optimizes for throughput

**Testing**: Run with `KRONOS_USE_TORCH_COMPILE=0` to disable if issues

---

### 1.4 Optimize File I/O (Estimated: 2-3% speedup)
**Risk**: LOW | **Impact**: LOW | **Effort**: LOW

**Current State**: 17k seconds across 17k+ write calls

**Implementation**:
```python
# Batch writes, use async I/O
import asyncio
from collections import deque

_pending_writes = deque()

def queue_write(path, data):
    """Queue write for batch processing."""
    _pending_writes.append((path, data))

def flush_writes():
    """Batch write queued data."""
    if not _pending_writes:
        return
    # Process in batches
    while _pending_writes:
        batch = [_pending_writes.popleft() for _ in range(min(100, len(_pending_writes)))]
        for path, data in batch:
            # Actual write
            pass
```

**Alternative**: Use environment variable to disable verbose logging in production

---

## 🟡 TIER 2: Medium-Risk, Medium-Impact

### 2.1 Reduce Prediction Frequency (Estimated: 20-30% speedup)
**Risk**: MEDIUM | **Impact**: HIGH | **Effort**: MEDIUM

**Current State**: Running predictions for all symbols in every analysis cycle

**Implementation Options**:

**Option A: Differential Updates**
- Only re-predict symbols with significant price changes
- Track last prediction price vs current price
- Re-predict if price moved >X% (e.g., 2%)

**Option B: Staged Analysis**
- Split symbols into tiers (high-priority, medium, low)
- Predict high-priority every cycle
- Medium priority every 2nd cycle
- Low priority every 3rd cycle

**Option C: Smart Scheduling**
- Track symbol volatility
- Predict volatile symbols more frequently
- Predict stable symbols less frequently

**Example Implementation**:
```python
# Add to trade_stock_e2e.py

_last_prediction_prices = {}
_prediction_thresholds = {
    'crypto': 0.02,  # 2% for crypto (more volatile)
    'equity': 0.01,  # 1% for equities
}

def should_repredict_symbol(symbol, current_price):
    """Check if symbol needs new prediction based on price change."""
    if symbol not in _last_prediction_prices:
        return True

    last_price, last_time = _last_prediction_prices[symbol]
    price_change = abs(current_price - last_price) / last_price

    threshold = _prediction_thresholds.get(
        'crypto' if symbol.endswith('USD') else 'equity',
        0.015
    )

    # Always repredict if >X time has passed (e.g., 30 min)
    time_elapsed = time.time() - last_time
    if time_elapsed > 1800:  # 30 minutes
        return True

    return price_change >= threshold
```

**Safety**: Track prediction freshness, ensure no stale predictions used for trading

---

### 2.2 Reduce Strategy Optimization Iterations (Estimated: 3-5% speedup)
**Risk**: MEDIUM | **Impact**: LOW | **Effort**: LOW

**Current State**: 472 grid search calls, 74,711 objective evaluations

**Implementation**:
```python
# In optimization_utils_fast.py
# Reduce DIRECT algorithm iterations/tolerance

# Add environment control
MAX_OPTIMIZER_ITERATIONS = int(os.getenv('OPTIMIZER_MAX_ITER', '50'))  # Reduce from default
OPTIMIZER_TOLERANCE = float(os.getenv('OPTIMIZER_TOLERANCE', '0.001'))  # Increase tolerance

# In direct() call:
result = direct(
    objective,
    bounds,
    maxfun=MAX_OPTIMIZER_ITERATIONS,  # Limit function evaluations
    eps=OPTIMIZER_TOLERANCE,
)
```

**Testing**: Compare strategy quality with reduced iterations vs full iterations

---

## 🔴 TIER 3: High-Risk, High-Impact (Test Thoroughly)

### 3.1 Model Quantization (Estimated: 20-30% speedup)
**Risk**: HIGH | **Impact**: HIGH | **Effort**: HIGH

**Implementation**: Use INT8 quantization for Kronos model
**Concerns**: May affect prediction quality, needs extensive backtesting
**Recommendation**: Test in paper trading first, compare returns

### 3.2 Reduce Prediction Horizon (Estimated: 15-25% speedup)
**Risk**: HIGH | **Impact**: HIGH | **Effort**: LOW

**Current State**: Predicting max_horizon=7 steps ahead

**Implementation**: Reduce to 3-5 steps if strategies don't use full horizon

**Safety**: Verify all strategies work with shorter horizon

---

## Implementation Roadmap

### Phase 1: Quick Wins (Week 1)
1. ✅ Enable PyTorch compilation (1.3)
2. ✅ Reduce GPU→CPU transfers (1.2)
3. ✅ Add prediction caching framework (1.1)
4. ✅ Optimize file I/O (1.4)

**Expected Speedup**: 25-35%
**Risk**: Very Low

### Phase 2: Smart Caching (Week 2)
1. ✅ Implement prediction cache with TTL
2. ✅ Add cache hit/miss monitoring
3. ✅ Tune cache TTL based on market conditions

**Expected Speedup**: Additional 15-20%
**Risk**: Low (with proper monitoring)

### Phase 3: Prediction Optimization (Week 3-4)
1. ⚠️ Test differential update strategy (2.1)
2. ⚠️ Implement smart prediction scheduling
3. ⚠️ Validate trading performance unchanged

**Expected Speedup**: Additional 20-30%
**Risk**: Medium (requires careful validation)

---

## Monitoring & Validation

### Key Metrics to Track
1. **Performance Metrics**:
   - Total analysis time per cycle
   - Prediction time per symbol
   - Cache hit rate
   - GPU memory usage

2. **Trading Metrics**:
   - Sharpe ratio (should not decrease)
   - Win rate (maintain or improve)
   - Average return per trade
   - Max drawdown

3. **Quality Metrics**:
   - Prediction freshness
   - Strategy quality scores
   - Number of missed opportunities

### Rollback Triggers
- Sharpe ratio drops >10%
- Win rate drops >5%
- Cache hit rate <30% (cache too aggressive)
- GPU OOM errors increase

---

## Environment Variables for Control

```bash
# Prediction caching
export KRONOS_PREDICTION_CACHE_TTL=300  # seconds (5 min default)
export KRONOS_PREDICTION_CACHE_ENABLED=1

# PyTorch compilation
export KRONOS_USE_TORCH_COMPILE=1
export TORCH_COMPILE_MODE=reduce-overhead

# Optimization tuning
export OPTIMIZER_MAX_ITER=50
export OPTIMIZER_TOLERANCE=0.001

# Differential predictions
export PREDICTION_PRICE_THRESHOLD=0.015  # 1.5%
export PREDICTION_TIME_THRESHOLD=1800     # 30 minutes

# Parallel analysis (already exists)
export MARKETSIM_PARALLEL_ANALYSIS=1
export MARKETSIM_PARALLEL_WORKERS=8
```

---

## Expected Results

### Conservative Estimate (Phase 1 only)
- **Before**: 2047s (~34 min)
- **After**: 1330s (~22 min)
- **Speedup**: 35%

### Aggressive Estimate (All phases)
- **Before**: 2047s (~34 min)
- **After**: 920s (~15 min)
- **Speedup**: 55%

### Safety Buffer
Start with Phase 1, validate trading performance unchanged, then proceed to Phase 2/3.

---

## Next Steps

1. **Review this plan** - validate assumptions about prediction patterns
2. **Choose tier** - start with Tier 1 (low risk)
3. **Implement in order** - test each change independently
4. **Monitor closely** - ensure no degradation in trading quality
5. **Profile again** - after each phase to validate improvements

**Critical**: Always test in PAPER=1 mode before real money. Compare P&L over at least 1 week.
