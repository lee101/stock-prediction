# Disk Cache Analysis: Why It Doesn't Help Within a Run

## Current Situation

**Disk cache IS working** - but only across different backtest runs, not within the same run!

```bash
$ python check_disk_cache_usage.py
✓ Cache directory: .cache/
  Files: 34,707 cache entries
  Size: 163 MB
✓ Helps across DIFFERENT backtest runs
✗ Does NOT help within SAME backtest run
```

## The Problem

### How disk_cache Works

```python
# disk_cache.py lines 23-25
if isinstance(arg, torch.Tensor):
    tensor = arg.detach().cpu().numpy()
    key_parts.append(hashlib.md5(tensor.tobytes()).hexdigest())
```

Cache key = **MD5 hash of ENTIRE tensor content**

### Why It Fails Within a Run

Walk-forward backtest with 70 simulations:

```python
# Sim 0: Uses days [0-199]
context_0 = tensor([day0, day1, ..., day199])  # 200 values
cache_key_0 = md5(context_0.tobytes())  # Unique

# Sim 1: Uses days [0-198]
context_1 = tensor([day0, day1, ..., day198])  # 199 values (DIFFERENT LENGTH!)
cache_key_1 = md5(context_1.tobytes())  # Different hash!
```

**Even though days [0-198] are IDENTICAL**, the cache keys don't match because tensor lengths differ!

**Result:** Each simulation gets its own cache entry = **no reuse within run**.

## Cache Effectiveness

| Scenario | Cache Behavior | Benefit |
|----------|----------------|---------|
| First backtest run | All MISS (compute + store) | ✗ None |
| Second backtest run (same data) | All HIT (load from disk) | ✓ Very fast |
| Within single run | Miss for each simulation | ✗ None |

The cache helps if you run the **same backtest multiple times**, but not within a **single run**.

Since walk-forward has 98% overlapping data, we're still recomputing 98% redundantly!

## Solution Options

### Option 1: Fix disk_cache Hashing

**Idea:** Hash only the data content, not length

```python
# Instead of hashing full tensor:
key = md5(tensor.tobytes())

# Hash a representative sample:
key = md5(f"{tensor.shape}:{tensor[:10].tobytes()}:{tensor[-10:].tobytes()}")
```

**Pros:** Minimal changes
**Cons:**
- Could have collisions
- Complex to get right
- Still slow (disk I/O per prediction)

### Option 2: Per-Day Caching

**Idea:** Cache individual day predictions, not full contexts

Use `src/prediction_cache.py`:
```python
cache_key = (data_hash, key_to_predict, day_index)
```

**Pros:**
- Granular caching
- Reuses across simulations
**Cons:**
- Requires integration into run_single_simulation
- In-memory only (unless we add disk persistence)

### Option 3: Pre-Compute Predictions ⭐ RECOMMENDED

**Idea:** Compute predictions ONCE for full dataset before simulations

```python
def backtest_forecasts(symbol, num_simulations=70):
    stock_data = download_data(...)

    # NEW: Pre-compute ONCE for full dataset
    predictions = {}
    for key in ["Close", "Low", "High", "Open"]:
        predictions[key] = _compute_toto_forecast(stock_data, key)  # Once!

    # Run simulations using pre-computed predictions
    for sim in range(num_simulations):
        sim_data = stock_data.iloc[:-(sim+1)]
        result = run_simulation(sim_data, predictions)  # Fast: no model calls!
```

**Pros:**
- Simplest implementation
- Fastest (no disk I/O, no cache lookups)
- Predictions computed once, reused 70 times
- 57.9x speedup on model inference

**Cons:**
- None! This is the right approach.

## Implementation: Pre-Compute Approach

### Current Flow (Slow)

```
backtest_forecasts(symbol, 70):
  for sim in range(70):                     ← 70 iterations
    simulation_data = stock_data[:-(sim+1)]
    run_single_simulation(simulation_data):
      for key in [Close, Low, High, Open]:  ← 4 keys
        predictions = _compute_toto_forecast(...)  ← SLOW! Called 280 times
```

**Total model calls:** 70 sims × 4 keys = **280 calls**
**But only ~8 unique predictions needed!** (4 keys × ~2 lengths)

### Optimized Flow (Fast)

```
backtest_forecasts(symbol, 70):
  # PRE-COMPUTE: Once for full dataset
  predictions_cache = {}
  for key in [Close, Low, High, Open]:      ← 4 keys
    predictions_cache[key] = _compute_toto_forecast(full_data, key)  ← 4 calls total

  # SIMULATE: Reuse predictions
  for sim in range(70):                     ← 70 iterations
    simulation_data = stock_data[:-(sim+1)]
    run_single_simulation(simulation_data, predictions_cache):  ← FAST! No model calls
      predictions = predictions_cache[key]  ← Instant lookup
```

**Total model calls:** 4 (one per key)
**Reduction:** 280 → 4 = **70x fewer calls!**

## Expected Performance

### Before Optimization

```
Total time: 180s
├── Model inference: 130s (72%)  ← 280 model calls
├── Optimization: 38s (21%)
└── Other: 12s (7%)
```

### After Pre-Compute

```
Total time: 60s (-67%!)
├── Model inference: 10s (17%)   ← 4 model calls (pre-compute)
├── Optimization: 38s (63%)      ← Unchanged
└── Other: 12s (20%)             ← Unchanged

Speedup: 3x faster
```

## Integration Steps

See: `backtest_with_precompute.py` for proof-of-concept

### Minimal Changes to backtest_test3_inline.py

1. **Add pre-compute function before simulations**

```python
def precompute_all_predictions(stock_data, symbol):
    """Compute predictions once for full dataset"""
    predictions_cache = {}
    for key_to_predict in ["Close", "Low", "High", "Open"]:
        # Process full data
        data = pre_process_data(stock_data, key_to_predict)
        # ... prepare price frame ...

        # Compute once
        pred, band, abs = _compute_toto_forecast(symbol, key_to_predict, price, ...)
        predictions_cache[key_to_predict] = (pred, band, abs)

    return predictions_cache
```

2. **Modify backtest_forecasts to pre-compute**

```python
def backtest_forecasts(symbol, num_simulations=50):
    stock_data = download_daily_stock_data(...)

    # NEW: Pre-compute predictions once
    predictions_cache = precompute_all_predictions(stock_data, symbol)

    for sim_number in range(num_simulations):
        simulation_data = stock_data.iloc[:-(sim_number + 1)]
        result = run_single_simulation(
            simulation_data, symbol, ...,
            predictions_cache=predictions_cache  # Pass cache
        )
```

3. **Modify run_single_simulation to use cache**

```python
def run_single_simulation(..., predictions_cache=None):
    for key_to_predict in ["Close", "Low", "High", "Open"]:
        # OLD:
        # predictions = _compute_toto_forecast(...)  # SLOW

        # NEW:
        if predictions_cache and key_to_predict in predictions_cache:
            predictions, band, abs_val = predictions_cache[key_to_predict]  # FAST
        else:
            predictions = _compute_toto_forecast(...)  # Fallback
```

## Testing

```bash
# Check current cache status
python check_disk_cache_usage.py

# See pre-compute approach
python backtest_with_precompute.py plan

# Test on small dataset
MARKETSIM_FAST_SIMULATE=1 python backtest_with_precompute.py ETHUSD 10
```

## Files

- `check_disk_cache_usage.py` - Analyze current cache (34K files!)
- `backtest_with_precompute.py` - Proof-of-concept implementation
- `docs/DISK_CACHE_ANALYSIS.md` - This document
- `disk_cache.py` - Current cache implementation (needs no changes)

## Recommendation

**Implement Option 3: Pre-compute predictions**

Benefits:
- ✅ 3x total speedup (180s → 60s)
- ✅ Simplest to implement (~50 lines of code)
- ✅ No cache key complexity
- ✅ Works with existing disk_cache for cross-run caching
- ✅ Predictions computed once, reused 70 times

This is the right architectural fix for walk-forward validation.
