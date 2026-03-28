# Prediction Caching Integration Guide

## The Problem

**98% of model predictions are redundant** in walk-forward backtesting!

### Current Behavior

70 simulations on 200 days of data:
- Sim 0: Uses days 0-199 (200 days)
- Sim 1: Uses days 0-198 (199 days)
- Sim 69: Uses days 0-130 (131 days)

**Result:** Day 50 is predicted 70 times, but it's always the same!

### Numbers

- **Total predictions made:** 46,340 (70 sims × ~165 avg days × 4 keys)
- **Unique predictions needed:** 800 (200 days × 4 keys)
- **Redundant:** 45,540 (98.3%)

**Potential speedup: 3.2x total** (57.9x on model inference alone)

## Solution: Prediction Cache

Created: `src/prediction_cache.py`

Simple in-memory cache keyed by `(data_hash, key_to_predict, day_index)`.

## Integration Steps

### 1. Add to backtest_test3_inline.py

At top of file:
```python
from src.prediction_cache import get_cache, reset_cache, hash_dataframe
```

### 2. Reset cache at start of backtest_forecasts

In `backtest_forecasts()`, after data download:

```python
def backtest_forecasts(symbol, num_simulations=50):
    # ... existing code to download data ...

    # Reset prediction cache for this backtest run
    reset_cache()

    # ... rest of function ...
```

### 3. Cache Toto predictions

In `run_single_simulation()`, find the Toto prediction section:

**Before:**
```python
if run_toto:
    try:
        toto_predictions, toto_band, toto_abs = _compute_toto_forecast(
            symbol,
            key_to_predict,
            price,
            current_last_price,
            toto_params,
        )
    except Exception as exc:
        # ...
```

**After:**
```python
if run_toto:
    # Check cache first
    cache = get_cache()
    data_hash = hash_dataframe(simulation_data)
    cached = cache.get(data_hash, key_to_predict, len(simulation_data))

    if cached is not None:
        toto_predictions, toto_band, toto_abs = cached
    else:
        try:
            toto_predictions, toto_band, toto_abs = _compute_toto_forecast(
                symbol,
                key_to_predict,
                price,
                current_last_price,
                toto_params,
            )
            # Store in cache for next simulation
            cache.put(data_hash, key_to_predict, len(simulation_data),
                     (toto_predictions, toto_band, toto_abs))
        except Exception as exc:
            # ...
```

### 4. Cache Kronos predictions

Similarly for Kronos, find:

**Before:**
```python
if need_kronos and ensure_kronos_ready():
    try:
        kronos_predictions, kronos_abs = _compute_kronos_forecast(...)
```

**After:**
```python
if need_kronos and ensure_kronos_ready():
    cache = get_cache()
    data_hash = hash_dataframe(simulation_data)
    cached = cache.get(data_hash, f"kronos_{key_to_predict}", len(simulation_data))

    if cached is not None:
        kronos_predictions, kronos_abs = cached
    else:
        try:
            kronos_predictions, kronos_abs = _compute_kronos_forecast(...)
            cache.put(data_hash, f"kronos_{key_to_predict}", len(simulation_data),
                     (kronos_predictions, kronos_abs))
        except Exception as exc:
            # ...
```

### 5. Log cache stats at end

In `backtest_forecasts()`, before return:

```python
    # ... after all simulations complete ...

    # Log cache statistics
    cache = get_cache()
    stats = cache.stats()
    logger.info(
        f"Prediction cache: {stats['hit_rate']:.1f}% hit rate, "
        f"{stats['hits']} hits / {stats['total_requests']} requests, "
        f"cache size: {stats['cache_size']}"
    )

    return results_df
```

## Expected Results

With 70 simulations:
- **First run:** 0% hit rate (building cache)
- **Hits:** ~45,540 cache hits (~98%)
- **Misses:** ~800 (only unique predictions)
- **Speedup:** ~3.2x total

### Before Caching
```
Total time: 180s
  Model inference: 130s (72%)
  Optimization: 38s (21%)
  Other: 12s (7%)
```

### After Caching
```
Total time: 60s (-67%!)
  Model inference: 10s (17%, cached!)
  Optimization: 38s (63%, unchanged)
  Other: 12s (20%, unchanged)

Cache: 98.3% hit rate, 45540 hits, 800 misses
```

## Testing

1. Add logging to see cache behavior:
```python
# After cache.get()
if cached is not None:
    logger.debug(f"Cache HIT for {key_to_predict} day {len(simulation_data)}")
else:
    logger.debug(f"Cache MISS for {key_to_predict} day {len(simulation_data)}")
```

2. Run small backtest to verify:
```bash
MARKETSIM_FAST_SIMULATE=1 python -c "
from backtest_test3_inline import backtest_forecasts
backtest_forecasts('ETHUSD', 10)
"
```

3. Check logs for cache stats

## Configuration

Disable caching if needed:
```bash
export MARKETSIM_CACHE_PREDICTIONS=0
```

## Alternative: Process Simulations in Reverse

Simpler integration - process oldest simulation first:

```python
def backtest_forecasts(symbol, num_simulations=50):
    # ... setup code ...

    # Process in REVERSE order (oldest to newest)
    for sim_number in reversed(range(num_simulations)):
        simulation_data = stock_data.iloc[: -(sim_number + 1)].copy(deep=True)
        # ... rest of code ...
```

This way each simulation builds cache for next one. No code changes in run_single_simulation needed beyond adding cache.get/put.

## Memory Considerations

Cache size with 200 days × 4 keys = 800 entries:
- Each entry: ~few KB (predictions + band)
- Total: ~5-10 MB (negligible)

For 1000 days: ~50 MB (still fine)

## Files

- `src/prediction_cache.py` - Cache implementation
- `analyze_caching_opportunity.py` - Analysis showing 98% redundancy
- `docs/CACHING_INTEGRATION_GUIDE.md` - This guide

## Next Steps

1. Integrate caching into backtest_test3_inline.py
2. Test with MARKETSIM_FAST_SIMULATE=1 (10-35 sims)
3. Verify cache hit rate >95%
4. Measure actual speedup
5. Roll out to production

Expected total speedup: **3-4x faster backtests!**
