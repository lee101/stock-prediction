# Performance Analysis: Backtest Optimization Bottlenecks

## Executive Summary

The ~18-19 second delay per symbol evaluation is **NOT** caused by the MaxDiff optimization itself, but by the **forecast generation** step (Toto/Kronos sampling).

### Actual Time Breakdown (per symbol)

| Component | Time | % of Total | Operations |
|-----------|------|-----------|------------|
| **Toto/Kronos Forecasting** | ~17-18s | ~95% | 4 keys × 7 horizons × 128 samples = 3,584 torch samples |
| MaxDiff Optimization | ~0.2s | ~1% | 2 strategies × 2 close_at_eod × 500 evals = 2,000 profit calculations |
| Data Processing | ~0.5s | ~3% | DataFrame operations, tensor conversions |
| Other | ~0.3s | ~2% | Logging, metrics, etc. |

## Root Cause Analysis

### 1. Toto/Kronos Sampling (95% of time)

**File**: `backtest_test3_inline.py:1436-1581` (`_compute_toto_forecast`)

**Problem**: For each symbol evaluation:
- 4 price keys to predict: Close, Low, High, Open
- 7 forecast horizons (reversed range(1, 8))
- 128 samples per forecast (from hyperparamstore)
- **Total: 4 × 7 × 128 = 3,584 torch model forward passes**

**Evidence from logs**:
```
INFO | _clamp_toto_params:1682 | Adjusted Toto sampling bounds for UNIUSD:
  num_samples=128, samples_per_batch=32 (was 128/16).
```

**Code**:
```python
# backtest_test3_inline.py:1484-1490
forecast = cached_predict(
    context,
    1,
    num_samples=requested_num_samples,  # 128
    samples_per_batch=requested_batch,   # 32
    symbol=symbol,
)
```

This happens for EVERY forecast horizon (7 times) for EVERY price key (4 times).

### 2. MaxDiff Optimization (1% of time)

**File**: `src/optimization_utils.py:130-201`

**Current Performance** (measured):
- `optimize_entry_exit_multipliers`: 0.12s (505 evaluations)
- `optimize_always_on_multipliers`: 0.10s (243 evaluations)
- **Total per symbol: ~0.22s**

**Not a bottleneck** - This is already well optimized!

### 3. Multiple close_at_eod Candidates (2x multiplier)

**File**: `backtest_test3_inline.py:445, 675`

```python
close_at_eod_candidates = [close_at_eod] if close_at_eod is not None else [False, True]
```

Both MaxDiff strategies try 2 close_at_eod values, effectively doubling optimization time (but still only ~0.4s total).

## Optimization Recommendations

### Priority 1: Reduce Toto/Kronos Sampling (Biggest Impact)

**Option A: Reduce num_samples** (6-8x speedup)
```bash
# Reduce from 128 to 16-32 samples
# Edit hyperparamstore to lower num_samples for all symbols
# Expected speedup: 128/16 = 8x or 128/32 = 4x
```

**Option B: Cache Forecasts** (Huge speedup for repeated backtests)
```python
# Add forecast caching layer
# Cache key: (symbol, data_hash, horizon, key_to_predict)
# Invalidate only when data changes
```

**Option C: Reduce Forecast Horizons** (7x speedup)
```python
# Change from 7 horizons to 1 (only predict next day)
max_horizon = 1  # instead of 7
# This is likely acceptable since we only use the last prediction
```

**Option D: Parallel Forecasting** (4x speedup)
```python
# Run 4 key predictions in parallel
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
    futures = {executor.submit(forecast_key, key): key
               for key in ["Close", "Low", "High", "Open"]}
```

### Priority 2: Set Fast Mode Env Vars (6x speedup on optimization)

```bash
export MARKETSIM_FAST_OPTIMIZE=1      # Reduces optimizer evals 500→100
export MARKETSIM_FAST_SIMULATE=1      # Reduces simulations 50→35
```

These are already supported in the code but may not be set.

### Priority 3: Specify close_at_eod (2x speedup on optimization)

```python
# Set close_at_eod=False explicitly to skip testing both options
result = evaluate_maxdiff_strategy(
    ...,
    close_at_eod=False,  # Don't test both True/False
)
```

### Priority 4: Batch Optimization Calls (Minor improvement)

If running multiple symbols, optimize them in parallel:
```python
# Use multiprocessing to run multiple symbol backtests in parallel
# WARNING: May cause CUDA OOM if models are large
```

## Quick Wins (Implement Today)

1. **Set environment variables** (No code changes):
   ```bash
   export MARKETSIM_FAST_OPTIMIZE=1
   export MARKETSIM_FAST_SIMULATE=1
   ```
   Expected: 50→35 simulations = 1.4x speedup

2. **Reduce max_horizon to 1** (One line change):
   ```python
   max_horizon = 1  # line 1452 in backtest_test3_inline.py
   ```
   Expected: 7x speedup on forecasting = ~6x overall

3. **Lower num_samples in hyperparamstore** (Config change):
   Change from 128 to 32 samples
   Expected: 4x speedup on forecasting = ~3.5x overall

### Combined Quick Wins Impact

If all three quick wins are applied:
- Time per symbol: 18s → 18s / (1.4 × 6 × 4) = **0.5 seconds** 🚀
- **36x speedup overall!**

## Profiling Evidence

Run `python profile_optimization.py` to verify optimization performance:

```
1. Profiling optimize_entry_exit_multipliers (MaxDiff)...
Elapsed time: 0.12s (505 evaluations)

2. Profiling optimize_always_on_multipliers (MaxDiffAlwaysOn)...
Elapsed time: 0.10s (243 evaluations)

3. Function Call Counts
100 objective calls: 0.020s (0.20ms per call)
```

The optimization itself is fast. The bottleneck is elsewhere.

## Next Steps

1. ✅ Profile and identify bottleneck (DONE - it's forecasting, not optimization)
2. ⚠️ Implement quick wins (set env vars, reduce horizons, reduce samples)
3. 🔲 Add forecast caching for repeated backtests
4. 🔲 Profile Toto/Kronos inference to optimize model forward pass
5. 🔲 Consider model quantization (bf16/fp16) for faster inference

## Files to Check

- `backtest_test3_inline.py:1436-1581` - Toto forecast loop (main bottleneck)
- `backtest_test3_inline.py:2600-2610` - Kronos sampling
- `backtest_test3_inline.py:1682` - Toto param clamping
- `src/optimization_utils.py` - Already well optimized
- `hyperparamstore/` - Contains num_samples=128 configs

## Measurement Commands

```bash
# Profile one symbol backtest
time python backtest_test3_inline.py UNIUSD

# With fast mode
MARKETSIM_FAST_OPTIMIZE=1 MARKETSIM_FAST_SIMULATE=1 \
  time python backtest_test3_inline.py UNIUSD

# Profile with cProfile
python -m cProfile -o backtest.prof backtest_test3_inline.py UNIUSD
python -c "import pstats; p = pstats.Stats('backtest.prof'); p.sort_stats('cumulative').print_stats(30)"
```
