# Backtest Inference Speedup Investigation

## Summary

Investigated GPU utilization issues in `backtest_test3_inline.py` and tested two optimization approaches.

## Baseline Performance

**Original code** (`backtest_test3_inline.py.pre_speedup`):
- **Total time (10 sims)**: 67.29 seconds
- **Per simulation**: 6.73 seconds
- **Forecast success**: 100% (all Toto forecasts working)
- **MaxDiffAlwaysOn Sharpe**: 32.46

## Optimization Attempts

### Attempt 1: Batched Predictions (FAILED)

**Approach**: Change from 7 sequential 1-step predictions to 1 batched 7-step prediction

**Implementation**:
```python
# Original (7 GPU calls)
for pred_idx in range(1, 8):
    context = price_frame[:-pred_idx]
    forecast = cached_predict(context, prediction_length=1)  # 7 calls

# Optimized attempt (1 GPU call)
context = price_frame[:-7]
forecast = cached_predict(context, prediction_length=7)  # 1 call
```

**Results**:
- **Total time**: 84.32 seconds (**25% SLOWER!**)
- **Forecast failures**: ~90% failed with "list index out of range"
- **Root cause**: Semantic mismatch
  - Original: 7 separate 1-step predictions with increasing context (walk-forward validation)
  - Optimized: 1 multi-step 7-horizon prediction (multi-step forecasting)
  - **These are fundamentally different approaches!**

**Status**: ❌ **REJECTED** - Breaks functionality and is slower

### Attempt 2: Async GPU Transfers (TESTING)

**Approach**: Add `non_blocking=True` to GPU transfers only, keep same prediction logic

**Implementation**:
```python
context = torch.tensor(current_context["y"].values, dtype=torch.float32)
if torch.cuda.is_available() and context.device.type == 'cpu':
    context = context.to('cuda', non_blocking=True)  # ← Added this
```

**Expected speedup**: 1.2-1.5x (modest but safe)

**Status**: 🔄 **TESTING** - Results pending

## Technical Analysis

### Why Batched Predictions Failed

The original code performs **walk-forward validation**:
1. Use first 93 rows → predict row 94 (1 step ahead)
2. Use first 94 rows → predict row 95 (1 step ahead)
3. ...
4. Use first 99 rows → predict row 100 (1 step ahead)

The batched approach attempted **multi-step forecasting**:
1. Use first 93 rows → predict rows 94-100 (7 steps ahead)

These are semantically different:
- **Walk-forward**: Each prediction has maximum available context
- **Multi-step**: All predictions from same (earliest) context
- **Result**: Different forecast quality, different MAE, different PnL

### Why It Was Slower

1. **Model inefficiency**: Toto model not optimized for multi-horizon predictions
2. **Return structure mismatch**: Expected `forecast[0..6]`, got different structure
3. **Error overhead**: Failures triggered expensive Kronos fallback

## Recommendations

### Immediate: Apply Async-Only Optimization

- ✅ Safe: No semantic changes
- ✅ Simple: Single line addition
- ✅ Expected: 1.2-1.5x speedup
- ⚠️ Modest: Won't achieve 5-7x target

### Future: Alternative Approaches

If more speedup needed, consider:

1. **Batch across symbols**: Process multiple symbols in parallel on GPU
2. **Reduce num_samples**: Test if fewer Monte Carlo samples maintain quality
3. **Model quantization**: Use int8/fp16 (requires MAE validation)
4. **Compiled model caching**: Improve torch.compile reuse across calls
5. **KV cache optimization**: If Toto supports it, reuse cached computations

## Files

- `backtest_test3_inline.py.pre_speedup` - Original backup
- `optimized_compute_toto_forecast.py` - Batched version (broken)
- `optimized_async_only.py` - Async-only version (testing)
- `baseline_test_output.log` - Baseline timing results
- `speedup_test_output.log` - Batched optimization test (failed)
- `async_test_output.log` - Async-only test (pending)

## Conclusion

Batched predictions fundamentally changed the forecasting semantics from walk-forward validation to multi-step forecasting. This not only broke functionality but was actually slower due to model inefficiency and error overhead.

The async-only optimization is a safer, more conservative approach that should provide modest speedup without changing forecast quality.
