# Torch Compilation Optimization Analysis

## Current Issues

### 1. CUDAGraphs Skipping Due to Mutated Inputs
**Location**: Throughout model execution
**Symptom**: `skipping cudagraphs due to mutated inputs (2 instances)` appears repeatedly

**Root Cause**: The `KVCache` class in `toto/toto/model/util.py` uses mutable state:
- `_current_idx: List[int]` (line 128) - Python list that gets mutated during inference
- `append()` method (line 225) - Updates `_current_idx[cache_idx] = int(end_idx)`
- `_keys` and `_values` tensors are updated in-place (lines 219-223)

**Impact**: CUDAGraphs provide significant performance improvements by recording and replaying GPU operations. Skipping them results in 20-40% slower inference.

### 2. Recompilation Limit Exceeded
**Location**: `positional_embedding` function in `toto/toto/model/attention.py:105`
**Symptom**:
```
W1101 01:56:35.074000 101341 torch/_dynamo/convert_frame.py:1358] [6/8] torch._dynamo hit config.recompile_limit (8)
last reason: 6/7: kv_cache._current_idx[7] == 0
```

**Root Cause**:
- The `_current_idx` list values change between inference calls
- Each unique value triggers a new compilation
- Default recompile limit is 8, which gets quickly exhausted

**Impact**:
- Additional compilation overhead on each unique sequence length
- Unpredictable performance after 8th recompilation
- Increased memory usage from multiple compiled versions

### 3. Explicit Graph Breaks
**Location**: `toto/toto/model/util.py:192-193`
**Code**:
```python
if _torch_graph_break is not None and _torch_compiler_is_compiling():
    _torch_graph_break()
```

**Impact**: Forces the compiler to split the computation graph, preventing end-to-end optimizations

## Proposed Solutions

### Solution 1: Tensor-Based Index Tracking
**Priority**: HIGH
**Complexity**: Medium

Replace Python list `_current_idx` with a tensor:
```python
self._current_idx = torch.zeros(time_layer_count, dtype=torch.int32, device=self.device)
```

Benefits:
- Tensors are better handled by torch.compile
- Can be marked as dynamic or static as needed
- More efficient on GPU

### Solution 2: Static Shape Annotations
**Priority**: HIGH
**Complexity**: Low

Use `torch._dynamo.mark_static` to mark certain dimensions:
```python
import torch._dynamo as dynamo
dynamo.mark_static(self._keys, 0)  # batch dimension
dynamo.mark_static(self._keys, 2)  # max_seq_len dimension
```

### Solution 3: Remove Explicit Graph Breaks
**Priority**: MEDIUM
**Complexity**: Low

The graph break in `KVCache.append()` may not be necessary. Test removing it:
- Originally added to prevent compilation issues
- Modern torch.compile may handle this better
- Can use `torch.compiler.disable` on specific functions if needed

### Solution 4: Increase Recompile Limits
**Priority**: LOW
**Complexity**: Low

Temporary workaround while fixing root causes:
```python
torch._dynamo.config.cache_size_limit = 256
torch._dynamo.config.accumulated_cache_size_limit = 256
```

### Solution 5: Mark Dynamic Dimensions Explicitly
**Priority**: HIGH
**Complexity**: Medium

Use `torch._dynamo.mark_dynamic` for sequence lengths that legitimately vary:
```python
dynamo.mark_dynamic(tensor, 1)  # Mark seq_len as dynamic
```

This tells the compiler to expect variations without recompiling.

### Solution 6: Compilation Mode Tuning
**Priority**: MEDIUM
**Complexity**: Low

Current: `mode=max-autotune` which is aggressive but may cause issues
Try: `mode=reduce-overhead` or `mode=default` for better stability

## Testing Strategy

### Integration Test Requirements
1. **Prediction Equivalence Test**
   - Run same inputs through compiled and non-compiled models
   - Compare outputs with strict tolerances (MAE < 1e-6)
   - Test across multiple sequence lengths and batch sizes

2. **Performance Regression Test**
   - Benchmark inference time with/without compilation
   - Track memory usage
   - Monitor compilation time

3. **Determinism Test**
   - Multiple runs with same seed should produce identical results
   - Both compiled and non-compiled versions

### Test Implementation
```python
def test_compilation_equivalence():
    # Load model without compilation
    model_no_compile = load_toto_pipeline(compile=False)

    # Load model with compilation
    model_compile = load_toto_pipeline(compile=True)

    # Test data
    test_inputs = generate_test_data(n_samples=10)

    for inputs in test_inputs:
        output_no_compile = model_no_compile(inputs)
        output_compile = model_compile(inputs)

        mae = torch.abs(output_no_compile - output_compile).mean()
        assert mae < 1e-6, f"MAE {mae} exceeds threshold"
```

## Experimentation Plan

### Phase 1: Diagnosis (Current)
- [x] Identify root causes of compilation issues
- [x] Document current behavior
- [ ] Create baseline performance metrics

### Phase 2: Quick Wins
1. Remove explicit graph break (test for side effects)
2. Try different compilation modes
3. Increase recompile limits temporarily

### Phase 3: Structural Fixes
1. Replace Python list with tensor for `_current_idx`
2. Add proper static/dynamic shape annotations
3. Test KV cache modifications thoroughly

### Phase 4: Validation
1. Run full integration test suite
2. Compare predictions against baseline
3. Measure performance improvements
4. Deploy to staging environment

## Expected Outcomes

### Performance Targets
- **Inference Speed**: 30-50% improvement with CUDAGraphs enabled
- **Compilation Time**: Reduced recompilations (< 3 per model)
- **Memory Usage**: Similar or better than current

### Accuracy Requirements
- **MAE**: < 1e-6 between compiled and non-compiled
- **Max Diff**: < 1e-5 for any single prediction
- **Determinism**: Bit-exact results across runs with same seed

## Monitoring

Track these metrics during experimentation:
1. Number of graph breaks
2. Recompilation count
3. CUDAGraph usage (enabled/disabled)
4. Peak GPU memory
5. Inference latency (p50, p95, p99)
6. Compilation overhead

## References

- PyTorch Compilation Troubleshooting: https://pytorch.org/docs/main/torch.compiler_troubleshooting.html
- CUDAGraphs Documentation: https://pytorch.org/docs/stable/notes/cuda.html#cuda-graphs
- Dynamic Shapes: https://pytorch.org/docs/stable/generated/torch._dynamo.mark_dynamic.html
