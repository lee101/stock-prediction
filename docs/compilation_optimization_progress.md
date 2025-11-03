# Torch Compilation Optimization - Progress Report

## Summary

We've completed investigation and initial optimization attempts for improving torch.compile performance for the Toto and Kronos models. This document summarizes findings, work completed, and next steps.

## Work Completed

### 1. Root Cause Analysis ✓

**Identified Issues:**

1. **CUDAGraphs Skipping** (`toto/toto/model/util.py`):
   - KVCache uses Python list `_current_idx` which gets mutated during inference
   - In-place tensor updates for `_keys` and `_values`
   - Result: "skipping cudagraphs due to mutated inputs" warnings

2. **Recompilation Limit Exceeded** (`toto/toto/model/attention.py:105`):
   - Function `positional_embedding` hits 8 recompiles
   - Caused by changing values in `kv_cache._current_idx[7]`
   - Each unique sequence length triggers new compilation

3. **Explicit Graph Breaks** (`toto/toto/model/util.py:192-193`):
   - Manual `torch._dynamo.graph_break()` calls
   - Prevents end-to-end optimization

### 2. Created Comprehensive Test Suite ✓

**Location:** `toto/toto/test/integration/test_compilation_optimization.py`

**Test Coverage:**
- Single-step prediction equivalence (compiled vs non-compiled)
- Multi-step autoregressive equivalence
- KV cache state equivalence
- Performance benchmarking (with warmup)
- Memory overhead measurement
- Different compilation modes (default, reduce-overhead, max-autotune)
- Dynamic shape handling

**Status:** ✓ All tests passing on current code

### 3. Created Optimization Implementation

**Location:** `toto/toto/model/util_optimized.py`

**Key Changes:**
1. Replaced `_current_idx: List[int]` with `_current_idx: torch.Tensor`
2. Removed explicit graph breaks
3. Updated index operations to use tensors

**Current Limitation:**
- Still uses `.item()` to convert tensor→int for slicing
- This causes graph breaks and prevents full optimization

### 4. Created Experiment Framework ✓

**Location:** `toto/experiments/test_kvcache_optimization.py`

**Capabilities:**
- Prediction equivalence testing
- Performance benchmarking (warmup + timing)
- Memory profiling
- Configurable compilation modes
- Side-by-side comparison of original vs optimized

## Key Findings

### Issue #1: Mutated State in KVCache

**Original Code:**
```python
class KVCache:
    _current_idx: List[int] = field(init=False, repr=False)

    def append(self, layer_idx: int, kv: KV):
        start_idx = self._current_idx[cache_idx]
        end_idx = start_idx + keys.shape[1]
        self._keys[cache_idx, :, start_idx:end_idx, :, :] = keys  # In-place update
        self._current_idx[cache_idx] = int(end_idx)  # Python list mutation
```

**Problems:**
1. Python list doesn't play well with torch.compile
2. In-place tensor updates mark tensors as mutated
3. Triggers recompilation on every unique index value

**First Optimization Attempt:**
```python
class KVCacheOptimized:
    _current_idx: torch.Tensor = field(init=False, repr=False)  # Tensor instead of list

    def append(self, layer_idx: int, kv: KV):
        start_idx = self._current_idx[cache_idx].item()  # Still needs .item()
        end_idx = start_idx + keys.shape[1]
        self._keys[cache_idx, :, start_idx:end_idx, :, :] = keys
        self._current_idx[cache_idx] = end_idx
```

**Remaining Issue:**
- `.item()` causes graph breaks
- Tensor indexing with dynamic values still problematic

### Issue #2: Dynamic Slicing with Mutable Indices

The core challenge is that we need to:
1. Maintain a growing cache (dynamic length)
2. Slice tensors with runtime-determined indices
3. Update indices during forward pass

This is fundamentally difficult for static graph compilation.

## Proposed Solutions

### Solution A: Functional KVCache (No Mutations)

Instead of mutating cache in-place, return new cache state:

```python
def append(cache, layer_idx, kv):
    # Return new cache instead of mutating
    new_cache = cache.copy()
    new_cache.update(layer_idx, kv)
    return new_cache
```

**Pros:**
- No mutations → better for compilation
- Pure functions easier to optimize

**Cons:**
- Memory overhead from copying
- API change required
- May be slower without optimizations

### Solution B: Fixed-Size Pre-Allocation with Masking

Use full-size cache with attention masking:

```python
class KVCacheFixed:
    def __init__(self, max_seq_len):
        # Allocate full size upfront
        self._keys = torch.zeros(..., max_seq_len, ...)
        self._valid_mask = torch.zeros(max_seq_len, dtype=torch.bool)

    def append(self, kv):
        # No dynamic slicing needed
        self._keys[..., self.current_pos, :] = kv
        self._valid_mask[self.current_pos] = True
        self.current_pos += 1  # Still dynamic, but simpler
```

**Pros:**
- No dynamic slicing of cache tensors
- Index tracking simpler
- Better for compilation

**Cons:**
- Always uses max memory
- Still has mutable counter

### Solution C: Mark Dynamic Dimensions Explicitly

Keep current design but annotate properly:

```python
import torch._dynamo as dynamo

class KVCache:
    def __init__(self, ...):
        # ... initialization ...
        dynamo.mark_dynamic(self._current_idx, 0)  # Mark as dynamic

    def __getitem__(self, layer_idx):
        # Tell compiler this dimension varies
        end_idx = self._current_idx[cache_idx]
        keys = dynamo.maybe_mark_dynamic(
            self._keys[cache_idx, :, :end_idx, :, :],
            2  # seq_len dimension
        )
        return keys, values
```

**Pros:**
- Minimal code changes
- Tells compiler what to expect
- May reduce recompilations

**Cons:**
- Still has mutations
- May not enable CUDAGraphs
- Performance gain uncertain

### Solution D: Separate Compilation Units

Exclude KVCache operations from compilation:

```python
@torch.compiler.disable
def kv_cache_append(cache, layer_idx, kv):
    # This won't be compiled
    cache.append(layer_idx, kv)

def compiled_model_forward(...):
    # Main model is compiled
    out = model(x)
    # Cache updates not compiled
    kv_cache_append(cache, 0, (k, v))
    return out
```

**Pros:**
- Isolates problematic code
- Rest of model can still benefit
- Simple to implement

**Cons:**
- Doesn't fix the underlying issue
- Loses potential speedup from cache ops
- Still has graph breaks at boundaries

## Recommended Approach

**Phase 1: Quick Wins (Immediate)**
1. Use `torch.compiler.disable` on KVCache methods
2. Increase recompile limits: `torch._dynamo.config.cache_size_limit = 256`
3. Test with `mode="reduce-overhead"` instead of `max-autotune`
4. Add dynamic shape annotations where appropriate

**Phase 2: Deeper Optimization (Week 1-2)**
1. Implement Solution B (fixed-size with masking) as experiment
2. Benchmark against current implementation
3. If faster, gradually roll out to codebase

**Phase 3: Architectural Changes (Week 3-4)**
1. Evaluate functional KVCache design (Solution A)
2. Consider redesigning attention mechanism for better compilation
3. Explore alternative caching strategies (e.g., paged attention)

## Testing Strategy

For each optimization attempt:

1. **Correctness** (REQUIRED):
   ```bash
   pytest toto/test/integration/test_compilation_optimization.py::TestCompilationEquivalence -v
   ```
   Must pass with MAE < 1e-6

2. **Performance** (MEASURE):
   ```bash
   python toto/experiments/test_kvcache_optimization.py
   ```
   Track: inference time, memory usage, compilation time

3. **Integration** (VALIDATE):
   ```bash
   python backtest_test3_inline.py
   ```
   Run actual backtest with TOTO_COMPILE=1

## Metrics to Track

| Metric | Baseline | Target | Current |
|--------|----------|--------|---------|
| CUDAGraphs enabled | No | Yes | No |
| Recompilations per model | ~8 | <3 | ~8 |
| Inference speedup | 1.0x | 1.3-1.5x | TBD |
| MAE vs non-compiled | 0 | <1e-6 | <1e-6 ✓ |
| Peak memory overhead | - | <30% | TBD |

## Next Steps

1. **Immediate (Today)**:
   - ✓ Document findings
   - ✓ Create test infrastructure
   - ✓ Analyze root causes
   - [ ] Implement Quick Win #1 (compiler.disable)
   - [ ] Run full benchmark suite

2. **Short-term (This Week)**:
   - [ ] Implement fixed-size cache experiment
   - [ ] Benchmark all solutions
   - [ ] Pick best approach
   - [ ] Apply to Kronos model as well

3. **Medium-term (Next Week)**:
   - [ ] Validate in production backtest
   - [ ] Document best practices
   - [ ] Update architecture docs
   - [ ] Train team on compilation patterns

## References

- Analysis doc: `docs/compilation_optimization_analysis.md`
- Test suite: `toto/toto/test/integration/test_compilation_optimization.py`
- Experiment script: `toto/experiments/test_kvcache_optimization.py`
- Optimized implementation: `toto/toto/model/util_optimized.py`

## Open Questions

1. Can we use `torch.jit.script` for KVCache instead of torch.compile?
2. Would switching to Flash Attention 2 help with compilation?
3. Is there a way to mark tensors as "safe to mutate" for CUDAGraphs?
4. Should we consider different strategies for training vs inference?

## Conclusion

We've successfully identified the root causes of compilation issues and created a comprehensive testing framework. The main challenge is handling mutable state (KVCache) in a way that's compatible with torch.compile and CUDAGraphs.

The path forward involves either:
- **Conservative**: Exclude problematic code from compilation (quick, safe)
- **Moderate**: Redesign cache to use fixed sizes (more work, likely faster)
- **Aggressive**: Move to functional design (big change, uncertain benefit)

Recommendation: Start with conservative approach for immediate wins, while prototyping the moderate approach for longer-term gains.
