# Torch Compilation Optimization - Summary

## Executive Summary

We've completed a comprehensive investigation into torch.compile optimization issues for the Toto and Kronos models. We've identified the root causes, created extensive testing infrastructure, and developed several solution paths.

## What We Found

### Root Causes of Compilation Issues

1. **KVCache State Mutations** (PRIMARY ISSUE)
   - Location: `toto/toto/model/util.py` KVCache class
   - Problem: Python list `_current_idx` mutated during inference
   - Impact: Prevents CUDAGraphs, causes recompilations
   - Evidence: "skipping cudagraphs due to mutated inputs" in logs

2. **Recompilation Limit Exceeded**
   - Location: `toto/toto/model/attention.py:105` (positional_embedding function)
   - Problem: Hits 8 recompiles due to changing cache indices
   - Impact: Unpredictable performance, additional compile overhead
   - Evidence: `torch._dynamo hit config.recompile_limit (8)` warnings

3. **Explicit Graph Breaks**
   - Location: `toto/toto/model/util.py:192-193`
   - Problem: Manual `torch._dynamo.graph_break()` calls
   - Impact: Prevents end-to-end optimization

## What We Built

### 1. Comprehensive Test Suite ✓
**File:** `toto/toto/test/integration/test_compilation_optimization.py`

- Tests prediction equivalence (compiled vs non-compiled)
- Validates MAE < 1e-6 across all test cases
- Benchmarks performance improvements
- Tests multiple compilation modes
- **Status:** All tests passing ✓

```bash
# Run tests:
cd toto && pytest toto/test/integration/test_compilation_optimization.py -v
```

### 2. Detailed Analysis Documentation ✓
**Files:**
- `docs/compilation_optimization_analysis.md` - Technical deep-dive
- `docs/compilation_optimization_progress.md` - Work completed + next steps
- `docs/COMPILATION_OPTIMIZATION_SUMMARY.md` - This file

### 3. Optimized KVCache Implementation ✓
**File:** `toto/toto/model/util_optimized.py`

- Replaces Python list with torch.Tensor for indices
- Removes explicit graph breaks
- **Limitation:** Still uses `.item()` which causes graph breaks

### 4. Experiment Framework ✓
**File:** `toto/experiments/test_kvcache_optimization.py`

- Side-by-side comparison of implementations
- Performance benchmarking with proper warmup
- Memory profiling
- Equivalence validation

```bash
# Run experiments:
cd toto && python experiments/test_kvcache_optimization.py
```

## Recommended Solutions

### Option 1: Quick Win (RECOMMENDED FOR IMMEDIATE USE) ⚡

**Approach:** Exclude KVCache from compilation using `@torch.compiler.disable`

**Implementation:**
```python
# In toto/toto/model/util.py

@torch.compiler.disable
def __getitem__(self, layer_idx: int) -> KV:
    # ... existing code ...

@torch.compiler.disable
def append(self, layer_idx: int, kv: KV):
    # ... existing code ...
```

**Pros:**
- Minimal code changes (2 decorators)
- No risk of breaking existing functionality
- Rest of model still benefits from compilation
- Can implement today

**Cons:**
- Doesn't fully optimize KV cache operations
- Still has some graph breaks at method boundaries

**Expected Impact:**
- Reduce recompilations from ~8 to ~2-3
- Stabilize compilation behavior
- 10-20% inference speedup (conservative estimate)

### Option 2: Fixed-Size Cache with Masking (RECOMMENDED FOR LONG-TERM)

**Approach:** Pre-allocate full cache, use masking instead of dynamic slicing

**Example:**
```python
class KVCacheFixed:
    def __init__(self, max_seq_len):
        # Allocate full size upfront
        self._keys = torch.zeros(..., max_seq_len, ...)
        self._current_pos = 0  # Simple counter, not indexing

    def append(self, kv):
        # No dynamic slicing needed
        self._keys[..., self.current_pos, :] = kv
        self.current_pos += 1
```

**Pros:**
- Eliminates dynamic slicing
- Better compilation compatibility
- Potentially enables CUDAGraphs
- 30-50% inference speedup (optimistic estimate)

**Cons:**
- Always uses max memory
- Requires more extensive testing
- API might need small changes
- 1-2 weeks to implement properly

### Option 3: Increase Recompile Limits (IMMEDIATE WORKAROUND)

**Approach:** Configure torch to allow more recompilations

**Implementation:**
```python
# In backtest_test3_inline.py or model loading code:
import torch._dynamo

torch._dynamo.config.cache_size_limit = 256
torch._dynamo.config.accumulated_cache_size_limit = 256
```

**Pros:**
- One-line fix
- No code changes to model
- Immediate relief from recompile errors

**Cons:**
- Doesn't fix root cause
- Increased memory usage
- Compilation time still high

## Immediate Action Items

### This Week:
1. ✅ **Done:** Complete analysis and documentation
2. ✅ **Done:** Create test suite
3. **TODO:** Implement Option 1 (Quick Win) in separate branch
4. **TODO:** Run full benchmark comparison
5. **TODO:** If successful, merge to main

### Next Week:
1. **TODO:** Start prototyping Option 2 (Fixed-Size Cache)
2. **TODO:** Apply same patterns to Kronos model
3. **TODO:** Run production backtests with optimizations
4. **TODO:** Measure end-to-end impact

### Commands to Run

```bash
# 1. Run baseline tests (current code)
cd /nvme0n1-disk/code/stock-prediction/toto
source ../.venv/bin/activate
pytest toto/test/integration/test_compilation_optimization.py::TestCompilationEquivalence -v

# 2. Run experiments
python experiments/test_kvcache_optimization.py

# 3. Test with actual model (after applying changes)
cd /nvme0n1-disk/code/stock-prediction
export TOTO_COMPILE=1
export TORCH_COMPILE_MODE=reduce-overhead
python backtest_test3_inline.py  # Your actual inference script
```

## Expected Outcomes

| Optimization | Compilation Time | Inference Speed | Memory | Risk | Effort |
|--------------|------------------|-----------------|--------|------|--------|
| Baseline (current) | High | 1.0x | 1.0x | - | - |
| Option 1 (compiler.disable) | Medium | 1.1-1.2x | 1.0x | **Low** | **1 day** |
| Option 2 (fixed-size cache) | Low | 1.3-1.5x | 1.2x | Medium | 1-2 weeks |
| Option 3 (config only) | High | 1.0x | 1.1x | **Very Low** | **1 hour** |

## Key Files Reference

```
toto/
├── toto/model/util.py                              # Original KVCache (needs optimization)
├── toto/model/util_optimized.py                    # Optimized version (experimental)
├── toto/test/integration/
│   ├── test_compilation_optimization.py            # Test suite ✓
│   └── test_kv_cache_compile.py                   # Existing tests ✓
├── experiments/
│   ├── test_kvcache_optimization.py               # Benchmark framework ✓
│   └── quick_win_kvcache.py                       # Quick win guide ✓
└── docs/
    ├── compilation_optimization_analysis.md        # Technical analysis ✓
    ├── compilation_optimization_progress.md        # Progress tracking ✓
    └── COMPILATION_OPTIMIZATION_SUMMARY.md         # This file ✓

backtest_test3_inline.py                            # Main inference script (uses Toto/Kronos)
external/kronos/                                     # Kronos model (similar issues)
```

## Success Criteria

A successful optimization will achieve:

1. **Correctness:** MAE < 1e-6 between compiled and non-compiled predictions ✓ (Achieved in tests)
2. **Performance:** 20-40% inference speedup
3. **Stability:** < 3 recompilations per model load
4. **CUDAGraphs:** Enabled (no "skipping cudagraphs" warnings)
5. **Memory:** < 30% overhead vs baseline

## Current Status

- ✅ Analysis complete
- ✅ Root causes identified
- ✅ Test infrastructure in place
- ✅ Multiple solution paths documented
- ⏳ Optimization implementation in progress
- ⏳ Performance validation pending
- ⏳ Production deployment pending

## Questions?

For detailed technical information, see:
- `docs/compilation_optimization_analysis.md`
- `docs/compilation_optimization_progress.md`

To run experiments:
```bash
cd toto && python experiments/test_kvcache_optimization.py
```

To run tests:
```bash
cd toto && pytest toto/test/integration/test_compilation_optimization.py -v
```

## Next Conversation

When we resume work on this:
1. Review benchmark results from experiments
2. Implement chosen solution (likely Option 1 first)
3. Validate with integration tests
4. Run production backtests
5. Measure impact and iterate

---

**Last Updated:** 2025-11-01
**Status:** Investigation Complete, Ready for Implementation
