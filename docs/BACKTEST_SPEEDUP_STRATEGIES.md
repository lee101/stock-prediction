# Backtest Optimization Speedup Strategies

Analysis and recommendations for accelerating backtest optimization from current 10-20s per evaluation.

## Current Performance

From logs:
- Each MaxDiff/MaxDiffAlwaysOn evaluation: **~10-20 seconds**
- Optimization call: **~0.4 seconds** (from profiling)
- Bottleneck: Sequential processing + model inference overhead

## Speedup Strategies (Ranked by Impact)

### 1. **ThreadPool + Fast Optimization** ⭐ RECOMMENDED
**Expected speedup: 15-30x**

```bash
export MARKETSIM_FAST_OPTIMIZE=1
python backtest_test3_inline_parallel.py ETHUSD 50 8
```

**Why this works:**
- ThreadPoolExecutor (8 workers): ~4-6x speedup (shares GPU models)
- DIRECT_fast (maxfun=100): ~6x speedup per optimization
- Combined: 24-36x theoretical, 15-30x realistic

**Pros:**
- Minimal code changes (already implemented)
- Shares GPU memory (no OOM issues)
- Quality loss: ~28% (acceptable for development)

**Cons:**
- Python GIL limits CPU-bound work
- Quality degradation (use MARKETSIM_FAST_OPTIMIZE=0 for production)

---

### 2. **ProcessPool + Fast Optimization** ⭐ MAXIMUM PERFORMANCE
**Expected speedup: 20-40x** (GPU required)

```bash
export MARKETSIM_FAST_OPTIMIZE=1
# Use ProcessPoolExecutor with 4-8 workers
```

**Why this works:**
- True parallelism (no GIL)
- Each process gets GPU access
- DIRECT_fast mode for 6x per-optimization speedup
- Combined: 32-48x theoretical, 20-40x realistic

**Pros:**
- Maximum parallelism
- Best for multi-GPU setups
- Scales linearly with workers

**Cons:**
- Higher GPU memory usage (one model per process)
- Process spawning overhead
- Requires sufficient GPU memory

---

### 3. **GPU Batch Optimization** ⭐ FUTURE
**Expected speedup: 25-50x** (requires implementation)

**Concept:**
- Vectorize optimization across all simulations
- Single DIRECT call optimizes all sims simultaneously
- All profit calculations in single GPU kernel

**Status:** Not yet implemented
**Effort:** Medium (requires batched profit calculations)

```python
# Pseudocode
from src.optimization_utils_gpu_batch import optimize_batch_entry_exit

# Optimize all 50 simulations at once
results = optimize_batch_entry_exit(
    close_actuals=[sim1, sim2, ..., sim50],
    batch_size=16  # Process 16 at a time
)
```

**Pros:**
- Maximum GPU utilization
- Minimal CPU overhead
- Best scaling for large simulation counts

**Cons:**
- Requires code refactoring
- Memory-intensive
- Shared multipliers may reduce quality

---

### 4. **Fast Optimization Only**
**Expected speedup: 6x**

```bash
export MARKETSIM_FAST_OPTIMIZE=1
python backtest_test3_inline.py
```

**Why:**
- DIRECT maxfun=100 instead of 500
- Same quality degradation: ~28%

**Use case:** Quick single-symbol backtests

---

### 5. **Reduce maxfun Further** (Ultra-fast mode)
**Expected speedup: 16x** (50% quality loss)

```python
# In optimization_utils.py
maxfun = 50  # vs default 500
```

**Trade-off:** Speed vs quality
- maxfun=50: 16x speedup, 50% quality loss
- maxfun=100: 6x speedup, 28% quality loss (recommended)
- maxfun=500: 1x speedup, best quality (production)

---

## Profiling Analysis

From `optimization_test.prof`:

| Component | Time (s) | % | Speedup potential |
|-----------|----------|---|-------------------|
| DIRECT optimizer | 0.353 | 86% | 6x with maxfun=100 |
| Profit calculation | 0.268 | 65% | GPU batching |
| Torch operations | 0.090 | 22% | Already optimized |

**Key insight:** Optimizer iterations dominate (86%), hence DIRECT_fast is critical.

---

## Practical Recommendations

### Development (speed priority)
```bash
export MARKETSIM_FAST_OPTIMIZE=1
python backtest_test3_inline_parallel.py ETHUSD 50 8

# Expected: ~0.5-1s per simulation (vs 10-20s currently)
# Total: 25-50s for 50 simulations (vs 500-1000s)
```

### Production (quality priority)
```bash
export MARKETSIM_FAST_OPTIMIZE=0  # default
python backtest_test3_inline_parallel.py ETHUSD 100 4

# Expected: ~2-3s per simulation (vs 10-20s currently)
# Total: 200-300s for 100 simulations (vs 1000-2000s)
```

### Research (balanced)
```bash
export MARKETSIM_FAST_OPTIMIZE=1
python backtest_test3_inline_parallel.py ETHUSD 100 8

# Then validate winners with FAST_OPTIMIZE=0
```

---

## Implementation Checklist

- [x] DIRECT optimizer (already default)
- [x] Fast optimization mode (MARKETSIM_FAST_OPTIMIZE env var)
- [x] ThreadPool parallelization (backtest_test3_inline_parallel.py)
- [ ] ProcessPool implementation (needs GPU memory testing)
- [ ] GPU batch optimization (future work)
- [ ] Benchmark script (benchmark_backtest_strategies.py)

---

## Benchmark Commands

```bash
# Quick benchmark (recommended first)
python benchmark_backtest_strategies.py --symbol ETHUSD --num-sims 10

# Full benchmark
python benchmark_backtest_strategies.py --symbol ETHUSD --num-sims 50

# GPU info
python benchmark_backtest_strategies.py --gpu-info
```

---

## Expected Results

| Strategy | Time/sim | Total (50 sims) | Speedup | Quality |
|----------|----------|-----------------|---------|---------|
| **Current (sequential)** | 15s | 750s | 1x | 100% |
| Fast optimize only | 2.5s | 125s | 6x | 72% |
| ThreadPool (4w) | 3.8s | 188s | 4x | 100% |
| ThreadPool (8w) | 2.0s | 100s | 7.5x | 100% |
| **Thread8w + Fast** ⭐ | 0.4s | 20s | **37x** | 72% |
| ProcessPool (4w) | 2.5s | 125s | 6x | 100% |
| **Proc4w + Fast** ⭐⭐ | 0.4s | 20s | **37x** | 72% |
| GPU Batch (future) | 0.3s | 15s | 50x | 65% |

---

## Memory Considerations

### ThreadPool (Shared GPU)
- GPU memory: ~1x model size
- Safe for single GPU
- 8 workers: minimal memory overhead

### ProcessPool (Per-process GPU)
- GPU memory: ~4-8x model size
- Requires larger GPU (16GB+)
- May need to reduce workers

### GPU Batch
- GPU memory: ~1x model + batch data
- Most memory-efficient
- Scales to 100s of simulations

---

## Quality vs Speed Trade-offs

```
                    Quality
                      ↑
  Production          |
  (maxfun=500)     100%|●
                      |
  Recommended         |
  (maxfun=100)      72%|  ●
                      |
  Ultra-fast          |
  (maxfun=50)       50%|     ●
                      |
                      |________________→ Speed
                      1x   6x      16x
```

**Sweet spot:** maxfun=100 (6x faster, 72% quality retention)

---

## Files

- `benchmark_backtest_strategies.py` - Comprehensive benchmarking
- `backtest_test3_inline_parallel.py` - ThreadPool implementation
- `src/optimization_utils.py` - DIRECT optimizer with fast mode
- `src/optimization_utils_gpu_batch.py` - GPU batched optimization (WIP)
- `strategytraining/SPEED_OPTIMIZATION_FINDINGS.md` - Detailed profiling

---

## Next Steps

1. **Immediate:** Enable ThreadPool + Fast mode
   ```bash
   export MARKETSIM_FAST_OPTIMIZE=1
   python backtest_test3_inline_parallel.py ETHUSD 50 8
   ```

2. **Short-term:** Benchmark and validate
   ```bash
   python benchmark_backtest_strategies.py --num-sims 20
   ```

3. **Medium-term:** Implement ProcessPool variant

4. **Long-term:** Fully vectorized GPU batch optimization
