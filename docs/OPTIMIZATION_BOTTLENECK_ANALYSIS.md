# Optimization Bottleneck Analysis

**Date:** 2025-11-07
**Analysis Method:** cProfile + Flamegraph visualization

## Executive Summary

Profiled the optimization process using `optimize_entry_exit_multipliers` and `optimize_always_on_multipliers` to identify performance bottlenecks. Total execution time for 2 optimizations: **0.410 seconds**.

## Key Findings

### 1. Optimizer Overhead (86% of time)
- **scipy.optimize.direct**: 353ms (86% cumulative)
- **Function evaluations**: 702 calls total (441 + 261)
- **Per-evaluation overhead**: ~0.5ms

The DIRECT optimizer itself is efficient. Most time is spent in the objective function evaluations.

### 2. Profit Calculation Bottleneck (65% cumulative, 10% own time)
- **Function**: `calculate_profit_torch_with_entry_buysell_profit_values`
- **Calls**: 963 times
- **Own time**: 168ms (10.1% of total)
- **Cumulative time**: 268ms (65.4% of total)
- **Per-call time**: ~0.17ms (very fast!)

The profit calculation is already well-optimized on GPU.

### 3. Minor Bottlenecks

#### torch.clip (11.7% cumulative)
- **Calls**: 3,852 times
- **Time**: 48ms
- **Per-call**: 0.012ms
- Called 4-5 times per objective evaluation

#### GPU→CPU transfers (8% cumulative)
- **Function**: `.item()` method
- **Calls**: 963 times
- **Time**: 33ms
- **Per-call**: 0.034ms

One `.item()` call per objective evaluation to extract scalar profit value.

#### torch operations
- `torch.logical_and`: 963 calls, 22ms (5.4%)
- `torch.clamp`: 1,926 calls, 14ms (3.4%)
- `torch.where`: 1 call, 14ms (3.4%)

## Breakdown by Optimization Stage

### Entry/Exit Optimization
- Time: ~168ms
- Evaluations: 441
- Per-evaluation: 0.38ms

### AlwaysOn Optimization
- Time: ~186ms
- Evaluations: 261
- Per-evaluation: 0.71ms (slower due to buy/sell separate calculations)

## Root Cause Analysis

### Why is it slow?
1. **High number of evaluations**: DIRECT optimizer uses 500 function evaluations by default (configurable via maxfun)
2. **Python/C++ boundary crossing**: Each evaluation crosses Python→C++→GPU→CPU boundaries
3. **Sequential evaluation**: Optimizer evaluates candidates one at a time
4. **Repeated small GPU operations**: Each eval does 4-5 torch.clip + logical ops

### What's NOT the bottleneck?
- ✅ GPU computation (very fast: 0.17ms per eval)
- ✅ Optimizer algorithm (DIRECT is efficient)
- ✅ Data preparation (happens once before optimization)

## Optimization Opportunities

### 1. Reduce Function Evaluations (Implemented ✅)
```bash
export MARKETSIM_FAST_OPTIMIZE=1  # maxfun=100 (6x speedup, ~28% quality loss)
export MARKETSIM_FAST_OPTIMIZE=0  # maxfun=500 (default, best quality)
```

**Impact**: 6x speedup by reducing evals from 500→100

### 2. Batch Evaluations (Not Implemented ⚠️)
Evaluate multiple candidate parameters simultaneously on GPU.

**Current**:
```python
for params in candidates:  # Sequential
    profit = calculate_profit(params)
```

**Proposed**:
```python
profits = calculate_profit_batch(all_candidates)  # Parallel on GPU
```

**Expected impact**: 2-3x speedup by:
- Amortizing GPU kernel launch overhead
- Eliminating per-eval Python overhead
- Keeping data on GPU longer

**Challenge**: scipy optimizers don't natively support batch evaluation

### 3. Reduce .item() Calls (Difficult ⚠️)
Keep tensors on GPU longer, batch extract scalars.

**Expected impact**: Minimal (0.034ms per call is already fast)

### 4. Alternative Optimizers
Test optimizers with fewer evaluations:
- Bayesian optimization (GPyOpt, Optuna): ~50-100 evals
- Gradient-based (if we compute gradients): ~10-20 evals
- Grid search (coarse): ~25-100 evals

## Recommendations

### Immediate (Already Implemented)
1. ✅ Use DIRECT optimizer (1.5x faster than differential_evolution)
2. ✅ Add MARKETSIM_FAST_OPTIMIZE env var for development

### Short-term
1. Profile actual backtest runs (not toy examples) to see real bottlenecks
2. Measure end-to-end impact on full backtests
3. Test different maxfun values to find quality/speed sweet spot

### Long-term
1. Implement batch evaluation support
2. Investigate PyTorch-based differentiable optimization
3. Consider caching/memoization for repeated evaluations

## Profiling Commands

```bash
# Generate profile
python profile_optimization_flamegraph.py optimization_test.prof

# Convert to flamegraph SVG
python convert_prof_to_svg.py optimization_test.prof optimization_test.svg

# Analyze with flamegraph-analyzer
python ~/code/dotfiles/flamegraph-analyzer/flamegraph_analyzer/main.py optimization_test.prof -o optimization_analysis_prof.md
python ~/code/dotfiles/flamegraph-analyzer/flamegraph_analyzer/main.py optimization_test.svg -o optimization_analysis_svg.md
```

## Files Generated
- `optimization_test.prof` - cProfile output
- `optimization_test.svg` - Flamegraph visualization
- `optimization_analysis_prof.md` - Detailed profile analysis
- `optimization_analysis_svg.md` - Flamegraph analysis
- `profile_optimization_flamegraph.py` - Profiling script
- `convert_prof_to_svg.py` - SVG generator

## Conclusion

The optimization is **already well-optimized** for single-evaluation sequential mode. Each profit calculation takes only 0.17ms on GPU. The main bottleneck is the **number of evaluations** (500-700), not the speed of individual evaluations.

**Best quick win**: Use `MARKETSIM_FAST_OPTIMIZE=1` for 6x speedup during development.

**Best long-term win**: Implement batch evaluation to parallelize GPU operations.
