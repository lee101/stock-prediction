# Multi-Optimizer Benchmark Harness

A comprehensive framework for testing and comparing optimization algorithms on P&L maximization problems, with support for parallel execution and detailed performance analysis.

## Overview

This benchmarking suite allows you to systematically compare different optimization algorithms to find which one works best for your P&L maximization tasks. It supports:

- **7 optimization algorithms** (scipy + Optuna)
- **Parallel execution** using multiprocessing
- **Real P&L data integration** from your strategy datasets
- **Comprehensive performance metrics** (convergence speed, win rates, efficiency)
- **Statistical significance testing**

## Quick Start

### 1. Basic Synthetic Benchmark

Test optimizers on a synthetic P&L landscape:

```bash
# Quick test with 3 optimizers
python strategytraining/benchmark_optimizers.py \
    --num-trials 3 \
    --workers 4 \
    --optimizers DIRECT DifferentialEvolution OptunaTPE

# Full benchmark with all optimizers
python strategytraining/benchmark_optimizers.py \
    --num-trials 5 \
    --workers 8
```

### 2. Real P&L Benchmark

Test on actual strategy optimization problems:

```bash
# Quick test (2 symbols, 3 windows, 3 trials)
python strategytraining/benchmark_on_real_pnl.py \
    --num-symbols 2 \
    --num-windows 3 \
    --num-trials 3 \
    --workers 4

# Comprehensive test (5 symbols, 5 windows, 5 trials)
python strategytraining/benchmark_on_real_pnl.py \
    --num-symbols 5 \
    --num-windows 5 \
    --num-trials 5 \
    --workers 8 \
    --optimizers DIRECT DifferentialEvolution DualAnnealing OptunaTPE
```

### 3. Analyze Results

```bash
# Analyze single benchmark
python strategytraining/analyze_optimizer_benchmarks.py \
    strategytraining/benchmark_results/real_pnl_benchmark_*.parquet

# Analyze multiple benchmarks
python strategytraining/analyze_optimizer_benchmarks.py \
    strategytraining/benchmark_results/*.parquet \
    --output-dir analysis_output/
```

## Optimizers Tested

| Optimizer | Type | Best For | Speed | Notes |
|-----------|------|----------|-------|-------|
| **DIRECT** | Deterministic | Global optimization | Fast (1.5x faster than DE) | Current default, good balance |
| **DifferentialEvolution** | Evolutionary | Robust global search | Medium | Current fallback, reliable |
| **DualAnnealing** | Simulated annealing | Escaping local minima | Medium | Good for complex landscapes |
| **SHGO** | Simplicial homology | Multimodal functions | Slow | Best for functions with many local minima |
| **BasinHopping** | Random perturbation | Local refinement | Medium | Combines global + local search |
| **GridSearch** | Exhaustive | Baseline comparison | Slow | Simple but thorough |
| **OptunaTPE** | Bayesian | Hyperparameter tuning | Fast | Requires `optuna` package |

## Performance Metrics

The benchmark tracks:

### Primary Metrics
- **Best P&L**: Final objective value achieved
- **Time to converge**: Wall-clock time in seconds
- **Function evaluations**: Number of objective calls
- **Convergence status**: Whether optimizer succeeded

### Derived Metrics
- **Win rate**: % of problems where optimizer found best P&L
- **Relative improvement**: % better than baseline (DIRECT)
- **Efficiency**: P&L per second, P&L per evaluation
- **Statistical significance**: Mann-Whitney U tests between optimizers

### Composite Score
Overall ranking based on:
- 50% P&L quality
- 30% win rate
- 20% speed

## Example Output

```
================================================================================
REAL P&L BENCHMARK SUMMARY
================================================================================
                       best_value_mean  best_value_std  time_seconds_mean  wins  win_rate
optimizer_name
DIRECT                       -0.145623        0.082341              0.25    45     45.0%
DifferentialEvolution        -0.148291        0.085112              0.42    25     25.0%
DualAnnealing                -0.143177        0.079834              0.68    18     18.0%
OptunaTPE                    -0.141023        0.077291              0.35    12     12.0%

RECOMMENDATIONS
================================================================================
🏆 Best Overall: OptunaTPE
   Composite Score: 0.7234
   Win Rate: 12.0%
   Mean P&L: -0.141023
   Mean Time: 0.35s

⚡ Fastest: DIRECT
   Mean Time: 0.25s

💰 Best P&L: OptunaTPE
   Mean P&L: -0.141023

🎯 Most Wins: DIRECT
   Win Rate: 45.0%
```

## File Structure

```
strategytraining/
├── benchmark_optimizers.py           # Core benchmark harness
├── benchmark_on_real_pnl.py         # Real P&L data integration
├── analyze_optimizer_benchmarks.py  # Results analysis tools
├── benchmark_results/               # Output directory
│   ├── benchmark_synthetic_*.parquet
│   ├── real_pnl_benchmark_*.parquet
│   └── *_summary.csv
└── datasets/                        # P&L data for benchmarking
    └── full_strategy_dataset_*.parquet
```

## Configuration Options

### BenchmarkConfig

```python
config = BenchmarkConfig(
    bounds=[(-0.03, 0.03), (-0.03, 0.03)],  # Search bounds
    maxiter=50,                              # Max iterations
    popsize=10,                              # Population size
    function_budget=500,                     # Max function evals
    seed=42,                                 # Random seed
    atol=1e-5,                              # Convergence tolerance
)
```

### Command-Line Arguments

**benchmark_optimizers.py**:
- `--strategy`: Strategy name (default: synthetic)
- `--num-trials`: Trials per optimizer (default: 5)
- `--workers`: Parallel workers (default: 4)
- `--optimizers`: Specific optimizers to test (default: all)
- `--output-dir`: Results directory (default: strategytraining/benchmark_results)

**benchmark_on_real_pnl.py**:
- `--num-symbols`: Number of symbols (default: 5)
- `--num-windows`: Windows per symbol (default: 3)
- `--num-trials`: Trials per problem (default: 3)
- `--optimizers`: Optimizers to test (default: DIRECT, DE, DualAnnealing, OptunaTPE)
- `--workers`: Parallel workers (default: 4)

## Integration with Existing Code

The benchmarking harness integrates seamlessly with your existing optimization code:

### Using Real Objective Functions

```python
from src.optimization_utils import _EntryExitObjective
from strategytraining.benchmark_optimizers import OptimizerBenchmark, BenchmarkConfig

# Create objective from your data
objective = _EntryExitObjective(
    close_actual=close_actual,
    positions=positions,
    high_actual=high_actual,
    high_pred=high_pred,
    low_actual=low_actual,
    low_pred=low_pred,
    close_at_eod=False,
    trading_fee=0.001,
)

# Benchmark optimizers
config = BenchmarkConfig(bounds=[(-0.03, 0.03), (-0.03, 0.03)])
benchmark = OptimizerBenchmark(objective, config, strategy_name="my_strategy")

# Run all optimizers
results = benchmark.run_all_optimizers()

# Or run specific optimizer
direct_result = benchmark.run_direct()
de_result = benchmark.run_differential_evolution()
optuna_result = benchmark.run_optuna()
```

### Parallel Benchmarking

```python
from strategytraining.benchmark_optimizers import run_parallel_benchmarks

results_df = run_parallel_benchmarks(
    objective_fn=my_objective,
    config=config,
    strategy_name="maxdiff",
    num_trials=5,
    max_workers=8,
    optimizers=['DIRECT', 'OptunaTPE', 'DualAnnealing']
)

# Analyze
from strategytraining.benchmark_optimizers import analyze_benchmark_results
summary = analyze_benchmark_results(results_df)
print(summary)
```

## Performance Optimization Tips

1. **Use appropriate worker count**: Set `--workers` to your CPU core count for best parallelism
2. **Balance budget vs trials**: More trials = better statistics, higher budget = better solutions
3. **Filter optimizers**: Skip slow optimizers (SHGO, GridSearch) for quick tests
4. **Cache P&L data**: The real P&L benchmark loads data once per symbol
5. **Use Optuna for speed**: OptunaTPE is often fastest for low-dimensional problems

## Speedup Examples

With 8 cores and 6 optimizers × 5 trials = 30 benchmarks:

- **Sequential**: ~15 minutes (30s per benchmark)
- **Parallel (8 workers)**: ~4 minutes (3.75x speedup)
- **Real improvement**: Can test more optimizers in less time

## Next Steps

1. **Run initial benchmarks** to establish baseline performance
2. **Compare results** using the analysis tool
3. **Update optimization_utils.py** to use best optimizer
4. **Re-run strategy optimization** with improved optimizer
5. **Iterate** by testing new optimizers or tuning parameters

## Troubleshooting

### "No module named 'optuna'"
```bash
uv pip install optuna
```

### "Can't pickle objective function"
Ensure objective functions are either:
- Top-level functions (not closures)
- Picklable classes with `__call__` method
- Use `_EntryExitObjective` or `_AlwaysOnObjective` from `optimization_utils.py`

### "No P&L dataset found"
Run data collection first:
```bash
python strategytraining/collect_strategy_pnl_dataset.py
```

### Out of memory
- Reduce `--num-symbols` or `--num-windows`
- Decrease `--workers`
- Lower `function_budget` in config

## References

- **scipy.optimize.direct**: https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.direct.html
- **Differential Evolution**: https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.differential_evolution.html
- **Optuna**: https://optuna.readthedocs.io/
- **Original optimization code**: `src/optimization_utils.py`
- **Strategy P&L collection**: `strategytraining/collect_strategy_pnl_dataset.py`

## License

Part of the stock-prediction project.
