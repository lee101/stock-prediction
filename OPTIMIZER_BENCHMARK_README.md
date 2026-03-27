# Entry/Exit Optimizer Benchmarking and Configuration

This document describes the new optimizer benchmarking tools and configuration options for the entry/exit optimization in backtesting.

## Quick Start

### 1. Benchmark Different Optimizers

Run the benchmark script to compare different optimization backends:

```bash
# Run with default 10 runs
.venv313/bin/python benchmark_entry_exit_optimizers.py

# Run with custom number of runs
.venv313/bin/python benchmark_entry_exit_optimizers.py --runs 20
```

**Benchmark Results** (example from 3 runs):
```
Backend                        Time (ms)       Evals      Param Error     Speedup
----------------------------------------------------------------------------------------------------
grid-torch-50                        0.08 ±  0.0       50       0.002755        9.17x
grid-torch-100                       0.14 ±  0.0      100       0.001364        4.88x
grid-torch-500 (baseline)            0.70 ±  0.0      500       0.000331        1.00x
scipy-minimize-Powell                1.11 ±  0.0       59       0.001021        0.63x
scipy-minimize-LBFGSB                2.67 ±  1.0       42       0.014996        0.26x
scipy-differential-evolution         7.15 ±  0.9      145       0.001128        0.10x
scipy-dual-annealing                16.94 ±  0.5      429       0.002227        0.04x
```

**Key Insights:**
- `grid-torch-50` is ~9x faster than baseline with acceptable accuracy
- `grid-torch-100` is ~5x faster with better accuracy
- scipy optimizers are slower than grid search for this problem

### 2. Configure Optimizer Backend

Set the `ENTRY_EXIT_OPTIMIZER_BACKEND` environment variable to use a different optimizer:

```bash
# Use grid search with 50 points (9x faster)
export ENTRY_EXIT_OPTIMIZER_BACKEND=grid-50

# Use grid search with 100 points (5x faster, recommended)
export ENTRY_EXIT_OPTIMIZER_BACKEND=grid-100

# Use grid search with 500 points (baseline, most accurate)
export ENTRY_EXIT_OPTIMIZER_BACKEND=grid-500
```

**Default:** `grid-100` (changed from previous default of 500 for better performance)

### 3. Enable Fast Simulate Mode

For even faster execution in `trade_stock_e2e.py`, enable FAST_SIMULATE mode:

```bash
# Enable fast simulate mode (reduces simulations from 70 to 20)
export FAST_SIMULATE=1

# Or use MARKETSIM_BACKTEST_SIMULATIONS for custom value
export MARKETSIM_BACKTEST_SIMULATIONS=30
```

**FAST_SIMULATE mode:**
- Reduces number of backtest simulations from 70 to 20
- Approximately 3.5x faster overall execution
- Useful for development, testing, and rapid iteration

## Available Optimizer Backends

| Backend | Grid Points | Speed vs Baseline | Accuracy | Use Case |
|---------|-------------|-------------------|----------|----------|
| `grid-500` | 500 | 1.0x (baseline) | Highest | Production, maximum accuracy |
| `grid-100` | 100 | ~5x faster | Good | Default, recommended |
| `grid-50` | 50 | ~9x faster | Acceptable | Fast development, testing |

## Environment Variables Summary

| Variable | Description | Default | Example |
|----------|-------------|---------|---------|
| `ENTRY_EXIT_OPTIMIZER_BACKEND` | Optimizer backend to use | `grid-100` | `grid-50`, `grid-100`, `grid-500` |
| `FAST_SIMULATE` | Enable fast simulate mode | `0` (off) | `1`, `true`, `yes`, `on` |
| `MARKETSIM_BACKTEST_SIMULATIONS` | Number of backtest simulations | `70` (20 if FAST_SIMULATE=1) | `20`, `30`, `50` |

## Recommendations

### For Development & Testing
```bash
export ENTRY_EXIT_OPTIMIZER_BACKEND=grid-50
export FAST_SIMULATE=1
```
This gives you ~30x overall speedup (9x from optimizer + 3.5x from fewer simulations).

### For Production
```bash
export ENTRY_EXIT_OPTIMIZER_BACKEND=grid-100
# FAST_SIMULATE=0 (default)
```
This gives you ~5x speedup with minimal accuracy loss.

### For Maximum Accuracy
```bash
export ENTRY_EXIT_OPTIMIZER_BACKEND=grid-500
# FAST_SIMULATE=0 (default)
```
Use the original slow but accurate settings.

## Technical Details

### How Entry/Exit Optimization Works

The backtest optimizes entry/exit multipliers for high and low predictions:

1. **High Multiplier Optimization**: Grid search over `[-0.03, 0.03]` with N points
2. **Low Multiplier Optimization**: Grid search over `[-0.03, 0.03]` with N points
3. **Total Evaluations**: 2 × N evaluations per strategy

With `grid-500` (baseline), this is **1000 evaluations** per strategy.
With `grid-100`, this is **200 evaluations** per strategy (~5x faster).
With `grid-50`, this is **100 evaluations** per strategy (~10x faster).

### Benchmark Script Details

The `benchmark_entry_exit_optimizers.py` script:
- Creates synthetic test objectives that mimic real backtesting
- Tests each optimizer over multiple runs with different random seeds
- Measures execution time, number of evaluations, and accuracy
- Provides recommendations based on speed/accuracy tradeoff

## Files Modified

- `backtest_test3_inline.py`: Added `ENTRY_EXIT_OPTIMIZER_BACKEND` configuration
- `trade_stock_e2e.py`: Added `FAST_SIMULATE` mode
- `benchmark_entry_exit_optimizers.py`: New benchmark script

## Changelog

### 2025-11-13
- Added `benchmark_entry_exit_optimizers.py` for comparing optimizer backends
- Changed default optimizer from `grid-500` to `grid-100` for 5x speedup
- Added `ENTRY_EXIT_OPTIMIZER_BACKEND` environment variable
- Added `FAST_SIMULATE` mode to `trade_stock_e2e.py`
- Grid search optimizers: `grid-50`, `grid-100`, `grid-500`
