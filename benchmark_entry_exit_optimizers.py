#!/usr/bin/env python3
"""
Benchmark different optimization backends for entry/exit parameter optimization.

This script compares:
- grid-torch-500: Grid search with 500 points (current baseline)
- grid-torch-100: Grid search with 100 points (faster)
- grid-torch-50: Grid search with 50 points (even faster)
- scipy-minimize: scipy.optimize.minimize with L-BFGS-B
- scipy-dual-annealing: scipy.optimize.dual_annealing (global optimizer)
- scipy-differential-evolution: scipy.optimize.differential_evolution

Usage:
    python benchmark_entry_exit_optimizers.py --runs 10
"""

import argparse
import time
from typing import Callable, Tuple, Dict, Any
import numpy as np
import torch
from scipy import optimize


def create_test_objective() -> Tuple[Callable, float, float]:
    """
    Create a synthetic test objective that simulates the entry/exit optimization problem.

    Returns:
        objective: Function to minimize
        true_opt_val: True optimal value for benchmarking
        true_opt_param: True optimal parameter
    """
    # Simulate a noisy profit function similar to real backtesting
    # The real function has local maxima and noise from market data
    np.random.seed(42)
    noise_scale = 0.001

    # True optimum at mult=0.015
    true_opt_param = 0.015

    def objective(mult: float) -> float:
        """Simulates profit calculation with noise."""
        # Quadratic with peak at 0.015, plus noise
        base = -((mult - true_opt_param) ** 2) * 100
        noise = np.random.normal(0, noise_scale)
        return -(base + noise)  # Negative because we minimize

    true_opt_val = objective(true_opt_param)
    return objective, true_opt_val, true_opt_param


def optimize_grid_torch(
    objective: Callable,
    n_points: int = 500,
    bounds: Tuple[float, float] = (-0.03, 0.03),
) -> Tuple[float, float, int]:
    """
    Grid search using numpy (current method in backtest).

    Returns:
        best_param: Best parameter found
        best_value: Best objective value
        n_evals: Number of function evaluations
    """
    best_value = float('inf')
    best_param = 0.0
    n_evals = 0

    for multiplier in np.linspace(bounds[0], bounds[1], n_points):
        value = objective(multiplier)
        n_evals += 1
        if value < best_value:
            best_value = value
            best_param = multiplier

    return best_param, best_value, n_evals


def optimize_scipy_minimize(
    objective: Callable,
    bounds: Tuple[float, float] = (-0.03, 0.03),
    method: str = 'L-BFGS-B',
) -> Tuple[float, float, int]:
    """
    Scipy minimize with bounds.

    Returns:
        best_param: Best parameter found
        best_value: Best objective value
        n_evals: Number of function evaluations
    """
    n_evals = {'count': 0}

    def counted_objective(x):
        n_evals['count'] += 1
        if isinstance(x, np.ndarray):
            return objective(float(x[0]))
        return objective(float(x))

    result = optimize.minimize(
        counted_objective,
        x0=[0.0],
        method=method,
        bounds=[bounds],
        options={'maxiter': 100},
    )

    return float(result.x[0]), float(result.fun), n_evals['count']


def optimize_scipy_dual_annealing(
    objective: Callable,
    bounds: Tuple[float, float] = (-0.03, 0.03),
) -> Tuple[float, float, int]:
    """
    Scipy dual annealing (global optimization).

    Returns:
        best_param: Best parameter found
        best_value: Best objective value
        n_evals: Number of function evaluations
    """
    n_evals = {'count': 0}

    def counted_objective(x):
        n_evals['count'] += 1
        if isinstance(x, np.ndarray):
            return objective(float(x[0]))
        return objective(float(x))

    result = optimize.dual_annealing(
        counted_objective,
        bounds=[bounds],
        maxiter=100,
        seed=42,
    )

    return float(result.x[0]), float(result.fun), n_evals['count']


def optimize_scipy_differential_evolution(
    objective: Callable,
    bounds: Tuple[float, float] = (-0.03, 0.03),
) -> Tuple[float, float, int]:
    """
    Scipy differential evolution (global optimization).

    Returns:
        best_param: Best parameter found
        best_value: Best objective value
        n_evals: Number of function evaluations
    """
    n_evals = {'count': 0}

    def counted_objective(x):
        n_evals['count'] += 1
        if isinstance(x, np.ndarray):
            return objective(float(x[0]))
        return objective(float(x))

    result = optimize.differential_evolution(
        counted_objective,
        bounds=[bounds],
        maxiter=20,
        seed=42,
        strategy='best1bin',
        popsize=5,
    )

    return float(result.x[0]), float(result.fun), n_evals['count']


def benchmark_optimizer(
    name: str,
    optimizer_fn: Callable,
    runs: int = 10,
) -> Dict[str, Any]:
    """
    Benchmark an optimizer over multiple runs.

    Returns:
        Dictionary with timing and accuracy stats
    """
    times = []
    n_evals_list = []
    param_errors = []
    value_errors = []

    for _ in range(runs):
        objective, true_opt_val, true_opt_param = create_test_objective()

        start_time = time.time()
        best_param, best_value, n_evals = optimizer_fn(objective)
        elapsed = time.time() - start_time

        times.append(elapsed)
        n_evals_list.append(n_evals)
        param_errors.append(abs(best_param - true_opt_param))
        value_errors.append(abs(best_value - true_opt_val))

    return {
        'name': name,
        'mean_time': np.mean(times),
        'std_time': np.std(times),
        'mean_n_evals': np.mean(n_evals_list),
        'mean_param_error': np.mean(param_errors),
        'mean_value_error': np.mean(value_errors),
        'times': times,
    }


def print_results(results: list[Dict[str, Any]]):
    """Print benchmark results in a nice table."""
    print("\n" + "="*100)
    print("ENTRY/EXIT OPTIMIZER BACKEND BENCHMARK RESULTS")
    print("="*100)

    # Sort by mean time
    results = sorted(results, key=lambda x: x['mean_time'])

    print(f"\n{'Backend':<30} {'Time (ms)':<15} {'Evals':<10} {'Param Error':<15} {'Speedup':<10}")
    print("-"*100)

    baseline_time = None
    for r in results:
        if 'grid-torch-500' in r['name']:
            baseline_time = r['mean_time']
            break

    for r in results:
        time_ms = r['mean_time'] * 1000
        speedup = baseline_time / r['mean_time'] if baseline_time else 1.0

        print(
            f"{r['name']:<30} "
            f"{time_ms:>10.2f} ± {r['std_time']*1000:>4.1f}   "
            f"{r['mean_n_evals']:>6.0f}     "
            f"{r['mean_param_error']:>10.6f}      "
            f"{speedup:>6.2f}x"
        )

    print("\n" + "="*100)
    print("KEY INSIGHTS:")
    print("-"*100)

    # Find fastest
    fastest = min(results, key=lambda x: x['mean_time'])
    print(f"• Fastest: {fastest['name']} ({fastest['mean_time']*1000:.2f}ms)")

    # Find most accurate
    most_accurate = min(results, key=lambda x: x['mean_param_error'])
    print(f"• Most Accurate: {most_accurate['name']} (error: {most_accurate['mean_param_error']:.6f})")

    # Find best tradeoff (speed vs accuracy)
    for r in results:
        r['score'] = r['mean_time'] * (1 + r['mean_param_error'] * 1000)

    best_tradeoff = min(results, key=lambda x: x['score'])
    speedup = baseline_time / best_tradeoff['mean_time'] if baseline_time else 1.0
    print(f"• Best Tradeoff: {best_tradeoff['name']} ({speedup:.1f}x speedup)")

    print("\nRECOMMENDATION:")
    if best_tradeoff['mean_param_error'] < 0.001:
        print(f"✓ Use '{best_tradeoff['name']}' for {speedup:.1f}x speedup with minimal accuracy loss")
    else:
        print(f"⚠ '{best_tradeoff['name']}' is fastest but check if accuracy is acceptable for your use case")

    print("="*100 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark entry/exit optimization backends"
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=10,
        help="Number of benchmark runs per optimizer (default: 10)",
    )
    args = parser.parse_args()

    print("\nStarting benchmark with {} runs per optimizer...".format(args.runs))
    print("This will take approximately {} seconds.\n".format(args.runs * 7))

    # Define optimizers to benchmark
    optimizers = [
        ("grid-torch-500 (baseline)", lambda obj: optimize_grid_torch(obj, n_points=500)),
        ("grid-torch-100", lambda obj: optimize_grid_torch(obj, n_points=100)),
        ("grid-torch-50", lambda obj: optimize_grid_torch(obj, n_points=50)),
        ("scipy-minimize-LBFGSB", lambda obj: optimize_scipy_minimize(obj, method='L-BFGS-B')),
        ("scipy-minimize-Powell", lambda obj: optimize_scipy_minimize(obj, method='Powell')),
        ("scipy-dual-annealing", optimize_scipy_dual_annealing),
        ("scipy-differential-evolution", optimize_scipy_differential_evolution),
    ]

    results = []
    for name, optimizer_fn in optimizers:
        print(f"Benchmarking {name}...")
        result = benchmark_optimizer(name, optimizer_fn, runs=args.runs)
        results.append(result)

    print_results(results)


if __name__ == '__main__':
    main()
