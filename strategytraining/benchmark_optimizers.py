#!/usr/bin/env python3
"""
Multi-Optimizer Benchmark Harness

Tests multiple optimization algorithms on real P&L maximization tasks.
Supports parallel execution and comprehensive performance tracking.

Optimizers tested:
- scipy DIRECT (Dividing Rectangles) - Current default
- scipy Differential Evolution - Current fallback
- scipy Dual Annealing - Simulated annealing
- scipy SHGO - Simplicial Homology Global Optimization
- scipy Basin Hopping - Random perturbation + local search
- Optuna TPE - Tree-structured Parzen Estimator (if installed)
- Grid Search - Baseline

Usage:
    python benchmark_optimizers.py --strategy maxdiff --num-trials 10 --workers 4
    python benchmark_optimizers.py --all-strategies --workers 8
"""

import argparse
import json
import time
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Any
import sys

import numpy as np
import pandas as pd
from scipy.optimize import (
    direct,
    differential_evolution,
    dual_annealing,
    shgo,
    basinhopping,
    OptimizeResult,
)

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

warnings.filterwarnings('ignore')


@dataclass
class OptimizerResult:
    """Results from a single optimizer run"""
    optimizer_name: str
    best_params: List[float]
    best_value: float
    num_evaluations: int
    time_seconds: float
    converged: bool
    strategy_name: str
    trial_id: int

    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class BenchmarkConfig:
    """Configuration for optimizer benchmark"""
    bounds: List[Tuple[float, float]]
    maxiter: int = 50
    popsize: int = 10
    seed: int = 42
    function_budget: int = 500  # Max function evaluations
    atol: float = 1e-5


class OptimizerBenchmark:
    """Benchmark suite for comparing optimization algorithms"""

    def __init__(
        self,
        objective_fn: Callable,
        config: BenchmarkConfig,
        strategy_name: str = "unknown",
        trial_id: int = 0,
    ):
        self.objective_fn = objective_fn
        self.config = config
        self.strategy_name = strategy_name
        self.trial_id = trial_id
        self.eval_count = 0

    def _reset_counter(self):
        """Reset evaluation counter"""
        self.eval_count = 0

    def _counted_objective(self, x):
        """Objective function with evaluation counting"""
        self.eval_count += 1
        return self.objective_fn(x)

    def run_direct(self) -> OptimizerResult:
        """DIRECT (Dividing Rectangles) - Current default optimizer"""
        self._reset_counter()
        start_time = time.time()

        try:
            result = direct(
                self._counted_objective,
                bounds=self.config.bounds,
                maxfun=self.config.function_budget,
            )
            converged = result.success
            best_x = result.x.tolist()
            best_val = result.fun
        except Exception as e:
            print(f"DIRECT failed: {e}")
            converged = False
            best_x = [0.0] * len(self.config.bounds)
            best_val = float('inf')

        elapsed = time.time() - start_time

        return OptimizerResult(
            optimizer_name="DIRECT",
            best_params=best_x,
            best_value=best_val,
            num_evaluations=self.eval_count,
            time_seconds=elapsed,
            converged=converged,
            strategy_name=self.strategy_name,
            trial_id=self.trial_id,
        )

    def run_differential_evolution(self) -> OptimizerResult:
        """Differential Evolution - Current fallback optimizer"""
        self._reset_counter()
        start_time = time.time()

        try:
            result = differential_evolution(
                self._counted_objective,
                bounds=self.config.bounds,
                maxiter=self.config.maxiter,
                popsize=self.config.popsize,
                atol=self.config.atol,
                seed=self.config.seed,
                workers=1,  # Single worker for fair comparison
                updating="immediate",
            )
            converged = result.success
            best_x = result.x.tolist()
            best_val = result.fun
        except Exception as e:
            print(f"Differential Evolution failed: {e}")
            converged = False
            best_x = [0.0] * len(self.config.bounds)
            best_val = float('inf')

        elapsed = time.time() - start_time

        return OptimizerResult(
            optimizer_name="DifferentialEvolution",
            best_params=best_x,
            best_value=best_val,
            num_evaluations=self.eval_count,
            time_seconds=elapsed,
            converged=converged,
            strategy_name=self.strategy_name,
            trial_id=self.trial_id,
        )

    def run_dual_annealing(self) -> OptimizerResult:
        """Dual Annealing - Simulated annealing with local search"""
        self._reset_counter()
        start_time = time.time()

        try:
            result = dual_annealing(
                self._counted_objective,
                bounds=self.config.bounds,
                maxfun=self.config.function_budget,
                seed=self.config.seed,
            )
            converged = result.success
            best_x = result.x.tolist()
            best_val = result.fun
        except Exception as e:
            print(f"Dual Annealing failed: {e}")
            converged = False
            best_x = [0.0] * len(self.config.bounds)
            best_val = float('inf')

        elapsed = time.time() - start_time

        return OptimizerResult(
            optimizer_name="DualAnnealing",
            best_params=best_x,
            best_value=best_val,
            num_evaluations=self.eval_count,
            time_seconds=elapsed,
            converged=converged,
            strategy_name=self.strategy_name,
            trial_id=self.trial_id,
        )

    def run_shgo(self) -> OptimizerResult:
        """SHGO - Simplicial Homology Global Optimization"""
        self._reset_counter()
        start_time = time.time()

        try:
            result = shgo(
                self._counted_objective,
                bounds=self.config.bounds,
                options={
                    'maxfev': self.config.function_budget,
                    'f_tol': self.config.atol,
                },
            )
            converged = result.success
            best_x = result.x.tolist()
            best_val = result.fun
        except Exception as e:
            print(f"SHGO failed: {e}")
            converged = False
            best_x = [0.0] * len(self.config.bounds)
            best_val = float('inf')

        elapsed = time.time() - start_time

        return OptimizerResult(
            optimizer_name="SHGO",
            best_params=best_x,
            best_value=best_val,
            num_evaluations=self.eval_count,
            time_seconds=elapsed,
            converged=converged,
            strategy_name=self.strategy_name,
            trial_id=self.trial_id,
        )

    def run_basinhopping(self) -> OptimizerResult:
        """Basin Hopping - Random perturbation + local minimization"""
        self._reset_counter()
        start_time = time.time()

        # Starting point: center of bounds
        x0 = np.array([(b[0] + b[1]) / 2 for b in self.config.bounds])

        try:
            result = basinhopping(
                self._counted_objective,
                x0,
                niter=self.config.maxiter,
                seed=self.config.seed,
            )
            converged = True  # Basin hopping doesn't have success flag
            best_x = result.x.tolist()
            best_val = result.fun
        except Exception as e:
            print(f"Basin Hopping failed: {e}")
            converged = False
            best_x = [0.0] * len(self.config.bounds)
            best_val = float('inf')

        elapsed = time.time() - start_time

        return OptimizerResult(
            optimizer_name="BasinHopping",
            best_params=best_x,
            best_value=best_val,
            num_evaluations=self.eval_count,
            time_seconds=elapsed,
            converged=converged,
            strategy_name=self.strategy_name,
            trial_id=self.trial_id,
        )

    def run_grid_search(self, grid_points: int = 10) -> OptimizerResult:
        """Grid Search - Exhaustive search baseline"""
        self._reset_counter()
        start_time = time.time()

        # Create grid
        grids = [np.linspace(b[0], b[1], grid_points) for b in self.config.bounds]

        best_x = None
        best_val = float('inf')

        # Exhaustive search
        if len(self.config.bounds) == 2:
            for x1 in grids[0]:
                for x2 in grids[1]:
                    x = [x1, x2]
                    val = self._counted_objective(x)
                    if val < best_val:
                        best_val = val
                        best_x = x
        else:
            # For higher dimensions, sample randomly from grid
            import itertools
            for point in itertools.product(*grids):
                val = self._counted_objective(list(point))
                if val < best_val:
                    best_val = val
                    best_x = list(point)

                if self.eval_count >= self.config.function_budget:
                    break

        elapsed = time.time() - start_time

        return OptimizerResult(
            optimizer_name="GridSearch",
            best_params=best_x if best_x else [0.0] * len(self.config.bounds),
            best_value=best_val,
            num_evaluations=self.eval_count,
            time_seconds=elapsed,
            converged=True,
            strategy_name=self.strategy_name,
            trial_id=self.trial_id,
        )

    def run_optuna(self) -> Optional[OptimizerResult]:
        """Optuna TPE - Tree-structured Parzen Estimator (if installed)"""
        try:
            import optuna
            optuna.logging.set_verbosity(optuna.logging.WARNING)
        except ImportError:
            print("Optuna not installed, skipping...")
            return None

        self._reset_counter()
        start_time = time.time()

        def optuna_objective(trial):
            params = []
            for i, (low, high) in enumerate(self.config.bounds):
                params.append(trial.suggest_float(f'x{i}', low, high))
            return self._counted_objective(params)

        try:
            study = optuna.create_study(
                direction='minimize',
                sampler=optuna.samplers.TPESampler(seed=self.config.seed)
            )
            study.optimize(
                optuna_objective,
                n_trials=self.config.function_budget,
                show_progress_bar=False,
            )

            best_x = [study.best_params[f'x{i}'] for i in range(len(self.config.bounds))]
            best_val = study.best_value
            converged = True

        except Exception as e:
            print(f"Optuna failed: {e}")
            converged = False
            best_x = [0.0] * len(self.config.bounds)
            best_val = float('inf')

        elapsed = time.time() - start_time

        return OptimizerResult(
            optimizer_name="OptunaTPE",
            best_params=best_x,
            best_value=best_val,
            num_evaluations=self.eval_count,
            time_seconds=elapsed,
            converged=converged,
            strategy_name=self.strategy_name,
            trial_id=self.trial_id,
        )

    def run_all_optimizers(self) -> List[OptimizerResult]:
        """Run all available optimizers"""
        results = []

        print(f"  Running DIRECT...")
        results.append(self.run_direct())

        print(f"  Running Differential Evolution...")
        results.append(self.run_differential_evolution())

        print(f"  Running Dual Annealing...")
        results.append(self.run_dual_annealing())

        print(f"  Running SHGO...")
        results.append(self.run_shgo())

        print(f"  Running Basin Hopping...")
        results.append(self.run_basinhopping())

        print(f"  Running Grid Search...")
        results.append(self.run_grid_search())

        print(f"  Running Optuna TPE...")
        optuna_result = self.run_optuna()
        if optuna_result:
            results.append(optuna_result)

        return results


class SyntheticPnLObjective:
    """Picklable synthetic P&L objective function for testing"""

    def __init__(self, dim: int = 2, noise_level: float = 0.01, seed: int = 42):
        self.dim = dim
        self.noise_level = noise_level
        self.seed = seed

        rng = np.random.RandomState(seed)

        # Random quadratic with noise to simulate real P&L landscape
        A = rng.randn(dim, dim)
        self.A = A.T @ A  # Make positive semi-definite
        self.b = rng.randn(dim)
        self.c = rng.randn()
        self.rng = np.random.RandomState(seed + 1)  # Separate RNG for noise

    def __call__(self, x):
        x = np.array(x)
        base_val = 0.5 * x.T @ self.A @ x + self.b.T @ x + self.c
        noise = self.rng.randn() * self.noise_level
        return base_val + noise


def run_single_benchmark(
    optimizer_name: str,
    objective_fn: Callable,
    config: BenchmarkConfig,
    strategy_name: str,
    trial_id: int,
) -> OptimizerResult:
    """Run a single optimizer (for parallel execution)"""
    benchmark = OptimizerBenchmark(objective_fn, config, strategy_name, trial_id)

    optimizer_methods = {
        'DIRECT': benchmark.run_direct,
        'DifferentialEvolution': benchmark.run_differential_evolution,
        'DualAnnealing': benchmark.run_dual_annealing,
        'SHGO': benchmark.run_shgo,
        'BasinHopping': benchmark.run_basinhopping,
        'GridSearch': benchmark.run_grid_search,
        'OptunaTPE': benchmark.run_optuna,
    }

    if optimizer_name not in optimizer_methods:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")

    return optimizer_methods[optimizer_name]()


def run_parallel_benchmarks(
    objective_fn: Callable,
    config: BenchmarkConfig,
    strategy_name: str = "test",
    num_trials: int = 5,
    max_workers: int = 4,
    optimizers: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Run multiple optimizer benchmarks in parallel across multiple trials.

    Args:
        objective_fn: Objective function to minimize
        config: Benchmark configuration
        strategy_name: Name of strategy being optimized
        num_trials: Number of independent trials per optimizer
        max_workers: Number of parallel workers
        optimizers: List of optimizer names to test (None = all)

    Returns:
        DataFrame with results
    """
    if optimizers is None:
        optimizers = [
            'DIRECT', 'DifferentialEvolution', 'DualAnnealing',
            'SHGO', 'BasinHopping', 'GridSearch', 'OptunaTPE'
        ]

    print(f"Running {len(optimizers)} optimizers x {num_trials} trials = {len(optimizers) * num_trials} total benchmarks")
    print(f"Using {max_workers} parallel workers")
    print("=" * 80)

    all_results = []

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {}

        # Submit all tasks
        for optimizer_name in optimizers:
            for trial_id in range(num_trials):
                future = executor.submit(
                    run_single_benchmark,
                    optimizer_name,
                    objective_fn,
                    config,
                    strategy_name,
                    trial_id,
                )
                futures[future] = (optimizer_name, trial_id)

        # Collect results
        completed = 0
        total = len(futures)

        for future in as_completed(futures):
            optimizer_name, trial_id = futures[future]
            try:
                result = future.result()
                if result:  # Optuna might return None
                    all_results.append(result)
                    completed += 1
                    print(f"✓ [{completed}/{total}] {optimizer_name} trial {trial_id}: "
                          f"val={result.best_value:.6f} time={result.time_seconds:.2f}s evals={result.num_evaluations}")
            except Exception as e:
                print(f"✗ {optimizer_name} trial {trial_id} failed: {e}")

    # Convert to DataFrame
    df = pd.DataFrame([r.to_dict() for r in all_results])
    return df


def analyze_benchmark_results(df: pd.DataFrame) -> pd.DataFrame:
    """Analyze benchmark results and compute summary statistics"""
    summary = df.groupby('optimizer_name').agg({
        'best_value': ['mean', 'std', 'min', 'max'],
        'time_seconds': ['mean', 'std'],
        'num_evaluations': ['mean', 'std'],
        'converged': 'mean',
    }).round(6)

    summary.columns = ['_'.join(col) for col in summary.columns]
    summary = summary.sort_values('best_value_mean')

    return summary


def main():
    parser = argparse.ArgumentParser(description="Benchmark optimization algorithms")
    parser.add_argument(
        '--strategy',
        type=str,
        default='synthetic',
        help='Strategy to optimize (default: synthetic test function)'
    )
    parser.add_argument(
        '--num-trials',
        type=int,
        default=5,
        help='Number of trials per optimizer (default: 5)'
    )
    parser.add_argument(
        '--workers',
        type=int,
        default=4,
        help='Number of parallel workers (default: 4)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='strategytraining/benchmark_results',
        help='Output directory for results'
    )
    parser.add_argument(
        '--optimizers',
        nargs='+',
        default=None,
        help='Specific optimizers to test (default: all)'
    )

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create objective function
    print(f"Creating objective function for strategy: {args.strategy}")
    objective_fn = SyntheticPnLObjective(dim=2, noise_level=0.01, seed=42)

    # Create config
    config = BenchmarkConfig(
        bounds=[(-0.03, 0.03), (-0.03, 0.03)],
        maxiter=50,
        popsize=10,
        function_budget=500,
        seed=42,
    )

    # Run benchmarks
    start_time = time.time()
    results_df = run_parallel_benchmarks(
        objective_fn,
        config,
        strategy_name=args.strategy,
        num_trials=args.num_trials,
        max_workers=args.workers,
        optimizers=args.optimizers,
    )
    total_time = time.time() - start_time

    # Analyze results
    print("\n" + "=" * 80)
    print("BENCHMARK SUMMARY")
    print("=" * 80)
    summary = analyze_benchmark_results(results_df)
    print(summary)

    print(f"\nTotal benchmark time: {total_time:.2f}s")

    # Save results
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    results_file = output_dir / f"benchmark_{args.strategy}_{timestamp}.parquet"
    summary_file = output_dir / f"benchmark_{args.strategy}_{timestamp}_summary.csv"

    results_df.to_parquet(results_file)
    summary.to_csv(summary_file)

    print(f"\nResults saved to:")
    print(f"  {results_file}")
    print(f"  {summary_file}")


if __name__ == "__main__":
    main()
