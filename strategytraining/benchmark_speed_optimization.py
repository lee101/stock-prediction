#!/usr/bin/env python3
"""
Speed-Focused Optimization Benchmark

Tests different optimizer configurations to find the fastest way to achieve
high-quality P&L results. Focus on DIRECT and Differential Evolution with
various parallelization strategies.

Key Questions:
1. Can we get DIRECT results faster with different configs?
2. How much speedup does DE get with multicore?
3. What's the optimal speed/quality tradeoff?

Usage:
    # Test DIRECT speed configs
    python strategytraining/benchmark_speed_optimization.py --mode direct --workers 8

    # Test DE with different worker counts
    python strategytraining/benchmark_speed_optimization.py --mode de-parallel --workers 8

    # Full speed comparison
    python strategytraining/benchmark_speed_optimization.py --mode full --workers 8
"""

import argparse
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Dict, Tuple, Optional

import numpy as np
import pandas as pd
from scipy.optimize import direct, differential_evolution

sys.path.insert(0, str(Path(__file__).parent.parent))

from strategytraining.benchmark_optimizers import (
    SyntheticPnLObjective,
    OptimizerResult,
)


@dataclass
class SpeedConfig:
    """Configuration for speed-focused optimization"""
    name: str
    optimizer_type: str  # 'direct' or 'de'
    maxfun: Optional[int] = None  # For DIRECT
    eps: Optional[float] = None  # For DIRECT
    maxiter: Optional[int] = None  # For DE
    popsize: Optional[int] = None  # For DE
    workers: int = 1  # For DE parallelism (-1 = all cores)
    atol: float = 1e-5


# DIRECT speed configurations
DIRECT_CONFIGS = [
    SpeedConfig("DIRECT_fast", "direct", maxfun=100, eps=0.01),
    SpeedConfig("DIRECT_default", "direct", maxfun=500, eps=0.001),
    SpeedConfig("DIRECT_thorough", "direct", maxfun=1000, eps=0.0001),
    SpeedConfig("DIRECT_ultra_fast", "direct", maxfun=50, eps=0.05),
]

# Differential Evolution with different worker counts
DE_PARALLEL_CONFIGS = [
    SpeedConfig("DE_sequential", "de", maxiter=50, popsize=10, workers=1),
    SpeedConfig("DE_2workers", "de", maxiter=50, popsize=10, workers=2),
    SpeedConfig("DE_4workers", "de", maxiter=50, popsize=10, workers=4),
    SpeedConfig("DE_8workers", "de", maxiter=50, popsize=10, workers=8),
    SpeedConfig("DE_allcores", "de", maxiter=50, popsize=10, workers=-1),
]

# DE speed configurations (sequential)
DE_SPEED_CONFIGS = [
    SpeedConfig("DE_fast", "de", maxiter=20, popsize=5, workers=1),
    SpeedConfig("DE_default", "de", maxiter=50, popsize=10, workers=1),
    SpeedConfig("DE_thorough", "de", maxiter=100, popsize=15, workers=1),
]


class SpeedBenchmark:
    """Benchmark focused on optimization speed"""

    def __init__(self, objective_fn: Callable, bounds: List[Tuple[float, float]], seed: int = 42):
        self.objective_fn = objective_fn
        self.bounds = bounds
        self.seed = seed
        self.eval_count = 0

    def _reset_counter(self):
        self.eval_count = 0

    def _counted_objective(self, x):
        self.eval_count += 1
        return self.objective_fn(x)

    def run_direct_config(self, config: SpeedConfig) -> OptimizerResult:
        """Run DIRECT with specific configuration"""
        self._reset_counter()
        start_time = time.time()

        try:
            result = direct(
                self._counted_objective,
                bounds=self.bounds,
                maxfun=config.maxfun,
                eps=config.eps,
            )
            converged = result.success
            best_x = result.x.tolist()
            best_val = result.fun
        except Exception as e:
            print(f"  {config.name} failed: {e}")
            converged = False
            best_x = [0.0] * len(self.bounds)
            best_val = float('inf')

        elapsed = time.time() - start_time

        return OptimizerResult(
            optimizer_name=config.name,
            best_params=best_x,
            best_value=best_val,
            num_evaluations=self.eval_count,
            time_seconds=elapsed,
            converged=converged,
            strategy_name="speed_test",
            trial_id=0,
        )

    def run_de_config(self, config: SpeedConfig) -> OptimizerResult:
        """Run Differential Evolution with specific configuration"""
        self._reset_counter()
        start_time = time.time()

        try:
            result = differential_evolution(
                self._counted_objective,
                bounds=self.bounds,
                maxiter=config.maxiter,
                popsize=config.popsize,
                atol=config.atol,
                seed=self.seed,
                workers=config.workers,
                updating="deferred" if config.workers != 1 else "immediate",
            )
            converged = result.success
            best_x = result.x.tolist()
            best_val = result.fun
        except Exception as e:
            print(f"  {config.name} failed: {e}")
            converged = False
            best_x = [0.0] * len(self.bounds)
            best_val = float('inf')

        elapsed = time.time() - start_time

        return OptimizerResult(
            optimizer_name=config.name,
            best_params=best_x,
            best_value=best_val,
            num_evaluations=self.eval_count,
            time_seconds=elapsed,
            converged=converged,
            strategy_name="speed_test",
            trial_id=0,
        )

    def run_config(self, config: SpeedConfig) -> OptimizerResult:
        """Run any configuration"""
        if config.optimizer_type == "direct":
            return self.run_direct_config(config)
        elif config.optimizer_type == "de":
            return self.run_de_config(config)
        else:
            raise ValueError(f"Unknown optimizer type: {config.optimizer_type}")


def run_speed_benchmark_single(
    config: SpeedConfig,
    bounds: List[Tuple[float, float]],
    trial_id: int = 0,
    seed: int = 42,
) -> OptimizerResult:
    """Run single speed benchmark (for parallel execution)"""
    # Create fresh objective for each trial
    objective = SyntheticPnLObjective(dim=len(bounds), noise_level=0.01, seed=seed + trial_id)
    benchmark = SpeedBenchmark(objective, bounds, seed=seed + trial_id)
    result = benchmark.run_config(config)
    result.trial_id = trial_id
    return result


def run_parallel_speed_benchmark(
    configs: List[SpeedConfig],
    bounds: List[Tuple[float, float]],
    num_trials: int = 5,
    max_workers: int = 4,
) -> pd.DataFrame:
    """Run speed benchmarks in parallel"""
    print(f"Running {len(configs)} configs × {num_trials} trials = {len(configs) * num_trials} benchmarks")
    print(f"Using {max_workers} parallel workers")
    print("=" * 80)

    all_results = []

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {}

        # Submit all tasks
        for config in configs:
            for trial_id in range(num_trials):
                future = executor.submit(
                    run_speed_benchmark_single,
                    config,
                    bounds,
                    trial_id,
                    42 + trial_id,
                )
                futures[future] = (config.name, trial_id)

        # Collect results
        completed = 0
        total = len(futures)

        for future in as_completed(futures):
            config_name, trial_id = futures[future]
            try:
                result = future.result()
                all_results.append(result)
                completed += 1
                print(f"✓ [{completed}/{total}] {config_name} trial {trial_id}: "
                      f"val={result.best_value:.6f} time={result.time_seconds:.3f}s "
                      f"evals={result.num_evaluations}")
            except Exception as e:
                print(f"✗ {config_name} trial {trial_id} failed: {e}")

    df = pd.DataFrame([r.to_dict() for r in all_results])
    return df


def analyze_speed_results(df: pd.DataFrame) -> pd.DataFrame:
    """Analyze speed benchmark results"""
    summary = df.groupby('optimizer_name').agg({
        'best_value': ['mean', 'std', 'min', 'max'],
        'time_seconds': ['mean', 'std', 'min', 'max'],
        'num_evaluations': ['mean', 'std'],
        'converged': 'mean',
    }).round(6)

    summary.columns = ['_'.join(col) for col in summary.columns]

    # Add speed metrics
    summary['speedup_vs_slowest'] = (
        summary['time_seconds_mean'].max() / summary['time_seconds_mean']
    ).round(2)

    summary['evals_per_second'] = (
        summary['num_evaluations_mean'] / summary['time_seconds_mean']
    ).round(1)

    # Quality metric: lower is better
    summary['quality_loss_pct'] = (
        (summary['best_value_mean'] - summary['best_value_mean'].min()) /
        abs(summary['best_value_mean'].min()) * 100
    ).round(2)

    # Speed/quality score: prefer fast AND good
    # Normalize both metrics to 0-1, then combine
    time_norm = (summary['time_seconds_mean'] - summary['time_seconds_mean'].min()) / (
        summary['time_seconds_mean'].max() - summary['time_seconds_mean'].min()
    )
    quality_norm = (summary['best_value_mean'] - summary['best_value_mean'].min()) / (
        summary['best_value_mean'].max() - summary['best_value_mean'].min()
    )

    summary['speed_quality_score'] = (
        (1 - time_norm) * 0.6 + (1 - quality_norm) * 0.4  # 60% speed, 40% quality
    ).round(4)

    summary = summary.sort_values('speed_quality_score', ascending=False)

    return summary


def print_speed_analysis(df: pd.DataFrame, mode: str):
    """Print detailed speed analysis"""
    print("\n" + "=" * 80)
    print(f"SPEED BENCHMARK RESULTS - {mode.upper()} MODE")
    print("=" * 80)

    summary = analyze_speed_results(df)

    print("\nFull Results:")
    print(summary.to_string())

    # Highlight top performers
    print("\n" + "=" * 80)
    print("TOP PERFORMERS")
    print("=" * 80)

    best_speed = summary['time_seconds_mean'].idxmin()
    print(f"\n⚡ FASTEST:")
    print(f"   {best_speed}")
    print(f"   Time: {summary.loc[best_speed, 'time_seconds_mean']:.4f}s")
    print(f"   Quality Loss: {summary.loc[best_speed, 'quality_loss_pct']:.2f}%")
    print(f"   Speedup: {summary.loc[best_speed, 'speedup_vs_slowest']:.2f}x")

    best_quality = summary['best_value_mean'].idxmin()
    print(f"\n💎 BEST QUALITY:")
    print(f"   {best_quality}")
    print(f"   Value: {summary.loc[best_quality, 'best_value_mean']:.6f}")
    print(f"   Time: {summary.loc[best_quality, 'time_seconds_mean']:.4f}s")

    best_overall = summary.index[0]
    print(f"\n🏆 BEST OVERALL (Speed+Quality):")
    print(f"   {best_overall}")
    print(f"   Score: {summary.loc[best_overall, 'speed_quality_score']:.4f}")
    print(f"   Time: {summary.loc[best_overall, 'time_seconds_mean']:.4f}s")
    print(f"   Quality Loss: {summary.loc[best_overall, 'quality_loss_pct']:.2f}%")
    print(f"   Speedup: {summary.loc[best_overall, 'speedup_vs_slowest']:.2f}x")

    # If DE parallel mode, show scaling
    if 'DE_' in df['optimizer_name'].iloc[0] and 'worker' in df['optimizer_name'].iloc[0]:
        print("\n" + "=" * 80)
        print("PARALLEL SCALING ANALYSIS")
        print("=" * 80)

        baseline = summary.loc['DE_sequential', 'time_seconds_mean']
        print(f"Baseline (1 worker): {baseline:.4f}s")

        for config_name in summary.index:
            if config_name == 'DE_sequential':
                continue
            time_val = summary.loc[config_name, 'time_seconds_mean']
            speedup = baseline / time_val
            efficiency = speedup / int(config_name.split('workers')[0].split('_')[-1]) * 100 if 'workers' in config_name else 0
            print(f"{config_name:20s}: {time_val:.4f}s ({speedup:.2f}x speedup, {efficiency:.1f}% efficiency)")

    return summary


def main():
    parser = argparse.ArgumentParser(description="Speed-focused optimization benchmark")
    parser.add_argument(
        '--mode',
        choices=['direct', 'de-parallel', 'de-speed', 'full'],
        default='full',
        help='Benchmark mode (default: full)'
    )
    parser.add_argument(
        '--num-trials',
        type=int,
        default=5,
        help='Number of trials per config (default: 5)'
    )
    parser.add_argument(
        '--workers',
        type=int,
        default=8,
        help='Number of parallel benchmark workers (default: 8)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='strategytraining/benchmark_results',
        help='Output directory'
    )

    args = parser.parse_args()

    # Create output dir
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Select configs based on mode
    if args.mode == 'direct':
        configs = DIRECT_CONFIGS
    elif args.mode == 'de-parallel':
        configs = DE_PARALLEL_CONFIGS
    elif args.mode == 'de-speed':
        configs = DE_SPEED_CONFIGS
    else:  # full
        configs = DIRECT_CONFIGS + DE_PARALLEL_CONFIGS + DE_SPEED_CONFIGS

    # Standard bounds for all tests
    bounds = [(-0.03, 0.03), (-0.03, 0.03)]

    # Run benchmarks
    start_time = time.time()
    results_df = run_parallel_speed_benchmark(
        configs,
        bounds,
        num_trials=args.num_trials,
        max_workers=args.workers,
    )
    total_time = time.time() - start_time

    # Analyze and print
    summary = print_speed_analysis(results_df, args.mode)

    print(f"\n\nTotal benchmark time: {total_time:.2f}s")

    # Save results
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    results_file = output_dir / f"speed_benchmark_{args.mode}_{timestamp}.parquet"
    summary_file = output_dir / f"speed_benchmark_{args.mode}_{timestamp}_summary.csv"

    results_df.to_parquet(results_file)
    summary.to_csv(summary_file)

    print(f"\nResults saved to:")
    print(f"  {results_file}")
    print(f"  {summary_file}")


if __name__ == "__main__":
    main()
