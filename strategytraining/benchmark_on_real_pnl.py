#!/usr/bin/env python3
"""
Benchmark Optimizers on Real P&L Data

Tests optimization algorithms on actual strategy P&L maximization problems
using real historical data. This provides realistic performance comparisons.

Usage:
    # Benchmark on real strategy data
    python benchmark_on_real_pnl.py --num-symbols 5 --num-trials 3 --workers 8

    # Benchmark specific optimizer
    python benchmark_on_real_pnl.py --optimizers DIRECT OptunaTPE --workers 4

    # Quick test
    python benchmark_on_real_pnl.py --num-symbols 2 --num-trials 1 --workers 4
"""

import argparse
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import List, Optional, Tuple, Callable
import warnings

import numpy as np
import pandas as pd
import torch

# Add parent directory
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.optimization_utils import (
    _EntryExitObjective,
    _AlwaysOnObjective,
)
from strategytraining.benchmark_optimizers import (
    OptimizerBenchmark,
    BenchmarkConfig,
    OptimizerResult,
    analyze_benchmark_results,
)

warnings.filterwarnings('ignore')


def load_pnl_dataset(dataset_path: str = None) -> pd.DataFrame:
    """Load the strategy P&L dataset"""
    if dataset_path is None:
        # Find most recent dataset
        dataset_dir = Path("strategytraining/datasets")
        parquet_files = list(dataset_dir.glob("*_strategy_performance.parquet"))
        if not parquet_files:
            raise FileNotFoundError("No P&L dataset found in strategytraining/datasets/")
        dataset_path = sorted(parquet_files)[-1]

    print(f"Loading P&L data from: {dataset_path}")
    df = pd.read_parquet(dataset_path)
    print(f"Loaded {len(df)} strategy-window combinations")
    print(f"Symbols: {df['symbol'].nunique()}")
    print(f"Strategies: {df['strategy'].unique().tolist()}")
    return df


def create_pnl_objective_from_data(
    symbol: str,
    strategy: str,
    window_num: int,
    data_dir: str = "trainingdata/train",
) -> Optional[Callable]:
    """
    Create a real P&L objective function from historical data.

    This simulates the actual optimization problem: given predictions and
    actuals, find the best entry/exit multipliers to maximize P&L.
    """
    # Load symbol data
    csv_path = Path(data_dir) / f"{symbol}.csv"
    if not csv_path.exists():
        print(f"Warning: Data file not found for {symbol}")
        return None

    try:
        df = pd.read_csv(csv_path)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values('timestamp').reset_index(drop=True)

        # For this benchmark, we'll create synthetic predictions
        # In real usage, these would come from the actual model
        df['close_pred'] = df['Close'].pct_change().shift(-1)
        df['high_pred'] = df['High'].pct_change().shift(-1)
        df['low_pred'] = df['Low'].pct_change().shift(-1)

        # Create window data (7 days)
        window_size = 7 * 24  # Assuming hourly data
        start_idx = window_num * window_size
        end_idx = start_idx + window_size

        if end_idx > len(df):
            return None

        window_df = df.iloc[start_idx:end_idx].copy()

        # Convert to torch tensors
        close_actual = torch.tensor(window_df['close_pred'].fillna(0).values, dtype=torch.float32)
        high_actual = torch.tensor(window_df['high_pred'].fillna(0).values, dtype=torch.float32)
        low_actual = torch.tensor(window_df['low_pred'].fillna(0).values, dtype=torch.float32)

        # Create simple position signal (buy when prediction > 0)
        positions = torch.tensor(
            (window_df['close_pred'].fillna(0) > 0).astype(float).values,
            dtype=torch.float32
        )

        # Create objective using real optimization code
        high_pred = high_actual.clone()
        low_pred = low_actual.clone()

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

        return objective

    except Exception as e:
        print(f"Error creating objective for {symbol}: {e}")
        return None


def benchmark_real_pnl_optimization(
    symbol: str,
    strategy: str,
    window_num: int,
    trial_id: int,
    optimizers: List[str],
    config: BenchmarkConfig,
) -> List[OptimizerResult]:
    """Run benchmark on a real P&L optimization problem"""

    # Create objective
    objective_fn = create_pnl_objective_from_data(symbol, strategy, window_num)
    if objective_fn is None:
        print(f"Skipping {symbol} window {window_num}: no data")
        return []

    # Run benchmarks
    results = []
    problem_id = f"{symbol}_{strategy}_w{window_num}_t{trial_id}"

    for optimizer_name in optimizers:
        benchmark = OptimizerBenchmark(
            objective_fn=objective_fn,
            config=config,
            strategy_name=problem_id,
            trial_id=trial_id,
        )

        optimizer_methods = {
            'DIRECT': benchmark.run_direct,
            'DifferentialEvolution': benchmark.run_differential_evolution,
            'DualAnnealing': benchmark.run_dual_annealing,
            'SHGO': benchmark.run_shgo,
            'BasinHopping': benchmark.run_basinhopping,
            'GridSearch': benchmark.run_grid_search,
            'OptunaTPE': benchmark.run_optuna,
        }

        if optimizer_name in optimizer_methods:
            result = optimizer_methods[optimizer_name]()
            if result:
                result.strategy_name = problem_id
                results.append(result)

    return results


def run_parallel_real_pnl_benchmarks(
    num_symbols: int = 5,
    num_windows_per_symbol: int = 3,
    num_trials: int = 3,
    optimizers: Optional[List[str]] = None,
    max_workers: int = 4,
) -> pd.DataFrame:
    """
    Run benchmarks across multiple real P&L optimization problems in parallel.

    Args:
        num_symbols: Number of symbols to test
        num_windows_per_symbol: Number of time windows per symbol
        num_trials: Number of independent trials per (symbol, window)
        optimizers: List of optimizer names
        max_workers: Parallel workers

    Returns:
        DataFrame with all results
    """
    if optimizers is None:
        optimizers = ['DIRECT', 'DifferentialEvolution', 'DualAnnealing', 'OptunaTPE']

    # Load P&L dataset to get symbol list
    pnl_df = load_pnl_dataset()
    available_symbols = pnl_df['symbol'].unique()[:num_symbols]

    print(f"\nBenchmark Configuration:")
    print(f"  Symbols: {num_symbols} ({list(available_symbols)})")
    print(f"  Windows per symbol: {num_windows_per_symbol}")
    print(f"  Trials: {num_trials}")
    print(f"  Optimizers: {optimizers}")
    print(f"  Total problems: {num_symbols * num_windows_per_symbol * num_trials}")
    print(f"  Parallel workers: {max_workers}")
    print("=" * 80)

    # Create config
    config = BenchmarkConfig(
        bounds=[(-0.03, 0.03), (-0.03, 0.03)],
        maxiter=50,
        popsize=10,
        function_budget=500,
        seed=42,
    )

    all_results = []

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {}

        # Submit all benchmark tasks
        for symbol in available_symbols:
            for window_num in range(num_windows_per_symbol):
                for trial_id in range(num_trials):
                    future = executor.submit(
                        benchmark_real_pnl_optimization,
                        symbol,
                        "maxdiff",  # Use maxdiff strategy
                        window_num,
                        trial_id,
                        optimizers,
                        config,
                    )
                    futures[future] = (symbol, window_num, trial_id)

        # Collect results
        completed = 0
        total = len(futures)

        for future in as_completed(futures):
            symbol, window_num, trial_id = futures[future]
            try:
                results = future.result()
                all_results.extend(results)
                completed += 1

                if results:
                    best_result = min(results, key=lambda r: r.best_value)
                    print(f"✓ [{completed}/{total}] {symbol} w{window_num} t{trial_id}: "
                          f"best={best_result.optimizer_name} "
                          f"val={best_result.best_value:.6f}")
            except Exception as e:
                print(f"✗ {symbol} w{window_num} t{trial_id} failed: {e}")

    # Convert to DataFrame
    if not all_results:
        print("Warning: No results collected!")
        return pd.DataFrame()

    df = pd.DataFrame([r.to_dict() for r in all_results])
    return df


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark optimizers on real P&L data"
    )
    parser.add_argument(
        '--num-symbols',
        type=int,
        default=5,
        help='Number of symbols to test (default: 5)'
    )
    parser.add_argument(
        '--num-windows',
        type=int,
        default=3,
        help='Number of time windows per symbol (default: 3)'
    )
    parser.add_argument(
        '--num-trials',
        type=int,
        default=3,
        help='Number of trials per problem (default: 3)'
    )
    parser.add_argument(
        '--optimizers',
        nargs='+',
        default=None,
        help='Specific optimizers to test (default: DIRECT, DE, DualAnnealing, OptunaTPE)'
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

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Run benchmarks
    start_time = time.time()

    results_df = run_parallel_real_pnl_benchmarks(
        num_symbols=args.num_symbols,
        num_windows_per_symbol=args.num_windows,
        num_trials=args.num_trials,
        optimizers=args.optimizers,
        max_workers=args.workers,
    )

    total_time = time.time() - start_time

    if results_df.empty:
        print("No results to analyze!")
        return

    # Analyze results
    print("\n" + "=" * 80)
    print("REAL P&L BENCHMARK SUMMARY")
    print("=" * 80)
    summary = analyze_benchmark_results(results_df)
    print(summary)

    print(f"\nTotal benchmark time: {total_time:.2f}s")
    print(f"Average per problem: {total_time / len(results_df):.2f}s")

    # Additional analysis: best optimizer by problem
    best_by_problem = (
        results_df
        .sort_values('best_value')
        .groupby('strategy_name')
        .first()
        .reset_index()
    )

    optimizer_wins = best_by_problem['optimizer_name'].value_counts()
    print(f"\nOptimizer Wins (best P&L per problem):")
    for optimizer, wins in optimizer_wins.items():
        pct = 100 * wins / len(best_by_problem)
        print(f"  {optimizer}: {wins} wins ({pct:.1f}%)")

    # Save results
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    results_file = output_dir / f"real_pnl_benchmark_{timestamp}.parquet"
    summary_file = output_dir / f"real_pnl_benchmark_{timestamp}_summary.csv"
    wins_file = output_dir / f"real_pnl_benchmark_{timestamp}_wins.csv"

    results_df.to_parquet(results_file)
    summary.to_csv(summary_file)
    optimizer_wins.to_csv(wins_file)

    print(f"\nResults saved to:")
    print(f"  {results_file}")
    print(f"  {summary_file}")
    print(f"  {wins_file}")


if __name__ == "__main__":
    main()
