#!/usr/bin/env python3
"""
Example of parallelizing backtest simulations using multiprocessing.
Shows how to run multiple optimizations in parallel for massive speedup.
"""

import time
import numpy as np
from concurrent.futures import ProcessPoolExecutor
from src.optimization_utils import optimize_entry_exit_multipliers


def run_single_simulation(sim_id: int) -> dict:
    """Simulate one backtest with optimization"""
    import torch

    # Simulate different data for each sim (in reality, different time windows)
    np.random.seed(42 + sim_id)
    torch.manual_seed(42 + sim_id)

    n = 100
    close_actual = torch.randn(n) * 0.02
    high_actual = close_actual + torch.abs(torch.randn(n)) * 0.01
    low_actual = close_actual - torch.abs(torch.randn(n)) * 0.01
    high_pred = torch.randn(n) * 0.01 + 0.005
    low_pred = torch.randn(n) * 0.01 - 0.005
    positions = torch.where(high_pred > low_pred, torch.ones(n), -torch.ones(n))

    # Optimize (this is what takes time)
    h_mult, l_mult, profit = optimize_entry_exit_multipliers(
        close_actual, positions, high_actual, high_pred, low_actual, low_pred,
        maxiter=30,  # Reduced for speed
        popsize=8,
        workers=1,  # Each process handles one sim
    )

    return {
        'sim_id': sim_id,
        'h_mult': h_mult,
        'l_mult': l_mult,
        'profit': profit,
    }


def main():
    n_simulations = 70

    print(f"Running {n_simulations} simulations...")
    print("="*80)

    # Sequential (current approach)
    print("\n1. Sequential execution:")
    start = time.time()
    results_seq = []
    for i in range(n_simulations):
        result = run_single_simulation(i)
        results_seq.append(result)
    seq_time = time.time() - start
    print(f"   Time: {seq_time:.2f}s")

    # Parallel (proposed approach)
    print("\n2. Parallel execution (8 workers):")
    start = time.time()
    with ProcessPoolExecutor(max_workers=8) as executor:
        results_par = list(executor.map(run_single_simulation, range(n_simulations)))
    par_time = time.time() - start
    print(f"   Time: {par_time:.2f}s")

    # Summary
    print("\n" + "="*80)
    print(f"Speedup: {seq_time / par_time:.2f}x")
    print(f"Estimated time for your backtests:")
    print(f"  Current (sequential): ~{seq_time:.1f}s per run")
    print(f"  With parallelization: ~{par_time:.1f}s per run")
    print("\nTo use: Modify backtest_forecasts() to use ProcessPoolExecutor")


if __name__ == '__main__':
    main()
