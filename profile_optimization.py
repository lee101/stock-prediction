#!/usr/bin/env python3
"""
Profile the optimization bottlenecks in MaxDiff strategy evaluation.

Usage:
    python profile_optimization.py

This will:
1. Profile a single MaxDiff optimization run
2. Show detailed timing breakdown
3. Identify hotspots in torch operations
"""

import cProfile
import pstats
import io
import time
import torch
import numpy as np
from typing import Dict

# Import the optimization functions
from src.optimization_utils import optimize_entry_exit_multipliers, optimize_always_on_multipliers
from loss_utils import calculate_trading_profit_torch_with_entry_buysell


def create_mock_data(n_samples=100):
    """Create mock data similar to real backtesting data"""
    torch.manual_seed(42)

    close_actual = torch.randn(n_samples) * 0.01
    positions = torch.randint(-1, 2, (n_samples,)).float()
    high_actual = close_actual + torch.abs(torch.randn(n_samples)) * 0.005
    high_pred = torch.randn(n_samples) * 0.01
    low_actual = close_actual - torch.abs(torch.randn(n_samples)) * 0.005
    low_pred = torch.randn(n_samples) * 0.01

    buy_indicator = (positions > 0).float()
    sell_indicator = (positions < 0).float()

    return {
        'close_actual': close_actual,
        'positions': positions,
        'high_actual': high_actual,
        'high_pred': high_pred,
        'low_actual': low_actual,
        'low_pred': low_pred,
        'buy_indicator': buy_indicator,
        'sell_indicator': sell_indicator,
    }


def profile_optimization():
    """Profile the optimization process"""

    print("=" * 80)
    print("OPTIMIZATION PROFILING")
    print("=" * 80)

    # Create mock data
    data = create_mock_data(100)

    # Profile optimize_entry_exit_multipliers (MaxDiff strategy)
    print("\n1. Profiling optimize_entry_exit_multipliers (MaxDiff)...")
    print("-" * 80)

    profiler = cProfile.Profile()
    start_time = time.time()

    profiler.enable()
    high_mult, low_mult, profit = optimize_entry_exit_multipliers(
        data['close_actual'],
        data['positions'],
        data['high_actual'],
        data['high_pred'],
        data['low_actual'],
        data['low_pred'],
        close_at_eod=False,
        trading_fee=0.0005,
    )
    profiler.disable()

    elapsed = time.time() - start_time

    print(f"Elapsed time: {elapsed:.2f}s")
    print(f"Result: high_mult={high_mult:.4f}, low_mult={low_mult:.4f}, profit={profit:.4f}")

    # Print top 20 time-consuming functions
    s = io.StringIO()
    ps = pstats.Stats(profiler, stream=s).sort_stats('cumulative')
    ps.print_stats(20)
    print(s.getvalue())

    # Profile optimize_always_on_multipliers (MaxDiffAlwaysOn strategy)
    print("\n2. Profiling optimize_always_on_multipliers (MaxDiffAlwaysOn)...")
    print("-" * 80)

    profiler2 = cProfile.Profile()
    start_time = time.time()

    profiler2.enable()
    high_mult, low_mult, profit = optimize_always_on_multipliers(
        data['close_actual'],
        data['buy_indicator'],
        data['sell_indicator'],
        data['high_actual'],
        data['high_pred'],
        data['low_actual'],
        data['low_pred'],
        close_at_eod=False,
        trading_fee=0.0005,
        is_crypto=False,
    )
    profiler2.disable()

    elapsed = time.time() - start_time

    print(f"Elapsed time: {elapsed:.2f}s")
    print(f"Result: high_mult={high_mult:.4f}, low_mult={low_mult:.4f}, profit={profit:.4f}")

    # Print top 20 time-consuming functions
    s2 = io.StringIO()
    ps2 = pstats.Stats(profiler2, stream=s2).sort_stats('cumulative')
    ps2.print_stats(20)
    print(s2.getvalue())

    # Count function calls
    print("\n3. Function Call Counts")
    print("-" * 80)

    # Profile just one objective call to see torch overhead
    from src.optimization_utils import _EntryExitObjective

    obj = _EntryExitObjective(
        data['close_actual'],
        data['positions'],
        data['high_actual'],
        data['high_pred'],
        data['low_actual'],
        data['low_pred'],
        False,
        0.0005
    )

    profiler3 = cProfile.Profile()
    start_time = time.time()

    profiler3.enable()
    for _ in range(100):
        obj([0.01, -0.01])
    profiler3.disable()

    elapsed = time.time() - start_time

    print(f"100 objective calls: {elapsed:.3f}s ({elapsed/100*1000:.2f}ms per call)")

    s3 = io.StringIO()
    ps3 = pstats.Stats(profiler3, stream=s3).sort_stats('cumulative')
    ps3.print_stats(20)
    print(s3.getvalue())


def benchmark_torch_operations():
    """Benchmark individual torch operations to identify overhead"""

    print("\n4. Torch Operation Benchmarks")
    print("-" * 80)

    data = create_mock_data(100)
    n_iter = 10000

    operations = {
        'torch.view': lambda: data['close_actual'].view(-1),
        'torch.clamp': lambda: torch.clamp(data['high_pred'], 0, 10),
        'torch.clip': lambda: torch.clip(data['high_pred'], 0, 10),
        'torch.logical_or': lambda: torch.logical_or(data['buy_indicator'] > 0, data['sell_indicator'] > 0),
        'torch tensor add': lambda: data['close_actual'] + data['high_pred'],
        'torch tensor mul': lambda: data['close_actual'] * data['high_pred'],
        '.item()': lambda: data['close_actual'].sum().item(),
    }

    for name, op in operations.items():
        start = time.time()
        for _ in range(n_iter):
            op()
        elapsed = time.time() - start
        print(f"{name:20s}: {elapsed*1000:.2f}ms total, {elapsed/n_iter*1e6:.2f}µs per op")


if __name__ == '__main__':
    profile_optimization()
    benchmark_torch_operations()

    print("\n" + "=" * 80)
    print("RECOMMENDATIONS:")
    print("=" * 80)
    print("1. Check if MARKETSIM_FAST_OPTIMIZE=1 is set (reduces evals from 500 to 100)")
    print("2. Consider caching torch tensor views/reshapes")
    print("3. Reduce number of close_at_eod candidates if multiple are being tested")
    print("4. Profile with line_profiler for more detailed analysis")
    print("5. Check if data is on GPU when it should be on CPU (or vice versa)")
