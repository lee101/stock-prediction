"""Benchmark script to compare Python vs mypyc-compiled performance.

This script measures the performance improvement from mypyc compilation.
"""

import os
import sys
import time
from pathlib import Path

import numpy as np


def benchmark_function(func, *args, iterations=10000, warmup=1000):
    """Benchmark a function with warmup and multiple iterations."""
    # Warmup
    for _ in range(warmup):
        func(*args)

    # Actual benchmark
    start = time.perf_counter()
    for _ in range(iterations):
        func(*args)
    end = time.perf_counter()

    return (end - start) / iterations * 1_000_000  # Return microseconds per call


def benchmark_env_parsing():
    """Benchmark env_parsing module."""
    import src.env_parsing as ep

    results = {}

    # Set up environment
    os.environ["TEST_BOOL"] = "1"
    os.environ["TEST_INT"] = "42"
    os.environ["TEST_FLOAT"] = "3.14"

    print("\n📊 Benchmarking env_parsing module...")

    # Benchmark parse_bool_env
    time_us = benchmark_function(ep.parse_bool_env, "TEST_BOOL", iterations=50000)
    results["parse_bool_env"] = time_us
    print(f"  parse_bool_env:  {time_us:.3f} μs/call")

    # Benchmark parse_int_env
    time_us = benchmark_function(ep.parse_int_env, "TEST_INT", iterations=50000)
    results["parse_int_env"] = time_us
    print(f"  parse_int_env:   {time_us:.3f} μs/call")

    # Benchmark parse_float_env
    time_us = benchmark_function(ep.parse_float_env, "TEST_FLOAT", iterations=50000)
    results["parse_float_env"] = time_us
    print(f"  parse_float_env: {time_us:.3f} μs/call")

    return results


def benchmark_price_calculations():
    """Benchmark price_calculations module."""
    import src.price_calculations as pc

    results = {}

    # Create test data
    size = 1000
    close_vals = np.random.uniform(90, 110, size)
    high_vals = close_vals * np.random.uniform(1.0, 1.1, size)
    low_vals = close_vals * np.random.uniform(0.9, 1.0, size)

    print("\n📊 Benchmarking price_calculations module...")

    # Benchmark compute_close_to_extreme_movements
    time_us = benchmark_function(
        pc.compute_close_to_extreme_movements,
        close_vals,
        high_vals,
        low_vals,
        iterations=10000,
    )
    results["compute_close_to_extreme_movements"] = time_us
    print(f"  compute_close_to_extreme_movements: {time_us:.3f} μs/call")

    # Benchmark compute_price_range_pct
    time_us = benchmark_function(
        pc.compute_price_range_pct,
        high_vals,
        low_vals,
        close_vals,
        iterations=10000,
    )
    results["compute_price_range_pct"] = time_us
    print(f"  compute_price_range_pct:            {time_us:.3f} μs/call")

    # Benchmark safe_price_ratio
    time_us = benchmark_function(
        pc.safe_price_ratio,
        high_vals,
        low_vals,
        iterations=10000,
    )
    results["safe_price_ratio"] = time_us
    print(f"  safe_price_ratio:                   {time_us:.3f} μs/call")

    return results


def benchmark_strategy_price_lookup():
    """Benchmark strategy_price_lookup module."""
    import src.strategy_price_lookup as sp

    results = {}

    # Create test data
    data = {
        "maxdiffprofit_low_price": 95.0,
        "maxdiffprofit_high_price": 105.0,
        "predicted_low": 94.0,
        "predicted_high": 106.0,
    }

    print("\n📊 Benchmarking strategy_price_lookup module...")

    # Benchmark get_entry_price
    time_us = benchmark_function(
        sp.get_entry_price,
        data,
        "maxdiff",
        "buy",
        iterations=100000,
    )
    results["get_entry_price"] = time_us
    print(f"  get_entry_price:      {time_us:.3f} μs/call")

    # Benchmark get_takeprofit_price
    time_us = benchmark_function(
        sp.get_takeprofit_price,
        data,
        "maxdiff",
        "buy",
        iterations=100000,
    )
    results["get_takeprofit_price"] = time_us
    print(f"  get_takeprofit_price: {time_us:.3f} μs/call")

    # Benchmark is_limit_order_strategy
    time_us = benchmark_function(
        sp.is_limit_order_strategy,
        "maxdiff",
        iterations=100000,
    )
    results["is_limit_order_strategy"] = time_us
    print(f"  is_limit_order_strategy: {time_us:.3f} μs/call")

    return results


def main():
    """Run all benchmarks and display results."""
    print("=" * 70)
    print("MYPYC COMPILATION PERFORMANCE BENCHMARK")
    print("=" * 70)

    # Check if we're using compiled versions
    import src.env_parsing as ep

    is_compiled = hasattr(ep, "__mypyc__")
    print(f"\nUsing: {'✅ COMPILED (mypyc)' if is_compiled else '🐍 Pure Python'}")
    print(f"Python version: {sys.version.split()[0]}")

    if not is_compiled:
        print("\n⚠️  WARNING: Modules are not compiled!")
        print("To compile, run: python setup_mypyc.py build_ext --inplace")
        print()

    # Run benchmarks
    results = {}
    results["env_parsing"] = benchmark_env_parsing()
    results["price_calculations"] = benchmark_price_calculations()
    results["strategy_price_lookup"] = benchmark_strategy_price_lookup()

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    all_times = []
    for module, timings in results.items():
        for func, time_us in timings.items():
            all_times.append(time_us)

    avg_time = sum(all_times) / len(all_times)
    print(f"\nAverage function call time: {avg_time:.3f} μs")
    print(f"Total functions benchmarked: {len(all_times)}")

    if is_compiled:
        print("\n✅ Modules compiled with mypyc (opt_level=3)")
        print("   Expected speedup: 2-4x for numeric operations")
        print("   Expected speedup: 1.5-2x for string/logic operations")
    else:
        print("\n🐍 Running pure Python (no compilation)")
        print("   Compile with: python setup_mypyc.py build_ext --inplace")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
