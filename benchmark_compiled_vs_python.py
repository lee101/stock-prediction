"""Compare performance: Pure Python vs mypyc compiled.

This script runs the same benchmarks on both versions to measure speedup.
"""

import importlib.util
import os
import sys
import time
from pathlib import Path

import numpy as np


def load_module(name, path):
    """Load a module from a specific path."""
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def benchmark_function(func, *args, iterations=10000, warmup=1000):
    """Benchmark a function."""
    for _ in range(warmup):
        func(*args)

    start = time.perf_counter()
    for _ in range(iterations):
        func(*args)
    end = time.perf_counter()

    return (end - start) / iterations * 1_000_000


def main():
    """Run comparison benchmark."""
    print("=" * 80)
    print("PERFORMANCE COMPARISON: Pure Python vs mypyc Compiled")
    print("=" * 80)

    # Load both versions
    print("\n📦 Loading modules...")
    print("  Python: src/env_parsing.py")
    print("  Compiled: src/env_parsing.cpython-312-x86_64-linux-gnu.so")

    py_ep = load_module("env_parsing_py", "src/env_parsing.py")
    so_ep = load_module("env_parsing_so", "src/env_parsing.cpython-312-x86_64-linux-gnu.so")

    py_pc = load_module("price_calculations_py", "src/price_calculations.py")
    so_pc = load_module("price_calculations_so", "src/price_calculations.cpython-312-x86_64-linux-gnu.so")

    py_sp = load_module("strategy_price_lookup_py", "src/strategy_price_lookup.py")
    so_sp = load_module("strategy_price_lookup_so", "src/strategy_price_lookup.cpython-312-x86_64-linux-gnu.so")

    # Set up test data
    os.environ["TEST_BOOL"] = "1"
    os.environ["TEST_INT"] = "42"
    os.environ["TEST_FLOAT"] = "3.14"

    size = 1000
    close_vals = np.random.uniform(90, 110, size)
    high_vals = close_vals * np.random.uniform(1.0, 1.1, size)
    low_vals = close_vals * np.random.uniform(0.9, 1.0, size)

    data = {
        "maxdiffprofit_low_price": 95.0,
        "maxdiffprofit_high_price": 105.0,
    }

    print("\n" + "=" * 80)
    print("BENCHMARK RESULTS")
    print("=" * 80)

    results = []

    # env_parsing benchmarks
    print("\n📊 env_parsing.parse_bool_env")
    py_time = benchmark_function(py_ep.parse_bool_env, "TEST_BOOL", iterations=50000)
    so_time = benchmark_function(so_ep.parse_bool_env, "TEST_BOOL", iterations=50000)
    speedup = py_time / so_time
    results.append(("parse_bool_env", py_time, so_time, speedup))
    print(f"  Python:   {py_time:.3f} μs/call")
    print(f"  Compiled: {so_time:.3f} μs/call")
    print(f"  Speedup:  {speedup:.2f}x {'🚀' if speedup > 1.2 else '✓'}")

    print("\n📊 env_parsing.parse_int_env")
    py_time = benchmark_function(py_ep.parse_int_env, "TEST_INT", iterations=50000)
    so_time = benchmark_function(so_ep.parse_int_env, "TEST_INT", iterations=50000)
    speedup = py_time / so_time
    results.append(("parse_int_env", py_time, so_time, speedup))
    print(f"  Python:   {py_time:.3f} μs/call")
    print(f"  Compiled: {so_time:.3f} μs/call")
    print(f"  Speedup:  {speedup:.2f}x {'🚀' if speedup > 1.2 else '✓'}")

    # price_calculations benchmarks
    print("\n📊 price_calculations.compute_close_to_extreme_movements")
    py_time = benchmark_function(
        py_pc.compute_close_to_extreme_movements,
        close_vals,
        high_vals,
        low_vals,
        iterations=10000,
    )
    so_time = benchmark_function(
        so_pc.compute_close_to_extreme_movements,
        close_vals,
        high_vals,
        low_vals,
        iterations=10000,
    )
    speedup = py_time / so_time
    results.append(("compute_close_to_extreme_movements", py_time, so_time, speedup))
    print(f"  Python:   {py_time:.3f} μs/call")
    print(f"  Compiled: {so_time:.3f} μs/call")
    print(f"  Speedup:  {speedup:.2f}x {'🚀' if speedup > 1.2 else '✓'}")

    print("\n📊 price_calculations.safe_price_ratio")
    py_time = benchmark_function(
        py_pc.safe_price_ratio,
        high_vals,
        low_vals,
        iterations=10000,
    )
    so_time = benchmark_function(
        so_pc.safe_price_ratio,
        high_vals,
        low_vals,
        iterations=10000,
    )
    speedup = py_time / so_time
    results.append(("safe_price_ratio", py_time, so_time, speedup))
    print(f"  Python:   {py_time:.3f} μs/call")
    print(f"  Compiled: {so_time:.3f} μs/call")
    print(f"  Speedup:  {speedup:.2f}x {'🚀' if speedup > 1.2 else '✓'}")

    # strategy_price_lookup benchmarks
    print("\n📊 strategy_price_lookup.get_entry_price")
    py_time = benchmark_function(
        py_sp.get_entry_price,
        data,
        "maxdiff",
        "buy",
        iterations=100000,
    )
    so_time = benchmark_function(
        so_sp.get_entry_price,
        data,
        "maxdiff",
        "buy",
        iterations=100000,
    )
    speedup = py_time / so_time
    results.append(("get_entry_price", py_time, so_time, speedup))
    print(f"  Python:   {py_time:.3f} μs/call")
    print(f"  Compiled: {so_time:.3f} μs/call")
    print(f"  Speedup:  {speedup:.2f}x {'🚀' if speedup > 1.2 else '✓'}")

    print("\n📊 strategy_price_lookup.is_limit_order_strategy")
    py_time = benchmark_function(
        py_sp.is_limit_order_strategy,
        "maxdiff",
        iterations=100000,
    )
    so_time = benchmark_function(
        so_sp.is_limit_order_strategy,
        "maxdiff",
        iterations=100000,
    )
    speedup = py_time / so_time
    results.append(("is_limit_order_strategy", py_time, so_time, speedup))
    print(f"  Python:   {py_time:.3f} μs/call")
    print(f"  Compiled: {so_time:.3f} μs/call")
    print(f"  Speedup:  {speedup:.2f}x {'🚀' if speedup > 1.2 else '✓'}")

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    avg_speedup = sum(r[3] for r in results) / len(results)
    max_speedup = max(r[3] for r in results)
    min_speedup = min(r[3] for r in results)

    print(f"\nFunctions benchmarked: {len(results)}")
    print(f"Average speedup: {avg_speedup:.2f}x")
    print(f"Max speedup: {max_speedup:.2f}x ({[r[0] for r in results if r[3] == max_speedup][0]})")
    print(f"Min speedup: {min_speedup:.2f}x ({[r[0] for r in results if r[3] == min_speedup][0]})")

    improvements = [r for r in results if r[3] > 1.1]
    print(f"\nFunctions with >1.1x speedup: {len(improvements)}/{len(results)}")

    if avg_speedup > 1.5:
        print("\n🚀 Significant performance improvement from mypyc!")
    elif avg_speedup > 1.1:
        print("\n✅ Moderate performance improvement from mypyc")
    else:
        print("\n⚠️  Minimal speedup - these functions may be I/O bound or call numpy")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
