#!/usr/bin/env python3
"""Compare Chronos2 eager vs compiled: speed + MAE on training data."""
import json
import subprocess
import sys
from pathlib import Path

def run_bench(compile_mode: bool, symbol: str = "BTCUSD"):
    """Run benchmark with/without compile"""
    cmd = [
        sys.executable,
        "benchmark_chronos2.py",
        "--symbols", symbol,
        "--context-lengths", "512",
        "--batch-sizes", "128",
        "--aggregations", "median",
        "--sample-counts", "0",
        "--scalers", "none",
        "--verbose",
    ]
    if compile_mode:
        cmd.append("--torch-compile")

    mode_str = "compiled" if compile_mode else "eager"
    print(f"\n{'='*70}")
    print(f"Running {mode_str.upper()} mode")
    print(f"{'='*70}\n")

    result = subprocess.run(cmd, capture_output=False, text=True)
    if result.returncode != 0:
        print(f"❌ {mode_str} failed")
        return None

    # Find latest benchmark file
    bench_dir = Path("chronos2_benchmarks") / symbol
    if not bench_dir.exists():
        print(f"❌ No benchmark output found for {symbol}")
        return None

    files = sorted(bench_dir.glob(f"{symbol}_chronos2_bench_*.json"))
    if not files:
        print(f"❌ No benchmark files found")
        return None

    with open(files[-1]) as f:
        data = json.load(f)

    return data[0] if data else None

def compare_results(eager_data, compiled_data):
    """Compare and display results"""
    print(f"\n{'='*70}")
    print("COMPARISON RESULTS")
    print(f"{'='*70}\n")

    e_test = eager_data["test"]
    c_test = compiled_data["test"]

    print("📊 MAE Comparison (test set):")
    print(f"  Eager MAE:    {e_test['price_mae']:.2f}")
    print(f"  Compiled MAE: {c_test['price_mae']:.2f}")
    mae_delta = abs(c_test['price_mae'] - e_test['price_mae'])
    mae_delta_pct = (mae_delta / e_test['price_mae'] * 100) if e_test['price_mae'] > 0 else 0
    print(f"  Δ MAE:        {mae_delta:.2f} ({mae_delta_pct:.2f}%)")

    print(f"\n⚡ Speed Comparison:")
    print(f"  Eager time:    {e_test['latency_s']:.3f}s")
    print(f"  Compiled time: {c_test['latency_s']:.3f}s")
    speedup = e_test['latency_s'] / c_test['latency_s'] if c_test['latency_s'] > 0 else 0
    print(f"  Speedup:       {speedup:.2f}x")

    print(f"\n📈 Decision:")
    if mae_delta_pct < 5.0 and speedup > 1.1:
        print(f"✅ COMPILED mode recommended ({speedup:.2f}x faster, MAE delta {mae_delta_pct:.2f}%)")
    elif mae_delta_pct >= 5.0:
        print(f"❌ EAGER mode required (MAE delta {mae_delta_pct:.2f}% too high)")
    else:
        print(f"🟡 EAGER mode preferred (speedup {speedup:.2f}x not significant)")

def main():
    symbol = sys.argv[1] if len(sys.argv) > 1 else "BTCUSD"

    print("Chronos2 Compile Benchmark")
    print(f"Symbol: {symbol}\n")

    eager_data = run_bench(False, symbol)
    if not eager_data:
        sys.exit(1)

    compiled_data = run_bench(True, symbol)
    if not compiled_data:
        sys.exit(1)

    compare_results(eager_data, compiled_data)

if __name__ == "__main__":
    main()
