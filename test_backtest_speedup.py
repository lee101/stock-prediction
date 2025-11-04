#!/usr/bin/env python
"""
Test backtest speedup optimizations.

Compares original vs optimized inference timing on 10 simulations.
"""

import sys
import time
import torch
from pathlib import Path

# Add repo to path
repo_root = Path(__file__).resolve().parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

# Import after path setup
import backtest_test3_inline as bt

def run_timed_backtest(symbol: str, num_sims: int = 10) -> dict:
    """Run backtest and measure timing."""
    print(f"\n{'='*80}")
    print(f"Running {num_sims} simulations for {symbol}...")
    print(f"{'='*80}")

    start_time = time.time()

    # Run backtest (this will use current implementation)
    try:
        bt.backtest_forecasts(symbol, num_simulations=num_sims)
        elapsed = time.time() - start_time

        print(f"\n✓ Completed in {elapsed:.2f} seconds")
        print(f"  Average per simulation: {elapsed/num_sims:.2f} seconds")

        return {
            "success": True,
            "total_time": elapsed,
            "per_sim_time": elapsed / num_sims,
            "num_sims": num_sims,
        }
    except Exception as e:
        elapsed = time.time() - start_time
        print(f"\n✗ Failed after {elapsed:.2f} seconds: {e}")
        return {
            "success": False,
            "total_time": elapsed,
            "error": str(e),
        }


def main():
    symbol = "ETHUSD"
    num_sims = 10

    print("\n" + "="*80)
    print("BACKTEST SPEEDUP TEST")
    print("="*80)
    print(f"Symbol: {symbol}")
    print(f"Simulations: {num_sims}")
    print(f"Device: {'cuda' if torch.cuda.is_available() else 'cpu'}")

    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # Run baseline
    print("\n" + "="*80)
    print("BASELINE (current implementation)")
    print("="*80)
    baseline = run_timed_backtest(symbol, num_sims)

    if not baseline["success"]:
        print("\n✗ Baseline failed, cannot compare")
        return 1

    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"Baseline: {baseline['total_time']:.2f}s total, {baseline['per_sim_time']:.2f}s per sim")
    print(f"\nNow apply optimizations to backtest_test3_inline.py and rerun to compare!")

    return 0


if __name__ == "__main__":
    sys.exit(main())
