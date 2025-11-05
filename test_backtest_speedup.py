#!/usr/bin/env python3
"""
Test backtest speedup with Nevergrad optimizer.
Runs a small backtest to measure performance improvement.
"""

import time
import sys
from backtest_test3_inline import backtest_forecasts

def test_speedup(symbol="ETHUSD", num_sims=10):
    """Run backtest with small number of sims to test speedup"""
    print(f"Testing backtest speedup for {symbol} with {num_sims} simulations...")
    print("="*80)

    start = time.time()
    results_df = backtest_forecasts(symbol, num_simulations=num_sims)
    elapsed = time.time() - start

    print("\n" + "="*80)
    print(f"Completed {num_sims} simulations in {elapsed:.2f}s")
    print(f"Average: {elapsed/num_sims:.3f}s per simulation")
    print(f"\nEstimated time for 70 simulations: {(elapsed/num_sims)*70:.1f}s")
    print("\nResults shape:", results_df.shape)

    return elapsed, results_df

if __name__ == "__main__":
    symbol = sys.argv[1] if len(sys.argv) > 1 else "ETHUSD"
    num_sims = int(sys.argv[2]) if len(sys.argv) > 2 else 10

    elapsed, results = test_speedup(symbol, num_sims)
