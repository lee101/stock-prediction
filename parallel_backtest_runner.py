#!/usr/bin/env python3
"""
Parallel backtest runner for multiple symbols.
Each symbol's backtest runs in a separate process to fully utilize CPU cores.
"""

import argparse
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import List, Tuple
import pandas as pd


def run_backtest_for_symbol(symbol: str, num_simulations: int = 70) -> Tuple[str, float, pd.DataFrame]:
    """Run backtest for a single symbol (executed in separate process)"""
    import time
    from backtest_test3_inline import backtest_forecasts

    print(f"Starting backtest for {symbol}...")
    start = time.time()

    try:
        results = backtest_forecasts(symbol, num_simulations=num_simulations)
        elapsed = time.time() - start
        print(f"✓ {symbol} completed in {elapsed:.1f}s")
        return (symbol, elapsed, results)
    except Exception as e:
        elapsed = time.time() - start
        print(f"✗ {symbol} failed after {elapsed:.1f}s: {e}")
        return (symbol, elapsed, pd.DataFrame())


def run_parallel_backtests(
    symbols: List[str],
    num_simulations: int = 70,
    max_workers: int = 4
) -> dict:
    """
    Run backtests for multiple symbols in parallel.

    Args:
        symbols: List of symbols to backtest
        num_simulations: Number of simulations per symbol
        max_workers: Number of parallel processes (default 4)

    Returns:
        dict: {symbol: (elapsed_time, results_df)}
    """
    print(f"Running backtests for {len(symbols)} symbols in parallel")
    print(f"Max workers: {max_workers}")
    print(f"Simulations per symbol: {num_simulations}")
    print("="*80)

    start_time = time.time()
    results = {}

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_symbol = {
            executor.submit(run_backtest_for_symbol, symbol, num_simulations): symbol
            for symbol in symbols
        }

        # Collect results as they complete
        for future in as_completed(future_to_symbol):
            symbol, elapsed, df = future.result()
            results[symbol] = (elapsed, df)

    total_time = time.time() - start_time

    # Print summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    successful = sum(1 for _, df in results.values() if not df.empty)
    print(f"Completed: {successful}/{len(symbols)} symbols")
    print(f"Total time: {total_time:.1f}s")
    print(f"Avg per symbol: {total_time/len(symbols):.1f}s")

    if successful > 1:
        total_sequential = sum(elapsed for elapsed, _ in results.values())
        speedup = total_sequential / total_time
        print(f"Sequential would take: {total_sequential:.1f}s")
        print(f"Speedup: {speedup:.2f}x")

    print("\nIndividual times:")
    for symbol in symbols:
        if symbol in results:
            elapsed, df = results[symbol]
            status = "✓" if not df.empty else "✗"
            print(f"  {status} {symbol}: {elapsed:.1f}s")

    return results


def main():
    parser = argparse.ArgumentParser(description="Run parallel backtests")
    parser.add_argument(
        "symbols",
        nargs="+",
        help="Symbols to backtest (e.g., ETHUSD BTCUSD TSLA)"
    )
    parser.add_argument(
        "--num-simulations",
        type=int,
        default=70,
        help="Number of simulations per symbol (default: 70)"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of parallel workers (default: 4)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="backtest_results",
        help="Directory to save results (default: backtest_results)"
    )

    args = parser.parse_args()

    results = run_parallel_backtests(
        args.symbols,
        num_simulations=args.num_simulations,
        max_workers=args.workers
    )

    # Optionally save results
    if args.output_dir:
        from pathlib import Path
        output_path = Path(args.output_dir)
        output_path.mkdir(exist_ok=True)

        for symbol, (elapsed, df) in results.items():
            if not df.empty:
                csv_file = output_path / f"{symbol}_backtest.csv"
                df.to_csv(csv_file, index=False)
                print(f"Saved {symbol} results to {csv_file}")


if __name__ == "__main__":
    # Example usage if run without args
    import sys
    if len(sys.argv) == 1:
        print("Example: Run backtests for 3 symbols in parallel")
        print("python parallel_backtest_runner.py ETHUSD BTCUSD TSLA --workers 3 --num-simulations 10")
        print()
        sys.exit(1)

    main()
