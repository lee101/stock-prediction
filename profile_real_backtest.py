#!/usr/bin/env python3
"""
Profile actual backtest with model inference to find true bottlenecks.
"""
import cProfile
import pstats
import sys
import time

print("Starting real backtest profiling...")
print("This will profile the actual run_single_simulation with model inference")

output_file = sys.argv[1] if len(sys.argv) > 1 else "real_backtest.prof"

print(f"Profiling to {output_file}...")
profiler = cProfile.Profile()
profiler.enable()

start = time.time()

# Import and run actual backtest
from backtest_test3_inline import backtest_forecasts

# Run backtest on a single stock with reduced simulations
results = backtest_forecasts("AAPL", num_simulations=5)

elapsed = time.time() - start

profiler.disable()
profiler.dump_stats(output_file)

print(f"\nBacktest completed in {elapsed:.2f}s")
print(f"Profile saved to {output_file}")
print("\nTop 30 functions by cumulative time:")
stats = pstats.Stats(output_file)
stats.strip_dirs()
stats.sort_stats('cumulative')
stats.print_stats(30)

print("\n" + "="*80)
print("Now run:")
print(f"  python convert_prof_to_svg.py {output_file}")
print(f"  python ~/code/dotfiles/flamegraph-analyzer/flamegraph_analyzer/main.py {output_file} -o real_backtest_analysis.md")
