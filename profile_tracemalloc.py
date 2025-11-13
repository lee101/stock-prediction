#!/usr/bin/env python3
"""Memory allocation profiler using tracemalloc"""
import os
import tracemalloc
import sys

os.environ['PAPER'] = '1'

tracemalloc.start()

# Import and run
import trade_stock_e2e
try:
    trade_stock_e2e.main()
except KeyboardInterrupt:
    pass

# Take snapshot
snapshot = tracemalloc.take_snapshot()
top_stats = snapshot.statistics('lineno')

print("\n=== Top 50 Memory Allocations ===")
for stat in top_stats[:50]:
    print(stat)

print("\n=== Grouped by file ===")
top_stats = snapshot.statistics('filename')
for stat in top_stats[:20]:
    print(f"{stat.count:>7} allocs, {stat.size/1024/1024:>8.1f} MiB - {stat.traceback.format()[0]}")
