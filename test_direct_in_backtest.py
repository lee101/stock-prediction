#!/usr/bin/env python3
"""
Test that DIRECT optimizer is actually being used in backtest
and measure real speedup.
"""

import time
import os
import sys

# Test with DIRECT enabled (default)
print("="*80)
print("TEST 1: With DIRECT optimizer (default)")
print("="*80)

os.environ['MARKETSIM_USE_DIRECT_OPTIMIZER'] = '1'
os.environ['MARKETSIM_FAST_SIMULATE'] = '1'  # Use 35 sims for speed

# Force reimport to pick up env var
if 'backtest_test3_inline' in sys.modules:
    del sys.modules['backtest_test3_inline']
if 'src.optimization_utils' in sys.modules:
    del sys.modules['src.optimization_utils']

from backtest_test3_inline import backtest_forecasts

start = time.time()
result_direct = backtest_forecasts("ETHUSD", num_simulations=10)
time_direct = time.time() - start

print(f"\n✓ Completed with DIRECT: {time_direct:.1f}s")
print(f"  Results shape: {result_direct.shape}")
print()

# Test with differential_evolution
print("="*80)
print("TEST 2: With differential_evolution (old)")
print("="*80)

os.environ['MARKETSIM_USE_DIRECT_OPTIMIZER'] = '0'

# Force reimport
if 'backtest_test3_inline' in sys.modules:
    del sys.modules['backtest_test3_inline']
if 'src.optimization_utils' in sys.modules:
    del sys.modules['src.optimization_utils']

from backtest_test3_inline import backtest_forecasts

start = time.time()
result_de = backtest_forecasts("ETHUSD", num_simulations=10)
time_de = time.time() - start

print(f"\n✓ Completed with DE: {time_de:.1f}s")
print(f"  Results shape: {result_de.shape}")
print()

# Compare
print("="*80)
print("COMPARISON")
print("="*80)
print(f"DIRECT time:   {time_direct:.1f}s")
print(f"DE time:       {time_de:.1f}s")

if time_de > 0:
    speedup = time_de / time_direct
    print(f"Speedup:       {speedup:.2f}x")

    if speedup > 1.1:
        print("\n✓ DIRECT is faster!")
    elif speedup < 0.9:
        print("\n⚠️  DIRECT is slower (unexpected)")
    else:
        print("\n≈ Similar performance")
else:
    print("Could not compute speedup")

# Compare results
print("\n" + "="*80)
print("RESULTS QUALITY")
print("="*80)

if not result_direct.empty and not result_de.empty:
    cols = ['simple_strategy_return', 'maxdiff_return']
    for col in cols:
        if col in result_direct.columns and col in result_de.columns:
            direct_val = result_direct[col].mean()
            de_val = result_de[col].mean()
            diff_pct = abs(direct_val - de_val) / abs(de_val) * 100 if de_val != 0 else 0

            print(f"{col}:")
            print(f"  DIRECT: {direct_val:.6f}")
            print(f"  DE:     {de_val:.6f}")
            print(f"  Diff:   {diff_pct:.2f}%")

print("\nNote: Results should be similar in quality (within ~5%)")
