#!/usr/bin/env python3
"""
Quick test to verify DIRECT optimizer is active and faster.
"""

import time
import torch
import os

# Test data
torch.manual_seed(42)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
n = 100
close_actual = torch.randn(n, device=device) * 0.02
high_actual = close_actual + torch.abs(torch.randn(n, device=device)) * 0.01
low_actual = close_actual - torch.abs(torch.randn(n, device=device)) * 0.01
high_pred = torch.randn(n, device=device) * 0.01 + 0.005
low_pred = torch.randn(n, device=device) * 0.01 - 0.005
positions = torch.where(torch.abs(high_pred) > torch.abs(low_pred),
                        torch.ones(n, device=device),
                        -torch.ones(n, device=device))

print("="*80)
print("QUICK OPTIMIZER TEST")
print("="*80)
print()

# Test 1: DIRECT
print("1. Testing DIRECT optimizer...")
os.environ['MARKETSIM_USE_DIRECT_OPTIMIZER'] = '1'

from src.optimization_utils import optimize_entry_exit_multipliers

start = time.time()
h1, l1, p1 = optimize_entry_exit_multipliers(
    close_actual, positions, high_actual, high_pred, low_actual, low_pred,
    maxiter=50, popsize=10, workers=1
)
time_direct = time.time() - start

print(f"   Time: {time_direct:.3f}s")
print(f"   Result: h={h1:.6f}, l={l1:.6f}, profit={p1:.6f}")
print()

# Test 2: DE (force reimport)
print("2. Testing differential_evolution...")
os.environ['MARKETSIM_USE_DIRECT_OPTIMIZER'] = '0'

import sys
if 'src.optimization_utils' in sys.modules:
    del sys.modules['src.optimization_utils']

from src.optimization_utils import optimize_entry_exit_multipliers

start = time.time()
h2, l2, p2 = optimize_entry_exit_multipliers(
    close_actual, positions, high_actual, high_pred, low_actual, low_pred,
    maxiter=50, popsize=10, workers=1
)
time_de = time.time() - start

print(f"   Time: {time_de:.3f}s")
print(f"   Result: h={h2:.6f}, l={l2:.6f}, profit={p2:.6f}")
print()

# Compare
print("="*80)
print("RESULTS")
print("="*80)
speedup = time_de / time_direct if time_direct > 0 else 0
print(f"DIRECT:  {time_direct:.3f}s")
print(f"DE:      {time_de:.3f}s")
print(f"Speedup: {speedup:.2f}x")
print()

profit_diff = abs(p1 - p2) / abs(p2) * 100 if p2 != 0 else 0
print(f"Profit quality:")
print(f"  DIRECT: {p1:.6f}")
print(f"  DE:     {p2:.6f}")
print(f"  Diff:   {profit_diff:.2f}%")
print()

if speedup > 1.2:
    print("✓ DIRECT is significantly faster!")
elif speedup > 0.9:
    print("≈ Similar performance")
else:
    print("⚠️  DIRECT is slower (unexpected)")

if profit_diff < 5:
    print("✓ Quality is similar")
else:
    print("⚠️  Quality differs")
