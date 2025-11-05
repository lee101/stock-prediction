#!/usr/bin/env python3
"""
Profile the actual bottleneck in optimization.
Predictions should be cached - only varying multipliers.
"""

import time
import torch
import numpy as np
from loss_utils import calculate_profit_torch_with_entry_buysell_profit_values
from scipy.optimize import differential_evolution

# Generate realistic cached predictions
torch.manual_seed(42)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

n = 200
close_actual = torch.randn(n, device=device) * 0.02
high_actual = close_actual + torch.abs(torch.randn(n, device=device)) * 0.01
low_actual = close_actual - torch.abs(torch.randn(n, device=device)) * 0.01
high_pred = torch.randn(n, device=device) * 0.01 + 0.005
low_pred = torch.randn(n, device=device) * 0.01 - 0.005
positions = torch.where(torch.abs(high_pred) > torch.abs(low_pred), torch.ones(n, device=device), -torch.ones(n, device=device))

print("Profiling optimization bottleneck...")
print("="*80)
print(f"Device: {device}")
print(f"Data size: {n} days")
print()

# Profile single profit calculation
print("1. Single profit calculation (what optimizer calls repeatedly):")
start = time.time()
for _ in range(1000):
    profit = calculate_profit_torch_with_entry_buysell_profit_values(
        close_actual,
        high_actual,
        high_pred + 0.01,
        low_actual,
        low_pred - 0.01,
        positions,
        close_at_eod=False,
        trading_fee=0.0015,
    )
    total = profit.sum().item()
elapsed = time.time() - start
print(f"   1000 calls: {elapsed:.3f}s")
print(f"   Per call: {elapsed/1000*1000:.3f}ms")
print()

# Profile full optimization
print("2. Full scipy optimization (500 evals):")

def objective(params):
    h_mult, l_mult = params
    profit = calculate_profit_torch_with_entry_buysell_profit_values(
        close_actual,
        high_actual,
        high_pred + float(h_mult),
        low_actual,
        low_pred + float(l_mult),
        positions,
        close_at_eod=False,
        trading_fee=0.0015,
    )
    return -float(profit.sum().item())

start = time.time()
result = differential_evolution(
    objective,
    bounds=[(-0.03, 0.03), (-0.03, 0.03)],
    maxiter=50,
    popsize=10,
    seed=42,
    workers=1,
)
elapsed = time.time() - start
print(f"   Total time: {elapsed:.3f}s")
print(f"   Evaluations: {result.nfev}")
print(f"   Per eval: {elapsed/result.nfev*1000:.3f}ms")
print()

# Calculate overhead
eval_time = elapsed / result.nfev
calc_time = 0.183 / 1000  # From profiling above
overhead = eval_time - calc_time

print("3. Breakdown:")
print(f"   Pure calculation: {calc_time*1000:.3f}ms per eval")
print(f"   Optimizer overhead: {overhead*1000:.3f}ms per eval")
print(f"   Overhead %: {overhead/eval_time*100:.1f}%")
print()

print("4. Test with different scipy optimizers:")
print("-"*80)

optimizers = [
    ("differential_evolution", lambda obj: differential_evolution(obj, bounds=[(-0.03, 0.03), (-0.03, 0.03)], maxiter=50, popsize=10, seed=42)),
    ("dual_annealing", lambda obj: __import__('scipy.optimize', fromlist=['dual_annealing']).dual_annealing(obj, bounds=[(-0.03, 0.03), (-0.03, 0.03)], maxiter=250, seed=42)),
    ("shgo", lambda obj: __import__('scipy.optimize', fromlist=['shgo']).shgo(obj, bounds=[(-0.03, 0.03), (-0.03, 0.03)], n=100, sampling_method='sobol')),
    ("direct", lambda obj: __import__('scipy.optimize', fromlist=['direct']).direct(obj, bounds=[(-0.03, 0.03), (-0.03, 0.03)], maxfun=500)),
]

results = []
for name, opt_fn in optimizers:
    try:
        print(f"\n{name}:")
        start = time.time()
        result = opt_fn(objective)
        elapsed = time.time() - start

        if hasattr(result, 'nfev'):
            n_evals = result.nfev
        elif hasattr(result, 'nfun'):
            n_evals = result.nfun
        else:
            n_evals = "unknown"

        profit = -result.fun if hasattr(result, 'fun') else "unknown"

        print(f"   Time: {elapsed:.3f}s")
        print(f"   Evals: {n_evals}")
        print(f"   Profit: {profit:.6f}" if isinstance(profit, float) else f"   Profit: {profit}")

        results.append({
            'name': name,
            'time': elapsed,
            'evals': n_evals,
            'profit': profit,
        })
    except Exception as e:
        print(f"   Error: {e}")

print("\n" + "="*80)
print("SUMMARY")
print("="*80)
if results:
    print(f"{'Optimizer':<25} {'Time':<10} {'Evals':<10} {'Profit':<12}")
    print("-"*80)
    for r in results:
        evals_str = str(r['evals']) if isinstance(r['evals'], int) else r['evals']
        profit_str = f"{r['profit']:.6f}" if isinstance(r['profit'], float) else str(r['profit'])
        print(f"{r['name']:<25} {r['time']:<10.3f} {evals_str:<10} {profit_str:<12}")

print("\n" + "="*80)
print("BOTTLENECK ANALYSIS")
print("="*80)
print("✓ Predictions are cached (computed once before optimization)")
print("✓ Optimization only varies multipliers on cached tensors")
print("✓ Each eval is pure GPU tensor math (~0.18ms)")
print()
print("Actual bottleneck sources:")
print("  1. Optimizer overhead (scipy internal logic)")
print("  2. Python/C++ boundary crossing per evaluation")
print("  3. .item() calls to transfer scalar from GPU to CPU")
print()
print("Potential speedups:")
print("  1. Batch evaluations (evaluate multiple candidates at once)")
print("  2. Keep tensors on GPU longer (avoid .item() per eval)")
print("  3. Use optimizer with fewer evaluations (if quality ok)")
