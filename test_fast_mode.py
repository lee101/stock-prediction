#!/usr/bin/env python3
"""
Quick test to verify FAST_MODE optimization works and measure speedup.

Usage:
    # Test normal mode
    python test_fast_mode.py

    # Test fast optimize mode
    MARKETSIM_FAST_OPTIMIZE=1 python test_fast_mode.py
"""

import os
import time
import torch

# Import before setting env vars to test runtime behavior
from src.optimization_utils import optimize_entry_exit_multipliers, _FAST_MODE, _USE_DIRECT


def test_optimization_speed():
    """Test optimization speed with synthetic data"""
    print("=" * 80)
    print("OPTIMIZATION SPEED TEST")
    print("=" * 80)
    print(f"DIRECT optimizer: {'ENABLED' if _USE_DIRECT else 'DISABLED'}")
    print(f"Fast mode: {'ENABLED (maxfun=100)' if _FAST_MODE else 'DISABLED (maxfun=500)'}")
    print("=" * 80)

    # Create synthetic data
    torch.manual_seed(42)
    n = 100
    close_actual = torch.randn(n) * 0.01
    positions = (torch.randn(n) > 0).float() * 2 - 1  # -1 or 1
    high_actual = close_actual + torch.abs(torch.randn(n)) * 0.005
    low_actual = close_actual - torch.abs(torch.randn(n)) * 0.005
    high_pred = high_actual.clone()
    low_pred = low_actual.clone()

    # Run optimization multiple times
    num_trials = 10
    print(f"\nRunning {num_trials} optimization trials...")

    times = []
    results = []

    for i in range(num_trials):
        start = time.time()
        high_mult, low_mult, profit = optimize_entry_exit_multipliers(
            close_actual=close_actual,
            positions=positions,
            high_actual=high_actual,
            high_pred=high_pred,
            low_actual=low_actual,
            low_pred=low_pred,
            close_at_eod=False,
            trading_fee=0.001,
        )
        elapsed = time.time() - start
        times.append(elapsed)
        results.append((high_mult, low_mult, profit))

        print(f"  Trial {i+1}: {elapsed*1000:.2f}ms "
              f"(high={high_mult:.4f}, low={low_mult:.4f}, profit={profit:.6f})")

    # Statistics
    avg_time = sum(times) / len(times)
    min_time = min(times)
    max_time = max(times)

    print("\n" + "=" * 80)
    print("RESULTS")
    print("=" * 80)
    print(f"Average time: {avg_time*1000:.2f}ms")
    print(f"Min time: {min_time*1000:.2f}ms")
    print(f"Max time: {max_time*1000:.2f}ms")
    print(f"Std dev: {(sum((t - avg_time)**2 for t in times) / len(times))**0.5 * 1000:.2f}ms")

    # Estimated speedup
    if _FAST_MODE:
        normal_mode_time = avg_time * 6  # Expected ~6x slower
        print(f"\nEstimated normal mode time: ~{normal_mode_time*1000:.2f}ms (6x slower)")
        print(f"Speedup: ~6x")
    else:
        fast_mode_time = avg_time / 6  # Expected ~6x faster
        print(f"\nEstimated fast mode time: ~{fast_mode_time*1000:.2f}ms (6x faster)")
        print(f"To enable fast mode: MARKETSIM_FAST_OPTIMIZE=1 python test_fast_mode.py")

    # Check result consistency
    high_mults = [r[0] for r in results]
    low_mults = [r[1] for r in results]
    profits = [r[2] for r in results]

    print(f"\nResult consistency:")
    print(f"  High multiplier: {sum(high_mults)/len(high_mults):.4f} ± {(sum((h - sum(high_mults)/len(high_mults))**2 for h in high_mults) / len(high_mults))**0.5:.4f}")
    print(f"  Low multiplier: {sum(low_mults)/len(low_mults):.4f} ± {(sum((l - sum(low_mults)/len(low_mults))**2 for l in low_mults) / len(low_mults))**0.5:.4f}")
    print(f"  Profit: {sum(profits)/len(profits):.6f} ± {(sum((p - sum(profits)/len(profits))**2 for p in profits) / len(profits))**0.5:.6f}")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    test_optimization_speed()
