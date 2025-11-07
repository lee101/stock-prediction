#!/usr/bin/env python3
"""
Profile optimization tests and generate flamegraph data.
"""
import cProfile
import pstats
import sys
import torch
from src.optimization_utils import optimize_entry_exit_multipliers, optimize_always_on_multipliers

def run_optimization_test():
    """Run a representative optimization test."""
    torch.manual_seed(42)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    n = 200
    close_actual = torch.randn(n, device=device) * 0.02
    high_actual = close_actual + torch.abs(torch.randn(n, device=device)) * 0.01
    low_actual = close_actual - torch.abs(torch.randn(n, device=device)) * 0.01
    high_pred = torch.randn(n, device=device) * 0.01 + 0.005
    low_pred = torch.randn(n, device=device) * 0.01 - 0.005
    positions = torch.where(
        torch.abs(high_pred) > torch.abs(low_pred),
        torch.ones(n, device=device),
        -torch.ones(n, device=device)
    )

    print(f"Running optimization test on {device}...")

    # Profile entry/exit optimization
    h_mult, l_mult, profit = optimize_entry_exit_multipliers(
        close_actual,
        positions,
        high_actual,
        high_pred,
        low_actual,
        low_pred,
        close_at_eod=False,
        trading_fee=0.0015,
    )
    print(f"Entry/Exit: h={h_mult:.4f} l={l_mult:.4f} profit={profit:.4f}")

    # Profile always-on optimization
    buy_indicator = positions == 1
    sell_indicator = positions == -1

    h_mult2, l_mult2, profit2 = optimize_always_on_multipliers(
        close_actual,
        buy_indicator,
        sell_indicator,
        high_actual,
        high_pred,
        low_actual,
        low_pred,
        close_at_eod=False,
        trading_fee=0.0015,
        is_crypto=False,
    )
    print(f"AlwaysOn: h={h_mult2:.4f} l={l_mult2:.4f} profit={profit2:.4f}")

if __name__ == "__main__":
    output_file = sys.argv[1] if len(sys.argv) > 1 else "optimization.prof"

    print(f"Profiling to {output_file}...")
    profiler = cProfile.Profile()
    profiler.enable()

    run_optimization_test()

    profiler.disable()
    profiler.dump_stats(output_file)

    print(f"\nProfile saved to {output_file}")
    print("\nTop 20 functions by cumulative time:")
    stats = pstats.Stats(output_file)
    stats.strip_dirs()
    stats.sort_stats('cumulative')
    stats.print_stats(20)
