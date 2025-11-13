#!/usr/bin/env python3
"""
Profile trade_stock_e2e.py in PAPER mode with cProfile.

Usage:
    python profile_trade_stock.py

After profiling completes (or is interrupted with Ctrl+C):
    1. Generate flamegraph:
       .venv/bin/python -m flameprof trade_stock_e2e_paper.prof -o trade_stock_e2e_flamegraph.svg

    2. Analyze flamegraph:
       .venv/bin/flamegraph-analyzer trade_stock_e2e_flamegraph.svg -o docs/PAPER_MODE_PROFILING.md

    3. View in browser:
       xdg-open trade_stock_e2e_flamegraph.svg

Profiling Tips:
    - For startup/import profiling: Run for ~2-3 minutes
    - For runtime trading logic: Run for at least one full analysis cycle (~10-15 mins)
    - For comprehensive profiling: Run through multiple market phases (open, mid-day, close)
"""

import subprocess
import sys
import os

def main():
    """Profile the trade_stock_e2e script."""
    profile_file = 'trade_stock_e2e_paper.prof'

    print("=" * 80)
    print("PROFILING trade_stock_e2e.py in PAPER MODE")
    print("=" * 80)
    print(f"\nProfile will be saved to: {profile_file}")
    print("Press Ctrl+C to stop profiling and save results\n")
    print("=" * 80)

    # Set environment for PAPER mode
    env = os.environ.copy()
    env['PAPER'] = '1'

    # Use venv python if available, otherwise sys.executable
    venv_python = os.path.join(os.getcwd(), '.venv', 'bin', 'python')
    python_exec = venv_python if os.path.exists(venv_python) else sys.executable

    # Run with cProfile
    cmd = [
        python_exec,
        '-m', 'cProfile',
        '-o', profile_file,
        'trade_stock_e2e.py'
    ]

    print(f"Using Python: {python_exec}\n")

    try:
        subprocess.run(cmd, env=env)
    except KeyboardInterrupt:
        print("\n\n" + "=" * 80)
        print("Profiling interrupted by user")
        print("=" * 80)

    # Check if profile was created
    if os.path.exists(profile_file):
        print(f"\n✓ Profile data saved to {profile_file}")

        # Print basic stats
        import pstats
        print("\n=== Top 30 Functions by Cumulative Time ===\n")
        stats = pstats.Stats(profile_file)
        stats.strip_dirs()
        stats.sort_stats('cumulative')
        stats.print_stats(30)

        print("\n" + "=" * 80)
        print("NEXT STEPS")
        print("=" * 80)
        print(f"1. Generate flamegraph:")
        print(f"   python -m flameprof {profile_file} -o trade_stock_e2e_flamegraph.svg")
        print(f"\n2. Analyze flamegraph:")
        print(f"   flamegraph-analyzer trade_stock_e2e_flamegraph.svg -o docs/PAPER_MODE_PROFILING.md")
        print(f"\n3. View flamegraph in browser:")
        print(f"   xdg-open trade_stock_e2e_flamegraph.svg")
        print("=" * 80)
    else:
        print(f"\n✗ Profile file not created: {profile_file}")

if __name__ == '__main__':
    main()
