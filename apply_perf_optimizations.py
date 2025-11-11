#!/usr/bin/env python3
"""
Apply quick performance optimizations to backtest_test3_inline.py

This script implements the "Quick Wins" from PERFORMANCE_ANALYSIS.md:
1. Reduce max_horizon from 7 to 1 (7x speedup)
2. Add option to specify close_at_eod (2x speedup)
3. Instructions for env vars (additional speedup)

Usage:
    python apply_perf_optimizations.py --dry-run  # Preview changes
    python apply_perf_optimizations.py            # Apply changes
"""

import re
import argparse
from pathlib import Path


def apply_optimizations(dry_run=False):
    """Apply performance optimizations to backtest file"""

    target_file = Path(__file__).parent / "backtest_test3_inline.py"

    if not target_file.exists():
        print(f"Error: {target_file} not found")
        return False

    with open(target_file, 'r') as f:
        content = f.read()

    original_content = content
    changes = []

    # Optimization 1: Reduce max_horizon from 7 to 1
    # This gives ~7x speedup on forecasting
    pattern1 = r'max_horizon = 7'
    replacement1 = 'max_horizon = 1  # PERF: Reduced from 7 (only need next-day forecast)'
    if re.search(pattern1, content):
        content = re.sub(pattern1, replacement1, content)
        changes.append("✓ Reduced max_horizon from 7 to 1 (~7x speedup on forecasting)")

    # Optimization 2: Add environment variable check for max_horizon override
    # This allows easy tuning without code changes
    pattern2 = r'(max_horizon = 1.*\n)'
    replacement2 = (
        r'\1'
        '    # Allow override via environment variable for tuning\n'
        '    max_horizon = int(os.getenv("MARKETSIM_MAX_HORIZON", max_horizon))\n'
    )
    if re.search(pattern2, content):
        content = re.sub(pattern2, replacement2, content, count=1)
        changes.append("✓ Added MARKETSIM_MAX_HORIZON environment variable support")

    # Optimization 3: Document the close_at_eod parameter better
    # Users can pass close_at_eod=False to skip testing both options
    pattern3 = r'(close_at_eod_candidates = \[close_at_eod\] if close_at_eod is not None else \[False, True\])'
    replacement3 = (
        r'# PERF: Set close_at_eod explicitly (not None) to skip testing both options (2x faster)\n'
        r'        \1'
    )
    content = re.sub(pattern3, replacement3, content)
    if pattern3 in original_content:
        changes.append("✓ Added comment about close_at_eod performance optimization")

    if changes:
        if dry_run:
            print("\n=== DRY RUN MODE - No changes applied ===\n")
            print("Would apply the following changes:\n")
            for change in changes:
                print(f"  {change}")
            print("\nTo apply these changes, run without --dry-run")
            return True
        else:
            # Apply changes
            with open(target_file, 'w') as f:
                f.write(content)

            print("\n=== Applied Performance Optimizations ===\n")
            for change in changes:
                print(f"  {change}")
            print(f"\n✓ Changes written to {target_file}")
            return True
    else:
        print("No optimizations needed (may already be applied)")
        return False


def print_usage_instructions():
    """Print instructions for using the optimizations"""

    print("\n" + "=" * 80)
    print("PERFORMANCE OPTIMIZATION INSTRUCTIONS")
    print("=" * 80)

    print("\n1. Environment Variables (Set these for additional speedup):")
    print("-" * 80)
    print("""
# Fast optimize mode: Reduces optimizer evaluations from 500 to 100 (6x faster)
export MARKETSIM_FAST_OPTIMIZE=1

# Fast simulate mode: Reduces number of simulations from 50 to 35 (1.4x faster)
export MARKETSIM_FAST_SIMULATE=1

# Max horizon override: Set to 1 for fastest forecasting (7x faster than 7)
export MARKETSIM_MAX_HORIZON=1

# Combine all three for maximum speedup:
export MARKETSIM_FAST_OPTIMIZE=1
export MARKETSIM_FAST_SIMULATE=1
export MARKETSIM_MAX_HORIZON=1
""")

    print("\n2. Reduce num_samples in Hyperparamstore:")
    print("-" * 80)
    print("""
# Edit hyperparamstore configs to reduce num_samples from 128 to 32
# This gives ~4x speedup on Toto/Kronos forecasting

# Example: Edit hyperparamstore/UNIUSD.json
{
  "toto": {
    "num_samples": 32,        # <- Reduce from 128
    "samples_per_batch": 16   # <- Reduce from 32
  }
}
""")

    print("\n3. Expected Combined Speedup:")
    print("-" * 80)
    print("""
Without optimizations:  ~18 seconds per symbol
With all optimizations: ~0.5 seconds per symbol

Breakdown:
- max_horizon 7→1:           7x speedup
- num_samples 128→32:        4x speedup
- FAST_SIMULATE:             1.4x speedup
- FAST_OPTIMIZE:             Minor (optimization already fast)

Combined: ~36x speedup on forecasting-heavy backtests!
""")

    print("\n4. Testing:")
    print("-" * 80)
    print("""
# Before optimization
time python backtest_test3_inline.py UNIUSD

# After optimization (with env vars)
MARKETSIM_FAST_OPTIMIZE=1 MARKETSIM_FAST_SIMULATE=1 MARKETSIM_MAX_HORIZON=1 \\
  time python backtest_test3_inline.py UNIUSD

# Compare the times!
""")

    print("\n5. For Production Trading (trade_stock_e2e.py):")
    print("-" * 80)
    print("""
# Add these to your .bashrc or .env file for persistent speedup
echo 'export MARKETSIM_FAST_OPTIMIZE=1' >> ~/.bashrc
echo 'export MARKETSIM_FAST_SIMULATE=1' >> ~/.bashrc
echo 'export MARKETSIM_MAX_HORIZON=1' >> ~/.bashrc

# Or create a wrapper script:
cat > run_trading.sh << 'EOF'
#!/bin/bash
export MARKETSIM_FAST_OPTIMIZE=1
export MARKETSIM_FAST_SIMULATE=1
export MARKETSIM_MAX_HORIZON=1
PAPER=1 python trade_stock_e2e.py "$@"
EOF
chmod +x run_trading.sh

# Then use:
./run_trading.sh
""")

    print("=" * 80)
    print()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Apply performance optimizations to backtest code'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Preview changes without applying them'
    )
    args = parser.parse_args()

    success = apply_optimizations(dry_run=args.dry_run)

    if success:
        print_usage_instructions()

        if not args.dry_run:
            print("\n✅ Optimizations applied successfully!")
            print("📖 See PERFORMANCE_ANALYSIS.md for detailed analysis")
            print("🚀 Set environment variables above for maximum speedup")
    else:
        print("\n⚠️  No changes made")
