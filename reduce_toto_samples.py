#!/usr/bin/env python3
"""
Reduce num_samples in Toto hyperparameter configs for faster backtesting.

This script updates all Toto configs to use fewer samples, providing ~4-8x speedup
on forecast generation with minimal accuracy loss.

Usage:
    python reduce_toto_samples.py --samples 32 --batch 16 --dry-run  # Preview
    python reduce_toto_samples.py --samples 32 --batch 16            # Apply
"""

import json
import argparse
from pathlib import Path


def update_toto_configs(target_samples, target_batch, dry_run=False):
    """Update all Toto configs to use specified num_samples"""

    toto_dir = Path("hyperparams/toto")

    if not toto_dir.exists():
        print(f"Error: {toto_dir} not found")
        return False

    json_files = list(toto_dir.glob("*.json"))

    if not json_files:
        print(f"No JSON files found in {toto_dir}")
        return False

    print(f"Found {len(json_files)} Toto config files\n")

    updates = []
    errors = []

    for json_file in sorted(json_files):
        try:
            with open(json_file, 'r') as f:
                config = json.load(f)

            symbol = config.get("symbol", json_file.stem)
            old_samples = config["config"].get("num_samples", "unknown")
            old_batch = config["config"].get("samples_per_batch", "unknown")

            if old_samples == target_samples and old_batch == target_batch:
                continue  # Already at target

            # Update values
            config["config"]["num_samples"] = target_samples
            config["config"]["samples_per_batch"] = target_batch

            updates.append({
                'file': json_file,
                'symbol': symbol,
                'old_samples': old_samples,
                'old_batch': old_batch,
                'new_config': config,
            })

        except Exception as e:
            errors.append(f"{json_file.name}: {e}")

    if errors:
        print("Errors encountered:")
        for error in errors:
            print(f"  ❌ {error}")
        print()

    if not updates:
        print("✓ All configs already at target values")
        return True

    if dry_run:
        print("=== DRY RUN MODE - No changes applied ===\n")
        print(f"Would update {len(updates)} configs:\n")
        for update in updates[:10]:  # Show first 10
            print(f"  {update['symbol']:10s}: {update['old_samples']:4d}/{update['old_batch']:2d} → {target_samples}/{target_batch}")
        if len(updates) > 10:
            print(f"  ... and {len(updates) - 10} more")
        print(f"\nTo apply changes, run without --dry-run")
        return True

    # Apply updates
    for update in updates:
        with open(update['file'], 'w') as f:
            json.dump(update['new_config'], f, indent=2)

    print(f"✓ Updated {len(updates)} Toto configs:\n")
    for update in updates[:10]:  # Show first 10
        print(f"  {update['symbol']:10s}: {update['old_samples']:4d}/{update['old_batch']:2d} → {target_samples}/{target_batch}")
    if len(updates) > 10:
        print(f"  ... and {len(updates) - 10} more")

    return True


def estimate_speedup(old_samples, new_samples):
    """Estimate speedup from reducing samples"""
    if old_samples == 0:
        return 1.0
    return old_samples / new_samples


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Reduce num_samples in Toto configs for faster backtesting'
    )
    parser.add_argument(
        '--samples',
        type=int,
        default=32,
        help='Target num_samples (default: 32, was typically 1024)'
    )
    parser.add_argument(
        '--batch',
        type=int,
        default=16,
        help='Target samples_per_batch (default: 16, was typically 64)'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Preview changes without applying them'
    )
    args = parser.parse_args()

    print("=" * 80)
    print("TOTO SAMPLE REDUCTION")
    print("=" * 80)
    print(f"\nTarget configuration:")
    print(f"  num_samples:        {args.samples}")
    print(f"  samples_per_batch:  {args.batch}")
    print()

    # Typical values before optimization
    typical_old = 1024
    speedup = estimate_speedup(typical_old, args.samples)
    print(f"Expected speedup: ~{speedup:.1f}x (from {typical_old} samples to {args.samples})")
    print()

    success = update_toto_configs(args.samples, args.batch, dry_run=args.dry_run)

    if success and not args.dry_run:
        print("\n" + "=" * 80)
        print("NEXT STEPS")
        print("=" * 80)
        print("""
1. The Toto configs have been updated to use fewer samples
2. Set environment variables for additional speedup:

   export MARKETSIM_FAST_OPTIMIZE=1
   export MARKETSIM_FAST_SIMULATE=1
   export MARKETSIM_MAX_HORIZON=1

3. Test the performance improvement:

   # Before (if you have a backup):
   time python backtest_test3_inline.py UNIUSD

   # After:
   MARKETSIM_FAST_OPTIMIZE=1 MARKETSIM_FAST_SIMULATE=1 \\
     time python backtest_test3_inline.py UNIUSD

4. If accuracy is too low, increase samples back up:

   python reduce_toto_samples.py --samples 64 --batch 32
""")
        print("=" * 80)
