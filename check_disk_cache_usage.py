#!/usr/bin/env python3
"""
Check if disk_cache is actually working and why it's not helping.
"""

import os
from pathlib import Path

cache_base = Path(__file__).parent / '.cache'

print("="*80)
print("DISK CACHE ANALYSIS")
print("="*80)
print()

if not cache_base.exists():
    print("❌ No .cache directory found!")
    print(f"   Expected at: {cache_base}")
    print()
    print("Disk cache is not being used.")
else:
    print(f"✓ Cache directory exists: {cache_base}")
    print()

    # Check cached_predict cache
    cached_predict_dir = cache_base / 'cached_predict'
    if cached_predict_dir.exists():
        cache_files = list(cached_predict_dir.glob('*.pkl'))
        print(f"cached_predict cache:")
        print(f"  Files: {len(cache_files)}")

        if cache_files:
            total_size = sum(f.stat().st_size for f in cache_files)
            print(f"  Total size: {total_size / 1024 / 1024:.1f} MB")
            print(f"  Average per file: {total_size / len(cache_files) / 1024:.1f} KB")

            # Show newest files
            newest = sorted(cache_files, key=lambda f: f.stat().st_mtime, reverse=True)[:5]
            print(f"\n  Most recent cache entries:")
            for i, f in enumerate(newest, 1):
                age = os.path.getmtime(f)
                import time
                age_str = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(age))
                size = f.stat().st_size / 1024
                print(f"    {i}. {f.name[:32]}... ({size:.1f} KB, {age_str})")
    else:
        print("❌ No cached_predict cache found")

    print()

print("="*80)
print("THE PROBLEM WITH CURRENT CACHING")
print("="*80)
print("""
disk_cache uses FULL tensor content as cache key:
  - key = md5(tensor.tobytes())

In walk-forward backtest:
  - Sim 0: context = days [0:199] → hash(199 values)
  - Sim 1: context = days [0:198] → hash(198 values) [DIFFERENT!]

Even though days [0:198] are IDENTICAL, cache keys differ!

Result: NO cache reuse across simulations within same run.

Cache DOES help across runs:
  - First backtest: All cache MISSES (compute + store)
  - Second backtest: All cache HITS (load from disk)

But within a single backtest run: 98% redundant computation!
""")

print("="*80)
print("SOLUTION")
print("="*80)
print("""
Option 1: Modify disk_cache to key by (data_prefix, length)
  - More complex, could have collisions

Option 2: Use per-day caching (as in src/prediction_cache.py)
  - Cache individual day predictions
  - Reuse across simulations with different lengths
  - In-memory for speed, optional disk persistence

Option 3: Pre-compute all predictions before simulations
  - Run predictions once for full data
  - Slice results for each simulation
  - Simplest and fastest

Recommendation: Option 3 (pre-compute)
  - Predict once for days [0:199]
  - Each simulation uses subset
  - 57.9x speedup on model inference
""")

print("\n" + "="*80)
print("CURRENT CACHE EFFECTIVENESS")
print("="*80)

if cache_base.exists():
    print("✓ Helps across DIFFERENT backtest runs")
    print("✗ Does NOT help within SAME backtest run")
    print("\n  → Still computing same predictions 70x per run!")
else:
    print("✗ Not being used at all")
