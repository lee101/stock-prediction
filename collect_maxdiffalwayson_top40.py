#!/usr/bin/env python3
"""
Collect maxdiffalwayson strategy data for the top 40 maxdiff performers.
"""

import json
import subprocess
import sys

# Load the top 40 maxdiff symbols
with open('strategytraining/top_40_maxdiff_only.json') as f:
    data = json.load(f)

top_40_symbols = [s['symbol'] for s in data['top_40']]

print("=" * 80)
print("COLLECTING MAXDIFFALWAYSON STRATEGY DATA")
print("=" * 80)
print()
print(f"Collecting data for {len(top_40_symbols)} symbols:")
print(", ".join(top_40_symbols))
print()
print("This may take a while...")
print()

# Run the collector
cmd = [
    sys.executable,
    'strategytraining/collect_strategy_pnl_dataset.py',
    '--dataset-name', 'maxdiffalwayson_top40',
    '--window-days', '15',
    '--stride-days', '15',
    '--symbols'
] + top_40_symbols

print("Running command:")
print(" ".join(cmd))
print()

result = subprocess.run(cmd, capture_output=False)

if result.returncode == 0:
    print()
    print("=" * 80)
    print("✓ COLLECTION COMPLETE!")
    print("=" * 80)
    print()
    print("Dataset saved to: strategytraining/datasets/maxdiffalwayson_top40_*")
else:
    print()
    print("=" * 80)
    print("✗ COLLECTION FAILED")
    print("=" * 80)
    sys.exit(1)
