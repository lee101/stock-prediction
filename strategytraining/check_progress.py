"""
Check collection progress
"""

import json
from pathlib import Path
from datetime import datetime

progress_file = Path('strategytraining/datasets/collection_progress.json')

if not progress_file.exists():
    print("No collection in progress yet")
    exit(1)

with open(progress_file, 'r') as f:
    progress = json.load(f)

completed = progress['completed_count']
total = progress['total_symbols']
pct = (completed / total) * 100

print("="*80)
print("COLLECTION PROGRESS")
print("="*80)
print(f"Status: {completed}/{total} symbols ({pct:.1f}%)")
print(f"Last updated: {progress['last_updated']}")
print()

remaining = total - completed
if completed > 0:
    # Estimate based on ~17.5 min per symbol average
    est_time_remaining = remaining * 17.5
    print(f"Estimated time remaining: {est_time_remaining/60:.1f} hours")
else:
    print(f"Estimated total time: 33-44 hours")

print()
print("Recent symbols completed:")
for symbol in progress['completed_symbols'][-10:]:
    print(f"  ✓ {symbol}")

print()
print(f"Datasets saved in: strategytraining/datasets/")
