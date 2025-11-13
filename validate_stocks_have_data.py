#!/usr/bin/env python3
"""
Check which of the popular stocks we have data for.
"""

import json
from pathlib import Path

# Load popular stocks
with open('validated_popular_stocks.json') as f:
    stocks_data = json.load(f)

all_symbols = [s['symbol'] for s in stocks_data]

# Check which have data
train_dir = Path('trainingdata/train')
available_symbols = []
missing_symbols = []

for symbol in all_symbols:
    csv_path = train_dir / f"{symbol}.csv"
    if csv_path.exists():
        # Check file has enough data
        try:
            with open(csv_path) as f:
                lines = len(f.readlines())
            if lines > 100:  # At least 100 rows
                available_symbols.append(symbol)
            else:
                missing_symbols.append(f"{symbol} (only {lines} rows)")
        except:
            missing_symbols.append(f"{symbol} (read error)")
    else:
        missing_symbols.append(symbol)

print("=" * 80)
print("STOCK DATA AVAILABILITY CHECK")
print("=" * 80)
print()
print(f"Total symbols checked: {len(all_symbols)}")
print(f"Available with data: {len(available_symbols)}")
print(f"Missing or insufficient: {len(missing_symbols)}")
print()

print("=" * 80)
print(f"AVAILABLE SYMBOLS ({len(available_symbols)})")
print("=" * 80)
print(", ".join(available_symbols))
print()

print("=" * 80)
print("Python list:")
print("=" * 80)
print(repr(available_symbols))
print()

if missing_symbols[:20]:
    print("=" * 80)
    print(f"SAMPLE MISSING ({len(missing_symbols)} total, showing first 20):")
    print("=" * 80)
    for symbol in missing_symbols[:20]:
        print(f"  - {symbol}")
    print()

# Save available symbols
output = {
    'available_symbols': available_symbols,
    'total_available': len(available_symbols),
    'total_missing': len(missing_symbols),
    'data_dir': str(train_dir)
}

with open('available_stocks_with_data.json', 'w') as f:
    json.dump(output, f, indent=2)

print(f"✓ Saved to available_stocks_with_data.json")
