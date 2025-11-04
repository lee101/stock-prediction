#!/bin/bash
set -e

echo "=========================================="
echo "BACKTEST SPEEDUP: Apply & Test"
echo "=========================================="

# 1. Backup original
if [ ! -f backtest_test3_inline.py.pre_speedup ]; then
    echo "Creating backup..."
    cp backtest_test3_inline.py backtest_test3_inline.py.pre_speedup
else
    echo "Backup already exists"
fi

# 2. Extract and replace the function
echo ""
echo "Applying optimized _compute_toto_forecast()..."
python3 << 'EOF'
import re

# Read optimized function
with open('optimized_compute_toto_forecast.py', 'r') as f:
    opt_content = f.read()

# Extract just the function definition
func_match = re.search(r'(def _compute_toto_forecast\(.*?\n(?:.*?\n)*?    return predictions, bands, predicted_absolute_last)', opt_content, re.DOTALL)
if not func_match:
    print("ERROR: Could not extract function from optimized file")
    exit(1)

optimized_func = func_match.group(1)

# Read original file
with open('backtest_test3_inline.py', 'r') as f:
    orig_content = f.read()

# Find and replace the function
pattern = r'def _compute_toto_forecast\(.*?\n(?:.*?\n)*?    return predictions, bands, predicted_absolute_last'
match = re.search(pattern, orig_content, re.DOTALL)
if not match:
    print("ERROR: Could not find function in original file")
    exit(1)

# Replace
new_content = orig_content[:match.start()] + optimized_func + orig_content[match.end():]

# Write back
with open('backtest_test3_inline.py', 'w') as f:
    f.write(new_content)

print("✓ Applied optimized function")
print(f"  Original size: {len(match.group(0))} chars")
print(f"  Optimized size: {len(optimized_func)} chars")
EOF

echo ""
echo "=========================================="
echo "Running 10-simulation timing test..."
echo "=========================================="

# 3. Run timing test
.venv/bin/python test_backtest_speedup.py

echo ""
echo "=========================================="
echo "Done!"
echo "=========================================="
echo ""
echo "To restore original:"
echo "  cp backtest_test3_inline.py.pre_speedup backtest_test3_inline.py"
