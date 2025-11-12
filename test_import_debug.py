"""Debug import issue"""
import os
import sys
from pathlib import Path

print(f"Current file: {__file__}")
print(f"Parent: {Path(__file__).parent}")
print(f"Parent.parent: {Path(__file__).parent.parent}")

# Add parent directory to path (mimicking test file)
sys.path.insert(0, str(Path(__file__).parent))

# Set environment variables
os.environ["ONLY_CHRONOS2"] = "1"
os.environ["REAL_TESTING"] = "1"

print(f"sys.path[0]: {sys.path[0]}")

try:
    from backtest_test3_inline import (
        load_chronos2_wrapper,
        resolve_best_model,
        resolve_chronos2_params,
    )
    print("✓ Import successful!")
    print(f"load_chronos2_wrapper: {load_chronos2_wrapper}")
except ImportError as e:
    print(f"✗ Import failed: {e}")
    import traceback
    traceback.print_exc()
