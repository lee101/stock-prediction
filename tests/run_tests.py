#!/usr/bin/env python3
"""Test runner that handles torch import issues."""

import sys
import os
from pathlib import Path

# Try to import torch, if it fails use mock
try:
    import torch
    if not hasattr(torch, '__version__'):
        raise AttributeError("Torch not properly installed")
    print(f"Using real torch: {torch.__version__}")
except (ImportError, AttributeError) as e:
    print(f"Torch import failed ({e}), using mock...")
    # Add tests directory to path
    tests_dir = Path(__file__).parent
    sys.path.insert(0, str(tests_dir))
    import mock_torch as torch
    sys.modules['torch'] = torch
    print("Mock torch loaded")

# Now run pytest
import pytest

if __name__ == "__main__":
    # Run tests with verbose output
    test_files = [
        "tests/test_hfinference_comprehensive.py",
        "tests/test_hftraining_comprehensive.py",
        "tests/test_hfinference_engine_sim.py",
        "tests/test_hftraining_data_utils.py",
        "tests/test_hftraining_model.py",
        "tests/test_hftraining_training.py"
    ]
    
    # Filter to only existing test files
    existing_tests = [f for f in test_files if Path(f).exists()]
    
    print(f"\nRunning {len(existing_tests)} test files...")
    for test in existing_tests:
        print(f"  - {test}")
    
    # Run pytest
    exit_code = pytest.main(["-v", "--tb=short"] + existing_tests)
    
    print(f"\nTests completed with exit code: {exit_code}")
    sys.exit(exit_code)