#!/usr/bin/env python3
"""
Runner for realistic integration tests without mocking.
"""

import sys
import os
from pathlib import Path

# Add project root to path
TEST_DIR = Path(__file__).parent
REPO_ROOT = TEST_DIR.parent
sys.path.insert(0, str(REPO_ROOT))

import pytest


def run_realistic_tests():
    """Run all realistic integration tests."""
    
    test_files = [
        "tests/experimental/integration/integ/test_training_realistic.py",
        "tests/experimental/integration/integ/test_hftraining_realistic.py",
        "tests/experimental/integration/integ/test_totoembedding_realistic.py",
    ]
    
    print("=" * 60)
    print("Running Realistic Integration Tests (No Mocking)")
    print("=" * 60)
    
    # Run tests with verbose output
    args = [
        '-v',  # Verbose
        '-s',  # Show print statements
        '--tb=short',  # Short traceback format
        '--color=yes',  # Colored output
        '-x',  # Stop on first failure for debugging
    ]
    
    # Add test files
    args.extend(test_files)
    
    # Run pytest
    exit_code = pytest.main(args)
    
    if exit_code == 0:
        print("\n" + "=" * 60)
        print("✅ All realistic tests passed!")
        print("=" * 60)
    else:
        print("\n" + "=" * 60)
        print("❌ Some tests failed. Check output above.")
        print("=" * 60)
    
    return exit_code


def run_single_test_module(module_name):
    """Run tests for a single module."""
    
    module_map = {
        "training": "tests/experimental/integration/integ/test_training_realistic.py",
        "hftraining": "tests/experimental/integration/integ/test_hftraining_realistic.py",
        "totoembedding": "tests/experimental/integration/integ/test_totoembedding_realistic.py",
    }
    
    if module_name not in module_map:
        print(f"Unknown module: {module_name}")
        print(f"Available modules: {', '.join(module_map.keys())}")
        return 1
    
    test_file = module_map[module_name]
    
    print(f"Running tests for {module_name}...")
    args = ['-v', '-s', '--tb=short', '--color=yes', test_file]
    return pytest.main(args)


if __name__ == '__main__':
    if len(sys.argv) > 1:
        # Run specific module tests
        module = sys.argv[1]
        exit_code = run_single_test_module(module)
    else:
        # Run all tests
        exit_code = run_realistic_tests()
    
    sys.exit(exit_code)
