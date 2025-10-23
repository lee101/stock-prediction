#!/usr/bin/env python3
"""
Run realistic integration tests in isolation to avoid mock interference.
"""

import subprocess
import sys
from pathlib import Path

def run_isolated_test(test_file):
    """Run a test file in a separate process to avoid import pollution."""
    
    cmd = [
        sys.executable,
        '-m', 'pytest',
        test_file,
        '-v',
        '--tb=short',
        '--color=yes',
        '-x'  # Stop on first failure
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    print(f"\n{'='*60}")
    print(f"Testing: {test_file}")
    print(f"{'='*60}")
    print(result.stdout)
    if result.stderr:
        print("STDERR:", result.stderr)
    
    return result.returncode


def main():
    """Run all realistic tests in isolation."""
    
    test_files = [
        "tests/experimental/integration/integ/test_training_realistic.py",
        "tests/experimental/integration/integ/test_hftraining_realistic.py",
        "tests/experimental/integration/integ/test_totoembedding_realistic.py",
    ]
    
    print("=" * 60)
    print("Running Realistic Integration Tests (Isolated)")
    print("=" * 60)
    
    all_passed = True
    results = {}
    
    for test_file in test_files:
        if Path(test_file).exists():
            exit_code = run_isolated_test(test_file)
            results[test_file] = exit_code == 0
            if exit_code != 0:
                all_passed = False
        else:
            print(f"Warning: {test_file} not found")
            results[test_file] = False
            all_passed = False
    
    # Summary
    print("\n" + "=" * 60)
    print("Test Summary:")
    print("=" * 60)
    
    for test_file, passed in results.items():
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"{status}: {test_file}")
    
    if all_passed:
        print("\n✅ All realistic tests passed!")
        return 0
    else:
        print("\n❌ Some tests failed.")
        return 1


if __name__ == '__main__':
    sys.exit(main())
