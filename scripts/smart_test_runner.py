#!/usr/bin/env python3
"""
Smart test runner that prioritizes tests based on changed files.

This script:
1. Detects changed files from git (vs main branch or last commit)
2. Maps changed files to their corresponding test files
3. Runs tests for changed files first (fail-fast on critical paths)
4. Then runs remaining tests

Usage:
    python scripts/smart_test_runner.py [--verbose] [--dry-run]
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path
from typing import List, Set, Tuple


def get_changed_files(base_branch: str = "main") -> Set[str]:
    """Get list of changed files compared to base branch or last commit."""
    try:
        # Try to get changed files vs base branch
        result = subprocess.run(
            ["git", "diff", "--name-only", f"{base_branch}...HEAD"],
            capture_output=True,
            text=True,
            check=True
        )
        files = result.stdout.strip().split("\n")
        if files and files[0]:  # Check if we got results
            return set(f for f in files if f)
    except subprocess.CalledProcessError:
        pass

    try:
        # Fallback: get changed files in working directory + staged
        result = subprocess.run(
            ["git", "diff", "--name-only", "HEAD"],
            capture_output=True,
            text=True,
            check=True
        )
        staged_result = subprocess.run(
            ["git", "diff", "--name-only", "--cached"],
            capture_output=True,
            text=True,
            check=True
        )
        files = set(result.stdout.strip().split("\n") + staged_result.stdout.strip().split("\n"))
        return set(f for f in files if f)
    except subprocess.CalledProcessError:
        return set()


def map_file_to_tests(file_path: str) -> List[str]:
    """Map a source file to its corresponding test files."""
    tests = []
    path = Path(file_path)

    # Skip if already a test file
    if "test" in path.parts or file_path.startswith("tests/"):
        return [file_path] if path.exists() else []

    # Skip non-Python files
    if path.suffix != ".py":
        return []

    stem = path.stem

    # Direct test file mappings
    test_patterns = [
        f"tests/test_{stem}.py",
        f"tests/prod/test_{stem}.py",
        f"tests/prod/**/test_{stem}.py",
    ]

    # Special mappings for known critical files
    special_mappings = {
        "loss_utils.py": [
            "tests/test_close_at_eod.py",
            "tests/test_maxdiff_pnl.py",
        ],
        "trade_stock_e2e.py": [
            "tests/prod/trading/test_trade_stock_e2e.py",
            "tests/experimental/integration/integ/test_trade_stock_e2e_integ.py",
        ],
        "backtest_test3_inline.py": [
            "tests/prod/backtesting/test_backtest3.py",
        ],
    }

    if path.name in special_mappings:
        tests.extend(special_mappings[path.name])

    # Search for direct test files
    for pattern in test_patterns:
        if "*" in pattern:
            # Use glob for patterns
            for test_file in Path(".").glob(pattern):
                tests.append(str(test_file))
        else:
            if Path(pattern).exists():
                tests.append(pattern)

    # Search for tests that import this file
    if stem and path.exists():
        try:
            result = subprocess.run(
                ["grep", "-r", "-l", f"from {stem} import\\|import {stem}", "tests/"],
                capture_output=True,
                text=True,
            )
            if result.returncode == 0:
                for line in result.stdout.strip().split("\n"):
                    if line and line.endswith(".py"):
                        tests.append(line)
        except subprocess.CalledProcessError:
            pass

    return list(set(tests))  # Remove duplicates


def prioritize_tests(changed_files: Set[str]) -> Tuple[List[str], List[str]]:
    """
    Prioritize tests based on changed files.

    Returns:
        (priority_tests, remaining_tests): Lists of test file paths
    """
    priority_tests = set()

    # Map changed files to tests
    for file_path in changed_files:
        tests = map_file_to_tests(file_path)
        priority_tests.update(tests)

    # Critical tests that should always run first (prod-like tests)
    critical_tests = [
        "tests/prod/trading/test_trade_stock_e2e.py",
        "tests/prod/backtesting/test_backtest3.py",
        "tests/test_close_at_eod.py",
        "tests/test_maxdiff_pnl.py",
    ]

    # Add critical tests to priority if they exist
    for test in critical_tests:
        if Path(test).exists():
            priority_tests.add(test)

    # Get all test files
    all_tests = set()
    for test_dir in ["tests/prod", "tests"]:
        if Path(test_dir).exists():
            for test_file in Path(test_dir).rglob("test_*.py"):
                all_tests.add(str(test_file))

    # Remaining tests
    remaining_tests = all_tests - priority_tests

    return sorted(priority_tests), sorted(remaining_tests)


def run_tests(test_files: List[str], label: str, verbose: bool = False, dry_run: bool = False) -> bool:
    """
    Run tests and return success status.

    Returns:
        True if all tests passed, False otherwise
    """
    if not test_files:
        print(f"No {label} tests to run")
        return True

    print(f"\n{'='*80}")
    print(f"{label.upper()}")
    print(f"{'='*80}")
    print(f"Running {len(test_files)} test(s):")
    for test in test_files:
        print(f"  - {test}")
    print()

    if dry_run:
        print("DRY RUN: Would execute:")
        print(f"  python -m pytest {' '.join(test_files)} -v")
        return True

    # Run pytest with the test files
    cmd = ["python", "-m", "pytest"] + test_files
    if verbose:
        cmd.append("-v")

    # Add fail-fast for priority tests
    if label == "priority":
        cmd.append("-x")  # Stop on first failure

    result = subprocess.run(cmd)

    if result.returncode != 0:
        print(f"\n❌ {label.upper()} TESTS FAILED")
        return False

    print(f"\n✅ {label.upper()} TESTS PASSED")
    return True


def main():
    parser = argparse.ArgumentParser(description="Smart test runner with change-based prioritization")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose pytest output")
    parser.add_argument("--dry-run", "-n", action="store_true", help="Show what would be tested without running")
    parser.add_argument("--base-branch", "-b", default="main", help="Base branch for comparison (default: main)")
    args = parser.parse_args()

    print("Smart Test Runner")
    print("=" * 80)

    # Get changed files
    changed_files = get_changed_files(args.base_branch)
    if changed_files:
        print(f"\nDetected {len(changed_files)} changed file(s):")
        for f in sorted(changed_files)[:10]:  # Show first 10
            print(f"  - {f}")
        if len(changed_files) > 10:
            print(f"  ... and {len(changed_files) - 10} more")
    else:
        print("\nNo changed files detected (running all tests)")

    # Prioritize tests
    priority_tests, remaining_tests = prioritize_tests(changed_files)

    print(f"\nTest execution plan:")
    print(f"  Priority tests (fail-fast): {len(priority_tests)}")
    print(f"  Remaining tests: {len(remaining_tests)}")

    # Run priority tests first (with fail-fast)
    if not run_tests(priority_tests, "priority", args.verbose, args.dry_run):
        print("\n❌ PRIORITY TESTS FAILED - Stopping here (fail-fast)")
        sys.exit(1)

    # Run remaining tests
    if not run_tests(remaining_tests, "remaining", args.verbose, args.dry_run):
        print("\n❌ SOME REMAINING TESTS FAILED")
        sys.exit(1)

    print("\n" + "=" * 80)
    print("✅ ALL TESTS PASSED")
    print("=" * 80)
    sys.exit(0)


if __name__ == "__main__":
    main()
