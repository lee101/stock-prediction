#!/usr/bin/env python3
"""
Shared implementation for the smart test runner.

Keep the actual logic here so both the root-level compatibility module and the
scripts/ CLI entry point use the same code path.
"""

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path


_IGNORED_TEST_PARTS = {".uvcache", ".pytest_cache", "__pycache__"}
_IGNORED_TEST_ROOTS = {"tests/experimental"}
_REPO_ROOT = Path(__file__).resolve().parents[1]
_NESTED_PYTEST_BASETEMP_ROOT = _REPO_ROOT / ".pytest_tmp" / "smart-test-runner"


def _is_project_test_file(path: Path) -> bool:
    """Return True for repo-owned pytest modules, excluding cached/vendor files."""
    if path.suffix != ".py" or not path.name.startswith("test_"):
        return False
    normalized = path.as_posix()
    if any(normalized == root or normalized.startswith(f"{root}/") for root in _IGNORED_TEST_ROOTS):
        return False
    return not any(part in _IGNORED_TEST_PARTS or part.startswith(".") for part in path.parts)


def get_changed_files(base_branch: str = "main") -> set[str]:
    """Get list of changed files compared to base branch or last commit."""
    try:
        result = subprocess.run(
            ["git", "diff", "--name-only", f"{base_branch}...HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        files = result.stdout.strip().split("\n")
        if files and files[0]:
            return {f for f in files if f}
    except (subprocess.CalledProcessError, OSError):
        pass

    try:
        result = subprocess.run(
            ["git", "diff", "--name-only", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        staged_result = subprocess.run(
            ["git", "diff", "--name-only", "--cached"],
            capture_output=True,
            text=True,
            check=True,
        )
        files = set(result.stdout.strip().split("\n") + staged_result.stdout.strip().split("\n"))
        return {f for f in files if f}
    except (subprocess.CalledProcessError, OSError):
        return set()


def map_file_to_tests(file_path: str) -> list[str]:
    """Map a source file to its corresponding test files."""
    tests: list[str] = []
    path = Path(file_path)

    if "test" in path.parts or file_path.startswith("tests/"):
        return [file_path] if path.exists() and _is_project_test_file(path) else []

    if path.suffix != ".py":
        return []

    stem = path.stem
    test_patterns = [
        f"tests/test_{stem}.py",
        f"tests/prod/test_{stem}.py",
        f"tests/prod/**/test_{stem}.py",
    ]

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

    for pattern in test_patterns:
        if "*" in pattern:
            for test_file in Path().glob(pattern):
                if _is_project_test_file(test_file):
                    tests.append(str(test_file))
        elif Path(pattern).exists() and _is_project_test_file(Path(pattern)):
            tests.append(pattern)

    if stem and path.exists():
        try:
            result = subprocess.run(
                ["grep", "-r", "-l", f"from {stem} import\\|import {stem}", "tests/"],
                check=False,
                capture_output=True,
                text=True,
            )
            if result.returncode == 0:
                for line in result.stdout.strip().split("\n"):
                    test_path = Path(line)
                    if line and line.endswith(".py") and _is_project_test_file(test_path):
                        tests.append(line)
        except (subprocess.CalledProcessError, OSError):
            pass

    return list(set(tests))


def prioritize_tests(changed_files: set[str]) -> tuple[list[str], list[str]]:
    """Return priority and remaining test file lists for the current diff."""
    priority_tests = set()

    for file_path in changed_files:
        tests = map_file_to_tests(file_path)
        priority_tests.update(tests)

    critical_tests = [
        "tests/prod/trading/test_trade_stock_e2e.py",
        "tests/test_close_at_eod.py",
        "tests/test_maxdiff_pnl.py",
    ]

    for test in critical_tests:
        if Path(test).exists():
            priority_tests.add(test)

    all_tests = set()
    for test_dir in ["tests/prod", "tests"]:
        if Path(test_dir).exists():
            for test_file in Path(test_dir).rglob("test_*.py"):
                if _is_project_test_file(test_file):
                    all_tests.add(str(test_file))

    remaining_tests = all_tests - priority_tests
    return sorted(priority_tests), sorted(remaining_tests)


def run_tests(test_files: list[str], label: str, verbose: bool = False, dry_run: bool = False) -> bool:
    """Run pytest for the selected files and return success status."""
    if not test_files:
        print(f"No {label} tests to run")
        return True

    print(f"\n{'=' * 80}")
    print(f"{label.upper()}")
    print(f"{'=' * 80}")
    print(f"Running {len(test_files)} test(s):")
    for test in test_files:
        print(f"  - {test}")
    print()

    if dry_run:
        print("DRY RUN: Would execute:")
        print(f"  {sys.executable} -m pytest {' '.join(test_files)} -v")
        return True

    _NESTED_PYTEST_BASETEMP_ROOT.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=f"{label}-",
        dir=_NESTED_PYTEST_BASETEMP_ROOT,
    ) as nested_basetemp:
        cmd = [sys.executable, "-m", "pytest", "--basetemp", nested_basetemp, *test_files]
        if verbose:
            cmd.append("-v")
        cmd.extend(["--ignore=tests/experimental"])
        cmd.append("-x" if label == "priority" else "--maxfail=20")

        try:
            result = subprocess.run(cmd, check=False)
        except OSError as exc:
            print(f"\n❌ {label.upper()} TESTS FAILED TO START: {exc}")
            print("Rerun command:")
            print(f"  {format_rerun_command(test_files, label=label, verbose=verbose)}")
            return False
    if result.returncode != 0:
        print(f"\n❌ {label.upper()} TESTS FAILED")
        print("Rerun command:")
        print(f"  {format_rerun_command(test_files, label=label, verbose=verbose)}")
        return False

    print(f"\n✅ {label.upper()} TESTS PASSED")
    return True


def format_rerun_command(test_files: list[str], *, label: str, verbose: bool = False) -> str:
    cmd = [sys.executable, "-m", "pytest", *test_files]
    if verbose:
        cmd.append("-v")
    cmd.extend(["--ignore=tests/experimental"])
    cmd.append("-x" if label == "priority" else "--maxfail=20")
    return " ".join(cmd)


def main() -> None:
    parser = argparse.ArgumentParser(description="Smart test runner with change-based prioritization")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose pytest output")
    parser.add_argument("--dry-run", "-n", action="store_true", help="Show what would be tested without running")
    parser.add_argument("--base-branch", "-b", default="main", help="Base branch for comparison (default: main)")
    lane_group = parser.add_mutually_exclusive_group()
    lane_group.add_argument(
        "--priority-only",
        action="store_true",
        help="Run only the fail-fast priority lane",
    )
    lane_group.add_argument(
        "--remaining-only",
        action="store_true",
        help="Run only the remaining non-priority lane",
    )
    args = parser.parse_args()

    print("Smart Test Runner")
    print("=" * 80)

    changed_files = get_changed_files(args.base_branch)
    if changed_files:
        print(f"\nDetected {len(changed_files)} changed file(s):")
        for file_path in sorted(changed_files)[:10]:
            print(f"  - {file_path}")
        if len(changed_files) > 10:
            print(f"  ... and {len(changed_files) - 10} more")
    else:
        print("\nNo changed files detected (running all tests)")

    priority_tests, remaining_tests = prioritize_tests(changed_files)

    print("\nTest execution plan:")
    print(f"  Priority tests (fail-fast): {len(priority_tests)}")
    print(f"  Remaining tests: {len(remaining_tests)}")
    if args.priority_only:
        print("  Selected lane: priority only")
    elif args.remaining_only:
        print("  Selected lane: remaining only")
    else:
        print("  Selected lane: priority + remaining")

    if args.remaining_only:
        remaining_ok = run_tests(remaining_tests, "remaining", args.verbose, args.dry_run)
    else:
        if not run_tests(priority_tests, "priority", args.verbose, args.dry_run):
            print("\n❌ PRIORITY TESTS FAILED - Stopping here (fail-fast)")
            sys.exit(1)
        if args.priority_only:
            remaining_ok = True
        else:
            remaining_ok = run_tests(remaining_tests, "remaining", args.verbose, args.dry_run)
            if not remaining_ok:
                print("\n⚠️  SOME REMAINING TESTS FAILED (non-fatal)")

    exit_code = 0
    print("\n" + "=" * 80)
    if args.priority_only:
        print("✅ PRIORITY TESTS PASSED")
    elif args.remaining_only:
        if remaining_ok:
            print("✅ REMAINING TESTS PASSED")
        else:
            print("❌ REMAINING TESTS FAILED")
            exit_code = 1
    elif remaining_ok:
        print("✅ ALL TESTS PASSED")
    else:
        print("⚠️  PRIORITY TESTS PASSED; REMAINING TESTS HAD FAILURES")
    print("=" * 80)
    sys.exit(exit_code)
