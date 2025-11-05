# Smart Test Runner

## Overview

The smart test runner (`scripts/smart_test_runner.py`) is a change-aware test execution system that prioritizes running tests based on which files have changed. This enables faster CI feedback by running the most relevant tests first with fail-fast behavior.

## Key Features

1. **Change Detection**: Automatically detects changed files by comparing against the main branch (or last commit)
2. **Intelligent Test Mapping**: Maps source files to their corresponding test files
3. **Priority Testing**: Runs tests for changed code first with fail-fast (`-x` flag)
4. **Critical Test Prioritization**: Always includes critical prod-like tests in priority set
5. **Comprehensive Coverage**: Runs all remaining tests after priority tests pass

## Critical Tests

The following tests are always run first as they validate production-critical paths:

- `tests/prod/trading/test_trade_stock_e2e.py` - End-to-end trading system tests
- `tests/prod/backtesting/test_backtest3.py` - Backtesting validation
- `tests/test_close_at_eod.py` - Loss calculation with EOD exit logic
- `tests/test_maxdiff_pnl.py` - MaxDiff PnL calculations

## File-to-Test Mapping

The script includes special mappings for key files:

- `loss_utils.py` → `test_close_at_eod.py`, `test_maxdiff_pnl.py`
- `trade_stock_e2e.py` → `test_trade_stock_e2e.py`, `test_trade_stock_e2e_integ.py`
- `backtest_test3_inline.py` → `test_backtest3.py`

It also automatically searches for:
- Direct test file mappings (e.g., `foo.py` → `test_foo.py`)
- Files that import the changed module

## Usage

### CI Integration

The smart test runner is integrated into the CI workflow at `.github/workflows/ci.yml`:

```yaml
- name: Run smart test suite (change-aware, fail-fast)
  run: |
    python scripts/smart_test_runner.py --verbose
```

### Manual Execution

```bash
# Run with verbose output
python scripts/smart_test_runner.py --verbose

# Dry run to see what would be tested
python scripts/smart_test_runner.py --dry-run

# Compare against a different base branch
python scripts/smart_test_runner.py --base-branch develop
```

## CI Configuration

The workflow requires `fetch-depth: 0` in the checkout action to access full git history:

```yaml
- name: Checkout repository
  uses: actions/checkout@v4
  with:
    fetch-depth: 0  # Fetch all history for smart test detection
```

## Execution Flow

1. **Detect Changes**: Get list of changed files from git
2. **Map to Tests**: Convert source files to their test files
3. **Prioritize**: Combine mapped tests with critical tests
4. **Execute Priority**: Run priority tests with fail-fast (`-x`)
5. **Execute Remaining**: Run all other tests if priority tests pass
6. **Report**: Return exit code 0 if all pass, 1 if any fail

## Benefits

- **Faster Feedback**: Critical tests run first, catching issues early
- **Fail Fast**: CI stops immediately if priority tests fail, saving time and resources
- **Comprehensive**: Still runs all tests, just in a smarter order
- **Context-Aware**: Prioritizes tests related to your changes
- **Production Safety**: Always validates critical trading and backtesting paths

## Example Output

```
Smart Test Runner
================================================================================

Detected 2 changed file(s):
  - trade_stock_e2e.py
  - loss_utils.py

Test execution plan:
  Priority tests (fail-fast): 6
  Remaining tests: 188

================================================================================
PRIORITY
================================================================================
Running 6 test(s):
  - tests/prod/trading/test_trade_stock_e2e.py
  - tests/test_close_at_eod.py
  - tests/test_maxdiff_pnl.py
  - tests/prod/backtesting/test_backtest3.py
  - tests/prod/trading/test_trade_stock_e2e_helpers.py
  - tests/experimental/integration/integ/test_trade_stock_e2e_integ.py

✅ PRIORITY TESTS PASSED

================================================================================
REMAINING
================================================================================
Running 188 test(s):
  ...

✅ ALL TESTS PASSED
```

## Adding Custom Mappings

To add custom file-to-test mappings, edit `scripts/smart_test_runner.py` and update the `special_mappings` dict in the `map_file_to_tests()` function:

```python
special_mappings = {
    "your_module.py": [
        "tests/path/to/test_your_module.py",
        "tests/integration/test_your_module_integration.py",
    ],
}
```

## Troubleshooting

### No tests run in priority
- Check that changed files are mapped correctly
- Verify git history is available (fetch-depth: 0)
- Ensure test files exist at expected paths

### Tests not being detected
- Check file naming conventions (test_*.py)
- Verify import statements in test files
- Add explicit mappings in special_mappings dict

### False positives in change detection
- May occur with rebases or force pushes
- Use `--base-branch` to specify comparison point
- Consider adding `.git-blame-ignore-revs` for bulk changes
