# Testing Quick Start Guide

Quick reference for running tests and linters in the stock-prediction project.

## TL;DR - Most Common Commands

```bash
# Run all unit tests (fast)
pytest -m "unit"

# Run linter and formatter
ruff check src tests
ruff format src tests

# Auto-fix linting and formatting
ruff check --fix src tests
ruff format src tests

# Run tests like Fast CI (GitHub runners)
FAST_CI=1 CPU_ONLY=1 pytest -m "unit and not slow and not model_required"
```

## Running Tests

### By Category

```bash
# Unit tests only (fast, no models)
pytest -m "unit"

# Integration tests (may use models)
pytest -m "integration"

# Smoke tests (quick validation)
pytest -m "smoke"

# All tests except slow ones
pytest -m "not slow"

# CPU-only tests (skip GPU tests)
pytest -m "not cuda_required and not gpu_required"
```

### By File or Pattern

```bash
# Run specific test file
pytest tests/test_price_calculations.py

# Run tests matching pattern
pytest -k "test_price" -v

# Run all tests in a directory
pytest tests/prod/
```

### With Environment Flags

```bash
# Fast CI mode (like GitHub runners)
FAST_CI=1 CPU_ONLY=1 FAST_SIMULATE=1 pytest -m "unit and not slow"

# CPU-only mode
CPU_ONLY=1 pytest

# Fast simulation (fewer iterations)
FAST_SIMULATE=1 pytest tests/test_backtest.py
```

## Linting and Formatting

### Check (No Changes)

```bash
# Lint code
ruff check src tests

# Check formatting
ruff format --check src tests
```

### Fix (Make Changes)

```bash
# Auto-fix linting issues
ruff check --fix src tests

# Auto-format code
ruff format src tests

# Do both
ruff check --fix src tests && ruff format src tests
```

### Specific Paths

```bash
# Lint single file
ruff check src/models/chronos2_wrapper.py

# Format single directory
ruff format src/models/
```

## Type Checking

```bash
# Type check with ty (fast)
ty check

# Type check with pyright
pyright src

# Type check with mypy
mypy src
```

## Common Test Scenarios

### Before Committing

```bash
# Run fast checks
ruff check --fix src tests
ruff format src tests
pytest -m "unit and not slow" -v
```

### Full Local Validation

```bash
# Run everything like CI
ruff check src tests
ruff format --check src tests
ty check
pytest -m "not slow"
```

### Testing Model Changes

```bash
# Run model tests only
pytest -m "model_required" -v

# Run with fast config
FAST_SIMULATE=1 pytest -m "model_required"
```

### Testing on CPU

```bash
# Ensure tests work without GPU
CPU_ONLY=1 pytest -m "not cuda_required"
```

## Pytest Options

```bash
# Verbose output
pytest -v

# Show print statements
pytest -s

# Stop on first failure
pytest -x

# Show only short traceback
pytest --tb=short

# Run last failed tests
pytest --lf

# Run tests in parallel (if pytest-xdist installed)
pytest -n auto
```

## Test Markers Reference

| Marker | Description | Fast CI? |
|--------|-------------|----------|
| `unit` | Fast unit tests | ✅ Yes |
| `integration` | Integration tests | ❌ No |
| `slow` | Tests >10s | ❌ No |
| `model_required` | Needs ML models | ❌ No (except smoke) |
| `smoke` | Quick validation | ✅ Yes |
| `cuda_required` | Needs CUDA GPU | ❌ No (CPU only) |
| `external` | Calls external APIs | ❌ No |

## Environment Variables Reference

| Variable | Use Case | Value |
|----------|----------|-------|
| `FAST_CI=1` | Simulate GitHub runner | Skip slow/model tests |
| `CPU_ONLY=1` | Force CPU mode | Skip GPU tests |
| `FAST_SIMULATE=1` | Reduce iterations | Use minimal settings |
| `CI=1` | CI environment | Enable CI-specific behavior |
| `SELF_HOSTED=1` | Self-hosted runner | Enable GPU tests |

## Examples

### Run Tests Before Push

```bash
# Quick validation
ruff check --fix src tests
ruff format src tests
pytest -m "unit" -x -v
```

### Test Specific Feature

```bash
# Test chronos2 wrapper with verbose output
pytest tests/test_chronos2_wrapper.py -v -s

# With fast simulation
FAST_SIMULATE=1 pytest tests/test_chronos2_wrapper.py -v
```

### Debug Failing Test

```bash
# Run with full output and stop on failure
pytest tests/test_failing.py -vvs -x --tb=long

# Run only the specific failing test
pytest tests/test_failing.py::TestClass::test_method -vvs
```

### Simulate CI Locally

```bash
# Like Fast CI (GitHub runner)
FAST_CI=1 CPU_ONLY=1 FAST_SIMULATE=1 \
  pytest -m "unit and not slow and not model_required and not cuda_required" \
  --tb=short --maxfail=10

# Like Full CI (self-hosted)
CI=1 SELF_HOSTED=1 pytest --tb=short
```

## Troubleshooting

### Tests Are Too Slow

```bash
# Profile test execution time
pytest --durations=10

# Run only fast tests
pytest -m "unit and not slow"

# Use FAST_SIMULATE
FAST_SIMULATE=1 pytest
```

### Import Errors

```bash
# Ensure dependencies are installed
uv pip install --system -r requirements.txt

# Check PYTHONPATH
export PYTHONPATH=.
pytest
```

### GPU/CUDA Errors

```bash
# Force CPU mode
CPU_ONLY=1 pytest

# Skip GPU tests
pytest -m "not cuda_required and not gpu_required"
```

## More Information

For complete documentation, see [TESTING_AND_CI.md](TESTING_AND_CI.md)
