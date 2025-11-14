# CI Modernization Summary

## Overview

This document summarizes the modernization of the testing and CI infrastructure for the stock-prediction project. The goal was to create a flexible, efficient CI system that can run on both self-hosted GPU runners and standard GitHub CPU runners.

## What Was Done

### 1. Test Categorization System

**File:** `pytest.ini`

Added comprehensive pytest markers to categorize tests:
- **Test Categories:** `unit`, `integration`, `slow`, `model_required`, `smoke`
- **Resource Markers:** `cuda_required`, `gpu_required`, `cpu_only`, `memory_intensive`
- **External Markers:** `external`, `requires_openai`, `requires_alpaca`, `network_required`
- **CI Markers:** `ci_skip`, `self_hosted_only`, `benchmark`, `experimental`

This allows fine-grained control over which tests run in different environments.

### 2. Linting Configuration

**File:** `pyproject.toml`

Configured Ruff as the primary linter and formatter:
- **Replaced:** Black, isort, and most Flake8 functionality
- **Rules:** Pycodestyle, Pyflakes, pep8-naming, pyupgrade, flake8-bugbear, flake8-comprehensions
- **Features:** Built-in formatter, import sorter, comprehensive linting
- **Performance:** Extremely fast Rust-based tool

### 3. CI Mode Fixtures

**File:** `tests/conftest.py`

Added pytest fixtures and automatic test filtering:
- **Session Fixtures:** `ci_mode`, `fast_ci_mode`, `cpu_only_mode`, `fast_simulate_mode`
- **Config Fixtures:** `fast_model_config`, `fast_simulation_config`, `torch_device`
- **Auto Fixtures:** `setup_fast_simulate_env` (auto-applies FAST_SIMULATE)
- **Smart Filtering:** Automatically skips slow/GPU/model tests based on environment

### 4. Fast CI Workflow (GitHub Runners)

**File:** `.github/workflows/ci-fast.yml`

New lightweight CI for standard GitHub runners:
- **Runners:** `ubuntu-latest` (GitHub-hosted)
- **Python:** 3.11 and 3.13
- **PyTorch:** CPU-only version
- **Jobs:**
  - Lint & Format Check (Ruff)
  - Fast Unit Tests (unit tests only, no models)
  - Type Checking (ty, Pyright, mypy)
- **Environment:**
  - `CI=1`, `FAST_CI=1`, `CPU_ONLY=1`, `FAST_SIMULATE=1`
- **Test Selection:**
  - `-m "unit and not slow and not model_required and not cuda_required"`
  - Minimal smoke tests with models

### 5. Updated Full CI Workflow (Self-Hosted)

**File:** `.github/workflows/ci.yml`

Enhanced existing self-hosted CI:
- **Runners:** Self-hosted with GPU
- **Python:** 3.13
- **Environment:**
  - `CI=1`, `SELF_HOSTED=1`
  - FAST_CI NOT set (runs full suite)
  - CPU_ONLY NOT set (GPU enabled)
- **Enhanced Linting:** Ruff check + format
- **Full Test Suite:** All tests including slow, model, and GPU tests

### 6. Documentation

Created comprehensive documentation:

- **`docs/TESTING_AND_CI.md`** - Complete guide to testing and CI
  - Test categorization reference
  - Running tests guide
  - CI workflows explanation
  - Environment variables reference
  - Writing tests best practices
  - Linting and formatting guide

- **`docs/TESTING_QUICK_START.md`** - Quick reference
  - Common commands
  - Test running patterns
  - Linting commands
  - Environment variable reference
  - Examples and troubleshooting

- **`docs/CI_MODERNIZATION_SUMMARY.md`** - This file
  - Overview of changes
  - Migration guide
  - Benefits summary

- **`tests/example_test_markers.py`** - Example test file
  - Demonstrates all marker types
  - Shows fixture usage
  - Provides copy-paste templates

## Key Features

### 1. Dual CI Strategy

- **Fast CI:** Runs on every push/PR, provides quick feedback (<5 min)
- **Full CI:** Comprehensive testing on self-hosted hardware (slower but complete)

### 2. Smart Test Selection

Tests are automatically filtered based on environment:
- Fast CI skips slow, model, and GPU tests
- CPU-only mode skips CUDA tests
- CI mode skips external/network tests
- Self-hosted tests can use GPU features

### 3. Environment-Aware Testing

Tests can adapt their behavior using fixtures:
```python
def test_model(fast_ci_mode, fast_model_config):
    if fast_ci_mode:
        config = fast_model_config  # Minimal settings
    else:
        config = production_config  # Full settings
```

### 4. Modern Tooling

- **Ruff:** Single tool replaces Black, isort, Flake8, pyupgrade
- **Fast:** Rust-based, extremely quick linting and formatting
- **Consistent:** Same tool for linting, formatting, and import sorting

## Environment Variables

### CI Mode Control

| Variable | Purpose | Fast CI | Full CI |
|----------|---------|---------|---------|
| `CI` | Running in CI | ✅ 1 | ✅ 1 |
| `FAST_CI` | Limited test suite | ✅ 1 | ❌ 0 |
| `CPU_ONLY` | No GPU available | ✅ 1 | ❌ 0 |
| `FAST_SIMULATE` | Minimal iterations | ✅ 1 | ❌ 0 |
| `SELF_HOSTED` | Self-hosted runner | ❌ 0 | ✅ 1 |

### Test Filtering

| Variable | Purpose | Default |
|----------|---------|---------|
| `RUN_EXTERNAL_TESTS` | Enable external tests | 0 |
| `USE_REAL_ENV` | Use real credentials | 0 |
| `SKIP_TORCH_CHECK` | Skip PyTorch check | 0 |

## Benefits

### For Developers

1. **Faster Feedback:** Fast CI provides quick feedback on unit tests
2. **Local Testing:** Can simulate CI environments locally
3. **Flexible:** Run exactly the tests you need
4. **Clear Markers:** Easy to understand what each test does

### For CI/CD

1. **Cost Efficient:** Fast CI runs on free GitHub runners
2. **Parallel Execution:** Fast and Full CI can run simultaneously
3. **Resource Optimization:** GPU tests only run where GPUs are available
4. **Fail Fast:** Quick unit tests catch common errors early

### For Codebase

1. **Organized:** Clear test categorization
2. **Maintainable:** Well-documented testing practices
3. **Scalable:** Easy to add new test categories
4. **Consistent:** Standardized linting and formatting

## Migration Guide

### For Existing Tests

1. **Add markers to tests:**
   ```python
   # Before
   def test_something():
       ...

   # After
   @pytest.mark.unit
   def test_something():
       ...
   ```

2. **Mark slow tests:**
   ```python
   @pytest.mark.slow
   @pytest.mark.integration
   def test_long_operation():
       ...
   ```

3. **Mark model tests:**
   ```python
   @pytest.mark.model_required
   @pytest.mark.integration
   def test_chronos_model():
       ...
   ```

4. **Mark GPU tests:**
   ```python
   @pytest.mark.cuda_required
   def test_gpu_feature():
       ...
   ```

### For New Tests

1. **Always add appropriate markers**
2. **Use CI fixtures for adaptive behavior**
3. **Keep unit tests fast (<1s each)**
4. **Use `fast_model_config` for model tests**

See `tests/example_test_markers.py` for complete examples.

## Running Tests

### Locally

```bash
# Fast unit tests only
pytest -m "unit"

# Simulate Fast CI
FAST_CI=1 CPU_ONLY=1 pytest -m "unit and not slow and not model_required"

# Simulate Full CI
CI=1 SELF_HOSTED=1 pytest

# All tests except slow
pytest -m "not slow"
```

### Linting

```bash
# Check
ruff check src tests
ruff format --check src tests

# Fix
ruff check --fix src tests
ruff format src tests
```

## CI Workflows

### When They Run

Both workflows run on:
- Push to `main`
- Pull requests

They run in parallel, providing both quick feedback (Fast CI) and comprehensive validation (Full CI).

### Expected Timing

- **Fast CI:** ~3-5 minutes
  - Lint: ~30s
  - Unit Tests: ~2-3 min
  - Type Check: ~1-2 min

- **Full CI:** ~15-30 minutes (depending on suite)
  - Full linting: ~1 min
  - All tests: ~10-20 min
  - Benchmarks: ~5 min
  - Simulator: ~5-10 min

## Future Improvements

### Potential Enhancements

1. **Test Sharding:** Split tests across multiple runners
2. **Cache Optimization:** Cache dependencies and models
3. **Parallel Execution:** Use pytest-xdist for parallel tests
4. **Coverage Reports:** Add coverage tracking and reporting
5. **Performance Tracking:** Track test execution time trends
6. **Matrix Testing:** Test more Python versions
7. **Nightly Builds:** Comprehensive nightly test runs

### Test Organization

1. **Mark More Tests:** Continue adding markers to existing tests
2. **Smoke Tests:** Identify and mark minimal validation tests
3. **Benchmark Suite:** Expand performance benchmark tests
4. **Integration Groups:** Create test groups for different systems

## Resources

- **Testing Guide:** `docs/TESTING_AND_CI.md`
- **Quick Start:** `docs/TESTING_QUICK_START.md`
- **Example Tests:** `tests/example_test_markers.py`
- **Pytest Docs:** https://docs.pytest.org/
- **Ruff Docs:** https://docs.astral.sh/ruff/

## Questions?

For questions or issues:
1. Check the documentation in `docs/`
2. Look at example tests in `tests/example_test_markers.py`
3. Review CI workflow files in `.github/workflows/`
4. Consult pytest and Ruff documentation

## Summary

The modernized CI system provides:
- ✅ Fast feedback with CPU-only Fast CI
- ✅ Comprehensive testing with GPU Full CI
- ✅ Clear test categorization with pytest markers
- ✅ Modern linting with Ruff
- ✅ Environment-aware testing with fixtures
- ✅ Comprehensive documentation

This enables efficient development while maintaining high code quality and test coverage.
