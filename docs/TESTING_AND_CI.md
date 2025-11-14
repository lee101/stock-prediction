# Testing and CI Guide

This document describes the testing infrastructure, test categorization, and CI/CD setup for the stock-prediction project.

## Table of Contents

- [Overview](#overview)
- [Test Categorization](#test-categorization)
- [Running Tests](#running-tests)
- [CI/CD Workflows](#cicd-workflows)
- [Environment Variables](#environment-variables)
- [Writing Tests](#writing-tests)
- [Linting and Formatting](#linting-and-formatting)

## Overview

The project uses a comprehensive testing and CI strategy with two main CI workflows:

1. **Fast CI (GitHub Runners)** - Runs on standard GitHub runners with CPU-only, focusing on fast unit tests and linting
2. **Full CI (Self-Hosted GPU)** - Runs on self-hosted runners with GPU support, executing the full test suite including model and integration tests

## Test Categorization

Tests are organized using pytest markers. Each test should be marked with appropriate categories to enable efficient test selection in CI.

### Primary Test Categories

#### `@pytest.mark.unit`
Fast unit tests that don't require external services or heavy models. These run by default in Fast CI.

```python
@pytest.mark.unit
def test_price_calculation():
    result = compute_price_ratio(100, 110)
    assert result == 1.1
```

#### `@pytest.mark.integration`
Integration tests that may use models, databases, or coordinate multiple components. These run in Full CI.

```python
@pytest.mark.integration
def test_backtest_pipeline():
    # Test full backtest workflow
    ...
```

#### `@pytest.mark.slow`
Tests that take >10 seconds to run. Skipped in Fast CI mode.

```python
@pytest.mark.slow
@pytest.mark.integration
def test_full_training_loop():
    # Long-running training test
    ...
```

#### `@pytest.mark.model_required`
Tests that require loading ML models (Chronos, NeuralForecast, etc.). Only smoke-marked model tests run in Fast CI.

```python
@pytest.mark.model_required
def test_chronos_forecast():
    model = load_chronos_model()
    ...
```

#### `@pytest.mark.smoke`
Minimal smoke tests for quick validation. Even model tests marked as smoke can run in Fast CI.

```python
@pytest.mark.smoke
@pytest.mark.model_required
def test_model_loads():
    # Quick check that model can be initialized
    model = ChronosModel()
    assert model is not None
```

### Resource Requirement Markers

#### `@pytest.mark.cuda_required`
Tests that require CUDA-enabled PyTorch. Skipped on CPU-only runners.

```python
@pytest.mark.cuda_required
def test_gpu_training():
    assert torch.cuda.is_available()
    ...
```

#### `@pytest.mark.gpu_required`
Tests that require any GPU (CUDA or other). Skipped on CPU-only runners.

#### `@pytest.mark.cpu_only`
Tests that should only run on CPU (for testing CPU fallback paths).

```python
@pytest.mark.cpu_only
def test_cpu_fallback():
    # Ensure model works without GPU
    ...
```

#### `@pytest.mark.memory_intensive`
Tests requiring significant RAM (>8GB).

### External Dependency Markers

#### `@pytest.mark.external`
Tests that hit external services. Skipped in CI unless `RUN_EXTERNAL_TESTS=1`.

```python
@pytest.mark.external
def test_api_integration():
    # Calls real API
    ...
```

#### `@pytest.mark.requires_openai`
Tests needing live OpenAI API access.

#### `@pytest.mark.requires_alpaca`
Tests needing Alpaca API access.

#### `@pytest.mark.requires_binance`
Tests needing Binance API access.

#### `@pytest.mark.network_required`
Tests requiring internet connectivity.

### CI-Specific Markers

#### `@pytest.mark.ci_skip`
Tests to skip in all CI environments (e.g., tests that require manual intervention).

```python
@pytest.mark.ci_skip
def test_manual_verification():
    # Requires human verification
    ...
```

#### `@pytest.mark.self_hosted_only`
Tests that require self-hosted runner features (GPU, specific hardware).

```python
@pytest.mark.self_hosted_only
@pytest.mark.cuda_required
def test_multi_gpu_training():
    # Requires multiple GPUs
    ...
```

### Special Test Types

#### `@pytest.mark.benchmark`
Performance benchmark tests.

#### `@pytest.mark.experimental`
Tests in `tests/experimental/` (skipped unless `--run-experimental` flag is used).

#### `@pytest.mark.auto_generated`
Auto-generated coverage tests (off by default).

## Running Tests

### Run All Tests
```bash
pytest
```

### Run Only Unit Tests
```bash
pytest -m "unit"
```

### Run Fast Tests (No Slow, No Models)
```bash
pytest -m "unit and not slow and not model_required"
```

### Run Integration Tests
```bash
pytest -m "integration"
```

### Run Tests Without CUDA
```bash
CPU_ONLY=1 pytest -m "not cuda_required and not gpu_required"
```

### Run in Fast CI Mode (Simulating GitHub Runner)
```bash
FAST_CI=1 CPU_ONLY=1 FAST_SIMULATE=1 pytest
```

### Run Experimental Tests
```bash
pytest --run-experimental
```

### Run Specific Test File
```bash
pytest tests/test_price_calculations.py -v
```

## CI/CD Workflows

### Fast CI (`.github/workflows/ci-fast.yml`)

**Triggers:** Push to main, pull requests
**Runner:** `ubuntu-latest` (GitHub-hosted)
**Python Versions:** 3.11, 3.13

**Jobs:**
1. **Lint** - Ruff, Black, isort checks
2. **Fast Unit Tests** - Only unit tests, CPU-only PyTorch
3. **Type Check** - ty, Pyright, mypy

**Environment:**
```yaml
CI: "1"
FAST_CI: "1"
CPU_ONLY: "1"
FAST_SIMULATE: "1"
```

**Test Selection:**
```bash
pytest -m "unit and not slow and not model_required and not cuda_required"
```

### Full CI (`.github/workflows/ci.yml`)

**Triggers:** Push to main, pull requests
**Runner:** Self-hosted with GPU
**Python Version:** 3.13

**Jobs:**
1. **Quality** - Full linting and type checking
2. **Smart Test Suite** - Change-aware test selection
3. **Integration Tests** - Full model and integration tests
4. **Benchmarks** - Fast env benchmark, PPO smoke run
5. **Simulator** - Market simulator reports and trend analysis

**Environment:**
```yaml
CI: "1"
SELF_HOSTED: "1"
# FAST_CI and CPU_ONLY are NOT set
```

**Test Selection:**
- Runs all tests including slow, model_required, and cuda_required
- Uses smart test runner for efficient change detection

## Environment Variables

### CI Mode Control

| Variable | Description | Values |
|----------|-------------|--------|
| `CI` | Running in CI environment | `0` or `1` |
| `FAST_CI` | Fast CI mode (limited tests) | `0` or `1` |
| `CPU_ONLY` | CPU-only mode (no GPU) | `0` or `1` |
| `FAST_SIMULATE` | Fast simulation (minimal iterations) | `0` or `1` |
| `SELF_HOSTED` | Running on self-hosted runner | `0` or `1` |

### Test Filtering

| Variable | Description | Values |
|----------|-------------|--------|
| `RUN_EXTERNAL_TESTS` | Enable external/network tests | `0` or `1` |
| `USE_REAL_ENV` | Use real environment (credentials) | `0` or `1` |
| `SKIP_TORCH_CHECK` | Skip PyTorch requirement check | `0` or `1` |

### Market Simulator

| Variable | Description | Default |
|----------|-------------|---------|
| `MARKETSIM_ALLOW_MOCK_ANALYTICS` | Allow mock analytics | `1` |
| `MARKETSIM_SKIP_REAL_IMPORT` | Skip real imports | `1` |
| `MARKETSIM_ALLOW_CPU_FALLBACK` | Allow CPU fallback | `1` |

## Writing Tests

### Best Practices

1. **Always mark your tests** with appropriate categories:
   ```python
   @pytest.mark.unit
   def test_my_function():
       ...
   ```

2. **Use fixtures for CI modes** to adapt test behavior:
   ```python
   def test_model_inference(fast_ci_mode, fast_model_config):
       if fast_ci_mode:
           config = fast_model_config  # Use minimal settings
       else:
           config = full_model_config  # Use production settings
       ...
   ```

3. **Mark resource requirements** explicitly:
   ```python
   @pytest.mark.cuda_required
   def test_gpu_feature():
       ...
   ```

4. **Skip expensive tests conditionally**:
   ```python
   def test_expensive_operation(fast_simulate_mode):
       if fast_simulate_mode:
           iterations = 10
       else:
           iterations = 10000
       ...
   ```

### Available Fixtures

#### CI Mode Fixtures

- `ci_mode` - Returns True if running in CI
- `fast_ci_mode` - Returns True if Fast CI mode
- `cpu_only_mode` - Returns True if CPU-only
- `fast_simulate_mode` - Returns True if fast simulation enabled

#### Configuration Fixtures

- `fast_model_config` - Dict with minimal model settings
- `fast_simulation_config` - Dict with minimal simulation settings
- `torch_device` - Returns appropriate torch device (`'cpu'` or `'cuda'`)

#### Auto-Applied Fixtures

- `setup_fast_simulate_env` - Automatically sets `FAST_SIMULATE` env var when enabled

### Example Test Structure

```python
import pytest

@pytest.mark.unit
def test_pure_calculation():
    """Fast unit test with no dependencies."""
    result = my_calculation(10, 20)
    assert result == 30

@pytest.mark.integration
@pytest.mark.model_required
def test_model_integration(fast_ci_mode, fast_model_config):
    """Integration test that adapts to CI mode."""
    if fast_ci_mode:
        # Use minimal config in Fast CI
        config = fast_model_config
    else:
        # Use full config in Full CI
        config = {
            "context_length": 512,
            "prediction_length": 96,
            "num_samples": 100,
        }

    model = MyModel(**config)
    result = model.predict(data)
    assert result is not None

@pytest.mark.slow
@pytest.mark.integration
@pytest.mark.self_hosted_only
def test_full_training():
    """Long-running test, only runs on self-hosted."""
    # This test will be skipped in Fast CI
    ...
```

## Linting and Formatting

### Tools

The project uses modern Python tooling with Ruff as the primary linter and formatter:

- **Ruff** - Fast all-in-one linter and formatter (replaces Black, isort, Flake8, and more)
- **Flake8** - Additional linting (optional, mostly replaced by Ruff)
- **Pyright** - Type checker
- **mypy** - Type checker
- **ty** - Type checker

Ruff is the recommended tool as it's extremely fast and handles:
- Code linting (E, W, F rules from Flake8/Pyflakes)
- Code formatting (Black-compatible)
- Import sorting (isort-compatible)
- Additional rules (pyupgrade, flake8-bugbear, etc.)

### Running Linters Locally

```bash
# Ruff - lint and format (primary tool)
ruff check src tests
ruff format src tests

# Type checking
ty check
pyright src
mypy src
```

### Auto-Fix

```bash
# Ruff auto-fix linting issues
ruff check --fix src tests

# Ruff auto-format code
ruff format src tests
```

### Configuration

All linter configuration is in `pyproject.toml`:

- `[tool.ruff]` - Ruff settings (excludes, line length, target version)
- `[tool.ruff.lint]` - Ruff linting rules (select/ignore)
- `[tool.ruff.format]` - Ruff formatting (quote style, indentation)
- `[tool.ruff.lint.isort]` - Import sorting configuration
- `[tool.mypy]` - mypy settings

### CI Linting

Fast CI runs:
```bash
ruff check src tests --output-format=github
ruff format --check src tests
```

Full CI runs the same plus type checkers:
```bash
ruff check src tests
ruff format --check src tests
ty check
pyright src
```

## Tips and Tricks

### Running Tests Like CI

**Simulate Fast CI:**
```bash
FAST_CI=1 CPU_ONLY=1 FAST_SIMULATE=1 \
  pytest -m "unit and not slow and not model_required and not cuda_required"
```

**Simulate Full CI:**
```bash
CI=1 SELF_HOSTED=1 pytest
```

### Debugging CI Failures

1. Check which markers are applied to failing tests
2. Run with the same environment variables as CI
3. Use `-v` for verbose output
4. Use `--tb=short` for concise tracebacks

### Performance Optimization

- Mark slow tests with `@pytest.mark.slow` so they're skipped in Fast CI
- Use `fast_model_config` fixture in tests that can run with minimal settings
- Parametrize tests to run multiple scenarios efficiently
- Use `@pytest.mark.parametrize` instead of loops

### Common Issues

**Test skipped in Fast CI but should run:**
- Ensure it's marked with `@pytest.mark.unit`
- Remove `@pytest.mark.slow` if it's actually fast
- Remove `@pytest.mark.model_required` if it doesn't need models

**Test fails only in CI:**
- Check environment variables match
- Verify GPU/CPU assumptions
- Check for hardcoded paths
- Ensure deterministic behavior

**Test too slow for Fast CI:**
- Mark with `@pytest.mark.slow`
- Use `fast_simulate_mode` fixture to reduce iterations
- Consider splitting into unit and integration versions
