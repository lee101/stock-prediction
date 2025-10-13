# Toto Retraining System Testing Framework

A comprehensive testing framework for the Toto retraining system, designed for reliability, performance, and CI/CD integration.

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- uv (recommended) or pip for package management

### Setup
```bash
# Install test dependencies
./run_tests.sh deps

# Validate setup
./run_tests.sh validate

# Run development tests (fast)
./run_tests.sh dev
```

### Run All Tests
```bash
# Fast tests only (recommended for development)
./run_tests.sh fast

# All tests including slow ones
./run_tests.sh all --slow
```

## 📋 Test Structure

The testing framework is organized into several categories:

### Test Files
- **`test_toto_trainer.py`** - Unit tests for trainer components
- **`test_integration.py`** - End-to-end integration tests
- **`test_data_quality.py`** - Data validation and preprocessing tests
- **`test_performance.py`** - Performance and scalability tests
- **`test_regression.py`** - Regression tests for consistent behavior
- **`test_fixtures.py`** - Reusable test fixtures and utilities

### Configuration Files
- **`pytest.ini`** - Pytest configuration and markers
- **`conftest.py`** - Global fixtures and test setup
- **`test_runner.py`** - Python test runner with advanced options
- **`run_tests.sh`** - Bash convenience script

## 🏷️ Test Categories

Tests are organized using pytest markers:

### `@pytest.mark.unit`
Unit tests for individual components:
- Configuration classes
- Data preprocessing
- Model initialization
- Loss computation

### `@pytest.mark.integration`
Integration tests for system components:
- End-to-end training pipeline
- Data loading workflows
- Component interaction

### `@pytest.mark.data_quality`
Data validation and preprocessing tests:
- OHLC data consistency
- Missing value handling
- Outlier detection
- Feature engineering

### `@pytest.mark.performance`
Performance and scalability tests:
- Memory usage validation
- Training speed benchmarks
- Resource utilization
- Scalability characteristics

### `@pytest.mark.regression`
Regression tests for consistent behavior:
- Model output consistency
- Data processing determinism
- Configuration stability

### `@pytest.mark.slow`
Tests that take longer to run:
- Large dataset processing
- Extended training scenarios
- Stress testing

### `@pytest.mark.gpu`
GPU-specific tests (requires CUDA):
- GPU memory management
- CUDA computations

## 🛠️ Running Tests

### Using the Shell Script (Recommended)

```bash
# Individual test categories
./run_tests.sh unit                # Unit tests only
./run_tests.sh integration        # Integration tests only
./run_tests.sh data-quality        # Data quality tests
./run_tests.sh performance         # Performance tests (slow)
./run_tests.sh regression          # Regression tests

# Combined test suites
./run_tests.sh fast                # Fast tests (excludes slow)
./run_tests.sh all                 # All tests except slow
./run_tests.sh all --slow          # All tests including slow ones

# Special test suites
./run_tests.sh dev                 # Development suite (fast)
./run_tests.sh ci                  # CI/CD suite (comprehensive)

# Coverage and reporting
./run_tests.sh coverage            # Run with coverage report
./run_tests.sh smoke               # Quick smoke test

# Utilities
./run_tests.sh list                # List all tests
./run_tests.sh cleanup             # Clean up artifacts
```

### Using the Python Runner

```bash
# Basic commands
python test_runner.py unit
python test_runner.py integration --verbose
python test_runner.py performance --output perf_results/

# Specific tests
python test_runner.py specific test_toto_trainer.py
python test_runner.py specific test_data_quality.py::TestOHLCDataValidation

# Advanced options
python test_runner.py all --slow
python test_runner.py coverage --output htmlcov_custom
python test_runner.py report --output detailed_report.json
```

### Using Pytest Directly

```bash
# Basic pytest commands
pytest -v                         # All tests, verbose
pytest -m "unit"                  # Unit tests only
pytest -m "not slow"              # Exclude slow tests
pytest -k "data_quality"          # Tests matching keyword

# Advanced pytest options
pytest --tb=short                 # Short traceback format
pytest -x                         # Stop on first failure
pytest --lf                       # Run last failed tests only
pytest --co                       # Collect tests only (dry run)

# Parallel execution (if pytest-xdist installed)
pytest -n auto                    # Run tests in parallel

# Coverage reporting (if pytest-cov installed)
pytest --cov=. --cov-report=html
```

## 🔧 Configuration

### Pytest Configuration (`pytest.ini`)

Key settings:
- Test discovery patterns
- Default options and markers
- Timeout settings (5 minutes default)
- Warning filters
- Output formatting

### Global Fixtures (`conftest.py`)

Provides:
- Random seed management for reproducibility
- Environment setup and cleanup
- Mock configurations for external dependencies
- Performance tracking
- Memory management

### Test Markers

Configure which tests to run:
```bash
# Run only fast unit tests
pytest -m "unit and not slow"

# Run integration tests excluding GPU tests
pytest -m "integration and not gpu"

# Run all tests except performance tests
pytest -m "not performance"
```

## 📊 Test Data

The testing framework uses synthetic data generation for reliable, reproducible tests:

### Synthetic Data Features
- **Realistic OHLC patterns** - Generated using geometric Brownian motion
- **Configurable parameters** - Volatility, trends, correlations
- **Data quality issues** - Missing values, outliers, invalid relationships
- **Multiple timeframes** - Different frequencies and date ranges
- **Deterministic generation** - Same seed produces identical data

### Test Data Categories
- **Clean data** - Perfect OHLC relationships, no issues
- **Problematic data** - Missing values, outliers, violations
- **Multi-symbol data** - Correlated price series
- **Large datasets** - For performance and memory testing
- **Edge cases** - Empty data, single rows, extreme values

## 🏃‍♂️ Performance Testing

Performance tests validate:

### Memory Usage
- Peak memory consumption
- Memory growth over time
- Memory leak detection
- Batch processing efficiency

### Execution Speed
- Data loading performance
- Model initialization time
- Training step duration
- Preprocessing overhead

### Scalability
- Linear scaling with data size
- Batch size impact
- Sequence length effects
- Multi-symbol handling

### Resource Utilization
- CPU usage patterns
- GPU memory management (if available)
- I/O efficiency

## 🔄 Regression Testing

Regression tests ensure consistent behavior across changes:

### Data Processing
- Deterministic preprocessing
- Consistent feature extraction
- Stable technical indicators

### Model Behavior
- Deterministic forward passes
- Consistent loss computation
- Reproducible training steps

### Configuration Management
- Stable configuration hashing
- Consistent serialization
- Parameter preservation

## 🚨 CI/CD Integration

### GitHub Actions Example
```yaml
name: Tests
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v3
    - uses: actions/setup-python@v4
      with:
        python-version: '3.9'
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip uv
        ./run_tests.sh deps
    - name: Run CI test suite
      run: ./run_tests.sh ci
    - name: Upload coverage reports
      uses: codecov/codecov-action@v3
      if: success()
```

### Test Stages
1. **Validation** - Environment and dependency check
2. **Unit Tests** - Fast component tests
3. **Integration Tests** - System interaction tests
4. **Data Quality Tests** - Data validation tests
5. **Regression Tests** - Consistency verification

## 🔍 Debugging Tests

### Common Issues

**Import Errors**
```bash
# Check Python path
python -c "import sys; print(sys.path)"

# Verify dependencies
./run_tests.sh validate
```

**Memory Issues**
```bash
# Run with memory monitoring
pytest --tb=short -v test_performance.py::TestMemoryUsage
```

**Slow Tests**
```bash
# Profile test execution
pytest --durations=10

# Run only fast tests
./run_tests.sh fast
```

**Random Failures**
```bash
# Check for non-deterministic behavior
pytest test_regression.py -v --tb=long
```

### Debug Mode
```bash
# Run with Python debugger
pytest --pdb test_toto_trainer.py::test_failing_function

# Capture output (disable capture)
pytest -s test_integration.py
```

## 📈 Coverage Reporting

Generate coverage reports:

```bash
# HTML coverage report
./run_tests.sh coverage

# Terminal coverage report
pytest --cov=. --cov-report=term-missing

# XML coverage report (for CI)
pytest --cov=. --cov-report=xml
```

Coverage reports show:
- Line coverage percentage
- Branch coverage
- Missing lines
- Excluded files

## 🛡️ Mocking and Fixtures

The testing framework provides comprehensive mocking:

### Model Mocking
- **MockTotoModel** - Complete Toto model mock
- **Deterministic outputs** - Consistent predictions
- **Configurable behavior** - Customize for test scenarios

### Data Mocking
- **SyntheticDataFactory** - Generate test data
- **Configurable patterns** - Control data characteristics
- **Issue injection** - Add data quality problems

### External Dependencies
- **MLflow mocking** - Avoid external service calls
- **TensorBoard mocking** - Mock logging functionality
- **CUDA mocking** - Test GPU code without GPU

### Global Fixtures
Available fixtures:
- `sample_ohlc_data` - Basic OHLC dataset
- `mock_toto_model` - Mocked Toto model
- `temp_test_directory` - Temporary directory
- `regression_manager` - Regression test utilities

## 📝 Writing New Tests

### Test Structure
```python
import pytest
from test_fixtures import SyntheticDataFactory, MockTotoModel

class TestNewFeature:
    """Test new feature functionality"""
    
    @pytest.fixture
    def test_data(self):
        """Create test data"""
        factory = SyntheticDataFactory(seed=42)
        return factory.create_basic_ohlc_data(100)
    
    @pytest.mark.unit
    def test_basic_functionality(self, test_data):
        """Test basic functionality"""
        # Test implementation
        assert True
    
    @pytest.mark.integration
    def test_system_integration(self, test_data, mock_toto_model):
        """Test system integration"""
        # Integration test implementation
        assert True
    
    @pytest.mark.slow
    def test_large_scale_processing(self):
        """Test with large datasets"""
        # Slow test implementation
        pytest.skip("Slow test - run with --runslow")
```

### Best Practices
1. **Use descriptive names** - Clear test and function names
2. **Test single concepts** - One assertion per test when possible
3. **Use appropriate markers** - Categorize tests correctly
4. **Mock dependencies** - Isolate units under test
5. **Generate deterministic data** - Use fixed seeds
6. **Clean up resources** - Use fixtures for setup/teardown
7. **Document test intent** - Clear docstrings and comments

### Adding New Test Categories
1. Add marker to `pytest.ini`
2. Update `test_runner.py` with new command
3. Add shell script command in `run_tests.sh`
4. Document in this README

## 🔧 Maintenance

### Regular Tasks
- **Update test data** - Refresh synthetic datasets periodically
- **Review performance baselines** - Adjust thresholds as system evolves
- **Update regression references** - When intentional changes occur
- **Clean up artifacts** - Remove old test outputs

### Monitoring Test Health
- **Test execution times** - Watch for performance degradation
- **Memory usage trends** - Monitor for memory leaks
- **Flaky test detection** - Identify non-deterministic tests
- **Coverage trends** - Maintain good test coverage

## 📞 Support

### Common Commands Quick Reference
```bash
./run_tests.sh help           # Show help
./run_tests.sh validate       # Check setup
./run_tests.sh dev            # Quick development tests
./run_tests.sh ci             # Full CI suite
./run_tests.sh cleanup        # Clean up artifacts
```

### Getting Help
- Check test output for specific error messages
- Run validation to verify environment setup
- Use verbose mode (`-v`) for detailed output
- Check pytest documentation for advanced features

### Contributing
When adding new tests:
1. Follow existing patterns and conventions
2. Add appropriate test markers
3. Include documentation
4. Verify tests pass in clean environment
5. Update this README if needed

---

**Happy Testing! 🧪✨**