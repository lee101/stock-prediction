#!/usr/bin/env python3
"""
Global pytest configuration and shared fixtures for Toto retraining system tests.
"""

import pytest
import torch
import numpy as np
import pandas as pd
import warnings
import os
import sys
import tempfile
import shutil
from pathlib import Path
from unittest.mock import patch

# Configure warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)


def pytest_configure(config):
    """Configure pytest settings"""
    # Set random seeds for reproducibility
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
    
    # Configure torch for testing
    set_deterministic = getattr(torch, "set_deterministic", None)
    if callable(set_deterministic):
        set_deterministic(True, warn_only=True)
    else:
        use_deterministic = getattr(torch, "use_deterministic_algorithms", None)
        if callable(use_deterministic):
            use_deterministic(True, warn_only=True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # Set environment variables for testing
    os.environ['TESTING'] = '1'
    os.environ['PYTHONHASHSEED'] = '0'

    for marker in (
        "unit: Unit tests for individual components",
        "integration: Integration tests for system components",
        "performance: Performance and scalability tests",
        "regression: Regression tests to detect behavior changes",
        "slow: Tests that take a long time to run",
        "gpu: Tests that require GPU hardware",
        "data_quality: Tests for data validation and preprocessing",
        "training: Tests related to model training",
    ):
        config.addinivalue_line("markers", marker)


def pytest_unconfigure(config):
    """Cleanup after all tests"""
    # Clean up any test artifacts
    pass


@pytest.fixture(scope="session", autouse=True)
def setup_test_environment():
    """Setup global test environment"""
    # Set up logging for tests
    import logging
    logging.basicConfig(
        level=logging.WARNING,  # Reduce noise during testing
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Disable GPU for consistent testing (unless explicitly testing GPU)
    if not os.environ.get('PYTEST_GPU_TESTS'):
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
    
    yield
    
    # Cleanup
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


@pytest.fixture(scope="session")
def test_data_dir():
    """Create temporary directory for test data"""
    temp_dir = Path(tempfile.mkdtemp(prefix="toto_test_"))
    yield temp_dir
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture(autouse=True)
def reset_random_state():
    """Reset random state before each test for reproducibility"""
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)


@pytest.fixture
def mock_cuda_unavailable():
    """Mock CUDA as unavailable for CPU-only testing"""
    with patch('torch.cuda.is_available', return_value=False):
        yield


@pytest.fixture
def suppress_logging():
    """Suppress logging during tests"""
    import logging
    logging.disable(logging.CRITICAL)
    yield
    logging.disable(logging.NOTSET)


# Skip markers for conditional testing
def pytest_collection_modifyitems(config, items):
    """Modify test collection based on markers and environment"""
    
    # Skip slow tests by default unless --runslow is given
    if not config.getoption("--runslow"):
        skip_slow = pytest.mark.skip(reason="need --runslow option to run")
        for item in items:
            if "slow" in item.keywords:
                item.add_marker(skip_slow)
    
    # Skip GPU tests if CUDA is not available
    if not torch.cuda.is_available():
        skip_gpu = pytest.mark.skip(reason="CUDA not available")
        for item in items:
            if "gpu" in item.keywords:
                item.add_marker(skip_gpu)
    
    # Skip performance tests in CI unless explicitly requested
    if os.environ.get('CI') and not config.getoption("--runperf"):
        skip_perf = pytest.mark.skip(reason="Performance tests skipped in CI")
        for item in items:
            if "performance" in item.keywords:
                item.add_marker(skip_perf)


def pytest_addoption(parser):
    """Add custom command line options"""
    parser.addoption(
        "--runslow", 
        action="store_true", 
        default=False, 
        help="run slow tests"
    )
    parser.addoption(
        "--runperf", 
        action="store_true", 
        default=False, 
        help="run performance tests"
    )
    parser.addoption(
        "--rungpu", 
        action="store_true", 
        default=False, 
        help="run GPU tests"
    )


# Custom pytest markers
pytest_plugins = []

# Fixtures for mocking external dependencies
@pytest.fixture
def mock_mlflow():
    """Mock MLflow tracking"""
    with patch('mlflow.start_run'), \
         patch('mlflow.end_run'), \
         patch('mlflow.log_param'), \
         patch('mlflow.log_metric'), \
         patch('mlflow.log_artifact'):
        yield


@pytest.fixture  
def mock_tensorboard():
    """Mock TensorBoard writer"""
    mock_writer = patch('torch.utils.tensorboard.SummaryWriter')
    with mock_writer as mock_tb:
        mock_instance = mock_tb.return_value
        mock_instance.add_scalar.return_value = None
        mock_instance.add_histogram.return_value = None
        mock_instance.close.return_value = None
        yield mock_instance


@pytest.fixture
def mock_toto_import():
    """Mock Toto model import to avoid dependency"""
    from unittest.mock import MagicMock
    
    mock_toto = MagicMock()
    mock_model = MagicMock()
    mock_model.parameters.return_value = [torch.randn(10, requires_grad=True)]
    mock_model.train.return_value = None
    mock_model.eval.return_value = None
    mock_model.to.return_value = mock_model
    
    # Mock the model output
    mock_output = MagicMock()
    mock_output.loc = torch.randn(1, 10)  # Default shape
    mock_model.model.return_value = mock_output
    
    mock_toto.return_value = mock_model
    
    with patch('toto_ohlc_trainer.Toto', mock_toto):
        yield mock_toto


# Global test configuration
@pytest.fixture(scope="session", autouse=True)
def configure_test_settings():
    """Configure global test settings"""
    # Set pandas options for testing
    pd.set_option('display.max_rows', 10)
    pd.set_option('display.max_columns', 10)
    
    # Configure numpy
    np.seterr(all='warn')
    
    # Configure PyTorch
    torch.set_printoptions(precision=4, sci_mode=False)
    
    yield
    
    # Reset options after tests
    pd.reset_option('display.max_rows')
    pd.reset_option('display.max_columns')


# Helper functions for test data creation
def create_sample_ohlc_data(n_samples=100, symbol="TEST", seed=42):
    """Create sample OHLC data for testing"""
    np.random.seed(seed)
    
    dates = pd.date_range('2023-01-01', periods=n_samples, freq='H')
    base_price = 100.0
    
    # Generate realistic price series
    returns = np.random.normal(0, 0.02, n_samples)
    prices = [base_price]
    
    for ret in returns[1:]:
        new_price = max(prices[-1] * (1 + ret), 0.01)
        prices.append(new_price)
    
    closes = np.array(prices)
    opens = np.concatenate([[closes[0]], closes[:-1]])
    opens += np.random.normal(0, 0.001, n_samples) * opens
    
    # Ensure OHLC relationships
    highs = np.maximum(np.maximum(opens, closes),
                      np.maximum(opens, closes) * (1 + np.abs(np.random.normal(0, 0.005, n_samples))))
    lows = np.minimum(np.minimum(opens, closes),
                     np.minimum(opens, closes) * (1 - np.abs(np.random.normal(0, 0.005, n_samples))))
    
    volumes = np.random.randint(1000, 100000, n_samples)
    
    return pd.DataFrame({
        'timestamp': dates,
        'Open': opens,
        'High': highs,
        'Low': lows,
        'Close': closes,
        'Volume': volumes,
        'Symbol': symbol
    })


@pytest.fixture
def sample_ohlc_data():
    """Fixture providing sample OHLC data"""
    return create_sample_ohlc_data()


@pytest.fixture(params=[100, 500, 1000], ids=["small", "medium", "large"])
def parameterized_ohlc_data(request):
    """Parametrized fixture for different data sizes"""
    n_samples = request.param
    return create_sample_ohlc_data(n_samples, f"TEST_{n_samples}")


# Memory management fixtures
@pytest.fixture(autouse=True)
def cleanup_memory():
    """Cleanup memory after each test"""
    yield
    
    # Force garbage collection
    import gc
    gc.collect()
    
    # Clear CUDA cache if available
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# Error handling for tests
@pytest.fixture
def assert_no_warnings():
    """Context manager to assert no warnings are raised"""
    import warnings
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        yield w
        if w:
            warning_messages = [str(warning.message) for warning in w]
            pytest.fail(f"Unexpected warnings: {warning_messages}")


# Test reporting
def pytest_terminal_summary(terminalreporter, exitstatus, config):
    """Add custom information to test summary"""
    if hasattr(config, 'workerinput'):
        return  # Skip for xdist workers
    
    tr = terminalreporter
    tr.section("Test Environment Summary")
    
    # PyTorch info
    tr.line(f"PyTorch version: {torch.__version__}")
    tr.line(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        tr.line(f"CUDA device count: {torch.cuda.device_count()}")
    
    # NumPy info
    tr.line(f"NumPy version: {np.__version__}")
    
    # Pandas info
    tr.line(f"Pandas version: {pd.__version__}")
    
    # Test counts by marker
    if terminalreporter.stats:
        tr.section("Test Categories")
        for outcome in ['passed', 'failed', 'skipped']:
            if outcome in terminalreporter.stats:
                tests = terminalreporter.stats[outcome]
                markers = {}
                for test in tests:
                    for marker in test.keywords:
                        if marker in ['unit', 'integration', 'performance', 'regression', 'slow', 'gpu']:
                            markers[marker] = markers.get(marker, 0) + 1
                
                if markers:
                    tr.line(f"{outcome.upper()} by category:")
                    for marker, count in markers.items():
                        tr.line(f"  {marker}: {count}")


# Performance tracking
@pytest.fixture
def performance_tracker():
    """Track test performance metrics"""
    import time
    import psutil
    
    start_time = time.time()
    start_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
    
    yield
    
    end_time = time.time()
    end_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
    
    duration = end_time - start_time
    memory_delta = end_memory - start_memory
    
    # Log performance if test took more than 5 seconds or used > 100MB
    if duration > 5.0 or abs(memory_delta) > 100:
        print(f"\nPerformance: {duration:.2f}s, Memory: {memory_delta:+.1f}MB")
# Ensure project root is importable
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

MODULE_ROOT = Path(__file__).resolve().parent
if str(MODULE_ROOT) not in sys.path:
    sys.path.insert(0, str(MODULE_ROOT))
