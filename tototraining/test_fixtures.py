#!/usr/bin/env python3
"""
Test fixtures and mocking utilities for reliable testing of the Toto retraining system.
Provides reusable fixtures, mocks, and test utilities.
"""

import pytest
import torch
import numpy as np
import pandas as pd
import tempfile
import shutil
import json
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, asdict
import warnings

# Import modules to create fixtures for
from toto_ohlc_trainer import TotoOHLCConfig, TotoOHLCTrainer
from toto_ohlc_dataloader import DataLoaderConfig, OHLCPreprocessor, TotoOHLCDataLoader
from enhanced_trainer import EnhancedTotoTrainer

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning)


@dataclass
class TestScenario:
    """Define test scenario parameters"""
    name: str
    data_size: int
    n_symbols: int
    sequence_length: int
    prediction_length: int
    batch_size: int
    has_missing_data: bool = False
    has_outliers: bool = False
    has_irregular_timestamps: bool = False


class MockTotoModel:
    """Comprehensive mock for Toto model"""
    
    def __init__(self, config: TotoOHLCConfig, input_dim: int = 5):
        self.config = config
        self.input_dim = input_dim
        self._create_mock_structure()
    
    def _create_mock_structure(self):
        """Create the mock model structure"""
        # Main model mock
        self.model = Mock()
        
        # Parameters mock
        self._parameters = [torch.randn(100, requires_grad=True) for _ in range(5)]
        
        # Training/eval modes
        self.train = Mock()
        self.eval = Mock()
        
        # Device handling
        self.to = Mock(return_value=self)
        self.device = torch.device('cpu')
        
        # Configure model forward pass
        self._setup_forward_pass()
    
    def _setup_forward_pass(self):
        """Setup realistic forward pass behavior"""
        def mock_forward(x_reshaped, input_padding_mask, id_mask):
            batch_size = x_reshaped.shape[0]
            
            # Create mock output with proper structure
            mock_output = Mock()
            
            # Location parameter (predictions)
            mock_output.loc = torch.randn(batch_size, self.config.prediction_length)
            
            # Scale parameter (uncertainty)
            mock_output.scale = torch.ones(batch_size, self.config.prediction_length) * 0.1
            
            # Distribution for sampling
            mock_output.distribution = Mock()
            mock_output.distribution.sample = Mock(
                return_value=torch.randn(batch_size, self.config.prediction_length)
            )
            
            return mock_output
        
        self.model.side_effect = mock_forward
    
    def parameters(self):
        """Return mock parameters"""
        return iter(self._parameters)
    
    def state_dict(self):
        """Return mock state dict"""
        return {f'layer_{i}.weight': param for i, param in enumerate(self._parameters)}
    
    def load_state_dict(self, state_dict):
        """Mock loading state dict"""
        pass


class SyntheticDataFactory:
    """Factory for creating various types of synthetic test data"""
    
    def __init__(self, seed: int = 42):
        self.seed = seed
        np.random.seed(seed)
    
    def create_basic_ohlc_data(
        self, 
        n_samples: int, 
        symbol: str = "TEST",
        base_price: float = 100.0,
        volatility: float = 0.02,
        start_date: str = "2023-01-01",
        freq: str = "H"
    ) -> pd.DataFrame:
        """Create basic OHLC data"""
        dates = pd.date_range(start_date, periods=n_samples, freq=freq)
        
        # Generate close prices using geometric Brownian motion
        dt = 1.0 / 252  # Daily time step
        drift = 0.05    # 5% annual drift
        
        prices = [base_price]
        for _ in range(n_samples - 1):
            random_shock = np.random.normal(0, 1)
            price_change = prices[-1] * (drift * dt + volatility * np.sqrt(dt) * random_shock)
            new_price = max(prices[-1] + price_change, 0.01)  # Ensure positive
            prices.append(new_price)
        
        close_prices = np.array(prices)
        
        # Generate OHLC from close prices
        opens = np.concatenate([[close_prices[0]], close_prices[:-1]])
        opens += np.random.normal(0, volatility * 0.1, n_samples) * opens  # Small gaps
        
        # Ensure realistic OHLC relationships
        highs = []
        lows = []
        volumes = []
        
        for i in range(n_samples):
            open_price = opens[i]
            close_price = close_prices[i]
            
            # High is max(open, close) + some upward movement
            high_addition = abs(np.random.normal(0, volatility * 0.3)) * max(open_price, close_price)
            high_price = max(open_price, close_price) + high_addition
            
            # Low is min(open, close) - some downward movement
            low_subtraction = abs(np.random.normal(0, volatility * 0.3)) * min(open_price, close_price)
            low_price = min(open_price, close_price) - low_subtraction
            
            # Volume follows log-normal distribution
            volume = max(int(np.random.lognormal(9, 1)), 1)
            
            highs.append(high_price)
            lows.append(max(low_price, 0.01))  # Ensure positive
            volumes.append(volume)
        
        return pd.DataFrame({
            'timestamp': dates,
            'Open': opens,
            'High': highs,
            'Low': lows,
            'Close': close_prices,
            'Volume': volumes,
            'Symbol': symbol
        })
    
    def create_data_with_issues(
        self, 
        n_samples: int,
        symbol: str = "PROBLEMATIC",
        issue_types: List[str] = None
    ) -> pd.DataFrame:
        """Create OHLC data with various data quality issues"""
        if issue_types is None:
            issue_types = ['missing', 'outliers', 'invalid_ohlc']
        
        # Start with basic data
        data = self.create_basic_ohlc_data(n_samples, symbol)
        
        if 'missing' in issue_types:
            # Add missing values
            missing_indices = np.random.choice(n_samples, size=max(1, n_samples // 20), replace=False)
            data.loc[missing_indices, 'Close'] = np.nan
            
            missing_indices = np.random.choice(n_samples, size=max(1, n_samples // 30), replace=False)
            data.loc[missing_indices, 'Volume'] = np.nan
        
        if 'outliers' in issue_types:
            # Add price outliers
            outlier_indices = np.random.choice(n_samples, size=max(1, n_samples // 50), replace=False)
            for idx in outlier_indices:
                multiplier = np.random.choice([10, 0.1])  # 10x or 0.1x normal price
                data.loc[idx, 'Close'] = data.loc[idx, 'Close'] * multiplier
            
            # Add volume outliers
            vol_outlier_indices = np.random.choice(n_samples, size=max(1, n_samples // 40), replace=False)
            for idx in vol_outlier_indices:
                data.loc[idx, 'Volume'] = data.loc[idx, 'Volume'] * np.random.uniform(50, 100)
        
        if 'invalid_ohlc' in issue_types:
            # Violate OHLC relationships
            violation_indices = np.random.choice(n_samples, size=max(1, n_samples // 30), replace=False)
            for idx in violation_indices:
                # Make high lower than close
                data.loc[idx, 'High'] = data.loc[idx, 'Close'] * 0.9
                # Make low higher than open
                data.loc[idx, 'Low'] = data.loc[idx, 'Open'] * 1.1
        
        if 'negative_prices' in issue_types:
            # Add negative prices
            neg_indices = np.random.choice(n_samples, size=max(1, n_samples // 100), replace=False)
            data.loc[neg_indices, 'Low'] = -abs(data.loc[neg_indices, 'Low'])
        
        if 'infinite_values' in issue_types:
            # Add infinite values
            inf_indices = np.random.choice(n_samples, size=max(1, n_samples // 200), replace=False)
            data.loc[inf_indices[0], 'High'] = np.inf
            if len(inf_indices) > 1:
                data.loc[inf_indices[1], 'Low'] = -np.inf
        
        return data
    
    def create_multi_symbol_data(
        self, 
        symbols: List[str], 
        n_samples: int = 1000,
        correlation: float = 0.3
    ) -> Dict[str, pd.DataFrame]:
        """Create correlated multi-symbol data"""
        data = {}
        base_returns = np.random.normal(0, 0.02, n_samples)
        
        for i, symbol in enumerate(symbols):
            # Create correlated returns
            symbol_returns = (
                correlation * base_returns + 
                (1 - correlation) * np.random.normal(0, 0.02, n_samples)
            )
            
            # Generate prices from returns
            base_price = 100 + i * 20  # Different base prices
            prices = [base_price]
            
            for ret in symbol_returns[1:]:
                new_price = max(prices[-1] * (1 + ret), 0.01)
                prices.append(new_price)
            
            # Create OHLC data
            data[symbol] = self.create_basic_ohlc_data(
                n_samples=n_samples,
                symbol=symbol,
                base_price=base_price,
                volatility=0.015 + i * 0.005  # Varying volatility
            )
            
            # Replace close prices with correlated ones
            data[symbol]['Close'] = prices
        
        return data
    
    def create_temporal_data_with_gaps(
        self, 
        n_samples: int,
        symbol: str = "GAPPED",
        gap_probability: float = 0.05
    ) -> pd.DataFrame:
        """Create data with temporal gaps"""
        # Start with regular data
        data = self.create_basic_ohlc_data(n_samples, symbol)
        
        # Introduce gaps
        gap_mask = np.random.random(n_samples) < gap_probability
        gap_indices = np.where(gap_mask)[0]
        
        # Remove rows to create gaps
        if len(gap_indices) > 0:
            data = data.drop(data.index[gap_indices]).reset_index(drop=True)
        
        return data


@pytest.fixture(scope="session")
def data_factory():
    """Provide synthetic data factory"""
    return SyntheticDataFactory(seed=42)


@pytest.fixture
def mock_toto_model():
    """Provide mock Toto model"""
    config = TotoOHLCConfig(embed_dim=32, num_layers=2)
    return MockTotoModel(config)


@pytest.fixture
def basic_test_data(data_factory):
    """Basic test data fixture"""
    return data_factory.create_basic_ohlc_data(500, "BASIC_TEST")


@pytest.fixture
def problematic_test_data(data_factory):
    """Test data with various issues"""
    return data_factory.create_data_with_issues(300, "PROBLEM_TEST")


@pytest.fixture
def multi_symbol_test_data(data_factory):
    """Multi-symbol test data"""
    symbols = ['SYMBOL_A', 'SYMBOL_B', 'SYMBOL_C']
    return data_factory.create_multi_symbol_data(symbols, 800)


@pytest.fixture
def temp_test_directory():
    """Temporary directory for test files"""
    temp_dir = Path(tempfile.mkdtemp())
    yield temp_dir
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def test_scenarios():
    """Predefined test scenarios"""
    return [
        TestScenario(
            name="small_clean",
            data_size=100,
            n_symbols=2,
            sequence_length=20,
            prediction_length=5,
            batch_size=4
        ),
        TestScenario(
            name="medium_with_issues",
            data_size=500,
            n_symbols=3,
            sequence_length=50,
            prediction_length=10,
            batch_size=8,
            has_missing_data=True,
            has_outliers=True
        ),
        TestScenario(
            name="large_complex",
            data_size=2000,
            n_symbols=5,
            sequence_length=100,
            prediction_length=25,
            batch_size=16,
            has_irregular_timestamps=True
        )
    ]


class ConfigurationFactory:
    """Factory for creating test configurations"""
    
    @staticmethod
    def create_minimal_trainer_config(**overrides) -> TotoOHLCConfig:
        """Create minimal trainer configuration for testing"""
        defaults = {
            'patch_size': 4,
            'stride': 2,
            'embed_dim': 32,
            'num_layers': 2,
            'num_heads': 4,
            'mlp_hidden_dim': 64,
            'dropout': 0.1,
            'sequence_length': 20,
            'prediction_length': 5,
            'validation_days': 5
        }
        defaults.update(overrides)
        return TotoOHLCConfig(**defaults)
    
    @staticmethod
    def create_minimal_dataloader_config(temp_dir: Path = None, **overrides) -> DataLoaderConfig:
        """Create minimal dataloader configuration for testing"""
        defaults = {
            'train_data_path': str(temp_dir / "train") if temp_dir else "test_train",
            'test_data_path': str(temp_dir / "test") if temp_dir else "test_test",
            'sequence_length': 20,
            'prediction_length': 5,
            'batch_size': 4,
            'validation_split': 0.2,
            'normalization_method': "robust",
            'add_technical_indicators': False,
            'min_sequence_length': 25,
            'num_workers': 0,  # Avoid multiprocessing in tests
            'max_symbols': 3   # Limit for testing
        }
        defaults.update(overrides)
        return DataLoaderConfig(**defaults)


@pytest.fixture
def config_factory():
    """Provide configuration factory"""
    return ConfigurationFactory()


class MockManager:
    """Manager for creating and configuring mocks"""
    
    @staticmethod
    def create_mock_trainer(config: TotoOHLCConfig) -> Mock:
        """Create mock trainer"""
        trainer = Mock(spec=TotoOHLCTrainer)
        trainer.config = config
        trainer.device = torch.device('cpu')
        trainer.model = None
        trainer.optimizer = None
        trainer.logger = Mock()
        
        return trainer
    
    @staticmethod
    def create_mock_dataloader(batch_size: int = 4, num_batches: int = 3) -> Mock:
        """Create mock dataloader with sample batches"""
        batches = []
        
        for _ in range(num_batches):
            # Create mock MaskedTimeseries batch
            batch = Mock()
            batch.series = torch.randn(batch_size, 5, 20)  # batch, features, time
            batch.padding_mask = torch.ones(batch_size, 5, 20, dtype=torch.bool)
            batch.id_mask = torch.ones(batch_size, 5, 1, dtype=torch.long)
            batch.timestamp_seconds = torch.randint(1000000, 2000000, (batch_size, 5, 20))
            batch.time_interval_seconds = torch.full((batch_size, 5), 3600)  # 1 hour
            
            batches.append(batch)
        
        mock_dataloader = Mock()
        mock_dataloader.__iter__ = Mock(return_value=iter(batches))
        mock_dataloader.__len__ = Mock(return_value=num_batches)
        
        return mock_dataloader
    
    @staticmethod
    def create_mock_dataset(length: int = 100) -> Mock:
        """Create mock dataset"""
        dataset = Mock()
        dataset.__len__ = Mock(return_value=length)
        
        def mock_getitem(idx):
            batch = Mock()
            batch.series = torch.randn(5, 20)  # features, time
            batch.padding_mask = torch.ones(5, 20, dtype=torch.bool)
            batch.id_mask = torch.ones(5, 1, dtype=torch.long)
            batch.timestamp_seconds = torch.randint(1000000, 2000000, (5, 20))
            batch.time_interval_seconds = torch.full((5,), 3600)
            return batch
        
        dataset.__getitem__ = Mock(side_effect=mock_getitem)
        
        return dataset


@pytest.fixture
def mock_manager():
    """Provide mock manager"""
    return MockManager()


class TestDataPersistence:
    """Utilities for saving and loading test data"""
    
    @staticmethod
    def save_test_data(data: Dict[str, pd.DataFrame], directory: Path):
        """Save test data to directory"""
        directory.mkdir(parents=True, exist_ok=True)
        
        for symbol, df in data.items():
            filepath = directory / f"{symbol}.csv"
            df.to_csv(filepath, index=False)
    
    @staticmethod
    def save_test_config(config: Union[TotoOHLCConfig, DataLoaderConfig], filepath: Path):
        """Save test configuration to JSON"""
        if isinstance(config, TotoOHLCConfig):
            config_dict = asdict(config)
        elif hasattr(config, 'save'):
            config.save(str(filepath))
            return
        else:
            config_dict = asdict(config)
        
        with open(filepath, 'w') as f:
            json.dump(config_dict, f, indent=2, default=str)
    
    @staticmethod
    def create_test_data_directory(
        temp_dir: Path,
        data_factory: SyntheticDataFactory,
        scenario: TestScenario
    ) -> Tuple[Path, Path]:
        """Create complete test data directory structure"""
        train_dir = temp_dir / "train"
        test_dir = temp_dir / "test"
        
        # Generate data according to scenario
        symbols = [f"SYM_{i:03d}" for i in range(scenario.n_symbols)]
        
        if scenario.has_missing_data or scenario.has_outliers:
            issue_types = []
            if scenario.has_missing_data:
                issue_types.append('missing')
            if scenario.has_outliers:
                issue_types.append('outliers')
            
            train_data = {}
            test_data = {}
            
            for symbol in symbols:
                full_data = data_factory.create_data_with_issues(
                    scenario.data_size,
                    symbol,
                    issue_types
                )
                
                # Split into train/test
                split_idx = int(len(full_data) * 0.8)
                train_data[symbol] = full_data.iloc[:split_idx].copy()
                test_data[symbol] = full_data.iloc[split_idx:].copy()
        else:
            # Clean data
            train_data = {}
            test_data = {}
            
            for symbol in symbols:
                full_data = data_factory.create_basic_ohlc_data(
                    scenario.data_size,
                    symbol
                )
                
                split_idx = int(len(full_data) * 0.8)
                train_data[symbol] = full_data.iloc[:split_idx].copy()
                test_data[symbol] = full_data.iloc[split_idx:].copy()
        
        # Save data
        TestDataPersistence.save_test_data(train_data, train_dir)
        TestDataPersistence.save_test_data(test_data, test_dir)
        
        return train_dir, test_dir


@pytest.fixture
def test_data_persistence():
    """Provide test data persistence utilities"""
    return TestDataPersistence()


class AssertionHelpers:
    """Helper functions for common test assertions"""
    
    @staticmethod
    def assert_tensor_valid(tensor: torch.Tensor, name: str = "tensor"):
        """Assert tensor is valid (no NaN, Inf, reasonable range)"""
        assert isinstance(tensor, torch.Tensor), f"{name} should be a tensor"
        assert not torch.isnan(tensor).any(), f"{name} contains NaN values"
        assert not torch.isinf(tensor).any(), f"{name} contains infinite values"
        assert tensor.numel() > 0, f"{name} should not be empty"
    
    @staticmethod
    def assert_dataframe_valid(df: pd.DataFrame, required_columns: List[str] = None):
        """Assert DataFrame is valid"""
        assert isinstance(df, pd.DataFrame), "Should be a DataFrame"
        assert len(df) > 0, "DataFrame should not be empty"
        
        if required_columns:
            missing_cols = set(required_columns) - set(df.columns)
            assert not missing_cols, f"Missing required columns: {missing_cols}"
    
    @staticmethod
    def assert_ohlc_valid(df: pd.DataFrame):
        """Assert OHLC data validity"""
        AssertionHelpers.assert_dataframe_valid(df, ['Open', 'High', 'Low', 'Close'])
        
        # OHLC relationships
        assert (df['High'] >= df['Open']).all(), "High should be >= Open"
        assert (df['High'] >= df['Close']).all(), "High should be >= Close"
        assert (df['Low'] <= df['Open']).all(), "Low should be <= Open"
        assert (df['Low'] <= df['Close']).all(), "Low should be <= Close"
        
        # Positive prices
        assert (df[['Open', 'High', 'Low', 'Close']] > 0).all().all(), "All prices should be positive"
    
    @staticmethod
    def assert_performance_acceptable(execution_time: float, memory_mb: float, max_time: float = 10.0, max_memory: float = 1000.0):
        """Assert performance is within acceptable bounds"""
        assert execution_time < max_time, f"Execution time too high: {execution_time:.2f}s > {max_time}s"
        assert memory_mb < max_memory, f"Memory usage too high: {memory_mb:.1f}MB > {max_memory}MB"


@pytest.fixture
def assertion_helpers():
    """Provide assertion helpers"""
    return AssertionHelpers()


# Parametrized fixture for different test scenarios
@pytest.fixture(params=[
    ("small", 100, 2, 20, 5),
    ("medium", 500, 3, 50, 10),
    ("large", 1000, 5, 100, 20)
], ids=["small", "medium", "large"])
def parametrized_test_data(request, data_factory):
    """Parametrized fixture for different data sizes"""
    name, n_samples, n_symbols, seq_len, pred_len = request.param
    
    symbols = [f"{name.upper()}_{i}" for i in range(n_symbols)]
    data = data_factory.create_multi_symbol_data(symbols, n_samples)
    
    return {
        'data': data,
        'scenario': TestScenario(
            name=name,
            data_size=n_samples,
            n_symbols=n_symbols,
            sequence_length=seq_len,
            prediction_length=pred_len,
            batch_size=4
        )
    }


# Conditional fixtures for optional dependencies
@pytest.fixture
def mock_tensorboard():
    """Mock TensorBoard writer if not available"""
    try:
        from torch.utils.tensorboard import SummaryWriter
        return None  # Use real TensorBoard
    except ImportError:
        # Create mock
        mock_writer = Mock()
        mock_writer.add_scalar = Mock()
        mock_writer.add_histogram = Mock()
        mock_writer.add_graph = Mock()
        mock_writer.close = Mock()
        return mock_writer


@pytest.fixture
def mock_mlflow():
    """Mock MLflow if not available"""
    try:
        import mlflow
        return None  # Use real MLflow
    except ImportError:
        # Create mock MLflow module
        mock_mlflow = Mock()
        mock_mlflow.start_run = Mock()
        mock_mlflow.end_run = Mock()
        mock_mlflow.log_param = Mock()
        mock_mlflow.log_metric = Mock()
        mock_mlflow.log_artifact = Mock()
        return mock_mlflow


if __name__ == "__main__":
    # Test the fixtures
    import pytest
    pytest.main([__file__, "-v", "--tb=short"])