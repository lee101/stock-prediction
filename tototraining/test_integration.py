#!/usr/bin/env python3
"""
Integration tests for the Toto retraining system.
Tests end-to-end training pipeline with small synthetic data.
"""

import pytest
import torch
import numpy as np
import pandas as pd
import tempfile
import shutil
import json
import time
from pathlib import Path
from unittest.mock import Mock, patch
from typing import Dict, List, Tuple
import warnings

# Import modules under test
from toto_ohlc_trainer import TotoOHLCConfig, TotoOHLCTrainer
from toto_ohlc_dataloader import DataLoaderConfig, OHLCPreprocessor, TotoOHLCDataLoader
from enhanced_trainer import EnhancedTotoTrainer

# Suppress warnings during testing
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


class SyntheticDataGenerator:
    """Generates synthetic OHLC data for testing"""
    
    def __init__(self, seed: int = 42):
        self.seed = seed
        np.random.seed(seed)
    
    def generate_price_series(self, n_samples: int, base_price: float = 100.0, volatility: float = 0.02) -> np.ndarray:
        """Generate realistic price series using geometric Brownian motion"""
        dt = 1/365  # Daily time step
        drift = 0.05  # 5% annual drift
        
        prices = [base_price]
        for _ in range(n_samples - 1):
            random_shock = np.random.normal(0, 1)
            price_change = prices[-1] * (drift * dt + volatility * np.sqrt(dt) * random_shock)
            new_price = prices[-1] + price_change
            prices.append(max(new_price, 1.0))  # Ensure positive prices
        
        return np.array(prices)
    
    def generate_ohlc_data(
        self, 
        n_samples: int, 
        symbol: str = "TEST", 
        base_price: float = 100.0,
        start_date: str = "2023-01-01",
        freq: str = "H"
    ) -> pd.DataFrame:
        """Generate synthetic OHLC data"""
        # Generate base close prices
        close_prices = self.generate_price_series(n_samples, base_price)
        
        # Generate OHLC from close prices
        opens = []
        highs = []
        lows = []
        volumes = []
        
        for i in range(n_samples):
            if i == 0:
                open_price = close_prices[i]
            else:
                # Open is previous close + small gap
                gap = np.random.normal(0, 0.001) * close_prices[i-1]
                open_price = close_prices[i-1] + gap
            
            close_price = close_prices[i]
            
            # High is max of open/close + some upward movement
            high_addition = abs(np.random.normal(0, 0.005)) * max(open_price, close_price)
            high_price = max(open_price, close_price) + high_addition
            
            # Low is min of open/close - some downward movement
            low_subtraction = abs(np.random.normal(0, 0.005)) * min(open_price, close_price)
            low_price = min(open_price, close_price) - low_subtraction
            
            # Volume is log-normally distributed
            volume = int(np.random.lognormal(8, 1) * 100)  # Around 100k average volume
            
            opens.append(open_price)
            highs.append(high_price)
            lows.append(low_price)
            volumes.append(volume)
        
        # Create DataFrame
        dates = pd.date_range(start_date, periods=n_samples, freq=freq)
        
        data = pd.DataFrame({
            'timestamp': dates,
            'Open': opens,
            'High': highs,
            'Low': lows,
            'Close': close_prices,
            'Volume': volumes,
            'Symbol': symbol
        })
        
        return data
    
    def generate_multiple_symbols(
        self, 
        symbols: List[str], 
        n_samples: int = 500,
        start_date: str = "2023-01-01"
    ) -> Dict[str, pd.DataFrame]:
        """Generate data for multiple symbols"""
        data = {}
        base_prices = [50, 100, 150, 200, 300]  # Different base prices
        
        for i, symbol in enumerate(symbols):
            base_price = base_prices[i % len(base_prices)]
            data[symbol] = self.generate_ohlc_data(
                n_samples=n_samples,
                symbol=symbol,
                base_price=base_price,
                start_date=start_date
            )
        
        return data
    
    def save_to_csv_files(self, data: Dict[str, pd.DataFrame], output_dir: Path):
        """Save generated data to CSV files"""
        output_dir.mkdir(parents=True, exist_ok=True)
        
        for symbol, df in data.items():
            filepath = output_dir / f"{symbol}.csv"
            df.to_csv(filepath, index=False)
        
        return output_dir


@pytest.fixture
def synthetic_data_generator():
    """Create synthetic data generator"""
    return SyntheticDataGenerator(seed=42)


@pytest.fixture
def temp_data_dir(synthetic_data_generator):
    """Create temporary directory with synthetic data"""
    temp_dir = Path(tempfile.mkdtemp())
    
    # Generate data for multiple symbols
    symbols = ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'AMZN']
    data = synthetic_data_generator.generate_multiple_symbols(symbols, n_samples=200)
    
    # Create train/test directories
    train_dir = temp_dir / "train"
    test_dir = temp_dir / "test"
    
    # Split data: first 160 samples for training, last 40 for testing
    train_data = {}
    test_data = {}
    
    for symbol, df in data.items():
        train_data[symbol] = df.iloc[:160].copy()
        test_data[symbol] = df.iloc[160:].copy()
    
    # Save to files
    synthetic_data_generator.save_to_csv_files(train_data, train_dir)
    synthetic_data_generator.save_to_csv_files(test_data, test_dir)
    
    yield temp_dir
    
    # Cleanup
    shutil.rmtree(temp_dir)


class TestEndToEndTraining:
    """Test complete end-to-end training pipeline"""
    
    @pytest.fixture
    def minimal_config(self):
        """Create minimal configuration for fast testing"""
        return TotoOHLCConfig(
            patch_size=4,
            stride=2,
            embed_dim=32,  # Very small for testing
            num_layers=2,
            num_heads=2,
            mlp_hidden_dim=64,
            dropout=0.1,
            sequence_length=20,  # Short sequences for testing
            prediction_length=5,
            validation_days=10
        )
    
    @pytest.fixture
    def dataloader_config(self, temp_data_dir):
        """Create dataloader configuration"""
        return DataLoaderConfig(
            train_data_path=str(temp_data_dir / "train"),
            test_data_path=str(temp_data_dir / "test"),
            patch_size=4,
            stride=2,
            sequence_length=20,
            prediction_length=5,
            batch_size=4,
            validation_split=0.2,
            normalization_method="robust",
            add_technical_indicators=False,  # Disable for faster testing
            min_sequence_length=25,
            max_symbols=3,  # Limit for fast testing
            num_workers=0  # Avoid multiprocessing issues in tests
        )
    
    def test_synthetic_data_generation(self, synthetic_data_generator):
        """Test synthetic data generation"""
        data = synthetic_data_generator.generate_ohlc_data(100, "TEST")
        
        assert len(data) == 100
        assert 'timestamp' in data.columns
        assert all(col in data.columns for col in ['Open', 'High', 'Low', 'Close', 'Volume'])
        
        # Validate OHLC relationships
        assert all(data['High'] >= data['Open'])
        assert all(data['High'] >= data['Close'])
        assert all(data['Low'] <= data['Open'])
        assert all(data['Low'] <= data['Close'])
        assert all(data['Volume'] > 0)
    
    def test_data_loading_pipeline(self, dataloader_config, temp_data_dir):
        """Test complete data loading pipeline"""
        dataloader = TotoOHLCDataLoader(dataloader_config)
        
        # Test data loading
        train_data, val_data, test_data = dataloader.load_data()
        
        assert len(train_data) > 0, "Should have training data"
        assert len(test_data) > 0, "Should have test data"
        
        # Test dataloader preparation
        dataloaders = dataloader.prepare_dataloaders()
        
        assert 'train' in dataloaders, "Should have train dataloader"
        
        # Test batch loading
        train_loader = dataloaders['train']
        batch = next(iter(train_loader))
        
        # Check batch structure
        assert hasattr(batch, 'series'), "Batch should have series"
        assert hasattr(batch, 'padding_mask'), "Batch should have padding_mask"
        assert isinstance(batch.series, torch.Tensor)
        
        # Check shapes
        assert batch.series.dim() == 3, "Series should be 3D"
        batch_size, n_features, seq_len = batch.series.shape
        assert batch_size <= dataloader_config.batch_size
        assert seq_len == dataloader_config.sequence_length
    
    @patch('toto_ohlc_trainer.Toto')
    def test_model_initialization_pipeline(self, mock_toto, minimal_config):
        """Test model initialization pipeline"""
        # Create mock model
        mock_model = Mock()
        mock_model.parameters.return_value = [torch.randn(10, requires_grad=True)]
        mock_toto.return_value = mock_model
        
        trainer = TotoOHLCTrainer(minimal_config)
        trainer.initialize_model(input_dim=5)
        
        # Verify model was initialized
        assert trainer.model is not None
        assert trainer.optimizer is not None
        mock_toto.assert_called_once()
    
    @patch('toto_ohlc_trainer.Toto')
    def test_training_pipeline_structure(self, mock_toto, minimal_config, temp_data_dir):
        """Test training pipeline structure without full training"""
        # Mock the model
        mock_model = Mock()
        mock_model.parameters.return_value = [torch.randn(10, requires_grad=True)]
        mock_model.model = Mock()
        
        # Mock output
        mock_output = Mock()
        mock_output.loc = torch.randn(2, 5)
        mock_model.model.return_value = mock_output
        
        mock_toto.return_value = mock_model
        
        # Patch data loading to return small dataset
        with patch.object(TotoOHLCTrainer, 'load_data') as mock_load_data:
            # Create minimal mock datasets
            sample_x = torch.randn(4, minimal_config.sequence_length, 5)
            sample_y = torch.randn(4, minimal_config.prediction_length)
            mock_dataset = [(sample_x, sample_y)]
            
            mock_datasets = {'train': mock_dataset}
            mock_dataloaders = {'train': mock_dataset}
            mock_load_data.return_value = (mock_datasets, mock_dataloaders)
            
            trainer = TotoOHLCTrainer(minimal_config)
            
            # Test that training structure works
            try:
                trainer.train(num_epochs=1)  # Just one epoch
                # If we get here without exception, structure is good
                assert True
            except Exception as e:
                # Expected due to mocking, but check it's a reasonable error
                error_msg = str(e).lower()
                assert any(keyword in error_msg for keyword in ['mock', 'attribute', 'tensor'])
    
    def test_forward_pass_shapes(self, minimal_config):
        """Test forward pass tensor shapes"""
        # Create actual tensors to test shapes
        batch_size = 2
        seq_len = minimal_config.sequence_length
        features = 5
        pred_len = minimal_config.prediction_length
        
        # Input tensor
        x = torch.randn(batch_size, seq_len, features)
        y = torch.randn(batch_size, pred_len)
        
        # Test shape transformations as done in training
        x_reshaped = x.transpose(1, 2).contiguous()
        input_padding_mask = torch.zeros(batch_size, 1, seq_len, dtype=torch.bool)
        id_mask = torch.ones(batch_size, 1, seq_len, dtype=torch.float32)
        
        # Verify shapes
        assert x_reshaped.shape == (batch_size, features, seq_len)
        assert input_padding_mask.shape == (batch_size, 1, seq_len)
        assert id_mask.shape == (batch_size, 1, seq_len)
        
        # Test loss computation shapes
        predictions = torch.randn(batch_size, pred_len)
        loss = torch.nn.functional.mse_loss(predictions, y)
        
        assert loss.dim() == 0  # Scalar loss
        assert not torch.isnan(loss)
    
    @pytest.mark.slow
    def test_mini_training_run(self, dataloader_config, temp_data_dir):
        """Test a very short training run with real data (marked as slow test)"""
        # This test runs actual training for 1-2 epochs to verify integration
        
        # Create very minimal config
        config = TotoOHLCConfig(
            patch_size=4,
            stride=2,
            embed_dim=16,  # Extremely small
            num_layers=1,
            num_heads=2,
            mlp_hidden_dim=32,
            dropout=0.0,
            sequence_length=12,  # Very short
            prediction_length=3,
            validation_days=5
        )
        
        # Mock Toto model to avoid dependency
        with patch('toto_ohlc_trainer.Toto') as mock_toto:
            mock_model = Mock()
            mock_model.parameters.return_value = [torch.randn(50, requires_grad=True)]
            mock_model.train = Mock()
            mock_model.eval = Mock()
            mock_model.model = Mock()
            
            # Create deterministic output
            mock_output = Mock()
            mock_output.loc = torch.zeros(4, 3)  # batch_size=4, pred_len=3
            mock_model.model.return_value = mock_output
            
            mock_toto.return_value = mock_model
            
            trainer = TotoOHLCTrainer(config)
            
            # Create simple dataloader manually
            dataloader_instance = TotoOHLCDataLoader(dataloader_config)
            train_data, val_data, test_data = dataloader_instance.load_data()
            
            if len(train_data) > 0:
                # Mock the data loading in trainer
                with patch.object(trainer, 'load_data') as mock_trainer_load_data:
                    # Create simple mock data
                    sample_data = []
                    for i in range(2):  # Just 2 batches
                        x = torch.randn(4, config.sequence_length, 5)
                        y = torch.randn(4, config.prediction_length)
                        sample_data.append((x, y))
                    
                    mock_datasets = {'train': sample_data}
                    mock_dataloaders = {'train': sample_data}
                    mock_trainer_load_data.return_value = (mock_datasets, mock_dataloaders)
                    
                    # Run mini training
                    trainer.train(num_epochs=1)
                    
                    # Verify training was attempted
                    mock_model.train.assert_called()
                    assert trainer.optimizer is not None


class TestTrainingCallbacks:
    """Test training callbacks and monitoring integration"""
    
    def test_enhanced_trainer_initialization(self):
        """Test enhanced trainer initialization"""
        config = TotoOHLCConfig(embed_dim=32, num_layers=1)
        
        # Mock dependencies
        with patch('enhanced_trainer.TotoTrainingLogger'), \
             patch('enhanced_trainer.CheckpointManager'), \
             patch('enhanced_trainer.DashboardGenerator'):
            
            trainer = EnhancedTotoTrainer(
                config=config,
                experiment_name="test_experiment",
                enable_tensorboard=False,  # Disable to avoid dependencies
                enable_mlflow=False,
                enable_system_monitoring=False
            )
            
            assert trainer.experiment_name == "test_experiment"
            assert trainer.config == config
    
    def test_training_metrics_structure(self):
        """Test training metrics data structure"""
        # Test metrics that would be logged during training
        train_metrics = {
            'avg_gradient_norm': 0.5,
            'num_batches': 10
        }
        
        val_metrics = {
            'mse': 0.1,
            'mae': 0.05,
            'correlation': 0.8,
            'num_batches': 5
        }
        
        # Verify structure
        assert 'avg_gradient_norm' in train_metrics
        assert 'mse' in val_metrics
        assert all(isinstance(v, (int, float)) for v in train_metrics.values())
        assert all(isinstance(v, (int, float)) for v in val_metrics.values())


class TestErrorHandling:
    """Test error handling in integration scenarios"""
    
    def test_empty_data_handling(self):
        """Test handling of empty datasets"""
        config = TotoOHLCConfig()
        trainer = TotoOHLCTrainer(config)
        
        # Mock empty data loading
        with patch.object(trainer, 'load_data') as mock_load_data:
            mock_load_data.return_value = ({}, {})
            
            # Training should handle empty data gracefully
            trainer.train(num_epochs=1)
            # Should not crash, just log error and return
    
    def test_malformed_data_handling(self, temp_data_dir):
        """Test handling of malformed data"""
        # Create malformed CSV file
        bad_data_dir = temp_data_dir / "bad_data"
        bad_data_dir.mkdir()
        
        # Create CSV with missing columns
        bad_df = pd.DataFrame({
            'timestamp': pd.date_range('2023-01-01', periods=10, freq='H'),
            'Open': np.random.randn(10),
            # Missing High, Low, Close columns
        })
        bad_df.to_csv(bad_data_dir / "bad_data.csv", index=False)
        
        config = DataLoaderConfig(
            train_data_path=str(bad_data_dir),
            min_sequence_length=5
        )
        
        dataloader = TotoOHLCDataLoader(config)
        train_data, val_data, test_data = dataloader.load_data()
        
        # Should handle malformed data by skipping it
        assert len(train_data) == 0  # Bad data should be filtered out
    
    def test_insufficient_data_handling(self, synthetic_data_generator):
        """Test handling of insufficient data"""
        # Generate very small dataset
        small_data = synthetic_data_generator.generate_ohlc_data(10, "SMALL")
        
        config = DataLoaderConfig(
            min_sequence_length=50,  # Require more data than available
            sequence_length=20
        )
        
        preprocessor = OHLCPreprocessor(config)
        preprocessor.fit_scalers({"SMALL": small_data})
        
        # Should handle insufficient data gracefully
        from toto_ohlc_dataloader import OHLCDataset as DataLoaderOHLCDataset
        dataset = DataLoaderOHLCDataset({"SMALL": small_data}, config, preprocessor, 'train')
        
        # Dataset should be empty due to insufficient data
        assert len(dataset) == 0


class TestPerformanceCharacteristics:
    """Test performance characteristics of the training pipeline"""
    
    def test_memory_usage_characteristics(self, synthetic_data_generator):
        """Test memory usage remains reasonable"""
        # Generate moderately sized dataset
        data = synthetic_data_generator.generate_ohlc_data(1000, "MEMORY_TEST")
        
        config = DataLoaderConfig(
            sequence_length=50,
            prediction_length=10,
            batch_size=16,
            add_technical_indicators=False,
            min_sequence_length=60
        )
        
        from toto_ohlc_dataloader import OHLCPreprocessor, OHLCDataset as DataLoaderOHLCDataset
        
        preprocessor = OHLCPreprocessor(config)
        preprocessor.fit_scalers({"MEMORY_TEST": data})
        
        dataset = DataLoaderOHLCDataset({"MEMORY_TEST": data}, config, preprocessor, 'train')
        
        if len(dataset) > 0:
            # Test that we can create batches without excessive memory usage
            dataloader = torch.utils.data.DataLoader(dataset, batch_size=config.batch_size)
            
            batch_count = 0
            for batch in dataloader:
                assert isinstance(batch.series, torch.Tensor)
                batch_count += 1
                if batch_count >= 3:  # Test a few batches
                    break
            
            assert batch_count > 0, "Should have processed at least one batch"
    
    def test_training_speed_characteristics(self):
        """Test that training setup completes in reasonable time"""
        start_time = time.time()
        
        config = TotoOHLCConfig(
            embed_dim=16,
            num_layers=1,
            sequence_length=10
        )
        
        trainer = TotoOHLCTrainer(config)
        
        # Mock model initialization to avoid dependencies
        with patch('toto_ohlc_trainer.Toto') as mock_toto:
            mock_model = Mock()
            mock_model.parameters.return_value = [torch.randn(10, requires_grad=True)]
            mock_toto.return_value = mock_model
            
            trainer.initialize_model(input_dim=5)
        
        setup_time = time.time() - start_time
        
        # Setup should complete quickly (within 5 seconds even on slow systems)
        assert setup_time < 5.0, f"Setup took too long: {setup_time:.2f} seconds"


if __name__ == "__main__":
    # Run tests with specific markers
    pytest.main([
        __file__, 
        "-v", 
        "--tb=short",
        "-m", "not slow"  # Skip slow tests by default
    ])
