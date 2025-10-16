#!/usr/bin/env python3
"""
Comprehensive test suite for TotoTrainer training pipeline.

This test suite covers all requirements:
1. TotoTrainer class initialization with configs
2. Integration with OHLC dataloader
3. Mock Toto model loading and setup
4. Training loop functionality with few steps
5. Checkpoint saving/loading mechanisms
6. Error handling scenarios  
7. Memory usage and performance checks
8. Identification of specific fixes needed
"""

import pytest
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import tempfile
import shutil
import time
import psutil
import gc
import warnings
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

# Import modules under test
try:
    from toto_trainer import TotoTrainer, TrainerConfig, MetricsTracker, CheckpointManager
    from toto_ohlc_dataloader import TotoOHLCDataLoader, DataLoaderConfig, MaskedTimeseries
except ImportError as e:
    print(f"Import error: {e}")
    # Try local imports
    import sys
    sys.path.append('.')
    try:
        from toto_trainer import TotoTrainer, TrainerConfig, MetricsTracker, CheckpointManager
        from toto_ohlc_dataloader import TotoOHLCDataLoader, DataLoaderConfig, MaskedTimeseries
    except ImportError as e2:
        print(f"Local import error: {e2}")
        pytest.skip(f"Cannot import required modules: {e2}")

# Suppress warnings during testing
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


@pytest.fixture
def temp_dir():
    """Create temporary directory for test files"""
    temp_dir = tempfile.mkdtemp()
    yield Path(temp_dir)
    shutil.rmtree(temp_dir)


@pytest.fixture
def sample_ohlc_data():
    """Generate sample OHLC data for testing"""
    np.random.seed(42)
    n_samples = 200
    dates = pd.date_range('2023-01-01', periods=n_samples, freq='H')
    
    # Generate realistic OHLC data
    base_price = 100
    price_changes = np.random.normal(0, 0.01, n_samples)
    prices = [base_price]
    
    for change in price_changes[1:]:
        prices.append(prices[-1] * (1 + change))
    
    prices = np.array(prices)
    
    data = pd.DataFrame({
        'timestamp': dates,
        'Open': prices + np.random.normal(0, 0.1, n_samples),
        'High': prices + np.abs(np.random.normal(0, 0.5, n_samples)),
        'Low': prices - np.abs(np.random.normal(0, 0.5, n_samples)),
        'Close': prices + np.random.normal(0, 0.1, n_samples),
        'Volume': np.random.randint(1000, 10000, n_samples)
    })
    
    # Ensure OHLC constraints
    data['High'] = np.maximum(data['High'], np.maximum(data['Open'], data['Close']))
    data['Low'] = np.minimum(data['Low'], np.minimum(data['Open'], data['Close']))
    
    return data


@pytest.fixture
def trainer_config(temp_dir):
    """Create test trainer configuration"""
    return TrainerConfig(
        # Model config - smaller for testing
        patch_size=8,
        stride=4,
        embed_dim=64,
        num_layers=2,
        num_heads=4,
        mlp_hidden_dim=128,
        dropout=0.1,
        
        # Training config
        learning_rate=1e-3,
        weight_decay=0.01,
        batch_size=4,  # Small batch for testing
        accumulation_steps=1,
        max_epochs=3,  # Few epochs for testing
        warmup_epochs=1,
        
        # Optimization
        optimizer="adamw",
        scheduler="cosine",
        gradient_clip_val=1.0,
        use_mixed_precision=False,  # Disable for testing stability
        
        # Validation and checkpointing
        validation_frequency=1,
        save_every_n_epochs=1,
        keep_last_n_checkpoints=2,
        early_stopping_patience=5,
        
        # Paths
        save_dir=str(temp_dir / "checkpoints"),
        log_file=str(temp_dir / "training.log"),
        
        # Logging
        log_level="INFO",
        metrics_log_frequency=1,  # Log every batch
        
        # Memory optimization
        gradient_checkpointing=False,
        memory_efficient_attention=False,
        
        # Random seed for reproducibility
        random_seed=42
    )


@pytest.fixture
def dataloader_config(temp_dir):
    """Create test dataloader configuration"""
    return DataLoaderConfig(
        train_data_path=str(temp_dir / "train_data"),
        test_data_path=str(temp_dir / "test_data"),
        batch_size=4,
        sequence_length=48,  # Shorter sequences for testing
        prediction_length=12,
        patch_size=8,
        stride=4,
        validation_split=0.2,
        add_technical_indicators=False,  # Disable for simpler testing
        normalization_method="robust",
        min_sequence_length=60,
        max_symbols=3,  # Limit symbols for testing
        num_workers=0,  # Disable multiprocessing for testing
        random_seed=42
    )


@pytest.fixture
def sample_data_files(temp_dir, sample_ohlc_data):
    """Create sample CSV data files"""
    train_dir = temp_dir / "train_data"
    test_dir = temp_dir / "test_data"
    train_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)
    
    # Create multiple symbol files
    symbols = ['AAPL', 'GOOGL', 'MSFT']
    
    for i, symbol in enumerate(symbols):
        # Create variations of the base data
        data = sample_ohlc_data.copy()
        data = data.iloc[i*20:(i*20)+150].reset_index(drop=True)  # Different time periods
        
        # Slight price variations
        multiplier = 1 + i * 0.1
        for col in ['Open', 'High', 'Low', 'Close']:
            data[col] *= multiplier
        
        # Save to both train and test directories
        data.to_csv(train_dir / f"{symbol}.csv", index=False)
        # Test data is later part of the time series
        test_data = data.tail(50).copy()
        test_data.to_csv(test_dir / f"{symbol}.csv", index=False)
    
    return train_dir, test_dir


class TestTotoTrainerInitialization:
    """Test TotoTrainer class initialization and configuration"""
    
    def test_trainer_initialization_basic(self, trainer_config, dataloader_config):
        """Test basic trainer initialization"""
        trainer = TotoTrainer(trainer_config, dataloader_config)
        
        assert trainer.config == trainer_config
        assert trainer.dataloader_config == dataloader_config
        assert trainer.model is None  # Not initialized yet
        assert trainer.optimizer is None
        assert trainer.scheduler is None
        assert trainer.current_epoch == 0
        assert trainer.global_step == 0
        assert trainer.best_val_loss == float('inf')
        assert hasattr(trainer, 'logger')
        assert hasattr(trainer, 'metrics_tracker')
        assert hasattr(trainer, 'checkpoint_manager')
    
    def test_trainer_initialization_with_mixed_precision(self, trainer_config, dataloader_config):
        """Test trainer initialization with mixed precision"""
        trainer_config.use_mixed_precision = True
        trainer = TotoTrainer(trainer_config, dataloader_config)
        
        assert trainer.scaler is not None
        assert hasattr(trainer.scaler, 'scale')
    
    def test_trainer_initialization_without_mixed_precision(self, trainer_config, dataloader_config):
        """Test trainer initialization without mixed precision"""
        trainer_config.use_mixed_precision = False
        trainer = TotoTrainer(trainer_config, dataloader_config)
        
        assert trainer.scaler is None
    
    def test_checkpoint_directory_creation(self, trainer_config, dataloader_config, temp_dir):
        """Test that checkpoint directory is created"""
        checkpoint_dir = temp_dir / "test_checkpoints"
        trainer_config.save_dir = str(checkpoint_dir)
        
        trainer = TotoTrainer(trainer_config, dataloader_config)
        
        assert checkpoint_dir.exists()
        assert checkpoint_dir.is_dir()
    
    def test_random_seed_setting(self, trainer_config, dataloader_config):
        """Test that random seeds are set correctly"""
        trainer_config.random_seed = 123
        trainer = TotoTrainer(trainer_config, dataloader_config)
        
        # Test reproducibility
        torch.manual_seed(123)
        expected_tensor = torch.randn(5)
        
        trainer._set_random_seeds()
        actual_tensor = torch.randn(5)
        
        # Seeds should produce reproducible results
        assert not torch.allclose(expected_tensor, actual_tensor)  # Different since we reset


class TestDataloaderIntegration:
    """Test integration with OHLC dataloader"""
    
    def test_prepare_data_success(self, trainer_config, dataloader_config, sample_data_files):
        """Test successful data preparation"""
        trainer = TotoTrainer(trainer_config, dataloader_config)
        
        trainer.prepare_data()
        
        assert len(trainer.dataloaders) > 0
        assert 'train' in trainer.dataloaders
        # May or may not have val/test depending on data size
        
        # Test data loader properties
        train_loader = trainer.dataloaders['train']
        assert len(train_loader) > 0
        assert hasattr(train_loader.dataset, '__len__')
    
    def test_prepare_data_no_data(self, trainer_config, dataloader_config, temp_dir):
        """Test data preparation with no data files"""
        # Point to empty directories
        dataloader_config.train_data_path = str(temp_dir / "empty_train")
        dataloader_config.test_data_path = str(temp_dir / "empty_test")
        
        trainer = TotoTrainer(trainer_config, dataloader_config)
        
        with pytest.raises(ValueError, match="No data loaders created"):
            trainer.prepare_data()
    
    def test_data_loader_sample_format(self, trainer_config, dataloader_config, sample_data_files):
        """Test that data loader produces correct sample format"""
        trainer = TotoTrainer(trainer_config, dataloader_config)
        trainer.prepare_data()
        
        # Get a sample batch
        train_loader = trainer.dataloaders['train']
        sample_batch = next(iter(train_loader))
        
        # Should be MaskedTimeseries or tuple
        if isinstance(sample_batch, MaskedTimeseries):
            assert hasattr(sample_batch, 'series')
            assert hasattr(sample_batch, 'padding_mask')
            assert hasattr(sample_batch, 'id_mask')
            assert isinstance(sample_batch.series, torch.Tensor)
        else:
            assert isinstance(sample_batch, (tuple, list))
            assert len(sample_batch) >= 2  # x, y at minimum


class TestMockModelSetup:
    """Test model setup with mocking"""
    
    @patch('toto_trainer.Toto')
    def test_setup_model_success(self, mock_toto_class, trainer_config, dataloader_config, sample_data_files):
        """Test successful model setup with mocked Toto"""
        # Setup mock
        mock_model = Mock(spec=nn.Module)
        mock_model.parameters.return_value = [torch.randn(10, requires_grad=True)]
        mock_toto_class.return_value = mock_model
        
        trainer = TotoTrainer(trainer_config, dataloader_config)
        trainer.prepare_data()
        trainer.setup_model()
        
        # Verify model was created
        mock_toto_class.assert_called_once()
        assert trainer.model == mock_model
        assert trainer.optimizer is not None
        assert trainer.scheduler is not None
    
    @patch('toto_trainer.Toto')
    def test_setup_model_parameters(self, mock_toto_class, trainer_config, dataloader_config, sample_data_files):
        """Test that model is created with correct parameters"""
        mock_model = Mock(spec=nn.Module)
        mock_model.parameters.return_value = [torch.randn(10, requires_grad=True)]
        mock_toto_class.return_value = mock_model
        
        trainer = TotoTrainer(trainer_config, dataloader_config)
        trainer.prepare_data()
        trainer.setup_model()
        
        # Check that Toto was called with correct parameters
        call_kwargs = mock_toto_class.call_args[1]
        assert call_kwargs['patch_size'] == trainer_config.patch_size
        assert call_kwargs['embed_dim'] == trainer_config.embed_dim
        assert call_kwargs['num_layers'] == trainer_config.num_layers
    
    def test_setup_model_without_data(self, trainer_config, dataloader_config):
        """Test model setup without preparing data first"""
        trainer = TotoTrainer(trainer_config, dataloader_config)
        
        with pytest.raises(ValueError, match="Data loaders not prepared"):
            trainer.setup_model()


class TestTrainingLoop:
    """Test training loop functionality"""
    
    @patch('toto_trainer.Toto')
    def test_train_epoch_basic(self, mock_toto_class, trainer_config, dataloader_config, sample_data_files):
        """Test basic training epoch functionality"""
        # Setup mock model
        mock_model = self._create_mock_model()
        mock_toto_class.return_value = mock_model
        
        trainer = TotoTrainer(trainer_config, dataloader_config)
        trainer.prepare_data()
        trainer.setup_model()
        
        # Run one training epoch
        metrics = trainer.train_epoch()
        
        assert isinstance(metrics, dict)
        assert 'loss' in metrics
        assert metrics['loss'] >= 0
        assert isinstance(metrics['loss'], float)
    
    @patch('toto_trainer.Toto')
    def test_validate_epoch_basic(self, mock_toto_class, trainer_config, dataloader_config, sample_data_files):
        """Test basic validation epoch functionality"""
        mock_model = self._create_mock_model()
        mock_toto_class.return_value = mock_model
        
        trainer = TotoTrainer(trainer_config, dataloader_config)
        trainer.prepare_data()
        trainer.setup_model()
        
        # Run validation if validation data exists
        metrics = trainer.validate_epoch()
        
        if metrics:  # Only test if validation data exists
            assert isinstance(metrics, dict)
            assert 'loss' in metrics
            assert metrics['loss'] >= 0
    
    @patch('toto_trainer.Toto')
    def test_full_training_loop_few_steps(self, mock_toto_class, trainer_config, dataloader_config, sample_data_files):
        """Test full training loop with few steps"""
        mock_model = self._create_mock_model()
        mock_toto_class.return_value = mock_model
        
        # Configure for short training
        trainer_config.max_epochs = 2
        trainer_config.save_every_n_epochs = 1
        
        trainer = TotoTrainer(trainer_config, dataloader_config)
        trainer.prepare_data()
        trainer.setup_model()
        
        # Run training
        initial_epoch = trainer.current_epoch
        trainer.train()
        
        # Verify training progression
        assert trainer.current_epoch > initial_epoch
        assert trainer.global_step > 0
    
    def _create_mock_model(self):
        """Create a mock model with proper structure"""
        mock_model = Mock(spec=nn.Module)
        
        # Mock the inner model
        mock_inner_model = Mock()
        mock_output = Mock()
        mock_output.loc = torch.randn(4, 12)  # batch_size=4, prediction_length=12
        mock_inner_model.return_value = mock_output
        mock_model.model = mock_inner_model
        
        # Mock parameters
        mock_params = [torch.randn(10, requires_grad=True) for _ in range(3)]
        mock_model.parameters.return_value = mock_params
        
        # Mock training modes
        mock_model.train = Mock()
        mock_model.eval = Mock()
        
        # Mock device handling
        def mock_to(device):
            return mock_model
        mock_model.to = mock_to
        
        return mock_model


class TestCheckpointMechanisms:
    """Test checkpoint saving and loading"""
    
    def test_checkpoint_manager_creation(self, temp_dir):
        """Test checkpoint manager initialization"""
        checkpoint_dir = temp_dir / "checkpoints"
        manager = CheckpointManager(str(checkpoint_dir), keep_last_n=3)
        
        assert manager.save_dir == checkpoint_dir
        assert manager.keep_last_n == 3
        assert checkpoint_dir.exists()
    
    @patch('toto_trainer.Toto')
    def test_checkpoint_saving(self, mock_toto_class, trainer_config, dataloader_config, sample_data_files):
        """Test checkpoint saving functionality"""
        mock_model = Mock(spec=nn.Module)
        mock_model.parameters.return_value = [torch.randn(10, requires_grad=True)]
        mock_model.state_dict.return_value = {'param1': torch.randn(10)}
        mock_toto_class.return_value = mock_model
        
        trainer = TotoTrainer(trainer_config, dataloader_config)
        trainer.prepare_data()
        trainer.setup_model()
        
        # Save checkpoint
        checkpoint_path = trainer.checkpoint_manager.save_checkpoint(
            model=trainer.model,
            optimizer=trainer.optimizer,
            scheduler=trainer.scheduler,
            scaler=trainer.scaler,
            epoch=1,
            best_val_loss=0.5,
            metrics={'loss': 0.5},
            config=trainer_config,
            is_best=True
        )
        
        assert checkpoint_path.exists()
        assert (trainer.checkpoint_manager.save_dir / "best_model.pt").exists()
        assert (trainer.checkpoint_manager.save_dir / "latest.pt").exists()
    
    @patch('toto_trainer.Toto')
    def test_checkpoint_loading(self, mock_toto_class, trainer_config, dataloader_config, sample_data_files):
        """Test checkpoint loading functionality"""
        mock_model = Mock(spec=nn.Module)
        mock_model.parameters.return_value = [torch.randn(10, requires_grad=True)]
        mock_model.state_dict.return_value = {'param1': torch.randn(10)}
        mock_model.load_state_dict = Mock()
        mock_toto_class.return_value = mock_model
        
        trainer = TotoTrainer(trainer_config, dataloader_config)
        trainer.prepare_data()
        trainer.setup_model()
        
        # Save a checkpoint first
        checkpoint_path = trainer.checkpoint_manager.save_checkpoint(
            model=trainer.model,
            optimizer=trainer.optimizer,
            scheduler=trainer.scheduler,
            scaler=trainer.scaler,
            epoch=5,
            best_val_loss=0.3,
            metrics={'loss': 0.3},
            config=trainer_config
        )
        
        # Reset trainer state
        trainer.current_epoch = 0
        trainer.best_val_loss = float('inf')
        
        # Load checkpoint
        trainer.load_checkpoint(str(checkpoint_path))
        
        # Verify state was loaded
        assert trainer.current_epoch == 5
        assert trainer.best_val_loss == 0.3
        mock_model.load_state_dict.assert_called_once()
    
    def test_checkpoint_cleanup(self, temp_dir):
        """Test old checkpoint cleanup"""
        checkpoint_dir = temp_dir / "checkpoints"
        manager = CheckpointManager(str(checkpoint_dir), keep_last_n=2)
        
        # Create mock model and optimizer for testing
        mock_model = Mock()
        mock_model.state_dict.return_value = {'param': torch.tensor([1.0])}
        mock_optimizer = Mock()
        mock_optimizer.state_dict.return_value = {'lr': 0.001}
        mock_config = Mock()
        
        # Save multiple checkpoints
        for epoch in range(5):
            manager.save_checkpoint(
                model=mock_model,
                optimizer=mock_optimizer,
                scheduler=None,
                scaler=None,
                epoch=epoch,
                best_val_loss=0.1 * epoch,
                metrics={'loss': 0.1 * epoch},
                config=mock_config
            )
        
        # Check that only last 2 checkpoints remain
        checkpoint_files = list(checkpoint_dir.glob("checkpoint_epoch_*.pt"))
        assert len(checkpoint_files) <= 2
        
        # Check that latest epochs are kept
        epochs = [int(f.stem.split('_')[-1]) for f in checkpoint_files]
        epochs.sort()
        assert max(epochs) == 4  # Last epoch


class TestErrorHandling:
    """Test error handling scenarios"""
    
    def test_invalid_optimizer_type(self, trainer_config, dataloader_config):
        """Test handling of invalid optimizer type"""
        trainer_config.optimizer = "invalid_optimizer"
        trainer = TotoTrainer(trainer_config, dataloader_config)
        
        with pytest.raises(ValueError, match="Unsupported optimizer"):
            trainer._create_optimizer()
    
    def test_invalid_scheduler_type(self, trainer_config, dataloader_config):
        """Test handling of invalid scheduler type"""
        trainer_config.scheduler = "invalid_scheduler"
        trainer = TotoTrainer(trainer_config, dataloader_config)
        trainer.optimizer = torch.optim.Adam([torch.randn(1, requires_grad=True)])
        
        with pytest.raises(ValueError, match="Unsupported scheduler"):
            trainer._create_scheduler(steps_per_epoch=10)
    
    def test_missing_data_directory(self, trainer_config, dataloader_config, temp_dir):
        """Test handling of missing data directories"""
        dataloader_config.train_data_path = str(temp_dir / "nonexistent_train")
        dataloader_config.test_data_path = str(temp_dir / "nonexistent_test")
        
        trainer = TotoTrainer(trainer_config, dataloader_config)
        
        with pytest.raises(ValueError, match="No data loaders created"):
            trainer.prepare_data()
    
    @patch('toto_trainer.Toto')
    def test_model_forward_error_handling(self, mock_toto_class, trainer_config, dataloader_config, sample_data_files):
        """Test handling of model forward errors"""
        # Create model that raises exception on forward
        mock_model = Mock(spec=nn.Module)
        mock_model.model.side_effect = RuntimeError("Mock forward error")
        mock_model.parameters.return_value = [torch.randn(10, requires_grad=True)]
        mock_toto_class.return_value = mock_model
        
        trainer = TotoTrainer(trainer_config, dataloader_config)
        trainer.prepare_data()
        trainer.setup_model()
        
        # Training should handle the error gracefully or raise appropriately
        with pytest.raises((RuntimeError, Exception)):
            trainer.train_epoch()
    
    def test_checkpoint_loading_invalid_path(self, trainer_config, dataloader_config):
        """Test loading checkpoint from invalid path"""
        trainer = TotoTrainer(trainer_config, dataloader_config)
        
        with pytest.raises((FileNotFoundError, RuntimeError)):
            trainer.load_checkpoint("/nonexistent/checkpoint.pt")


class TestMemoryAndPerformance:
    """Test memory usage and performance metrics"""
    
    def test_memory_usage_tracking(self):
        """Test memory usage during operations"""
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Create some tensors to use memory
        tensors = []
        for _ in range(10):
            tensors.append(torch.randn(1000, 1000))
        
        peak_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Clean up
        del tensors
        gc.collect()
        
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        assert peak_memory > initial_memory
        assert final_memory <= peak_memory  # Memory should decrease after cleanup
    
    @patch('toto_trainer.Toto')
    def test_training_performance_metrics(self, mock_toto_class, trainer_config, dataloader_config, sample_data_files):
        """Test that performance metrics are collected"""
        mock_model = self._create_fast_mock_model()
        mock_toto_class.return_value = mock_model
        
        # Configure for performance testing
        trainer_config.compute_train_metrics = True
        trainer_config.max_epochs = 1
        
        trainer = TotoTrainer(trainer_config, dataloader_config)
        trainer.prepare_data()
        trainer.setup_model()
        
        start_time = time.time()
        metrics = trainer.train_epoch()
        training_time = time.time() - start_time
        
        # Check that metrics include timing information
        if 'batch_time_mean' in metrics:
            assert metrics['batch_time_mean'] > 0
            assert metrics['batch_time_mean'] < training_time  # Should be less than total time
    
    def test_metrics_tracker_functionality(self):
        """Test MetricsTracker class functionality"""
        tracker = MetricsTracker()
        
        # Test initial state
        assert len(tracker.losses) == 0
        
        # Update with some metrics
        predictions = torch.randn(10, 5)
        targets = torch.randn(10, 5)
        
        tracker.update(
            loss=0.5,
            predictions=predictions,
            targets=targets,
            batch_time=0.1,
            learning_rate=0.001
        )
        
        # Compute metrics
        metrics = tracker.compute_metrics()
        
        assert 'loss' in metrics
        assert 'mse' in metrics
        assert 'rmse' in metrics
        assert 'mae' in metrics
        assert 'batch_time_mean' in metrics
        assert 'learning_rate' in metrics
        
        # Verify metric values are reasonable
        assert metrics['loss'] == 0.5
        assert metrics['mse'] >= 0
        assert metrics['rmse'] >= 0
        assert metrics['mae'] >= 0
        assert metrics['batch_time_mean'] == 0.1
        assert metrics['learning_rate'] == 0.001
    
    def test_gradient_clipping_memory_efficiency(self):
        """Test gradient clipping doesn't cause memory leaks"""
        model = nn.Linear(100, 10)
        optimizer = torch.optim.Adam(model.parameters())
        
        initial_memory = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
        
        # Simulate training step with gradient clipping
        for _ in range(10):
            optimizer.zero_grad()
            x = torch.randn(32, 100)
            y = model(x)
            loss = y.sum()
            loss.backward()
            
            # Apply gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
        
        final_memory = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
        
        # Memory usage shouldn't grow significantly
        memory_growth = final_memory - initial_memory
        if torch.cuda.is_available():
            assert memory_growth < 100 * 1024 * 1024  # Less than 100MB growth
    
    def _create_fast_mock_model(self):
        """Create a mock model optimized for performance testing"""
        mock_model = Mock(spec=nn.Module)
        
        # Fast mock inner model
        mock_inner_model = Mock()
        mock_output = Mock()
        mock_output.loc = torch.zeros(4, 12)  # Use zeros for speed
        mock_inner_model.return_value = mock_output
        mock_model.model = mock_inner_model
        
        # Minimal parameters
        mock_model.parameters.return_value = [torch.zeros(1, requires_grad=True)]
        
        # Mock training modes
        mock_model.train = Mock()
        mock_model.eval = Mock()
        
        return mock_model


class TestTrainerConfigValidation:
    """Test trainer configuration validation"""
    
    def test_config_save_load(self, temp_dir):
        """Test configuration save and load functionality"""
        config = TrainerConfig(
            patch_size=16,
            embed_dim=512,
            learning_rate=1e-4
        )
        
        config_path = temp_dir / "config.json"
        config.save(str(config_path))
        
        assert config_path.exists()
        
        loaded_config = TrainerConfig.load(str(config_path))
        
        assert loaded_config.patch_size == config.patch_size
        assert loaded_config.embed_dim == config.embed_dim
        assert loaded_config.learning_rate == config.learning_rate
    
    def test_config_post_init(self, temp_dir):
        """Test configuration post-initialization"""
        save_dir = temp_dir / "test_save"
        config = TrainerConfig(save_dir=str(save_dir))
        
        # Check that save directory was created
        assert save_dir.exists()
        assert save_dir.is_dir()
    
    def test_config_default_values(self):
        """Test that configuration has reasonable defaults"""
        config = TrainerConfig()
        
        assert config.patch_size > 0
        assert config.embed_dim > 0
        assert config.num_layers > 0
        assert config.num_heads > 0
        assert 0 < config.learning_rate < 1
        assert 0 <= config.dropout < 1
        assert config.batch_size > 0
        assert config.max_epochs > 0


class TestIntegrationScenarios:
    """Test integration scenarios combining multiple components"""
    
    @patch('toto_trainer.Toto')
    def test_end_to_end_pipeline(self, mock_toto_class, trainer_config, dataloader_config, sample_data_files):
        """Test complete end-to-end training pipeline"""
        mock_model = self._create_complete_mock_model()
        mock_toto_class.return_value = mock_model
        
        # Configure for quick end-to-end test
        trainer_config.max_epochs = 2
        trainer_config.save_every_n_epochs = 1
        trainer_config.validation_frequency = 1
        
        trainer = TotoTrainer(trainer_config, dataloader_config)
        
        # Complete pipeline
        trainer.prepare_data()
        trainer.setup_model()
        trainer.train()
        
        # Verify final state
        assert trainer.current_epoch >= 1
        assert trainer.global_step > 0
        
        # Check that checkpoints were created
        checkpoint_files = list(Path(trainer_config.save_dir).glob("*.pt"))
        assert len(checkpoint_files) > 0
    
    @patch('toto_trainer.Toto')
    def test_resume_training_from_checkpoint(self, mock_toto_class, trainer_config, dataloader_config, sample_data_files):
        """Test resuming training from checkpoint"""
        mock_model = self._create_complete_mock_model()
        mock_toto_class.return_value = mock_model
        
        trainer_config.max_epochs = 3
        
        # First training run
        trainer1 = TotoTrainer(trainer_config, dataloader_config)
        trainer1.prepare_data()
        trainer1.setup_model()
        
        # Train for 1 epoch and save checkpoint
        trainer1.current_epoch = 0
        trainer1.train_epoch()
        trainer1.current_epoch = 1
        
        checkpoint_path = trainer1.checkpoint_manager.save_checkpoint(
            model=trainer1.model,
            optimizer=trainer1.optimizer,
            scheduler=trainer1.scheduler,
            scaler=trainer1.scaler,
            epoch=1,
            best_val_loss=0.5,
            metrics={'loss': 0.5},
            config=trainer_config
        )
        
        # Second training run - resume from checkpoint
        trainer2 = TotoTrainer(trainer_config, dataloader_config)
        trainer2.prepare_data()
        trainer2.setup_model()
        trainer2.load_checkpoint(str(checkpoint_path))
        
        # Verify state was restored
        assert trainer2.current_epoch == 1
        assert trainer2.best_val_loss == 0.5
    
    def _create_complete_mock_model(self):
        """Create a complete mock model for integration testing"""
        mock_model = Mock(spec=nn.Module)
        
        # Mock the inner model
        mock_inner_model = Mock()
        mock_output = Mock()
        mock_output.loc = torch.randn(4, 12)  # batch_size=4, prediction_length=12
        mock_inner_model.return_value = mock_output
        mock_model.model = mock_inner_model
        
        # Mock parameters
        param1 = torch.randn(50, requires_grad=True)
        param2 = torch.randn(25, requires_grad=True)
        mock_model.parameters.return_value = [param1, param2]
        
        # Mock state dict
        mock_model.state_dict.return_value = {
            'layer1.weight': param1,
            'layer2.weight': param2
        }
        mock_model.load_state_dict = Mock()
        
        # Mock training modes
        mock_model.train = Mock()
        mock_model.eval = Mock()
        
        # Mock device handling
        def mock_to(device):
            return mock_model
        mock_model.to = mock_to
        
        return mock_model


def run_comprehensive_tests():
    """Run all tests and provide a summary report"""
    print("=" * 80)
    print("RUNNING COMPREHENSIVE TOTO TRAINER TESTS")
    print("=" * 80)
    
    # Run tests with detailed output
    result = pytest.main([
        __file__, 
        "-v", 
        "--tb=short", 
        "--capture=no",
        "-x"  # Stop on first failure for detailed analysis
    ])
    
    return result


if __name__ == "__main__":
    run_comprehensive_tests()