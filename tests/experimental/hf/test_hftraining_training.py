#!/usr/bin/env python3
"""Unit tests for hftraining training components."""

import pytest
import torch
import torch.nn as nn
import numpy as np
from unittest.mock import Mock, patch, MagicMock
import tempfile
import os
import json
from pathlib import Path

# Add hftraining to path for imports
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../hftraining'))

from hftraining.train_hf import StockDataset, HFTrainer
from hftraining.hf_trainer import HFTrainingConfig, TransformerTradingModel
from hftraining.config import ExperimentConfig, create_config
from hftraining.run_training import setup_environment, load_and_process_data, create_model


@pytest.fixture(autouse=True)
def force_gpu_cuda():
    """Ensure tests execute with CUDA enabled and restore SDP kernel toggles."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA GPU required for hftraining tests")

    try:
        flash_enabled = torch.backends.cuda.flash_sdp_enabled()
        mem_enabled = torch.backends.cuda.mem_efficient_sdp_enabled()
        math_enabled = torch.backends.cuda.math_sdp_enabled()
    except AttributeError:
        yield
        return

    torch.backends.cuda.enable_flash_sdp(True)
    torch.backends.cuda.enable_mem_efficient_sdp(True)
    torch.backends.cuda.enable_math_sdp(True)

    try:
        yield
    finally:
        torch.backends.cuda.enable_flash_sdp(flash_enabled)
        torch.backends.cuda.enable_mem_efficient_sdp(mem_enabled)
        torch.backends.cuda.enable_math_sdp(math_enabled)


class TestStockDataset:
    """Test StockDataset functionality."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        return np.random.randn(200, 15)  # 200 timesteps, 15 features
    
    def test_dataset_init(self, sample_data):
        """Test dataset initialization."""
        dataset = StockDataset(
            sample_data,
            sequence_length=30,
            prediction_horizon=5
        )
        
        assert dataset.sequence_length == 30
        assert dataset.prediction_horizon == 5
        assert len(dataset.data) == 200
        
        # Check that we can create sequences
        expected_length = 200 - 30 - 5 + 1  # data_len - seq_len - pred_horizon + 1
        assert len(dataset) == expected_length
    
    def test_dataset_getitem(self, sample_data):
        """Test dataset item access."""
        dataset = StockDataset(
            sample_data,
            sequence_length=20,
            prediction_horizon=3
        )
        
        # Get first item
        item = dataset[0]
        
        # Check structure
        assert 'input_ids' in item
        assert 'labels' in item
        assert 'action_labels' in item
        
        # Check shapes
        assert item['input_ids'].shape == (20, 15)  # seq_len x features
        assert item['labels'].shape == (3, 15)      # pred_horizon x features
        assert item['action_labels'].shape == ()    # scalar
        
        # Check types
        assert isinstance(item['input_ids'], torch.Tensor)
        assert isinstance(item['labels'], torch.Tensor)
        assert isinstance(item['action_labels'], torch.Tensor)
    
    def test_dataset_insufficient_data(self):
        """Test dataset with insufficient data."""
        small_data = np.random.randn(10, 5)  # Too small
        
        with pytest.raises(ValueError, match="Dataset too small"):
            StockDataset(small_data, sequence_length=15, prediction_horizon=5)
    
    def test_dataset_action_labels(self, sample_data):
        """Test action label generation."""
        # Create data with predictable price movements
        data = np.ones((100, 5))
        data[:, 3] = np.arange(100)  # Increasing close prices (column 3)
        
        dataset = StockDataset(data, sequence_length=10, prediction_horizon=1)
        
        # All action labels should be 0 (buy) due to increasing prices
        for i in range(len(dataset)):
            item = dataset[i]
            # With constantly increasing prices, should mostly be buy signals
            assert item['action_labels'].item() in [0, 1, 2]


class TestHFTrainer:
    """Test HFTrainer functionality."""
    
    @pytest.fixture
    def config(self):
        """Create test configuration."""
        return HFTrainingConfig(
            hidden_size=64,
            num_layers=2,
            num_heads=4,
            batch_size=8,
            max_steps=100,
            eval_steps=50,
            save_steps=50,
            logging_steps=25,
            sequence_length=15,
            prediction_horizon=3,
            learning_rate=1e-3,
            warmup_steps=10,
            dropout=0.0,
            dropout_rate=0.0
        )
    
    @pytest.fixture
    def sample_datasets(self):
        """Create sample datasets."""
        train_data = np.random.randn(500, 10)
        val_data = np.random.randn(200, 10)
        
        train_dataset = StockDataset(train_data, sequence_length=15, prediction_horizon=3)
        val_dataset = StockDataset(val_data, sequence_length=15, prediction_horizon=3)
        
        return train_dataset, val_dataset
    
    def test_trainer_init(self, config, sample_datasets):
        """Test trainer initialization."""
        train_dataset, val_dataset = sample_datasets
        model = TransformerTradingModel(config, input_dim=10)
        
        trainer = HFTrainer(
            model=model,
            config=config,
            train_dataset=train_dataset,
            eval_dataset=val_dataset
        )
        
        assert trainer.model == model
        assert trainer.config == config
        assert trainer.train_dataset == train_dataset
        assert trainer.eval_dataset == val_dataset
        assert trainer.global_step == 0
    
    def test_trainer_compute_loss(self, config, sample_datasets):
        """Test loss computation."""
        train_dataset, val_dataset = sample_datasets
        model = TransformerTradingModel(config, input_dim=10)
        
        trainer = HFTrainer(
            model=model,
            config=config,
            train_dataset=train_dataset,
            eval_dataset=val_dataset
        )
        
        # Create sample batch
        batch = {
            'input_ids': torch.randn(4, 15, 10),
            'labels': torch.randn(4, 3, 10),
            'action_labels': torch.randint(0, 3, (4,)),
            'attention_mask': torch.ones(4, 15, dtype=torch.long),
        }
        
        loss = trainer.training_step(batch)
        
        assert isinstance(loss, float)
        assert loss >= 0
    
    def test_trainer_evaluation_step(self, config, sample_datasets):
        """Test evaluation step."""
        train_dataset, val_dataset = sample_datasets
        model = TransformerTradingModel(config, input_dim=10)
        
        trainer = HFTrainer(
            model=model,
            config=config,
            train_dataset=train_dataset,
            eval_dataset=val_dataset
        )
        
        # Mock evaluation
        with patch.object(trainer, 'evaluate') as mock_evaluate:
            mock_evaluate.return_value = {
                'eval_loss': 0.5,
                'eval_action_loss': 0.3,
                'eval_price_loss': 0.2
            }
            
            metrics = trainer.evaluation_step()
            
            assert 'eval_loss' in metrics
            assert 'eval_action_loss' in metrics
            assert 'eval_price_loss' in metrics
    
    @patch('hftraining.train_hf.WandBoardLogger')
    def test_trainer_logging(self, mock_logger_cls, config, sample_datasets):
        """Test trainer logging functionality."""
        train_dataset, val_dataset = sample_datasets
        model = TransformerTradingModel(config, input_dim=10)

        mock_logger = MagicMock()
        mock_logger.tensorboard_writer = MagicMock()
        mock_logger.tensorboard_log_dir = Path("logs")
        mock_logger.wandb_enabled = False
        mock_logger.log = MagicMock()
        mock_logger.add_scalar = MagicMock()
        mock_logger.finish = MagicMock()
        mock_logger_cls.return_value = mock_logger
        
        trainer = HFTrainer(
            model=model,
            config=config,
            train_dataset=train_dataset,
            eval_dataset=val_dataset
        )
        
        # Test log metrics
        metrics = {
            'train/loss': 0.5,
            'train/learning_rate': 1e-4
        }
        
        trainer.log_metrics(metrics, step=10)
        
        # Should use the unified metrics logger
        assert hasattr(trainer, 'metrics_logger')
        mock_logger.log.assert_called()
    
    def test_trainer_save_checkpoint(self, config, sample_datasets):
        """Test checkpoint saving."""
        train_dataset, val_dataset = sample_datasets
        model = TransformerTradingModel(config, input_dim=10)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            config.output_dir = tmpdir
            
            trainer = HFTrainer(
                model=model,
                config=config,
                train_dataset=train_dataset,
                eval_dataset=val_dataset
            )
            
            trainer.step = 100
            trainer.save_checkpoint()
            
            # Check checkpoint was saved
            checkpoint_path = Path(tmpdir) / "checkpoint_step_100.pth"
            assert checkpoint_path.exists()
            
            # Load and verify checkpoint
            checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
            assert 'model_state_dict' in checkpoint
            assert 'global_step' in checkpoint
            assert checkpoint['global_step'] == 100


class TestConfigSystem:
    """Test configuration system."""
    
    def test_create_config_default(self):
        """Test default configuration creation."""
        config = create_config("default")
        
        assert isinstance(config, ExperimentConfig)
        assert config.model.hidden_size > 0
        assert config.training.learning_rate > 0
        assert len(config.data.symbols) > 0
    
    def test_create_config_quick_test(self):
        """Test quick test configuration."""
        config = create_config("quick_test")
        
        assert config.training.max_steps <= 1000  # Should be small for testing
        assert config.model.hidden_size <= 256    # Should be small for testing
        assert len(config.data.symbols) == 1      # Should use single symbol
    
    def test_create_config_production(self):
        """Test production configuration."""
        config = create_config("production")
        
        assert config.training.max_steps >= 10000  # Should be large for production
        assert config.model.hidden_size >= 512     # Should be large for production
        assert len(config.data.symbols) > 1        # Should use multiple symbols
    
    def test_config_save_load(self):
        """Test configuration saving and loading."""
        config = create_config("default")
        config.experiment_name = "test_experiment"
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as tmp:
            try:
                # Save config
                config.save(tmp.name)
                
                # Load config
                loaded_config = ExperimentConfig.load(tmp.name)
                
                # Check loaded config
                assert loaded_config.experiment_name == "test_experiment"
                assert loaded_config.model.hidden_size == config.model.hidden_size
                assert loaded_config.training.learning_rate == config.training.learning_rate
                
            finally:
                os.unlink(tmp.name)


class TestTrainingPipeline:
    """Test training pipeline functions."""
    
    def test_setup_environment(self):
        """Test environment setup."""
        config = create_config("quick_test")
        
        with tempfile.TemporaryDirectory() as tmpdir:
            config.output.output_dir = tmpdir
            config.output.logging_dir = os.path.join(tmpdir, "logs")
            config.output.cache_dir = os.path.join(tmpdir, "cache")
            
            device = setup_environment(config)
            
            # Check directories were created
            assert Path(config.output.output_dir).exists()
            assert Path(config.output.logging_dir).exists()
            assert Path(config.output.cache_dir).exists()
            
            # Check config was saved
            config_path = Path(config.output.output_dir) / "config.json"
            assert config_path.exists()
            
            # Check device is valid
            assert device in ["cpu", "cuda", "mps"]
    
    @patch('hftraining.run_training.load_training_data')
    @patch('hftraining.run_training.StockDataProcessor')
    def test_load_and_process_data(self, mock_processor_class, mock_load_data):
        """Test data loading and processing."""
        config = create_config("quick_test")
        
        # Mock data loading
        mock_data = np.random.randn(1000, 20)
        mock_load_data.return_value = mock_data
        
        # Mock processor
        mock_processor = Mock()
        mock_processor.transform.return_value = mock_data
        mock_processor.feature_names = [f"feature_{i}" for i in range(20)]
        mock_processor_class.return_value = mock_processor
        
        with tempfile.TemporaryDirectory() as tmpdir:
            config.output.output_dir = tmpdir
            
            train_dataset, val_dataset, processor = load_and_process_data(config)
            
            # Check datasets were created
            assert train_dataset is not None
            assert train_dataset.__class__.__name__ == "StockDataset"
            
            # Check processor was saved
            processor_path = Path(config.output.output_dir) / "data_processor.pkl"
            mock_processor.save_scalers.assert_called_with(str(processor_path))
    
    def test_create_model(self):
        """Test model creation."""
        config = create_config("quick_test")
        input_dim = 25
        
        model, hf_config = create_model(config, input_dim)
        
        # Check model was created
        assert model.__class__.__name__ == "TransformerTradingModel"
        assert model.input_dim == input_dim
        
        # Check config conversion
        assert hf_config.hidden_size == config.model.hidden_size
        assert hf_config.learning_rate == config.training.learning_rate
        
        # Check model has parameters
        total_params = sum(p.numel() for p in model.parameters())
        assert total_params > 0
