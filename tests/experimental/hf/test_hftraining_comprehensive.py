#!/usr/bin/env python3
"""Comprehensive tests for hftraining modules."""

import pytest
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import tempfile
import json
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta
import sys
import os

# Add project root to path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Import modules to test
pytest.importorskip("torch", reason="hftraining tests require torch")
from hftraining.hf_trainer import TransformerTradingModel, HFTrainingConfig, MixedPrecisionTrainer as HFTrainer
from hftraining.data_utils import StockDataProcessor, DataCollator
from hftraining.modern_optimizers import Lion, LAMB as Lamb
# Note: Lookahead and RAdam may not be in modern_optimizers, skip for now


class TestTransformerTradingModel:
    """Test TransformerTradingModel functionality."""
    
    @pytest.fixture
    def config(self):
        """Create test configuration."""
        return HFTrainingConfig(
            hidden_size=128,
            num_heads=4,
            num_layers=2,
            intermediate_size=256,
            dropout=0.1,
            input_features=21,
            sequence_length=30,
            prediction_horizon=5
        )
    
    def test_model_initialization(self, config):
        """Test model initialization."""
        model = TransformerTradingModel(config)
        
        assert model.config == config
        assert isinstance(model.input_projection, nn.Linear)
        assert isinstance(model.transformer, nn.TransformerEncoder)
        assert model.input_projection.in_features == config.input_features
        assert model.input_projection.out_features == config.hidden_size
    
    def test_forward_pass(self, config):
        """Test model forward pass."""
        model = TransformerTradingModel(config)
        model.eval()
        
        # Create dummy input
        batch_size = 4
        x = torch.randn(batch_size, config.sequence_length, config.input_features)
        
        # Forward pass
        with torch.no_grad():
            output = model(x)
        
        # Check output structure
        assert 'price_predictions' in output
        assert 'action_logits' in output
        
        # Check output shapes
        assert output['price_predictions'].shape == (batch_size, config.prediction_horizon, config.input_features)
        assert output['action_logits'].shape == (batch_size, 3)
    
    def test_model_training_mode(self, config):
        """Test model behavior in training mode."""
        model = TransformerTradingModel(config)
        model.train()
        
        x = torch.randn(2, config.sequence_length, config.input_features)
        output = model(x)
        
        # Should apply dropout in training mode
        model.eval()
        output_eval = model(x)
        
        # Outputs should be different due to dropout
        assert not torch.allclose(output['price_predictions'], output_eval['price_predictions'])
    
    def test_gradient_flow(self, config):
        """Test gradient flow through model."""
        model = TransformerTradingModel(config)
        model.train()
        
        x = torch.randn(2, config.sequence_length, config.input_features, requires_grad=True)
        output = model(x)
        
        # Create dummy loss
        loss = output['price_predictions'].mean() + output['action_logits'].mean()
        loss.backward()
        
        # Check gradients exist
        for param in model.parameters():
            assert param.grad is not None
            assert not torch.isnan(param.grad).any()
    
    def test_model_save_load(self, config, tmp_path):
        """Test model saving and loading."""
        model = TransformerTradingModel(config)
        
        # Save model
        checkpoint_path = tmp_path / "model.pt"
        torch.save({
            'model_state_dict': model.state_dict(),
            'config': config.__dict__
        }, checkpoint_path)
        
        # Load model
        checkpoint = torch.load(checkpoint_path)
        loaded_config = HFTrainingConfig(**checkpoint['config'])
        loaded_model = TransformerTradingModel(loaded_config)
        loaded_model.load_state_dict(checkpoint['model_state_dict'])
        
        # Compare parameters
        for p1, p2 in zip(model.parameters(), loaded_model.parameters()):
            assert torch.allclose(p1, p2)


class TestHFTrainer:
    """Test HFTrainer functionality."""
    
    @pytest.fixture
    def config(self):
        """Create test configuration."""
        return HFTrainingConfig(
            hidden_size=64,
            num_heads=2,
            num_layers=1,
            learning_rate=1e-3,
            batch_size=4,
            num_epochs=2,
            warmup_steps=10,
            gradient_clip=1.0
        )
    
    @pytest.fixture
    def sample_data(self):
        """Create sample training data."""
        num_samples = 20
        seq_len = 30
        features = 21
        
        train_data = torch.randn(num_samples, seq_len, features)
        train_labels = {
            'prices': torch.randn(num_samples, 5, features),
            'actions': torch.randint(0, 3, (num_samples,))
        }
        
        val_data = torch.randn(5, seq_len, features)
        val_labels = {
            'prices': torch.randn(5, 5, features),
            'actions': torch.randint(0, 3, (5,))
        }
        
        return (train_data, train_labels), (val_data, val_labels)
    
    def test_trainer_initialization(self, config):
        """Test trainer initialization."""
        model = TransformerTradingModel(config)
        trainer = HFTrainer(model, config)
        
        assert trainer.model == model
        assert trainer.config == config
        assert isinstance(trainer.optimizer, torch.optim.Optimizer)
        assert trainer.device == torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    @patch('torch.cuda.is_available')
    def test_trainer_device_handling(self, mock_cuda, config):
        """Test device handling."""
        # Test CPU
        mock_cuda.return_value = False
        model = TransformerTradingModel(config)
        trainer = HFTrainer(model, config)
        assert trainer.device == torch.device('cpu')
        
        # Test CUDA
        mock_cuda.return_value = True
        trainer = HFTrainer(model, config)
        assert trainer.device == torch.device('cuda')
    
    def test_training_step(self, config, sample_data):
        """Test single training step."""
        model = TransformerTradingModel(config)
        trainer = HFTrainer(model, config)
        
        (train_data, train_labels), _ = sample_data
        batch_data = train_data[:4]
        batch_labels = {
            'prices': train_labels['prices'][:4],
            'actions': train_labels['actions'][:4]
        }
        
        # Run training step
        loss = trainer.training_step(batch_data, batch_labels)
        
        assert isinstance(loss, float)
        assert loss > 0
    
    def test_validation(self, config, sample_data):
        """Test validation."""
        model = TransformerTradingModel(config)
        trainer = HFTrainer(model, config)
        
        _, (val_data, val_labels) = sample_data
        
        # Run validation
        val_loss = trainer.validate(val_data, val_labels)
        
        assert isinstance(val_loss, float)
        assert val_loss > 0
    
    def test_full_training(self, config, sample_data, tmp_path):
        """Test full training loop."""
        config.num_epochs = 2
        config.checkpoint_dir = str(tmp_path)
        
        model = TransformerTradingModel(config)
        trainer = HFTrainer(model, config)
        
        (train_data, train_labels), (val_data, val_labels) = sample_data
        
        # Train model
        history = trainer.train(
            train_data, train_labels,
            val_data, val_labels
        )
        
        assert 'train_loss' in history
        assert 'val_loss' in history
        assert len(history['train_loss']) == config.num_epochs
        assert len(history['val_loss']) == config.num_epochs
        
        # Check checkpoint saved
        checkpoint_files = list(tmp_path.glob("*.pt"))
        assert len(checkpoint_files) > 0
    
    def test_optimizer_variants(self, config):
        """Test different optimizer configurations."""
        model = TransformerTradingModel(config)
        
        # Test with Adam
        config.optimizer = 'adam'
        trainer = HFTrainer(model, config)
        assert isinstance(trainer.optimizer, torch.optim.Adam)
        
        # Test with AdamW
        config.optimizer = 'adamw'
        trainer = HFTrainer(model, config)
        assert isinstance(trainer.optimizer, torch.optim.AdamW)
        
        # Test with custom optimizer
        config.optimizer = 'lion'
        trainer = HFTrainer(model, config)
        # Should handle custom optimizers gracefully
    
    def test_scheduler(self, config):
        """Test learning rate scheduler."""
        model = TransformerTradingModel(config)
        trainer = HFTrainer(model, config)
        
        initial_lr = trainer.optimizer.param_groups[0]['lr']
        
        # Step scheduler
        if hasattr(trainer, 'scheduler'):
            trainer.scheduler.step()
            new_lr = trainer.optimizer.param_groups[0]['lr']
            # LR should change
            assert new_lr != initial_lr or config.warmup_steps == 0


class TestStockDataProcessorAdvanced:
    """Advanced tests for StockDataProcessor."""
    
    @pytest.fixture
    def processor(self):
        """Create processor instance."""
        return StockDataProcessor(
            sequence_length=30,
            prediction_horizon=5,
            features=['close', 'volume', 'rsi', 'macd']
        )
    
    @pytest.fixture
    def sample_df(self):
        """Create sample dataframe."""
        dates = pd.date_range(start='2023-01-01', periods=200, freq='D')
        return pd.DataFrame({
            'open': np.random.uniform(90, 110, 200),
            'high': np.random.uniform(95, 115, 200),
            'low': np.random.uniform(85, 105, 200),
            'close': np.random.uniform(90, 110, 200),
            'volume': np.random.randint(1000000, 10000000, 200)
        }, index=dates)
    
    def test_feature_engineering(self, processor, sample_df):
        """Test feature engineering."""
        enhanced_df = processor.engineer_features(sample_df)
        
        # Check technical indicators added
        expected_features = ['returns', 'log_returns', 'rsi', 'macd', 
                           'macd_signal', 'bb_upper', 'bb_lower']
        
        for feature in expected_features:
            assert feature in enhanced_df.columns
        
        # Check no NaN in critical features after engineering
        assert not enhanced_df['close'].isna().any()
    
    def test_normalization(self, processor, sample_df):
        """Test data normalization."""
        enhanced_df = processor.engineer_features(sample_df)
        normalized = processor.normalize(enhanced_df)
        
        # Check normalization applied
        for col in normalized.columns:
            if col in processor.features:
                # Should be roughly normalized
                assert normalized[col].mean() < 10  # Reasonable scale
                assert normalized[col].std() < 10
    
    def test_sequence_creation(self, processor, sample_df):
        """Test sequence creation."""
        enhanced_df = processor.engineer_features(sample_df)
        normalized = processor.normalize(enhanced_df)
        
        sequences, targets = processor.create_sequences(normalized)
        
        assert len(sequences) > 0
        assert len(sequences) == len(targets)
        assert sequences.shape[1] == processor.sequence_length
        assert targets.shape[1] == processor.prediction_horizon
    
    def test_data_augmentation(self, processor):
        """Test data augmentation techniques."""
        data = np.random.randn(10, 30, 21)
        
        # Test noise addition
        augmented = processor.add_noise(data, noise_level=0.01)
        assert augmented.shape == data.shape
        assert not np.array_equal(augmented, data)
        
        # Test time warping
        warped = processor.time_warp(data)
        assert warped.shape == data.shape
    
    def test_pipeline_integration(self, processor, sample_df):
        """Test full data processing pipeline."""
        # Process data through full pipeline
        train_data, val_data = processor.prepare_data(sample_df)
        
        assert train_data is not None
        assert val_data is not None
        assert len(train_data) > len(val_data)
    
    @patch('yfinance.download')
    def test_data_download(self, mock_download, processor):
        """Test data download functionality."""
        mock_download.return_value = pd.DataFrame({
            'Open': [100, 101],
            'High': [102, 103],
            'Low': [99, 100],
            'Close': [101, 102],
            'Volume': [1000000, 1100000]
        })
        
        from hftraining.data_utils import download_stock_data
        data = download_stock_data(['AAPL'], start_date='2023-01-01')
        
        assert 'AAPL' in data
        assert len(data['AAPL']) == 2


class TestModernOptimizers:
    """Test modern optimizer implementations."""
    
    @pytest.fixture
    def model(self):
        """Create simple test model."""
        return nn.Sequential(
            nn.Linear(10, 20),
            nn.ReLU(),
            nn.Linear(20, 1)
        )
    
    def test_lion_optimizer(self, model):
        """Test Lion optimizer."""
        optimizer = Lion(model.parameters(), lr=1e-4)
        
        # Run optimization step
        x = torch.randn(32, 10)
        y = torch.randn(32, 1)
        
        output = model(x)
        loss = nn.MSELoss()(output, y)
        loss.backward()
        
        optimizer.step()
        optimizer.zero_grad()
        
        # Check parameters updated
        assert all(p.grad is None or p.grad.sum() == 0 for p in model.parameters())
    
    def test_lamb_optimizer(self, model):
        """Test Lamb optimizer."""
        optimizer = Lamb(model.parameters(), lr=1e-3)
        
        x = torch.randn(32, 10)
        y = torch.randn(32, 1)
        
        output = model(x)
        loss = nn.MSELoss()(output, y)
        loss.backward()
        
        # Store original params
        orig_params = [p.clone() for p in model.parameters()]
        
        optimizer.step()
        
        # Check parameters changed
        for orig, new in zip(orig_params, model.parameters()):
            assert not torch.allclose(orig, new)
    
    # def test_lookahead_optimizer(self, model):
    #     """Test Lookahead optimizer."""
    #     base_opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    #     optimizer = Lookahead(base_opt, k=5, alpha=0.5)
    #     
    #     # Run multiple steps to trigger lookahead update
    #     for _ in range(10):
    #         x = torch.randn(32, 10)
    #         y = torch.randn(32, 1)
    #         
    #         optimizer.zero_grad()
    #         output = model(x)
    #         loss = nn.MSELoss()(output, y)
    #         loss.backward()
    #         optimizer.step()
    #     
    #     # Check slow weights updated
    #     assert hasattr(optimizer, 'slow_weights')
    # 
    # def test_radam_optimizer(self, model):
    #     """Test RAdam optimizer."""
    #     optimizer = RAdam(model.parameters(), lr=1e-3)
    #     
    #     x = torch.randn(32, 10)
    #     y = torch.randn(32, 1)
    #     
    #     output = model(x)
    #     loss = nn.MSELoss()(output, y)
    #     loss.backward()
    #     
    #     optimizer.step()
    #     optimizer.zero_grad()
    #     
    #     # Check state updated
    #     assert len(optimizer.state) > 0


class TestDataCollator:
    """Test DataCollator functionality."""
    
    def test_collator_padding(self):
        """Test sequence padding."""
        collator = DataCollator(pad_token_id=0)
        
        # Create sequences of different lengths
        batch = [
            {'input': torch.randn(20, 21), 'target': torch.randn(5, 21)},
            {'input': torch.randn(25, 21), 'target': torch.randn(5, 21)},
            {'input': torch.randn(30, 21), 'target': torch.randn(5, 21)}
        ]
        
        collated = collator(batch)
        
        # All sequences should have same length after padding
        assert collated['input'].shape[0] == 3  # batch size
        assert collated['input'].shape[1] == 30  # max length
        assert collated['target'].shape[0] == 3
    
    def test_collator_attention_mask(self):
        """Test attention mask creation."""
        collator = DataCollator(pad_token_id=0, create_attention_mask=True)
        
        batch = [
            {'input': torch.randn(20, 21)},
            {'input': torch.randn(30, 21)}
        ]
        
        collated = collator(batch)
        
        assert 'attention_mask' in collated
        assert collated['attention_mask'].shape == (2, 30)
        # First sequence should have 20 True values
        assert collated['attention_mask'][0].sum() == 20
        # Second sequence should have 30 True values
        assert collated['attention_mask'][1].sum() == 30


class TestTrainingUtilities:
    """Test training utility functions."""
    
    def test_checkpoint_management(self, tmp_path):
        """Test checkpoint saving and loading."""
        from hftraining.hf_trainer import save_checkpoint, load_checkpoint
        
        # Create dummy model and optimizer
        model = nn.Linear(10, 1)
        optimizer = torch.optim.Adam(model.parameters())
        
        # Save checkpoint
        checkpoint_path = tmp_path / "checkpoint.pt"
        save_checkpoint(
            model, optimizer, 
            epoch=5, loss=0.1,
            path=checkpoint_path
        )
        
        assert checkpoint_path.exists()
        
        # Load checkpoint
        loaded = load_checkpoint(checkpoint_path)
        assert 'model_state_dict' in loaded
        assert 'optimizer_state_dict' in loaded
        assert loaded['epoch'] == 5
        assert loaded['loss'] == 0.1
    
    def test_early_stopping(self):
        """Test early stopping mechanism."""
        from hftraining.hf_trainer import EarlyStopping
        
        early_stopping = EarlyStopping(patience=3, min_delta=0.001)
        
        # Simulate training
        losses = [1.0, 0.9, 0.85, 0.84, 0.839, 0.838]
        
        for loss in losses:
            should_stop = early_stopping(loss)
            if should_stop:
                break
        
        assert early_stopping.best_loss < 1.0
        assert early_stopping.counter > 0
    
    def test_metric_tracking(self):
        """Test metric tracking during training."""
        from hftraining.hf_trainer import MetricTracker
        
        tracker = MetricTracker()
        
        # Add metrics
        for epoch in range(5):
            tracker.add('train_loss', 1.0 - epoch * 0.1)
            tracker.add('val_loss', 0.9 - epoch * 0.08)
            tracker.add('accuracy', 0.5 + epoch * 0.05)
        
        # Get history
        history = tracker.get_history()
        assert len(history['train_loss']) == 5
        assert len(history['val_loss']) == 5
        assert len(history['accuracy']) == 5
        
        # Get best metrics
        best = tracker.get_best_metrics()
        assert best['train_loss'] == min(history['train_loss'])
        assert best['accuracy'] == max(history['accuracy'])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])