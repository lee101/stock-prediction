#!/usr/bin/env python3
"""Unit tests for hftraining model components."""

import pytest
import torch
import torch.nn as nn
import numpy as np
from unittest.mock import Mock, patch
import tempfile
import os

# Add hftraining to path for imports
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../hftraining'))

from hftraining.hf_trainer import (
    HFTrainingConfig,
    TransformerTradingModel,
    PositionalEncoding,
    GPro,
    AdamW,
    MixedPrecisionTrainer,
    EarlyStopping,
    get_linear_schedule_with_warmup,
    get_cosine_schedule_with_warmup
)


class TestHFTrainingConfig:
    """Test HFTrainingConfig functionality."""
    
    def test_default_init(self):
        """Test default configuration."""
        config = HFTrainingConfig()
        
        # Check default values
        assert config.hidden_size == 512
        assert config.num_layers == 8
        assert config.num_heads == 16
        assert config.learning_rate == 1e-4
        assert config.optimizer_name == "gpro"
        assert config.batch_size == 32
        assert config.sequence_length == 60
        assert config.use_mixed_precision == True
    
    def test_custom_init(self):
        """Test custom configuration."""
        config = HFTrainingConfig(
            hidden_size=1024,
            num_layers=12,
            learning_rate=5e-5,
            optimizer_name="adamw"
        )
        
        assert config.hidden_size == 1024
        assert config.num_layers == 12
        assert config.learning_rate == 5e-5
        assert config.optimizer_name == "adamw"


class TestTransformerTradingModel:
    """Test TransformerTradingModel functionality."""
    
    @pytest.fixture
    def config(self):
        """Create test configuration."""
        return HFTrainingConfig(
            hidden_size=128,
            num_layers=2,
            num_heads=4,
            sequence_length=20,
            prediction_horizon=3
        )
    
    def test_model_init(self, config):
        """Test model initialization."""
        input_dim = 10
        model = TransformerTradingModel(config, input_dim)
        
        # Check components exist
        assert hasattr(model, 'input_projection')
        assert hasattr(model, 'pos_encoding')
        assert hasattr(model, 'transformer')
        assert hasattr(model, 'action_head')
        assert hasattr(model, 'value_head')
        assert hasattr(model, 'price_prediction_head')
        
        # Check dimensions
        assert model.input_projection.in_features == input_dim
        assert model.input_projection.out_features == config.hidden_size
    
    def test_forward_pass(self, config):
        """Test forward pass."""
        input_dim = 15
        batch_size = 4
        seq_len = config.sequence_length
        
        model = TransformerTradingModel(config, input_dim)
        x = torch.randn(batch_size, seq_len, input_dim)
        
        # Forward pass
        outputs = model(x)
        
        # Check output structure
        assert 'action_logits' in outputs
        assert 'value' in outputs
        assert 'price_predictions' in outputs
        assert 'hidden_states' in outputs
        
        # Check output shapes
        assert outputs['action_logits'].shape == (batch_size, 3)  # 3 actions
        assert outputs['value'].shape == (batch_size,)
        assert outputs['price_predictions'].shape == (batch_size, config.prediction_horizon)
        assert outputs['hidden_states'].shape == (batch_size, seq_len, config.hidden_size)
    
    def test_forward_with_attention_mask(self, config):
        """Test forward pass with attention mask."""
        input_dim = 10
        batch_size = 2
        seq_len = config.sequence_length
        
        model = TransformerTradingModel(config, input_dim)
        x = torch.randn(batch_size, seq_len, input_dim)
        
        # Create attention mask (1 = attend, 0 = don't attend)
        attention_mask = torch.ones(batch_size, seq_len)
        attention_mask[0, -5:] = 0  # Mask last 5 positions for first batch
        
        outputs = model(x, attention_mask=attention_mask)
        
        # Should still produce valid outputs
        assert outputs['action_logits'].shape == (batch_size, 3)
        assert outputs['value'].shape == (batch_size,)
    
    def test_parameter_count(self, config):
        """Test parameter counting."""
        input_dim = 20
        model = TransformerTradingModel(config, input_dim)
        
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        assert total_params > 0
        assert trainable_params == total_params  # All parameters should be trainable
        assert total_params > 10000  # Should have reasonable number of parameters


class TestPositionalEncoding:
    """Test PositionalEncoding functionality."""
    
    def test_positional_encoding_init(self):
        """Test positional encoding initialization."""
        d_model = 128
        max_len = 100
        
        pos_enc = PositionalEncoding(d_model, max_len)
        
        # Check registered buffer
        assert hasattr(pos_enc, 'pe')
        assert pos_enc.pe.shape == (max_len, 1, d_model)
    
    def test_positional_encoding_forward(self):
        """Test positional encoding forward pass."""
        d_model = 64
        batch_size = 8
        seq_len = 50
        
        pos_enc = PositionalEncoding(d_model, max_len=100)
        x = torch.randn(batch_size, seq_len, d_model)
        
        output = pos_enc(x)
        
        # Check output shape
        assert output.shape == x.shape
        
        # Check that positional encoding was added
        assert not torch.equal(x, output)


class TestOptimizers:
    """Test custom optimizer implementations."""
    
    def test_gpro_optimizer(self):
        """Test GPro optimizer."""
        # Create simple model
        model = nn.Linear(10, 1)
        optimizer = GPro(model.parameters(), lr=0.001)
        
        # Test initialization
        assert optimizer.defaults['lr'] == 0.001
        assert optimizer.defaults['projection_factor'] == 0.5
        
        # Test optimization step
        x = torch.randn(32, 10)
        y = torch.randn(32, 1)
        
        initial_params = [p.clone() for p in model.parameters()]
        
        # Forward pass and backward
        loss = nn.MSELoss()(model(x), y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
        # Check parameters changed
        final_params = list(model.parameters())
        for initial, final in zip(initial_params, final_params):
            assert not torch.equal(initial, final)
    
    def test_adamw_optimizer(self):
        """Test AdamW optimizer."""
        model = nn.Linear(5, 1)
        optimizer = AdamW(model.parameters(), lr=0.01, weight_decay=0.001)
        
        # Test initialization
        assert optimizer.defaults['lr'] == 0.01
        assert optimizer.defaults['weight_decay'] == 0.001
        
        # Test optimization step
        x = torch.randn(16, 5)
        y = torch.randn(16, 1)
        
        initial_params = [p.clone() for p in model.parameters()]
        
        loss = nn.MSELoss()(model(x), y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
        # Check parameters changed
        final_params = list(model.parameters())
        for initial, final in zip(initial_params, final_params):
            assert not torch.equal(initial, final)
    
    def test_optimizer_invalid_params(self):
        """Test optimizer parameter validation."""
        model = nn.Linear(5, 1)
        
        # Test invalid learning rate
        with pytest.raises(ValueError, match="Invalid learning rate"):
            GPro(model.parameters(), lr=-0.001)
        
        # Test invalid beta parameters
        with pytest.raises(ValueError, match="Invalid beta parameter"):
            GPro(model.parameters(), betas=(1.5, 0.999))


class TestLearningRateSchedulers:
    """Test learning rate schedulers."""
    
    def test_linear_schedule_with_warmup(self):
        """Test linear scheduler with warmup."""
        model = nn.Linear(5, 1)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        
        num_warmup_steps = 100
        num_training_steps = 1000
        
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps, num_training_steps
        )
        
        # Test warmup phase
        initial_lr = scheduler.get_last_lr()[0]
        
        # Step through warmup
        for _ in range(num_warmup_steps):
            scheduler.step()
        
        warmup_lr = scheduler.get_last_lr()[0]
        assert warmup_lr > initial_lr
        
        # Step through decay phase
        for _ in range(num_training_steps - num_warmup_steps):
            scheduler.step()
        
        final_lr = scheduler.get_last_lr()[0]
        assert final_lr < warmup_lr
    
    def test_cosine_schedule_with_warmup(self):
        """Test cosine scheduler with warmup."""
        model = nn.Linear(5, 1)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        
        num_warmup_steps = 50
        num_training_steps = 500
        
        scheduler = get_cosine_schedule_with_warmup(
            optimizer, num_warmup_steps, num_training_steps
        )
        
        # Test warmup phase
        initial_lr = scheduler.get_last_lr()[0]
        
        for _ in range(num_warmup_steps):
            scheduler.step()
        
        warmup_lr = scheduler.get_last_lr()[0]
        assert warmup_lr > initial_lr
        
        # Test cosine decay
        mid_step_lr = warmup_lr
        for _ in range((num_training_steps - num_warmup_steps) // 2):
            scheduler.step()
        
        mid_lr = scheduler.get_last_lr()[0]
        assert mid_lr < mid_step_lr


class TestMixedPrecisionTrainer:
    """Test mixed precision training utilities."""
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_mixed_precision_enabled(self):
        """Test mixed precision with CUDA."""
        trainer = MixedPrecisionTrainer(enabled=True)
        
        assert trainer.enabled
        assert trainer.scaler is not None
        
        # Test autocast context
        with trainer.autocast():
            x = torch.randn(10, 5, device='cuda')
            y = x * 2
            assert y.device.type == 'cuda'
    
    def test_mixed_precision_disabled(self):
        """Test mixed precision disabled."""
        trainer = MixedPrecisionTrainer(enabled=False)
        
        assert not trainer.enabled
        assert trainer.scaler is None
        
        # Test dummy context
        with trainer.autocast():
            x = torch.randn(10, 5)
            y = x * 2
            assert y.shape == x.shape


class TestEarlyStopping:
    """Test early stopping functionality."""
    
    def test_early_stopping_init(self):
        """Test early stopping initialization."""
        early_stopping = EarlyStopping(patience=5, threshold=0.001)
        
        assert early_stopping.patience == 5
        assert early_stopping.threshold == 0.001
        assert not early_stopping.greater_is_better
        assert early_stopping.best_score is None
        assert early_stopping.counter == 0
        assert not early_stopping.should_stop
    
    def test_early_stopping_improvement(self):
        """Test early stopping with improvement."""
        early_stopping = EarlyStopping(patience=3, threshold=0.01, greater_is_better=False)
        
        # First score
        early_stopping(1.0)
        assert early_stopping.best_score == 1.0
        assert early_stopping.counter == 0
        
        # Improvement (lower is better)
        early_stopping(0.8)
        assert early_stopping.best_score == 0.8
        assert early_stopping.counter == 0
        
        # Another improvement
        early_stopping(0.6)
        assert early_stopping.best_score == 0.6
        assert early_stopping.counter == 0
        assert not early_stopping.should_stop
    
    def test_early_stopping_no_improvement(self):
        """Test early stopping without improvement."""
        early_stopping = EarlyStopping(patience=2, threshold=0.01, greater_is_better=False)
        
        # First score
        early_stopping(1.0)
        
        # No improvement
        early_stopping(1.1)
        assert early_stopping.counter == 1
        assert not early_stopping.should_stop
        
        # Still no improvement
        early_stopping(1.05)
        assert early_stopping.counter == 2
        assert early_stopping.should_stop
    
    def test_early_stopping_greater_is_better(self):
        """Test early stopping with greater_is_better=True."""
        early_stopping = EarlyStopping(patience=2, threshold=0.01, greater_is_better=True)
        
        # First score
        early_stopping(0.5)
        
        # Improvement (higher is better)
        early_stopping(0.7)
        assert early_stopping.best_score == 0.7
        assert early_stopping.counter == 0
        
        # No improvement
        early_stopping(0.6)
        assert early_stopping.counter == 1
        
        early_stopping(0.65)
        assert early_stopping.counter == 2
        assert early_stopping.should_stop