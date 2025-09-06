#!/usr/bin/env python3
"""
Realistic integration tests for hftraining/ directory.
Tests actual model training, data processing, and optimization without mocks.
"""

import os
import sys
import tempfile
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from pathlib import Path
import json

# Add paths
TEST_DIR = Path(__file__).parent.parent
REPO_ROOT = TEST_DIR.parent
sys.path.extend([str(REPO_ROOT), str(REPO_ROOT / 'hftraining')])

import pytest


class TestHFTrainer:
    """Test HuggingFace trainer with real training loops."""
    
    @pytest.fixture
    def training_data(self):
        """Generate realistic financial training data."""
        n_samples = 500
        seq_len = 30
        n_features = 10
        
        # Create time series data with trends
        data = []
        for _ in range(n_samples):
            trend = np.random.randn() * 0.01
            noise = np.random.randn(seq_len, n_features) * 0.1
            base = np.linspace(0, trend * seq_len, seq_len).reshape(-1, 1)
            sample = base + noise
            data.append(sample)
        
        X = np.array(data, dtype=np.float32)
        y = np.random.randn(n_samples, 1).astype(np.float32)
        
        return torch.from_numpy(X), torch.from_numpy(y)
    
    def test_hf_trainer_training_loop(self, training_data):
        """Test complete training loop with HF trainer."""
        from hftraining.hf_trainer import HFTrainer, HFTrainingConfig, TransformerTradingModel
        
        X, y = training_data
        
        config = HFTrainingConfig(
            hidden_size=64,
            num_layers=2,
            num_heads=4,
            dropout=0.1,
            sequence_length=30,
            prediction_horizon=1,
            learning_rate=1e-3,
            batch_size=32,
            num_epochs=3,
            use_mixed_precision=False,
            gradient_clip_val=1.0
        )
        
        model = TransformerTradingModel(config, input_dim=10)
        trainer = HFTrainer(model, config)
        
        # Split data
        split_idx = int(len(X) * 0.8)
        train_X, val_X = X[:split_idx], X[split_idx:]
        train_y, val_y = y[:split_idx], y[split_idx:]
        
        # Train
        initial_loss = trainer.evaluate(val_X, val_y)
        history = trainer.train(train_X, train_y, val_X, val_y)
        final_loss = trainer.evaluate(val_X, val_y)
        
        # Verify training improved model
        assert final_loss < initial_loss * 0.95
        assert len(history['train_loss']) == config.num_epochs
        assert all(loss > 0 for loss in history['train_loss'])
        
        # Test prediction
        predictions = trainer.predict(val_X[:10])
        assert predictions.shape == (10, 1)
        assert not torch.isnan(predictions).any()
    
    def test_hf_trainer_checkpoint_resume(self, training_data):
        """Test checkpoint saving and resuming."""
        from hftraining.hf_trainer import HFTrainer, HFTrainingConfig, TransformerTradingModel
        
        X, y = training_data
        
        with tempfile.TemporaryDirectory() as tmpdir:
            config = HFTrainingConfig(
                hidden_size=32,
                num_layers=1,
                num_heads=2,
                checkpoint_dir=tmpdir,
                save_every_n_steps=50
            )
            
            model = TransformerTradingModel(config, input_dim=10)
            trainer = HFTrainer(model, config)
            
            # Train partially
            trainer.train(X[:100], y[:100], max_steps=50)
            
            # Save checkpoint
            checkpoint_path = Path(tmpdir) / 'checkpoint.pt'
            trainer.save_checkpoint(checkpoint_path)
            
            # Create new trainer and load
            model2 = TransformerTradingModel(config, input_dim=10)
            trainer2 = HFTrainer(model2, config)
            trainer2.load_checkpoint(checkpoint_path)
            
            # Verify weights are same
            for p1, p2 in zip(model.parameters(), model2.parameters()):
                assert torch.allclose(p1, p2)


class TestDataUtils:
    """Test data utilities with real data processing."""
    
    def test_data_preprocessor_normalization(self):
        """Test data preprocessing and normalization."""
        from hftraining.data_utils import DataPreprocessor, create_sequences
        
        # Create realistic OHLCV data
        n_days = 1000
        dates = pd.date_range('2020-01-01', periods=n_days)
        
        data = pd.DataFrame({
            'open': 100 + np.random.randn(n_days).cumsum(),
            'high': 101 + np.random.randn(n_days).cumsum(),
            'low': 99 + np.random.randn(n_days).cumsum(),
            'close': 100 + np.random.randn(n_days).cumsum(),
            'volume': np.random.lognormal(10, 1, n_days)
        }, index=dates)
        
        preprocessor = DataPreprocessor(
            normalize_method='zscore',
            add_technical_indicators=True
        )
        
        processed = preprocessor.fit_transform(data)
        
        # Verify normalization
        assert processed.shape[0] == data.shape[0]
        assert processed.shape[1] > data.shape[1]  # Added indicators
        assert abs(processed.mean().mean()) < 0.1  # Roughly centered
        assert 0.5 < processed.std().mean() < 2.0  # Reasonable scale
        
        # Test sequence creation
        sequences, targets = create_sequences(processed.values, seq_len=20, horizon=5)
        assert sequences.shape[1] == 20
        assert targets.shape[0] == sequences.shape[0]
    
    def test_data_augmentation(self):
        """Test data augmentation techniques."""
        from hftraining.data_utils import DataAugmenter
        
        # Create sample data
        data = torch.randn(100, 30, 10)  # 100 samples, 30 timesteps, 10 features
        
        augmenter = DataAugmenter(
            noise_level=0.01,
            dropout_prob=0.1,
            mixup_alpha=0.2
        )
        
        augmented = augmenter.augment(data)
        
        # Verify augmentation changed data but preserved structure
        assert augmented.shape == data.shape
        assert not torch.allclose(augmented, data)
        assert torch.isfinite(augmented).all()
        
        # Verify augmentation is reasonable
        diff = (augmented - data).abs().mean()
        assert diff < 0.5  # Not too different


class TestModernOptimizers:
    """Test modern optimization algorithms."""
    
    def test_modern_optimizers_convergence(self):
        """Test that modern optimizers converge on simple problems."""
        from hftraining.modern_optimizers import (
            AdamW, 
            Lion, 
            Shampoo,
            create_optimizer
        )
        
        # Simple quadratic optimization problem
        x = torch.randn(10, requires_grad=True)
        target = torch.randn(10)
        
        optimizers_to_test = [
            ('adamw', {'lr': 0.01, 'weight_decay': 0.01}),
            ('lion', {'lr': 0.001, 'weight_decay': 0.01}),
            ('shampoo', {'lr': 0.01, 'eps': 1e-10})
        ]
        
        for opt_name, opt_params in optimizers_to_test:
            # Reset parameter
            x.data = torch.randn(10)
            
            optimizer = create_optimizer(opt_name, [x], **opt_params)
            
            losses = []
            for _ in range(100):
                optimizer.zero_grad()
                loss = ((x - target) ** 2).sum()
                loss.backward()
                optimizer.step()
                losses.append(loss.item())
            
            # Verify convergence
            assert losses[-1] < losses[0] * 0.1, f"{opt_name} should converge"
            assert losses[-1] < 0.1, f"{opt_name} should reach low loss"
    
    def test_optimizer_memory_efficiency(self):
        """Test memory efficiency of optimizers."""
        from hftraining.modern_optimizers import create_optimizer
        
        # Create a moderately sized model
        model = nn.Sequential(
            nn.Linear(100, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 10)
        )
        
        if torch.cuda.is_available():
            model = model.cuda()
            
        optimizer = create_optimizer('memory_efficient_adamw', model.parameters(), lr=1e-3)
        
        # Run a few steps
        for _ in range(10):
            data = torch.randn(32, 100)
            if torch.cuda.is_available():
                data = data.cuda()
            
            optimizer.zero_grad()
            output = model(data)
            loss = output.sum()
            loss.backward()
            optimizer.step()
        
        # Check optimizer state size
        state_size = sum(
            sum(t.numel() * t.element_size() for t in state.values() if isinstance(t, torch.Tensor))
            for state in optimizer.state.values()
        )
        param_size = sum(p.numel() * p.element_size() for p in model.parameters())
        
        # State should not be too much larger than params (< 3x for efficient optimizer)
        assert state_size < param_size * 3


class TestImprovedSchedulers:
    """Test learning rate schedulers."""
    
    def test_scheduler_warmup_behavior(self):
        """Test warmup behavior of schedulers."""
        from hftraining.improved_schedulers import (
            CosineAnnealingWarmup,
            OneCycleLR,
            create_scheduler
        )
        
        model = nn.Linear(10, 1)
        optimizer = torch.optim.SGD(model.parameters(), lr=1.0)
        
        scheduler = create_scheduler(
            'cosine_warmup',
            optimizer,
            warmup_steps=10,
            total_steps=100,
            min_lr=0.01
        )
        
        lrs = []
        for step in range(100):
            lrs.append(optimizer.param_groups[0]['lr'])
            scheduler.step()
        
        # Verify warmup
        assert lrs[0] < lrs[9], "LR should increase during warmup"
        assert lrs[9] > lrs[99], "LR should decrease after warmup"
        assert lrs[99] >= 0.01, "LR should not go below min_lr"
    
    def test_adaptive_scheduler(self):
        """Test adaptive scheduling based on metrics."""
        from hftraining.improved_schedulers import AdaptiveScheduler
        
        model = nn.Linear(10, 1)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
        
        scheduler = AdaptiveScheduler(
            optimizer,
            mode='min',
            factor=0.5,
            patience=5,
            threshold=0.01
        )
        
        initial_lr = optimizer.param_groups[0]['lr']
        
        # Simulate plateau in loss
        for epoch in range(20):
            loss = 1.0 + np.random.randn() * 0.001  # Stagnant loss
            scheduler.step(loss)
        
        final_lr = optimizer.param_groups[0]['lr']
        
        # LR should have decreased due to plateau
        assert final_lr < initial_lr * 0.3


class TestProductionEngine:
    """Test production training setup."""
    
    def test_production_training_pipeline(self):
        """Test full production training pipeline."""
        from hftraining.train_production import ProductionTrainer, ProductionConfig
        
        with tempfile.TemporaryDirectory() as tmpdir:
            config = ProductionConfig(
                data_path=tmpdir,
                model_name='transformer_small',
                batch_size=16,
                learning_rate=1e-3,
                num_epochs=2,
                use_wandb=False,  # Disable for testing
                checkpoint_dir=tmpdir,
                enable_profiling=False
            )
            
            # Create sample data files
            for i in range(3):
                data = pd.DataFrame({
                    'timestamp': pd.date_range('2023-01-01', periods=100, freq='1h'),
                    'price': 100 + np.random.randn(100).cumsum(),
                    'volume': np.random.lognormal(10, 1, 100)
                })
                data.to_csv(Path(tmpdir) / f'data_{i}.csv', index=False)
            
            trainer = ProductionTrainer(config)
            
            # Run training
            metrics = trainer.train()
            
            # Verify training completed
            assert 'final_loss' in metrics
            assert metrics['final_loss'] > 0
            assert 'best_epoch' in metrics
            
            # Verify model was saved
            model_path = Path(tmpdir) / 'best_model.pt'
            assert model_path.exists()
    
    def test_distributed_training_setup(self):
        """Test distributed training configuration."""
        from hftraining.train_production import setup_distributed, cleanup_distributed
        
        if torch.cuda.device_count() < 2:
            pytest.skip("Multi-GPU required for distributed training test")
        
        # This would normally be run in separate processes
        # Here we just test the setup doesn't crash
        try:
            rank = 0
            world_size = 2
            setup_distributed(rank, world_size)
            
            # Verify distributed is initialized
            assert torch.distributed.is_initialized()
            assert torch.distributed.get_world_size() == world_size
            
        finally:
            cleanup_distributed()


class TestAutoTune:
    """Test automatic hyperparameter tuning."""
    
    def test_auto_tune_finds_good_params(self):
        """Test that auto-tuning finds reasonable parameters."""
        from hftraining.auto_tune import AutoTuner, TuneConfig
        
        with tempfile.TemporaryDirectory() as tmpdir:
            config = TuneConfig(
                search_space={
                    'learning_rate': (1e-4, 1e-2),
                    'batch_size': [16, 32, 64],
                    'hidden_size': [64, 128, 256],
                    'dropout': (0.0, 0.3)
                },
                metric='val_loss',
                mode='min',
                n_trials=10,
                timeout=60,  # 1 minute timeout
                output_dir=tmpdir
            )
            
            # Simple objective function
            def train_fn(params):
                # Simulate training with these params
                lr = params['learning_rate']
                bs = params['batch_size']
                hs = params['hidden_size']
                dropout = params['dropout']
                
                # Better performance with certain combinations
                loss = (
                    abs(lr - 0.001) * 10 +
                    abs(bs - 32) / 100 +
                    abs(hs - 128) / 1000 +
                    abs(dropout - 0.1) * 5
                )
                return {'val_loss': loss + np.random.randn() * 0.01}
            
            tuner = AutoTuner(config, train_fn)
            best_params, best_metric = tuner.tune()
            
            # Verify found reasonable params
            assert 0.0005 < best_params['learning_rate'] < 0.002
            assert best_params['batch_size'] in [16, 32, 64]
            assert best_metric < 1.0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])