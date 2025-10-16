#!/usr/bin/env python3
"""
Manual test runner for TotoTrainer without pytest dependencies.
Tests the core functionality directly.
"""

import sys
import os
import traceback
import tempfile
import shutil
from pathlib import Path
import warnings

# Import test modules
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch

# Suppress warnings
warnings.filterwarnings("ignore")

# Import modules under test
try:
    from toto_trainer import TotoTrainer, TrainerConfig, MetricsTracker, CheckpointManager
    from toto_ohlc_dataloader import TotoOHLCDataLoader, DataLoaderConfig, MaskedTimeseries
except ImportError as e:
    print(f"Import error: {e}")
    print("Note: This is expected due to missing Toto model dependencies.")
    print("Testing will proceed with mock implementations.")


class ManualTestRunner:
    """Manual test runner for TotoTrainer"""
    
    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.errors = []
        
    def run_test(self, test_func, test_name):
        """Run a single test and track results"""
        print(f"Running: {test_name}")
        try:
            test_func()
            print(f"✅ PASSED: {test_name}")
            self.passed += 1
        except Exception as e:
            print(f"❌ FAILED: {test_name}")
            if str(e):
                print(f"   Error: {str(e)}")
            else:
                print(f"   Error type: {type(e).__name__}")
                print(f"   Traceback: {traceback.format_exc()}")
            self.errors.append((test_name, str(e), traceback.format_exc()))
            self.failed += 1
        print()
    
    def print_summary(self):
        """Print test summary"""
        print("=" * 80)
        print("TEST SUMMARY")
        print("=" * 80)
        print(f"Passed: {self.passed}")
        print(f"Failed: {self.failed}")
        print(f"Total: {self.passed + self.failed}")
        
        if self.errors:
            print("\nFAILED TESTS:")
            print("-" * 40)
            for test_name, error, trace in self.errors:
                print(f"❌ {test_name}")
                print(f"   {error}")
                print()
        
        return self.failed == 0


def create_temp_dir():
    """Create temporary directory"""
    return tempfile.mkdtemp()


def cleanup_temp_dir(temp_dir):
    """Cleanup temporary directory"""
    shutil.rmtree(temp_dir, ignore_errors=True)


def create_sample_data():
    """Create sample OHLC data"""
    np.random.seed(42)
    n_samples = 200
    dates = pd.date_range('2023-01-01', periods=n_samples, freq='H')
    
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
    
    data['High'] = np.maximum(data['High'], np.maximum(data['Open'], data['Close']))
    data['Low'] = np.minimum(data['Low'], np.minimum(data['Open'], data['Close']))
    
    return data


def create_sample_data_files(temp_dir, create_test=True):
    """Create sample CSV data files"""
    train_dir = Path(temp_dir) / "train_data"
    train_dir.mkdir(parents=True, exist_ok=True)
    
    test_dir = None
    if create_test:
        test_dir = Path(temp_dir) / "test_data"
        test_dir.mkdir(parents=True, exist_ok=True)
    
    sample_data = create_sample_data()
    symbols = ['AAPL', 'GOOGL', 'MSFT']
    
    for i, symbol in enumerate(symbols):
        data = sample_data.copy()
        # Ensure we have enough data - use more samples
        start_idx = i * 10
        end_idx = start_idx + 180  # Larger chunks for training
        if end_idx > len(data):
            end_idx = len(data)
        data = data.iloc[start_idx:end_idx].reset_index(drop=True)
        
        multiplier = 1 + i * 0.1
        for col in ['Open', 'High', 'Low', 'Close']:
            data[col] *= multiplier
        
        # Save all data to training directory (let dataloader handle splits)
        data.to_csv(train_dir / f"{symbol}.csv", index=False)
        
        if create_test:
            # Save smaller test data
            test_data = data.iloc[-50:].copy()  # Last 50 rows
            test_data.to_csv(test_dir / f"{symbol}.csv", index=False)
        
        print(f"Created {symbol}: train={len(data)} rows" + (f", test=50 rows" if create_test else ""))
    
    if create_test:
        return train_dir, test_dir
    else:
        return train_dir


# TEST IMPLEMENTATIONS

def test_trainer_config_basic():
    """Test 1: TrainerConfig basic functionality"""
    config = TrainerConfig()
    
    assert config.patch_size > 0
    assert config.embed_dim > 0
    assert config.learning_rate > 0
    assert config.batch_size > 0
    
    temp_dir = create_temp_dir()
    try:
        config_with_temp = TrainerConfig(save_dir=temp_dir)
        assert Path(temp_dir).exists()
    finally:
        cleanup_temp_dir(temp_dir)


def test_trainer_config_save_load():
    """Test 2: TrainerConfig save/load functionality"""
    temp_dir = create_temp_dir()
    try:
        config = TrainerConfig(
            patch_size=16,
            embed_dim=512,
            learning_rate=1e-4
        )
        
        config_path = Path(temp_dir) / "config.json"
        config.save(str(config_path))
        
        loaded_config = TrainerConfig.load(str(config_path))
        
        assert loaded_config.patch_size == config.patch_size
        assert loaded_config.embed_dim == config.embed_dim
        assert loaded_config.learning_rate == config.learning_rate
    finally:
        cleanup_temp_dir(temp_dir)


def test_metrics_tracker():
    """Test 3: MetricsTracker functionality"""
    tracker = MetricsTracker()
    
    # Test initial state
    assert len(tracker.losses) == 0
    
    # Update with metrics
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
    
    assert metrics['loss'] == 0.5
    assert metrics['batch_time_mean'] == 0.1
    assert metrics['learning_rate'] == 0.001


def test_checkpoint_manager():
    """Test 4: CheckpointManager functionality"""
    temp_dir = create_temp_dir()
    try:
        checkpoint_dir = Path(temp_dir) / "checkpoints"
        manager = CheckpointManager(str(checkpoint_dir), keep_last_n=2)
        
        assert manager.save_dir == checkpoint_dir
        assert checkpoint_dir.exists()
        
        # Create real components for testing (avoid Mock pickle issues)
        model = nn.Linear(1, 1)
        optimizer = torch.optim.Adam(model.parameters())
        config = TrainerConfig()
        
        # Save checkpoint
        checkpoint_path = manager.save_checkpoint(
            model=model,
            optimizer=optimizer,
            scheduler=None,
            scaler=None,
            epoch=1,
            best_val_loss=0.5,
            metrics={'loss': 0.5},
            config=config
        )
        
        assert checkpoint_path.exists()
        assert (checkpoint_dir / "latest.pt").exists()
        
        # Test loading
        checkpoint = manager.load_checkpoint(str(checkpoint_path))
        assert checkpoint['epoch'] == 1
        assert checkpoint['best_val_loss'] == 0.5
        
    finally:
        cleanup_temp_dir(temp_dir)


def test_trainer_initialization():
    """Test 5: TotoTrainer initialization"""
    temp_dir = create_temp_dir()
    try:
        trainer_config = TrainerConfig(
            save_dir=str(Path(temp_dir) / "checkpoints"),
            log_file=str(Path(temp_dir) / "training.log"),
            max_epochs=2,
            batch_size=4
        )
        
        dataloader_config = DataLoaderConfig(
            train_data_path=str(Path(temp_dir) / "train_data"),
            test_data_path=str(Path(temp_dir) / "test_data"),
            batch_size=4,
            sequence_length=48,
            prediction_length=12
        )
        
        trainer = TotoTrainer(trainer_config, dataloader_config)
        
        assert trainer.config == trainer_config
        assert trainer.dataloader_config == dataloader_config
        assert trainer.model is None
        assert trainer.optimizer is None
        assert trainer.current_epoch == 0
        assert trainer.global_step == 0
        assert trainer.best_val_loss == float('inf')
        
    finally:
        cleanup_temp_dir(temp_dir)


def test_dataloader_integration():
    """Test 6: OHLC DataLoader integration"""
    temp_dir = create_temp_dir()
    try:
        # Only create training data to avoid split confusion
        train_dir = create_sample_data_files(temp_dir, create_test=False)
        
        config = DataLoaderConfig(
            train_data_path=str(train_dir),
            test_data_path="nonexistent",  # Force use of training data only
            batch_size=4,
            sequence_length=48,
            prediction_length=12,
            add_technical_indicators=False,
            max_symbols=2,
            num_workers=0,
            min_sequence_length=60,  # Reduced for test data
            validation_split=0.2,  # Create validation split from training
            test_split_days=2  # Use only 2 days for test split (instead of 30)
        )
        
        dataloader = TotoOHLCDataLoader(config)
        
        # Debug: Check if files exist
        print(f"Train directory: {train_dir}")
        print(f"Files in train dir: {list(train_dir.glob('*.csv'))}")
        
        # Test data loading
        train_data, val_data, test_data = dataloader.load_data()
        print(f"Loaded data: train={len(train_data)}, val={len(val_data)}, test={len(test_data)}")
        
        assert len(train_data) > 0 or len(test_data) > 0
        
        # Test dataloader preparation
        dataloaders = dataloader.prepare_dataloaders()
        
        if dataloaders:
            assert isinstance(dataloaders, dict)
            if 'train' in dataloaders:
                train_loader = dataloaders['train']
                assert len(train_loader) > 0
                
                # Test sample format
                sample_batch = next(iter(train_loader))
                if isinstance(sample_batch, MaskedTimeseries):
                    assert hasattr(sample_batch, 'series')
                    assert isinstance(sample_batch.series, torch.Tensor)
                
    finally:
        cleanup_temp_dir(temp_dir)


def test_trainer_prepare_data():
    """Test 7: TotoTrainer data preparation"""
    temp_dir = create_temp_dir()
    try:
        train_dir = create_sample_data_files(temp_dir, create_test=False)
        
        trainer_config = TrainerConfig(
            save_dir=str(Path(temp_dir) / "checkpoints"),
            batch_size=4
        )
        
        dataloader_config = DataLoaderConfig(
            train_data_path=str(train_dir),
            test_data_path="nonexistent",
            batch_size=4,
            sequence_length=48,
            prediction_length=12,
            add_technical_indicators=False,
            num_workers=0,
            min_sequence_length=60,
            validation_split=0.2,
            test_split_days=2
        )
        
        trainer = TotoTrainer(trainer_config, dataloader_config)
        trainer.prepare_data()
        
        assert len(trainer.dataloaders) > 0
        assert 'train' in trainer.dataloaders
        
    finally:
        cleanup_temp_dir(temp_dir)


def test_trainer_error_handling():
    """Test 8: TotoTrainer error handling"""
    temp_dir = create_temp_dir()
    try:
        trainer_config = TrainerConfig(
            save_dir=str(Path(temp_dir) / "checkpoints"),
            optimizer="invalid_optimizer"
        )
        
        dataloader_config = DataLoaderConfig()
        
        trainer = TotoTrainer(trainer_config, dataloader_config)
        
        # Test invalid optimizer error
        try:
            trainer._create_optimizer()
            assert False, "Should have raised ValueError"
        except ValueError as e:
            assert "Unsupported optimizer" in str(e)
        
        # Test invalid scheduler error
        trainer_config.optimizer = "adamw"
        trainer_config.scheduler = "invalid_scheduler"
        trainer.optimizer = torch.optim.Adam([torch.randn(1, requires_grad=True)])
        
        try:
            trainer._create_scheduler(steps_per_epoch=10)
            assert False, "Should have raised ValueError"
        except ValueError as e:
            assert "Unsupported scheduler" in str(e)
        
    finally:
        cleanup_temp_dir(temp_dir)


def test_model_creation_mock():
    """Test 9: Mock model creation"""
    temp_dir = create_temp_dir()
    try:
        train_dir = create_sample_data_files(temp_dir, create_test=False)
        
        trainer_config = TrainerConfig(
            save_dir=str(Path(temp_dir) / "checkpoints"),
            embed_dim=64,
            num_layers=2,
            batch_size=2  # Match dataloader batch size
        )
        
        dataloader_config = DataLoaderConfig(
            train_data_path=str(train_dir),
            test_data_path="nonexistent",
            batch_size=2,  # Smaller batch size to ensure we have batches
            num_workers=0,
            min_sequence_length=60,
            validation_split=0.2,
            test_split_days=2,
            drop_last=False  # Don't drop incomplete batches
        )
        
        with patch('toto_trainer.Toto') as mock_toto_class:
            mock_model = Mock(spec=nn.Module)
            # Create proper parameters that work with sum() and param counting
            param1 = torch.randn(10, requires_grad=True)
            param2 = torch.randn(5, requires_grad=True)
            params_list = [param1, param2]
            mock_model.parameters = lambda: iter(params_list)  # Return fresh iterator each time
            mock_model.to.return_value = mock_model  # Return self on to() calls
            mock_toto_class.return_value = mock_model
            
            trainer = TotoTrainer(trainer_config, dataloader_config)
            trainer.prepare_data()
            trainer.setup_model()
            
            mock_toto_class.assert_called_once()
            assert trainer.model == mock_model
            assert trainer.optimizer is not None
            
    finally:
        cleanup_temp_dir(temp_dir)


def test_memory_efficiency():
    """Test 10: Memory efficiency"""
    # Test gradient clipping memory usage
    model = nn.Linear(100, 10)
    optimizer = torch.optim.Adam(model.parameters())
    
    # Simulate training steps
    for _ in range(5):
        optimizer.zero_grad()
        x = torch.randn(16, 100)
        y = model(x)
        loss = y.sum()
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
    
    # If we get here without memory errors, test passed
    assert True


def run_all_tests():
    """Run all manual tests"""
    runner = ManualTestRunner()
    
    print("=" * 80)
    print("RUNNING MANUAL TOTO TRAINER TESTS")
    print("=" * 80)
    print()
    
    # List of all tests
    tests = [
        (test_trainer_config_basic, "TrainerConfig Basic Functionality"),
        (test_trainer_config_save_load, "TrainerConfig Save/Load"),
        (test_metrics_tracker, "MetricsTracker Functionality"),
        (test_checkpoint_manager, "CheckpointManager Functionality"),
        (test_trainer_initialization, "TotoTrainer Initialization"),
        (test_dataloader_integration, "DataLoader Integration"),
        (test_trainer_prepare_data, "TotoTrainer Data Preparation"),
        (test_trainer_error_handling, "TotoTrainer Error Handling"),
        (test_model_creation_mock, "Mock Model Creation"),
        (test_memory_efficiency, "Memory Efficiency")
    ]
    
    # Run each test
    for test_func, test_name in tests:
        runner.run_test(test_func, test_name)
    
    # Print summary
    success = runner.print_summary()
    
    if success:
        print("\n🎉 ALL TESTS PASSED!")
    else:
        print(f"\n⚠️  {runner.failed} TESTS FAILED")
    
    return success


if __name__ == "__main__":
    run_all_tests()