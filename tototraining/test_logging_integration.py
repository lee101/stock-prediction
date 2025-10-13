#!/usr/bin/env python3
"""
Integration Test for Toto Training Logging System
Tests all logging components to ensure they work together properly.
"""

import os
import sys
import time
import json
import tempfile
import shutil
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List
import numpy as np

# Test individual components
def test_training_logger():
    """Test the training logger"""
    print("🧪 Testing Training Logger...")
    
    try:
        from training_logger import create_training_logger
        
        with tempfile.TemporaryDirectory() as temp_dir:
            with create_training_logger("test_logger", temp_dir) as logger:
                # Test basic logging
                logger.log_training_start({"learning_rate": 0.001, "batch_size": 32})
                
                for epoch in range(3):
                    for batch in range(5):
                        logger.log_training_metrics(
                            epoch=epoch,
                            batch=batch,
                            train_loss=1.0 - epoch * 0.1 - batch * 0.02,
                            val_loss=1.1 - epoch * 0.1 - batch * 0.015,
                            learning_rate=0.001,
                            gradient_norm=0.5 + np.random.normal(0, 0.1)
                        )
                    
                    # Test epoch summary
                    logger.log_epoch_summary(
                        epoch=epoch,
                        train_loss=1.0 - epoch * 0.1,
                        val_loss=1.1 - epoch * 0.1,
                        epoch_time=30.5 + np.random.normal(0, 5)
                    )
                
                # Test error logging
                try:
                    raise ValueError("Test error")
                except ValueError as e:
                    logger.log_error(e, "test context")
                
                # Test best model logging
                logger.log_best_model("test_model.pth", "val_loss", 0.75)
                
                # Test early stopping
                logger.log_early_stopping(5, 10, "val_loss", 0.75)
                
                logger.log_training_complete(3, 120.0, {"best_val_loss": 0.75})
                
        print("✅ Training Logger: PASSED")
        return True
        
    except Exception as e:
        print(f"❌ Training Logger: FAILED - {e}")
        return False


def test_tensorboard_monitor():
    """Test TensorBoard monitor"""
    print("🧪 Testing TensorBoard Monitor...")
    
    try:
        from tensorboard_monitor import create_tensorboard_monitor
        
        with tempfile.TemporaryDirectory() as temp_dir:
            with create_tensorboard_monitor("test_tb", temp_dir) as tb_monitor:
                # Test training metrics
                for epoch in range(3):
                    for batch in range(10):
                        tb_monitor.log_training_metrics(
                            epoch=epoch,
                            batch=batch,
                            train_loss=1.0 - epoch * 0.1 - batch * 0.01,
                            learning_rate=0.001,
                            accuracy=0.8 + epoch * 0.05
                        )
                    
                    # Test validation metrics
                    tb_monitor.log_validation_metrics(
                        epoch=epoch,
                        val_loss=1.1 - epoch * 0.1,
                        accuracy=0.75 + epoch * 0.05
                    )
                    
                    # Test system metrics
                    tb_monitor.log_system_metrics(
                        cpu_percent=50.0 + np.random.normal(0, 10),
                        memory_percent=60.0 + np.random.normal(0, 5),
                        gpu_utilization=80.0 + np.random.normal(0, 10),
                        gpu_temperature=65.0 + np.random.normal(0, 5)
                    )
                
                # Test loss curves
                train_losses = [1.0 - i * 0.1 for i in range(5)]
                val_losses = [1.1 - i * 0.1 for i in range(5)]
                tb_monitor.log_loss_curves(train_losses, val_losses)
                
                # Test hyperparameters
                tb_monitor.log_hyperparameters(
                    {"learning_rate": 0.001, "batch_size": 32},
                    {"final_loss": 0.5}
                )
        
        print("✅ TensorBoard Monitor: PASSED")
        return True
        
    except Exception as e:
        print(f"❌ TensorBoard Monitor: FAILED - {e}")
        return False


def test_mlflow_tracker():
    """Test MLflow tracker"""
    print("🧪 Testing MLflow Tracker...")
    
    try:
        from mlflow_tracker import create_mlflow_tracker
        
        with tempfile.TemporaryDirectory() as temp_dir:
            with create_mlflow_tracker("test_mlflow", temp_dir) as tracker:
                # Start run
                run_id = tracker.start_run("test_run")
                
                # Test config logging
                config = {
                    "learning_rate": 0.001,
                    "batch_size": 32,
                    "epochs": 10
                }
                tracker.log_config(config)
                
                # Test training metrics
                for epoch in range(3):
                    for batch in range(10):
                        tracker.log_training_metrics(
                            epoch=epoch,
                            batch=batch,
                            train_loss=1.0 - epoch * 0.1 - batch * 0.01,
                            val_loss=1.1 - epoch * 0.1 - batch * 0.01,
                            learning_rate=0.001
                        )
                    
                    # Test epoch summary
                    tracker.log_epoch_summary(
                        epoch=epoch,
                        train_loss=1.0 - epoch * 0.1,
                        val_loss=1.1 - epoch * 0.1,
                        epoch_time=30.0
                    )
                
                # Test predictions logging
                predictions = np.random.normal(0, 1, 100)
                actuals = np.random.normal(0, 1, 100)
                tracker.log_predictions(predictions, actuals, step=10)
                
                # Test system metrics
                tracker.log_system_metrics(
                    cpu_percent=50.0,
                    memory_percent=60.0,
                    memory_used_gb=8.0,
                    gpu_utilization=80.0
                )
                
                # Test tags
                tracker.set_tags({"test": "true", "version": "1.0"})
        
        print("✅ MLflow Tracker: PASSED")
        return True
        
    except Exception as e:
        print(f"❌ MLflow Tracker: FAILED - {e}")
        return False


def test_checkpoint_manager():
    """Test checkpoint manager"""
    print("🧪 Testing Checkpoint Manager...")
    
    try:
        import torch
        from checkpoint_manager import create_checkpoint_manager
        
        # Create a simple model
        model = torch.nn.Linear(10, 1)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = create_checkpoint_manager(temp_dir, "val_loss", "min")
            
            # Test checkpointing
            for epoch in range(5):
                train_loss = 1.0 - epoch * 0.1
                val_loss = train_loss + 0.05 + np.random.normal(0, 0.02)
                
                metrics = {
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                    'accuracy': 0.8 + epoch * 0.05
                }
                
                checkpoint_info = manager.save_checkpoint(
                    model, optimizer, epoch, epoch * 100, metrics,
                    tags={'test': 'true'}
                )
                
                if checkpoint_info:
                    print(f"    Saved checkpoint for epoch {epoch}: {Path(checkpoint_info.path).name}")
            
            # Test loading best checkpoint
            best_checkpoint = manager.load_best_checkpoint(model, optimizer)
            if best_checkpoint:
                print(f"    Loaded best checkpoint from epoch {best_checkpoint['epoch']}")
            
            # Test summary
            summary = manager.get_checkpoint_summary()
            print(f"    Summary: {summary['total_checkpoints']} regular, {summary['best_checkpoints']} best")
        
        print("✅ Checkpoint Manager: PASSED")
        return True
        
    except Exception as e:
        print(f"❌ Checkpoint Manager: FAILED - {e}")
        return False


def test_training_callbacks():
    """Test training callbacks"""
    print("🧪 Testing Training Callbacks...")
    
    try:
        import torch
        from training_callbacks import (
            CallbackManager, CallbackState, EarlyStopping, 
            ReduceLROnPlateau, MetricTracker
        )
        
        # Create model and optimizer
        model = torch.nn.Linear(10, 1)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        
        # Create callbacks
        callbacks = [
            EarlyStopping(patience=3, verbose=True),
            ReduceLROnPlateau(optimizer, patience=2, verbose=True),
            MetricTracker(['train_loss', 'val_loss'])
        ]
        
        manager = CallbackManager(callbacks)
        manager.on_training_start()
        
        # Simulate training
        stopped = False
        for epoch in range(10):
            train_loss = 1.0 - epoch * 0.05 if epoch < 5 else 0.75 + np.random.normal(0, 0.02)
            val_loss = train_loss + 0.1 + (0.02 if epoch > 5 else 0)  # Plateau after epoch 5
            
            state = CallbackState(
                epoch=epoch,
                step=epoch * 100,
                train_loss=train_loss,
                val_loss=val_loss,
                model_state_dict=model.state_dict(),
                optimizer_state_dict=optimizer.state_dict()
            )
            
            should_stop = manager.on_epoch_end(state)
            if should_stop:
                print(f"    Early stopping triggered at epoch {epoch}")
                stopped = True
                break
        
        manager.on_training_end()
        
        if stopped:
            print("    Early stopping worked correctly")
        
        print("✅ Training Callbacks: PASSED")
        return True
        
    except Exception as e:
        print(f"❌ Training Callbacks: FAILED - {e}")
        return False


def test_dashboard_config():
    """Test dashboard configuration"""
    print("🧪 Testing Dashboard Config...")
    
    try:
        from dashboard_config import create_dashboard_generator
        
        with tempfile.TemporaryDirectory() as temp_dir:
            generator = create_dashboard_generator("test_dashboard")
            generator.config_dir = Path(temp_dir)
            
            # Create dashboard
            dashboard_config = generator.create_training_dashboard()
            
            # Test saving configurations
            generator.save_configurations(dashboard_config)
            
            # Check files were created
            expected_files = [
                "test_dashboard_dashboard_config.json",
                "test_dashboard_grafana_dashboard.json",
                "prometheus.yml",
                "toto_training_alerts.yml",
                "docker-compose.yml"
            ]
            
            created_files = []
            for file in expected_files:
                file_path = Path(temp_dir) / file
                if file_path.exists():
                    created_files.append(file)
            
            print(f"    Created {len(created_files)}/{len(expected_files)} config files")
            
            # Test HTML dashboard
            generator.save_html_dashboard(dashboard_config)
            html_file = Path(temp_dir) / "test_dashboard_dashboard.html"
            if html_file.exists():
                print(f"    HTML dashboard created: {html_file.name}")
        
        print("✅ Dashboard Config: PASSED")
        return True
        
    except Exception as e:
        print(f"❌ Dashboard Config: FAILED - {e}")
        return False


def test_integration():
    """Test integration of all components"""
    print("🧪 Testing Full Integration...")
    
    try:
        # This is a simplified integration test
        # In a real scenario, you would run the enhanced trainer
        
        from training_logger import create_training_logger
        from checkpoint_manager import create_checkpoint_manager
        from dashboard_config import create_dashboard_generator
        
        with tempfile.TemporaryDirectory() as temp_dir:
            experiment_name = "integration_test"
            
            # Initialize components
            logger = create_training_logger(experiment_name, temp_dir)
            checkpoint_manager = create_checkpoint_manager(temp_dir)
            dashboard_generator = create_dashboard_generator(experiment_name)
            dashboard_generator.config_dir = Path(temp_dir)
            
            # Simulate training flow
            config = {"learning_rate": 0.001, "batch_size": 32, "epochs": 5}
            logger.log_training_start(config)
            
            # Create dashboard
            dashboard_config = dashboard_generator.create_training_dashboard()
            dashboard_generator.save_configurations(dashboard_config)
            
            # Simulate training epochs
            for epoch in range(3):
                train_loss = 1.0 - epoch * 0.2
                val_loss = train_loss + 0.05
                
                # Log metrics
                logger.log_training_metrics(
                    epoch=epoch,
                    batch=0,
                    train_loss=train_loss,
                    val_loss=val_loss,
                    learning_rate=0.001
                )
                
                # Log epoch summary
                logger.log_epoch_summary(epoch, train_loss, val_loss, epoch_time=30.0)
            
            # Complete training
            logger.log_training_complete(3, 90.0, {"best_val_loss": 0.6})
            
            # Check if logs were created
            log_files = list(Path(temp_dir).glob("**/*.log"))
            json_files = list(Path(temp_dir).glob("**/*.json"))
            
            print(f"    Created {len(log_files)} log files and {len(json_files)} JSON files")
        
        print("✅ Full Integration: PASSED")
        return True
        
    except Exception as e:
        print(f"❌ Full Integration: FAILED - {e}")
        return False


def run_all_tests():
    """Run all integration tests"""
    print("🚀 Running Toto Training Logging System Tests")
    print("=" * 60)
    
    tests = [
        ("Training Logger", test_training_logger),
        ("TensorBoard Monitor", test_tensorboard_monitor),
        ("MLflow Tracker", test_mlflow_tracker),
        ("Checkpoint Manager", test_checkpoint_manager),
        ("Training Callbacks", test_training_callbacks),
        ("Dashboard Config", test_dashboard_config),
        ("Full Integration", test_integration)
    ]
    
    passed = 0
    failed = 0
    
    for test_name, test_func in tests:
        print(f"\n📋 {test_name}")
        print("-" * 40)
        
        try:
            if test_func():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"❌ {test_name}: CRASHED - {e}")
            failed += 1
    
    print("\n" + "=" * 60)
    print("📊 TEST SUMMARY")
    print("=" * 60)
    print(f"✅ Passed: {passed}")
    print(f"❌ Failed: {failed}")
    print(f"📈 Success Rate: {passed/(passed+failed)*100:.1f}%")
    
    if failed == 0:
        print("\n🎉 All tests passed! The logging system is ready for production.")
    else:
        print(f"\n⚠️  {failed} test(s) failed. Please check the errors above.")
    
    return failed == 0


def test_dependencies():
    """Test if required dependencies are available"""
    print("🔍 Checking Dependencies...")
    
    dependencies = {
        "torch": "PyTorch",
        "pandas": "Pandas",
        "numpy": "NumPy",
        "psutil": "psutil (system monitoring)",
        "matplotlib": "Matplotlib (plotting) - OPTIONAL",
        "tensorboard": "TensorBoard - OPTIONAL",
        "mlflow": "MLflow - OPTIONAL",
        "GPUtil": "GPUtil (GPU monitoring) - OPTIONAL"
    }
    
    available = []
    missing = []
    
    for module, description in dependencies.items():
        try:
            __import__(module)
            available.append((module, description))
        except ImportError:
            missing.append((module, description))
    
    print(f"✅ Available ({len(available)}):")
    for module, desc in available:
        print(f"    - {desc}")
    
    if missing:
        print(f"⚠️  Missing ({len(missing)}):")
        for module, desc in missing:
            print(f"    - {desc}")
            if "OPTIONAL" not in desc:
                print(f"      Install with: uv pip install {module}")
    
    return len(missing) == 0 or all("OPTIONAL" in desc for _, desc in missing)


if __name__ == "__main__":
    print("🧪 Toto Training Logging System - Integration Tests")
    print("=" * 60)
    
    # Check dependencies first
    if not test_dependencies():
        print("\n❌ Missing required dependencies. Please install them first.")
        sys.exit(1)
    
    # Run all tests
    success = run_all_tests()
    
    if success:
        print("\n🎯 Next Steps:")
        print("  1. Run 'python enhanced_trainer.py' to test with real training")
        print("  2. Start monitoring with: tensorboard --logdir tensorboard_logs")
        print("  3. View MLflow with: mlflow ui --backend-store-uri mlruns")
        print("  4. Setup monitoring stack with docker-compose in dashboard_configs/")
        
        sys.exit(0)
    else:
        sys.exit(1)