#!/usr/bin/env python3
"""
Enhanced Toto Trainer with Comprehensive Logging and Monitoring
Integrates all logging components: structured logging, TensorBoard, MLflow, checkpoints, and callbacks.
"""

import os
import sys
import time
import torch
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any
import logging

# Import our logging components
from training_logger import TotoTrainingLogger
from tensorboard_monitor import TensorBoardMonitor
from mlflow_tracker import MLflowTracker
from checkpoint_manager import CheckpointManager
from training_callbacks import (
    CallbackManager, CallbackState, EarlyStopping, 
    ReduceLROnPlateau, MetricTracker
)
from dashboard_config import DashboardGenerator

# Import the original trainer components
from toto_ohlc_trainer import TotoOHLCConfig, OHLCDataset, TotoOHLCTrainer


class EnhancedTotoTrainer(TotoOHLCTrainer):
    """
    Enhanced version of the Toto trainer with comprehensive logging and monitoring.
    Integrates all logging systems for production-ready training.
    """
    
    def __init__(
        self, 
        config: TotoOHLCConfig,
        experiment_name: str,
        enable_tensorboard: bool = True,
        enable_mlflow: bool = True,
        enable_system_monitoring: bool = True,
        log_dir: str = "logs",
        checkpoint_dir: str = "checkpoints"
    ):
        # Initialize base trainer
        super().__init__(config)
        
        self.experiment_name = experiment_name
        self.enable_tensorboard = enable_tensorboard
        self.enable_mlflow = enable_mlflow
        self.enable_system_monitoring = enable_system_monitoring
        
        # Initialize logging systems
        self.training_logger = TotoTrainingLogger(
            experiment_name=experiment_name,
            log_dir=log_dir,
            enable_system_monitoring=enable_system_monitoring
        )
        
        self.tensorboard_monitor = None
        if enable_tensorboard:
            try:
                self.tensorboard_monitor = TensorBoardMonitor(
                    experiment_name=experiment_name,
                    log_dir="tensorboard_logs"
                )
            except Exception as e:
                self.logger.warning(f"TensorBoard not available: {e}")
                self.tensorboard_monitor = None
        
        self.mlflow_tracker = None
        if enable_mlflow:
            try:
                self.mlflow_tracker = MLflowTracker(
                    experiment_name=experiment_name,
                    tracking_uri="mlruns"
                )
            except Exception as e:
                self.logger.warning(f"MLflow not available: {e}")
                self.mlflow_tracker = None
        
        # Checkpoint management
        self.checkpoint_manager = CheckpointManager(
            checkpoint_dir=checkpoint_dir,
            monitor_metric="val_loss",
            mode="min",
            max_checkpoints=5,
            save_best_k=3
        )
        
        # Training callbacks
        self.callbacks = None
        
        # Dashboard configuration
        self.dashboard_generator = DashboardGenerator(experiment_name)
        
        # Training state
        self.training_start_time = None
        self.epoch_start_time = None
        self.best_metrics = {}
        self.training_history = {
            'train_loss': [],
            'val_loss': [],
            'learning_rate': [],
            'epoch_times': []
        }
    
    def setup_callbacks(self, patience: int = 10, lr_patience: int = 5):
        """Setup training callbacks"""
        if not torch.nn:
            self.logger.warning("PyTorch not available, callbacks disabled")
            return
        
        callbacks_list = [
            EarlyStopping(
                monitor="val_loss",
                patience=patience,
                min_delta=1e-6,
                restore_best_weights=True,
                save_best_model_path=str(Path(self.checkpoint_manager.checkpoint_dir) / "early_stopping_best.pth")
            ),
            ReduceLROnPlateau(
                optimizer=self.optimizer,
                monitor="val_loss",
                patience=lr_patience,
                factor=0.5,
                min_lr=1e-7,
                verbose=True
            ),
            MetricTracker(
                metrics_to_track=['train_loss', 'val_loss', 'learning_rate'],
                window_size=10,
                detect_plateaus=True
            )
        ]
        
        self.callbacks = CallbackManager(callbacks_list)
    
    def initialize_model(self, input_dim: int):
        """Initialize model with enhanced logging"""
        super().initialize_model(input_dim)
        
        # Setup callbacks after optimizer is created
        self.setup_callbacks()
        
        # Log model to TensorBoard
        if self.tensorboard_monitor:
            # Create sample input for model graph
            sample_input = torch.randn(1, input_dim, self.config.sequence_length)
            self.tensorboard_monitor.set_model(self.model, sample_input)
    
    def train(self, num_epochs: int = 50):
        """Enhanced training loop with comprehensive monitoring"""
        self.training_start_time = time.time()
        
        # Start experiment tracking
        config_dict = {
            'patch_size': self.config.patch_size,
            'stride': self.config.stride,
            'embed_dim': self.config.embed_dim,
            'num_layers': self.config.num_layers,
            'num_heads': self.config.num_heads,
            'mlp_hidden_dim': self.config.mlp_hidden_dim,
            'dropout': self.config.dropout,
            'sequence_length': self.config.sequence_length,
            'prediction_length': self.config.prediction_length,
            'validation_days': self.config.validation_days,
            'num_epochs': num_epochs,
            'learning_rate': 1e-4,
            'weight_decay': 0.01,
            'optimizer': 'AdamW'
        }
        
        # Start logging systems
        self.training_logger.log_training_start(config_dict)
        
        if self.mlflow_tracker:
            self.mlflow_tracker.start_run(f"{self.experiment_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
            self.mlflow_tracker.log_config(config_dict)
        
        # Generate dashboard
        dashboard_config = self.dashboard_generator.create_training_dashboard()
        self.dashboard_generator.save_configurations(dashboard_config)
        self.dashboard_generator.save_html_dashboard(dashboard_config)
        
        # Load data
        datasets, dataloaders = self.load_data()
        
        if 'train' not in dataloaders:
            self.logger.error("No training data found!")
            return
        
        # Initialize model with correct input dimension (5 for OHLCV)
        self.initialize_model(input_dim=5)
        
        # Start callbacks
        if self.callbacks:
            self.callbacks.on_training_start()
        
        best_val_loss = float('inf')
        
        try:
            for epoch in range(num_epochs):
                self.epoch_start_time = time.time()
                self.logger.info(f"Epoch {epoch + 1}/{num_epochs}")
                
                # Training phase
                train_loss, train_metrics = self.train_epoch_enhanced(dataloaders['train'], epoch)
                
                # Validation phase
                val_loss, val_metrics = None, None
                if 'val' in dataloaders:
                    val_loss, val_metrics = self.validate_enhanced(dataloaders['val'], epoch)
                
                # Calculate epoch time
                epoch_time = time.time() - self.epoch_start_time
                
                # Current learning rate
                current_lr = self.optimizer.param_groups[0]['lr']
                
                # Update training history
                self.training_history['train_loss'].append(train_loss)
                if val_loss is not None:
                    self.training_history['val_loss'].append(val_loss)
                self.training_history['learning_rate'].append(current_lr)
                self.training_history['epoch_times'].append(epoch_time)
                
                # Log to all systems
                self._log_epoch_metrics(epoch, train_loss, val_loss, current_lr, epoch_time, train_metrics, val_metrics)
                
                # Save checkpoint
                metrics_for_checkpoint = {
                    'train_loss': train_loss,
                    'val_loss': val_loss if val_loss is not None else float('inf'),
                    'learning_rate': current_lr,
                    'epoch_time': epoch_time
                }
                
                checkpoint_info = self.checkpoint_manager.save_checkpoint(
                    model=self.model,
                    optimizer=self.optimizer,
                    epoch=epoch,
                    step=epoch * len(dataloaders['train']),
                    metrics=metrics_for_checkpoint,
                    additional_state={'training_history': self.training_history}
                )
                
                # Check for best model
                if val_loss is not None and val_loss < best_val_loss:
                    best_val_loss = val_loss
                    self.best_metrics = metrics_for_checkpoint
                    
                    # Log best model
                    if self.mlflow_tracker:
                        self.mlflow_tracker.log_best_model(
                            self.model,
                            checkpoint_info.path if checkpoint_info else "",
                            "val_loss",
                            val_loss,
                            epoch
                        )
                    
                    self.training_logger.log_best_model(
                        checkpoint_info.path if checkpoint_info else "",
                        "val_loss",
                        val_loss
                    )
                
                # Callback processing
                should_stop = False
                if self.callbacks:
                    callback_state = CallbackState(
                        epoch=epoch,
                        step=epoch * len(dataloaders['train']),
                        train_loss=train_loss,
                        val_loss=val_loss,
                        train_metrics=train_metrics,
                        val_metrics=val_metrics,
                        model_state_dict=self.model.state_dict(),
                        optimizer_state_dict=self.optimizer.state_dict()
                    )
                    
                    should_stop = self.callbacks.on_epoch_end(callback_state)
                
                if should_stop:
                    self.training_logger.log_early_stopping(epoch, 10, "val_loss", best_val_loss)
                    break
                
                # Log epoch summary
                samples_per_sec = len(dataloaders['train']) * dataloaders['train'].batch_size / epoch_time
                self.training_logger.log_epoch_summary(
                    epoch, train_loss, val_loss, epoch_time, samples_per_sec
                )
        
        except Exception as e:
            self.training_logger.log_error(e, "training loop")
            raise
        
        finally:
            # End training
            total_time = time.time() - self.training_start_time
            
            if self.callbacks:
                self.callbacks.on_training_end()
            
            self.training_logger.log_training_complete(epoch + 1, total_time, self.best_metrics)
            
            if self.mlflow_tracker:
                final_metrics = {
                    'final_train_loss': self.training_history['train_loss'][-1] if self.training_history['train_loss'] else 0,
                    'final_val_loss': self.training_history['val_loss'][-1] if self.training_history['val_loss'] else 0,
                    'best_val_loss': best_val_loss,
                    'total_training_time_hours': total_time / 3600,
                    'total_epochs': epoch + 1
                }
                
                self.mlflow_tracker.log_hyperparameters(config_dict, final_metrics)
    
    def train_epoch_enhanced(self, dataloader, epoch) -> Tuple[float, Dict[str, float]]:
        """Enhanced training epoch with detailed logging"""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        gradient_norms = []
        
        for batch_idx, (x, y) in enumerate(dataloader):
            x, y = x.to(self.device), y.to(self.device)
            
            self.optimizer.zero_grad()
            
            try:
                # Forward pass
                batch_size, seq_len, features = x.shape
                input_padding_mask = torch.zeros(batch_size, 1, seq_len, dtype=torch.bool, device=x.device)
                id_mask = torch.ones(batch_size, 1, seq_len, dtype=torch.float32, device=x.device)
                x_reshaped = x.transpose(1, 2).contiguous()
                
                output = self.model.model(x_reshaped, input_padding_mask, id_mask)
                
                if hasattr(output, 'loc'):
                    predictions = output.loc
                elif isinstance(output, dict) and 'prediction' in output:
                    predictions = output['prediction']
                else:
                    predictions = output
                
                if predictions.dim() == 3:
                    predictions = predictions[:, -1, 0]
                elif predictions.dim() == 2:
                    predictions = predictions[:, 0]
                
                loss = torch.nn.functional.mse_loss(predictions, y)
                
                # Backward pass
                loss.backward()
                
                # Calculate gradient norm
                grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                gradient_norms.append(grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm)
                
                self.optimizer.step()
                
                total_loss += loss.item()
                num_batches += 1
                
                # Log batch metrics
                if self.tensorboard_monitor and batch_idx % 10 == 0:
                    current_lr = self.optimizer.param_groups[0]['lr']
                    self.tensorboard_monitor.log_training_metrics(
                        epoch, batch_idx, loss.item(), current_lr
                    )
                    
                    # Log gradients and weights periodically
                    self.tensorboard_monitor.log_gradients()
                    self.tensorboard_monitor.log_model_weights()
                
                if self.mlflow_tracker and batch_idx % 50 == 0:
                    self.mlflow_tracker.log_training_metrics(
                        epoch, batch_idx, loss.item(),
                        learning_rate=self.optimizer.param_groups[0]['lr'],
                        gradient_norm=gradient_norms[-1] if gradient_norms else 0
                    )
                
                # Log to structured logger
                if batch_idx % 10 == 0:
                    self.training_logger.log_training_metrics(
                        epoch, batch_idx, loss.item(),
                        learning_rate=self.optimizer.param_groups[0]['lr'],
                        gradient_norm=gradient_norms[-1] if gradient_norms else 0
                    )
                
            except Exception as e:
                self.logger.error(f"Error in batch {batch_idx}: {e}")
                continue
        
        avg_loss = total_loss / max(num_batches, 1)
        avg_grad_norm = np.mean(gradient_norms) if gradient_norms else 0
        
        metrics = {
            'avg_gradient_norm': avg_grad_norm,
            'num_batches': num_batches
        }
        
        return avg_loss, metrics
    
    def validate_enhanced(self, dataloader, epoch) -> Tuple[float, Dict[str, float]]:
        """Enhanced validation with detailed logging"""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for x, y in dataloader:
                x, y = x.to(self.device), y.to(self.device)
                
                try:
                    batch_size, seq_len, features = x.shape
                    input_padding_mask = torch.zeros(batch_size, 1, seq_len, dtype=torch.bool, device=x.device)
                    id_mask = torch.ones(batch_size, 1, seq_len, dtype=torch.float32, device=x.device)
                    x_reshaped = x.transpose(1, 2).contiguous()
                    
                    output = self.model.model(x_reshaped, input_padding_mask, id_mask)
                    
                    if hasattr(output, 'loc'):
                        predictions = output.loc
                    elif isinstance(output, dict) and 'prediction' in output:
                        predictions = output['prediction']
                    else:
                        predictions = output
                    
                    if predictions.dim() == 3:
                        predictions = predictions[:, -1, 0]
                    elif predictions.dim() == 2:
                        predictions = predictions[:, 0]
                    
                    loss = torch.nn.functional.mse_loss(predictions, y)
                    total_loss += loss.item()
                    num_batches += 1
                    
                    # Store predictions for analysis
                    all_predictions.extend(predictions.cpu().numpy())
                    all_targets.extend(y.cpu().numpy())
                    
                except Exception as e:
                    self.logger.error(f"Error in validation: {e}")
                    continue
        
        avg_loss = total_loss / max(num_batches, 1)
        
        # Calculate additional metrics
        if all_predictions and all_targets:
            predictions_array = np.array(all_predictions)
            targets_array = np.array(all_targets)
            
            mse = np.mean((predictions_array - targets_array) ** 2)
            mae = np.mean(np.abs(predictions_array - targets_array))
            correlation = np.corrcoef(predictions_array, targets_array)[0, 1] if len(predictions_array) > 1 else 0
            
            # Log predictions vs actual
            if self.tensorboard_monitor:
                self.tensorboard_monitor.log_predictions_vs_actual(
                    predictions_array[:1000], targets_array[:1000], epoch
                )
            
            if self.mlflow_tracker:
                self.mlflow_tracker.log_predictions(
                    predictions_array, targets_array, epoch, "validation"
                )
        else:
            mse, mae, correlation = 0, 0, 0
        
        metrics = {
            'mse': mse,
            'mae': mae,
            'correlation': correlation,
            'num_batches': num_batches
        }
        
        return avg_loss, metrics
    
    def _log_epoch_metrics(self, epoch, train_loss, val_loss, learning_rate, epoch_time, train_metrics, val_metrics):
        """Log metrics to all monitoring systems"""
        
        # TensorBoard
        if self.tensorboard_monitor:
            self.tensorboard_monitor.log_validation_metrics(epoch, val_loss or 0)
            
            # Log system metrics
            if hasattr(self.training_logger, 'get_system_metrics'):
                sys_metrics = self.training_logger.get_system_metrics()
                self.tensorboard_monitor.log_system_metrics(
                    sys_metrics.cpu_percent,
                    sys_metrics.memory_percent,
                    sys_metrics.gpu_utilization,
                    sys_metrics.gpu_memory_used_gb / sys_metrics.gpu_memory_total_gb * 100 if sys_metrics.gpu_memory_total_gb else None,
                    sys_metrics.gpu_temperature
                )
        
        # MLflow
        if self.mlflow_tracker:
            epoch_metrics = {
                'epoch_train_loss': train_loss,
                'epoch_val_loss': val_loss or 0,
                'learning_rate': learning_rate,
                'epoch_time_seconds': epoch_time
            }
            
            if train_metrics:
                epoch_metrics.update({f"train_{k}": v for k, v in train_metrics.items()})
            if val_metrics:
                epoch_metrics.update({f"val_{k}": v for k, v in val_metrics.items()})
            
            self.mlflow_tracker.log_epoch_summary(
                epoch, train_loss, val_loss, 
                epoch_time=epoch_time,
                additional_metrics=epoch_metrics
            )
    
    def __enter__(self):
        """Context manager entry"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        # Close all monitoring systems
        if self.tensorboard_monitor:
            self.tensorboard_monitor.close()
        
        if self.mlflow_tracker:
            status = "FAILED" if exc_type is not None else "FINISHED"
            self.mlflow_tracker.end_run(status)
        
        if self.training_logger:
            self.training_logger.stop_system_monitoring()
            self.training_logger.save_training_summary()
        
        if exc_type is not None:
            self.logger.error(f"Training failed with error: {exc_val}")


def main():
    """Main function to run enhanced training"""
    print("🚀 Starting Enhanced Toto Training with Comprehensive Monitoring")
    
    # Create config
    config = TotoOHLCConfig(
        patch_size=12,
        stride=6,
        embed_dim=128,
        num_layers=4,
        num_heads=8,
        dropout=0.1,
        sequence_length=96,
        prediction_length=24,
        validation_days=30
    )
    
    experiment_name = f"toto_enhanced_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    # Initialize enhanced trainer
    with EnhancedTotoTrainer(
        config=config,
        experiment_name=experiment_name,
        enable_tensorboard=True,
        enable_mlflow=True,
        enable_system_monitoring=True
    ) as trainer:
        
        # Start training
        trainer.train(num_epochs=20)  # Reduced for testing
    
    print("✅ Enhanced training completed!")
    print(f"📊 Check logs in: logs/{experiment_name}_*")
    print(f"📈 TensorBoard: tensorboard --logdir tensorboard_logs")
    print(f"🧪 MLflow: mlflow ui --backend-store-uri mlruns")
    print(f"🎛️ Dashboard: Open dashboard_configs/{experiment_name}_dashboard.html")


if __name__ == "__main__":
    main()