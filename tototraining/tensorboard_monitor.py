#!/usr/bin/env python3
"""
TensorBoard Integration for Toto Training Pipeline
Provides real-time monitoring of loss, accuracy, gradients, model weights, and system metrics.
"""

import os
import time
import threading
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, List, Union
import numpy as np

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False
    SummaryWriter = None

try:
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use('Agg')  # Use non-interactive backend
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    plt = None


class TensorBoardMonitor:
    """
    TensorBoard monitoring system for Toto training pipeline.
    Handles real-time logging of metrics, gradients, weights, and visualizations.
    """
    
    def __init__(
        self,
        experiment_name: str,
        log_dir: str = "tensorboard_logs",
        enable_model_graph: bool = True,
        enable_weight_histograms: bool = True,
        enable_gradient_histograms: bool = True,
        histogram_freq: int = 100,  # Log histograms every N batches
        image_freq: int = 500,      # Log images every N batches
        flush_secs: int = 30        # Flush to disk every N seconds
    ):
        if not TENSORBOARD_AVAILABLE:
            raise ImportError("TensorBoard not available. Install with: uv pip install tensorboard")
        
        self.experiment_name = experiment_name
        self.log_dir = Path(log_dir)
        self.enable_model_graph = enable_model_graph
        self.enable_weight_histograms = enable_weight_histograms
        self.enable_gradient_histograms = enable_gradient_histograms
        self.histogram_freq = histogram_freq
        self.image_freq = image_freq
        
        # Create timestamped experiment directory
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.experiment_dir = self.log_dir / f"{experiment_name}_{timestamp}"
        
        # Initialize TensorBoard writers
        self.train_writer = SummaryWriter(
            log_dir=str(self.experiment_dir / "train"),
            flush_secs=flush_secs
        )
        self.val_writer = SummaryWriter(
            log_dir=str(self.experiment_dir / "validation"),
            flush_secs=flush_secs
        )
        self.system_writer = SummaryWriter(
            log_dir=str(self.experiment_dir / "system"),
            flush_secs=flush_secs
        )
        
        # Step counters
        self.train_step = 0
        self.val_step = 0
        self.system_step = 0
        
        # Model reference for graph logging
        self.model = None
        self.model_graph_logged = False
        
        print(f"TensorBoard monitoring initialized for: {experiment_name}")
        print(f"Log directory: {self.experiment_dir}")
        print(f"Start TensorBoard with: tensorboard --logdir {self.experiment_dir}")
    
    def set_model(self, model, sample_input=None):
        """Set model reference for graph and weight logging"""
        self.model = model
        
        if self.enable_model_graph and not self.model_graph_logged and sample_input is not None:
            try:
                self.train_writer.add_graph(model, sample_input)
                self.model_graph_logged = True
                print("Model graph logged to TensorBoard")
            except Exception as e:
                print(f"Warning: Could not log model graph: {e}")
    
    def log_training_metrics(
        self,
        epoch: int,
        batch: int,
        train_loss: float,
        learning_rate: Optional[float] = None,
        accuracy: Optional[float] = None,
        additional_metrics: Optional[Dict[str, float]] = None
    ):
        """Log training metrics"""
        # Core metrics
        self.train_writer.add_scalar('Loss/Train', train_loss, self.train_step)
        
        if learning_rate is not None:
            self.train_writer.add_scalar('Learning_Rate', learning_rate, self.train_step)
        
        if accuracy is not None:
            self.train_writer.add_scalar('Accuracy/Train', accuracy, self.train_step)
        
        # Additional metrics
        if additional_metrics:
            for name, value in additional_metrics.items():
                self.train_writer.add_scalar(f'Metrics/{name}', value, self.train_step)
        
        # Epoch and batch info
        self.train_writer.add_scalar('Info/Epoch', epoch, self.train_step)
        self.train_writer.add_scalar('Info/Batch', batch, self.train_step)
        
        self.train_step += 1
    
    def log_validation_metrics(
        self,
        epoch: int,
        val_loss: float,
        accuracy: Optional[float] = None,
        additional_metrics: Optional[Dict[str, float]] = None
    ):
        """Log validation metrics"""
        self.val_writer.add_scalar('Loss/Validation', val_loss, self.val_step)
        
        if accuracy is not None:
            self.val_writer.add_scalar('Accuracy/Validation', accuracy, self.val_step)
        
        if additional_metrics:
            for name, value in additional_metrics.items():
                self.val_writer.add_scalar(f'Metrics/{name}', value, self.val_step)
        
        self.val_writer.add_scalar('Info/Epoch', epoch, self.val_step)
        self.val_step += 1
    
    def log_model_weights(self, step: Optional[int] = None):
        """Log model weights as histograms"""
        if not self.enable_weight_histograms or self.model is None:
            return
        
        if step is None:
            step = self.train_step
        
        if step % self.histogram_freq != 0:
            return
        
        try:
            for name, param in self.model.named_parameters():
                if param.data is not None:
                    self.train_writer.add_histogram(f'Weights/{name}', param.data, step)
                    
                    # Log weight statistics
                    weight_mean = param.data.mean().item()
                    weight_std = param.data.std().item()
                    weight_norm = param.data.norm().item()
                    
                    self.train_writer.add_scalar(f'Weight_Stats/{name}_mean', weight_mean, step)
                    self.train_writer.add_scalar(f'Weight_Stats/{name}_std', weight_std, step)
                    self.train_writer.add_scalar(f'Weight_Stats/{name}_norm', weight_norm, step)
        
        except Exception as e:
            print(f"Warning: Could not log model weights: {e}")
    
    def log_gradients(self, step: Optional[int] = None):
        """Log gradients as histograms"""
        if not self.enable_gradient_histograms or self.model is None:
            return
        
        if step is None:
            step = self.train_step
        
        if step % self.histogram_freq != 0:
            return
        
        total_grad_norm = 0.0
        param_count = 0
        
        try:
            for name, param in self.model.named_parameters():
                if param.grad is not None:
                    self.train_writer.add_histogram(f'Gradients/{name}', param.grad, step)
                    
                    # Log gradient statistics
                    grad_mean = param.grad.mean().item()
                    grad_std = param.grad.std().item()
                    grad_norm = param.grad.norm().item()
                    
                    self.train_writer.add_scalar(f'Gradient_Stats/{name}_mean', grad_mean, step)
                    self.train_writer.add_scalar(f'Gradient_Stats/{name}_std', grad_std, step)
                    self.train_writer.add_scalar(f'Gradient_Stats/{name}_norm', grad_norm, step)
                    
                    total_grad_norm += grad_norm ** 2
                    param_count += 1
            
            # Log total gradient norm
            if param_count > 0:
                total_grad_norm = np.sqrt(total_grad_norm)
                self.train_writer.add_scalar('Gradient_Stats/Total_Norm', total_grad_norm, step)
        
        except Exception as e:
            print(f"Warning: Could not log gradients: {e}")
    
    def log_loss_curves(self, train_losses: List[float], val_losses: List[float]):
        """Log loss curves as images"""
        if not MATPLOTLIB_AVAILABLE:
            return
        
        if self.train_step % self.image_freq != 0:
            return
        
        try:
            fig, ax = plt.subplots(figsize=(10, 6))
            
            epochs = range(1, len(train_losses) + 1)
            ax.plot(epochs, train_losses, 'b-', label='Training Loss', linewidth=2)
            if val_losses and len(val_losses) == len(train_losses):
                ax.plot(epochs, val_losses, 'r-', label='Validation Loss', linewidth=2)
            
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Loss')
            ax.set_title('Training and Validation Loss')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            self.train_writer.add_figure('Loss_Curves/Training_Progress', fig, self.train_step)
            plt.close(fig)
        
        except Exception as e:
            print(f"Warning: Could not log loss curves: {e}")
    
    def log_accuracy_curves(self, train_accuracies: List[float], val_accuracies: List[float]):
        """Log accuracy curves as images"""
        if not MATPLOTLIB_AVAILABLE:
            return
        
        if self.train_step % self.image_freq != 0:
            return
        
        try:
            fig, ax = plt.subplots(figsize=(10, 6))
            
            epochs = range(1, len(train_accuracies) + 1)
            ax.plot(epochs, train_accuracies, 'b-', label='Training Accuracy', linewidth=2)
            if val_accuracies and len(val_accuracies) == len(train_accuracies):
                ax.plot(epochs, val_accuracies, 'r-', label='Validation Accuracy', linewidth=2)
            
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Accuracy')
            ax.set_title('Training and Validation Accuracy')
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.set_ylim(0, 1)
            
            self.train_writer.add_figure('Accuracy_Curves/Training_Progress', fig, self.train_step)
            plt.close(fig)
        
        except Exception as e:
            print(f"Warning: Could not log accuracy curves: {e}")
    
    def log_system_metrics(
        self,
        cpu_percent: float,
        memory_percent: float,
        gpu_utilization: Optional[float] = None,
        gpu_memory_percent: Optional[float] = None,
        gpu_temperature: Optional[float] = None
    ):
        """Log system metrics"""
        self.system_writer.add_scalar('CPU/Usage_Percent', cpu_percent, self.system_step)
        self.system_writer.add_scalar('Memory/Usage_Percent', memory_percent, self.system_step)
        
        if gpu_utilization is not None:
            self.system_writer.add_scalar('GPU/Utilization_Percent', gpu_utilization, self.system_step)
        
        if gpu_memory_percent is not None:
            self.system_writer.add_scalar('GPU/Memory_Percent', gpu_memory_percent, self.system_step)
        
        if gpu_temperature is not None:
            self.system_writer.add_scalar('GPU/Temperature_C', gpu_temperature, self.system_step)
        
        self.system_step += 1
    
    def log_hyperparameters(self, hparams: Dict[str, Any], metrics: Dict[str, float]):
        """Log hyperparameters and final metrics"""
        # Convert all values to scalars for TensorBoard
        scalar_hparams = {}
        for key, value in hparams.items():
            if isinstance(value, (int, float, bool)):
                scalar_hparams[key] = value
            else:
                scalar_hparams[key] = str(value)
        
        try:
            self.train_writer.add_hparams(scalar_hparams, metrics)
        except Exception as e:
            print(f"Warning: Could not log hyperparameters: {e}")
    
    def log_predictions_vs_actual(
        self, 
        predictions: np.ndarray, 
        actuals: np.ndarray, 
        step: Optional[int] = None
    ):
        """Log predictions vs actual values as scatter plot"""
        if not MATPLOTLIB_AVAILABLE or step is None:
            return
        
        if step % self.image_freq != 0:
            return
        
        try:
            fig, ax = plt.subplots(figsize=(8, 8))
            
            # Sample data if too many points
            if len(predictions) > 1000:
                indices = np.random.choice(len(predictions), 1000, replace=False)
                predictions = predictions[indices]
                actuals = actuals[indices]
            
            ax.scatter(actuals, predictions, alpha=0.5, s=20)
            
            # Perfect prediction line
            min_val = min(np.min(actuals), np.min(predictions))
            max_val = max(np.max(actuals), np.max(predictions))
            ax.plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect Prediction')
            
            ax.set_xlabel('Actual Values')
            ax.set_ylabel('Predicted Values')
            ax.set_title('Predictions vs Actual Values')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # Calculate and display R²
            correlation_matrix = np.corrcoef(actuals, predictions)
            r_squared = correlation_matrix[0, 1] ** 2
            ax.text(0.05, 0.95, f'R² = {r_squared:.3f}', 
                   transform=ax.transAxes, fontsize=12, 
                   bbox=dict(boxstyle="round", facecolor='wheat', alpha=0.8))
            
            self.val_writer.add_figure('Predictions/Scatter_Plot', fig, step)
            plt.close(fig)
        
        except Exception as e:
            print(f"Warning: Could not log predictions scatter plot: {e}")
    
    def log_feature_importance(self, feature_names: List[str], importances: np.ndarray, step: int):
        """Log feature importance as bar chart"""
        if not MATPLOTLIB_AVAILABLE:
            return
        
        try:
            fig, ax = plt.subplots(figsize=(12, 8))
            
            # Sort by importance
            sorted_indices = np.argsort(importances)[::-1]
            sorted_names = [feature_names[i] for i in sorted_indices]
            sorted_importances = importances[sorted_indices]
            
            bars = ax.bar(range(len(sorted_names)), sorted_importances)
            ax.set_xlabel('Features')
            ax.set_ylabel('Importance')
            ax.set_title('Feature Importance')
            ax.set_xticks(range(len(sorted_names)))
            ax.set_xticklabels(sorted_names, rotation=45, ha='right')
            
            # Add value labels on bars
            for bar, importance in zip(bars, sorted_importances):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                       f'{importance:.3f}', ha='center', va='bottom')
            
            plt.tight_layout()
            self.train_writer.add_figure('Analysis/Feature_Importance', fig, step)
            plt.close(fig)
        
        except Exception as e:
            print(f"Warning: Could not log feature importance: {e}")
    
    def log_learning_rate_schedule(self, learning_rates: List[float], step: int):
        """Log learning rate schedule"""
        if not MATPLOTLIB_AVAILABLE:
            return
        
        try:
            fig, ax = plt.subplots(figsize=(10, 6))
            
            steps = range(len(learning_rates))
            ax.plot(steps, learning_rates, 'g-', linewidth=2)
            ax.set_xlabel('Step')
            ax.set_ylabel('Learning Rate')
            ax.set_title('Learning Rate Schedule')
            ax.set_yscale('log')
            ax.grid(True, alpha=0.3)
            
            self.train_writer.add_figure('Training/Learning_Rate_Schedule', fig, step)
            plt.close(fig)
        
        except Exception as e:
            print(f"Warning: Could not log learning rate schedule: {e}")
    
    def flush(self):
        """Flush all writers"""
        self.train_writer.flush()
        self.val_writer.flush()
        self.system_writer.flush()
    
    def close(self):
        """Close all writers"""
        self.train_writer.close()
        self.val_writer.close()
        self.system_writer.close()
    
    def __enter__(self):
        """Context manager entry"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.flush()
        self.close()


# Convenience function for quick TensorBoard setup
def create_tensorboard_monitor(
    experiment_name: str,
    log_dir: str = "tensorboard_logs",
    **kwargs
) -> TensorBoardMonitor:
    """Create a TensorBoard monitor with sensible defaults"""
    return TensorBoardMonitor(
        experiment_name=experiment_name,
        log_dir=log_dir,
        **kwargs
    )


if __name__ == "__main__":
    # Example usage
    if TORCH_AVAILABLE and TENSORBOARD_AVAILABLE:
        with create_tensorboard_monitor("test_experiment") as tb:
            # Simulate training
            for epoch in range(5):
                for batch in range(10):
                    train_loss = 1.0 - (epoch * 0.1 + batch * 0.01)
                    tb.log_training_metrics(
                        epoch=epoch,
                        batch=batch,
                        train_loss=train_loss,
                        learning_rate=0.001,
                        accuracy=train_loss * 0.8
                    )
                
                # Validation
                val_loss = train_loss + 0.1
                tb.log_validation_metrics(epoch, val_loss, accuracy=val_loss * 0.8)
            
            print("Example logging completed. Check TensorBoard!")
    else:
        print("PyTorch or TensorBoard not available for example")