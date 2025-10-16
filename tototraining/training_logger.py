#!/usr/bin/env python3
"""
Robust Training Logger for Toto Retraining Pipeline
Provides structured logging for training metrics, loss curves, validation scores, and system metrics.
"""

import os
import json
import time
import logging
import psutil
import threading
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List, Union
from dataclasses import dataclass, asdict
from collections import defaultdict, deque
import numpy as np

try:
    import GPUtil
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


@dataclass
class TrainingMetrics:
    """Container for training metrics"""
    epoch: int
    batch: int
    train_loss: float
    val_loss: Optional[float] = None
    learning_rate: float = 0.0
    train_accuracy: Optional[float] = None
    val_accuracy: Optional[float] = None
    gradient_norm: Optional[float] = None
    timestamp: str = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now().isoformat()


@dataclass
class SystemMetrics:
    """Container for system metrics"""
    cpu_percent: float
    memory_used_gb: float
    memory_total_gb: float
    memory_percent: float
    disk_used_gb: float
    disk_free_gb: float
    gpu_utilization: Optional[float] = None
    gpu_memory_used_gb: Optional[float] = None
    gpu_memory_total_gb: Optional[float] = None
    gpu_temperature: Optional[float] = None
    timestamp: str = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now().isoformat()


class TotoTrainingLogger:
    """
    Comprehensive logging system for Toto training pipeline.
    Handles structured logging, metrics tracking, and system monitoring.
    """
    
    def __init__(
        self, 
        experiment_name: str,
        log_dir: str = "logs",
        log_level: int = logging.INFO,
        enable_system_monitoring: bool = True,
        system_monitor_interval: float = 30.0,  # seconds
        metrics_buffer_size: int = 1000
    ):
        self.experiment_name = experiment_name
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        
        # Create experiment-specific directory
        self.experiment_dir = self.log_dir / f"{experiment_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.experiment_dir.mkdir(exist_ok=True)
        
        self.enable_system_monitoring = enable_system_monitoring
        self.system_monitor_interval = system_monitor_interval
        self.metrics_buffer_size = metrics_buffer_size
        
        # Initialize logging
        self._setup_logging(log_level)
        
        # Initialize metrics storage
        self.training_metrics = deque(maxlen=metrics_buffer_size)
        self.system_metrics = deque(maxlen=metrics_buffer_size)
        self.loss_history = defaultdict(list)
        self.accuracy_history = defaultdict(list)
        
        # System monitoring
        self._system_monitor_thread = None
        self._stop_monitoring = threading.Event()
        
        if self.enable_system_monitoring:
            self.start_system_monitoring()
        
        # Metrics files
        self.metrics_file = self.experiment_dir / "training_metrics.jsonl"
        self.system_metrics_file = self.experiment_dir / "system_metrics.jsonl"
        
        self.logger.info(f"Training logger initialized for experiment: {experiment_name}")
        self.logger.info(f"Log directory: {self.experiment_dir}")
    
    def _setup_logging(self, log_level: int):
        """Setup structured logging with multiple handlers"""
        # Create logger
        self.logger = logging.getLogger(f"toto_training_{self.experiment_name}")
        self.logger.setLevel(log_level)
        
        # Clear existing handlers
        self.logger.handlers.clear()
        
        # Create formatters
        detailed_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s - [%(filename)s:%(lineno)d]'
        )
        simple_formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s'
        )
        
        # File handler for detailed logs
        detailed_file_handler = logging.FileHandler(
            self.experiment_dir / "training_detailed.log"
        )
        detailed_file_handler.setLevel(logging.DEBUG)
        detailed_file_handler.setFormatter(detailed_formatter)
        
        # File handler for important events
        events_file_handler = logging.FileHandler(
            self.experiment_dir / "training_events.log"
        )
        events_file_handler.setLevel(logging.INFO)
        events_file_handler.setFormatter(simple_formatter)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(log_level)
        console_handler.setFormatter(simple_formatter)
        
        # Add handlers
        self.logger.addHandler(detailed_file_handler)
        self.logger.addHandler(events_file_handler)
        self.logger.addHandler(console_handler)
    
    def log_training_metrics(
        self,
        epoch: int,
        batch: int,
        train_loss: float,
        val_loss: Optional[float] = None,
        learning_rate: float = 0.0,
        train_accuracy: Optional[float] = None,
        val_accuracy: Optional[float] = None,
        gradient_norm: Optional[float] = None,
        additional_metrics: Optional[Dict[str, float]] = None
    ):
        """Log training metrics"""
        metrics = TrainingMetrics(
            epoch=epoch,
            batch=batch,
            train_loss=train_loss,
            val_loss=val_loss,
            learning_rate=learning_rate,
            train_accuracy=train_accuracy,
            val_accuracy=val_accuracy,
            gradient_norm=gradient_norm
        )
        
        # Store metrics
        self.training_metrics.append(metrics)
        self.loss_history['train'].append(train_loss)
        if val_loss is not None:
            self.loss_history['val'].append(val_loss)
        if train_accuracy is not None:
            self.accuracy_history['train'].append(train_accuracy)
        if val_accuracy is not None:
            self.accuracy_history['val'].append(val_accuracy)
        
        # Write to file
        metrics_dict = asdict(metrics)
        if additional_metrics:
            metrics_dict.update(additional_metrics)
        
        # Convert numpy/torch types to Python types for JSON serialization
        def convert_to_json_serializable(obj):
            if hasattr(obj, 'item'):  # numpy/torch scalar
                return obj.item()
            elif hasattr(obj, 'tolist'):  # numpy array
                return obj.tolist()
            return obj
        
        json_safe_dict = {}
        for k, v in metrics_dict.items():
            json_safe_dict[k] = convert_to_json_serializable(v)
        
        with open(self.metrics_file, 'a') as f:
            f.write(json.dumps(json_safe_dict, default=str) + '\n')
        
        # Log to console/files
        log_msg = f"Epoch {epoch}, Batch {batch}: Train Loss={train_loss:.6f}"
        if val_loss is not None:
            log_msg += f", Val Loss={val_loss:.6f}"
        if learning_rate > 0:
            log_msg += f", LR={learning_rate:.2e}"
        if gradient_norm is not None:
            log_msg += f", Grad Norm={gradient_norm:.4f}"
        if train_accuracy is not None:
            log_msg += f", Train Acc={train_accuracy:.4f}"
        if val_accuracy is not None:
            log_msg += f", Val Acc={val_accuracy:.4f}"
        
        self.logger.info(log_msg)
    
    def log_model_checkpoint(self, checkpoint_path: str, metrics: Dict[str, float]):
        """Log model checkpoint information"""
        self.logger.info(f"Model checkpoint saved: {checkpoint_path}")
        for metric_name, value in metrics.items():
            self.logger.info(f"  {metric_name}: {value:.6f}")
    
    def log_best_model(self, model_path: str, best_metric: str, best_value: float):
        """Log best model information"""
        self.logger.info(f"🏆 NEW BEST MODEL! {best_metric}={best_value:.6f}")
        self.logger.info(f"Best model saved: {model_path}")
    
    def log_early_stopping(self, epoch: int, patience: int, best_metric: str, best_value: float):
        """Log early stopping event"""
        self.logger.info(f"⏹️ Early stopping triggered at epoch {epoch}")
        self.logger.info(f"Patience reached: {patience}")
        self.logger.info(f"Best {best_metric}: {best_value:.6f}")
    
    def log_learning_rate_schedule(self, epoch: int, old_lr: float, new_lr: float, reason: str):
        """Log learning rate schedule changes"""
        self.logger.info(f"📉 Learning rate updated at epoch {epoch}: {old_lr:.2e} → {new_lr:.2e}")
        self.logger.info(f"Reason: {reason}")
    
    def log_epoch_summary(
        self, 
        epoch: int, 
        train_loss: float, 
        val_loss: Optional[float] = None,
        epoch_time: Optional[float] = None,
        samples_per_sec: Optional[float] = None
    ):
        """Log epoch summary"""
        summary = f"📊 Epoch {epoch} Summary: Train Loss={train_loss:.6f}"
        if val_loss is not None:
            summary += f", Val Loss={val_loss:.6f}"
        if epoch_time is not None:
            summary += f", Time={epoch_time:.2f}s"
        if samples_per_sec is not None:
            summary += f", Throughput={samples_per_sec:.1f} samples/s"
        
        self.logger.info(summary)
    
    def log_training_start(self, config: Dict[str, Any]):
        """Log training start with configuration"""
        self.logger.info("🚀 Starting Toto training...")
        self.logger.info("Training Configuration:")
        for key, value in config.items():
            self.logger.info(f"  {key}: {value}")
        
        # Save config to file
        config_file = self.experiment_dir / "config.json"
        with open(config_file, 'w') as f:
            json.dump(config, f, indent=2, default=str)
    
    def log_training_complete(self, total_epochs: int, total_time: float, best_metrics: Dict[str, float]):
        """Log training completion"""
        self.logger.info("✅ Training completed!")
        self.logger.info(f"Total epochs: {total_epochs}")
        self.logger.info(f"Total time: {total_time:.2f} seconds ({total_time/3600:.2f} hours)")
        self.logger.info("Best metrics:")
        for metric, value in best_metrics.items():
            self.logger.info(f"  {metric}: {value:.6f}")
    
    def log_error(self, error: Exception, context: str = ""):
        """Log training errors"""
        error_msg = f"❌ Error"
        if context:
            error_msg += f" in {context}"
        error_msg += f": {str(error)}"
        self.logger.error(error_msg, exc_info=True)
    
    def log_warning(self, message: str):
        """Log warnings"""
        self.logger.warning(f"⚠️ {message}")
    
    def get_system_metrics(self) -> SystemMetrics:
        """Collect current system metrics"""
        # CPU and Memory
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        metrics = SystemMetrics(
            cpu_percent=cpu_percent,
            memory_used_gb=memory.used / (1024**3),
            memory_total_gb=memory.total / (1024**3),
            memory_percent=memory.percent,
            disk_used_gb=disk.used / (1024**3),
            disk_free_gb=disk.free / (1024**3)
        )
        
        # GPU metrics if available
        if GPU_AVAILABLE:
            try:
                gpus = GPUtil.getGPUs()
                if gpus:
                    gpu = gpus[0]  # Use first GPU
                    metrics.gpu_utilization = gpu.load * 100
                    metrics.gpu_memory_used_gb = gpu.memoryUsed / 1024
                    metrics.gpu_memory_total_gb = gpu.memoryTotal / 1024
                    metrics.gpu_temperature = gpu.temperature
            except Exception:
                pass  # Ignore GPU errors
        
        return metrics
    
    def _system_monitor_loop(self):
        """Background system monitoring loop"""
        while not self._stop_monitoring.wait(self.system_monitor_interval):
            try:
                metrics = self.get_system_metrics()
                self.system_metrics.append(metrics)
                
                # Write to file
                with open(self.system_metrics_file, 'a') as f:
                    f.write(json.dumps(asdict(metrics)) + '\n')
                
                # Log warnings for high resource usage
                if metrics.memory_percent > 90:
                    self.log_warning(f"High memory usage: {metrics.memory_percent:.1f}%")
                if metrics.gpu_utilization is not None and metrics.gpu_utilization < 50:
                    self.log_warning(f"Low GPU utilization: {metrics.gpu_utilization:.1f}%")
                
            except Exception as e:
                self.logger.error(f"Error in system monitoring: {e}")
    
    def start_system_monitoring(self):
        """Start background system monitoring"""
        if self._system_monitor_thread is None or not self._system_monitor_thread.is_alive():
            self._stop_monitoring.clear()
            self._system_monitor_thread = threading.Thread(
                target=self._system_monitor_loop,
                daemon=True
            )
            self._system_monitor_thread.start()
            self.logger.info("System monitoring started")
    
    def stop_system_monitoring(self):
        """Stop background system monitoring"""
        if self._system_monitor_thread and self._system_monitor_thread.is_alive():
            self._stop_monitoring.set()
            self._system_monitor_thread.join()
            self.logger.info("System monitoring stopped")
    
    def get_loss_statistics(self) -> Dict[str, Dict[str, float]]:
        """Get loss statistics"""
        stats = {}
        for loss_type, losses in self.loss_history.items():
            if losses:
                stats[f"{loss_type}_loss"] = {
                    'mean': np.mean(losses),
                    'std': np.std(losses),
                    'min': np.min(losses),
                    'max': np.max(losses),
                    'current': losses[-1] if losses else None
                }
        return stats
    
    def get_accuracy_statistics(self) -> Dict[str, Dict[str, float]]:
        """Get accuracy statistics"""
        stats = {}
        for acc_type, accuracies in self.accuracy_history.items():
            if accuracies:
                stats[f"{acc_type}_accuracy"] = {
                    'mean': np.mean(accuracies),
                    'std': np.std(accuracies),
                    'min': np.min(accuracies),
                    'max': np.max(accuracies),
                    'current': accuracies[-1] if accuracies else None
                }
        return stats
    
    def save_training_summary(self):
        """Save comprehensive training summary"""
        summary = {
            'experiment_name': self.experiment_name,
            'start_time': self.experiment_dir.name.split('_')[-2] + '_' + self.experiment_dir.name.split('_')[-1],
            'total_training_samples': len(self.training_metrics),
            'total_system_samples': len(self.system_metrics),
            'loss_statistics': self.get_loss_statistics(),
            'accuracy_statistics': self.get_accuracy_statistics(),
        }
        
        # Add latest system metrics
        if self.system_metrics:
            latest_system = self.system_metrics[-1]
            summary['final_system_state'] = asdict(latest_system)
        
        summary_file = self.experiment_dir / "training_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        self.logger.info(f"Training summary saved: {summary_file}")
    
    def __enter__(self):
        """Context manager entry"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.stop_system_monitoring()
        self.save_training_summary()
        
        if exc_type is not None:
            self.log_error(exc_val, "training context")
        
        self.logger.info("Training logger session ended")


# Convenience function for quick logger setup
def create_training_logger(
    experiment_name: str,
    log_dir: str = "logs",
    **kwargs
) -> TotoTrainingLogger:
    """Create a training logger with sensible defaults"""
    return TotoTrainingLogger(
        experiment_name=experiment_name,
        log_dir=log_dir,
        **kwargs
    )


if __name__ == "__main__":
    # Example usage
    with create_training_logger("test_experiment") as logger:
        logger.log_training_start({"learning_rate": 0.001, "batch_size": 32})
        
        for epoch in range(3):
            for batch in range(5):
                train_loss = 1.0 - (epoch * 0.1 + batch * 0.02)
                val_loss = train_loss + 0.1
                
                logger.log_training_metrics(
                    epoch=epoch,
                    batch=batch,
                    train_loss=train_loss,
                    val_loss=val_loss,
                    learning_rate=0.001,
                    gradient_norm=0.5
                )
        
        logger.log_training_complete(3, 60.0, {"best_val_loss": 0.75})