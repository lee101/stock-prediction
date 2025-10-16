#!/usr/bin/env python3
"""
Enhanced logging utilities for training
"""

import logging
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional
import json


class ColoredFormatter(logging.Formatter):
    """Colored console formatter"""
    
    # ANSI color codes
    COLORS = {
        'DEBUG': '\033[36m',      # Cyan
        'INFO': '\033[32m',       # Green
        'WARNING': '\033[33m',    # Yellow
        'ERROR': '\033[31m',      # Red
        'CRITICAL': '\033[35m',   # Magenta
        'RESET': '\033[0m',       # Reset
        'BOLD': '\033[1m',        # Bold
    }
    
    def format(self, record):
        # Add color based on level
        level_color = self.COLORS.get(record.levelname, self.COLORS['RESET'])
        reset_color = self.COLORS['RESET']
        bold = self.COLORS['BOLD']
        
        # Format timestamp
        timestamp = datetime.fromtimestamp(record.created).strftime('%H:%M:%S')
        
        # Format message with colors
        if record.levelname in ['ERROR', 'CRITICAL']:
            formatted = f"{level_color}{bold}[{record.levelname}]{reset_color} {timestamp} | {level_color}{record.getMessage()}{reset_color}"
        elif record.levelname == 'WARNING':
            formatted = f"{level_color}[{record.levelname}]{reset_color} {timestamp} | {record.getMessage()}"
        elif record.levelname == 'INFO':
            formatted = f"{level_color}[INFO]{reset_color} {timestamp} | {record.getMessage()}"
        else:
            formatted = f"[{record.levelname}] {timestamp} | {record.getMessage()}"
        
        return formatted


class TrainingLogger:
    """Enhanced training logger with multiple outputs"""
    
    def __init__(self, log_dir: str, experiment_name: str = "training"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Create timestamped log file
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.log_file = self.log_dir / f"{experiment_name}_{timestamp}.log"
        self.metrics_file = self.log_dir / f"{experiment_name}_metrics_{timestamp}.jsonl"
        
        # Setup loggers
        self.setup_loggers()
        
        # Training metrics storage
        self.metrics_history = []
        self.best_metrics = {}
        
    def setup_loggers(self):
        """Setup console and file loggers"""
        
        # Main logger
        self.logger = logging.getLogger('training')
        self.logger.setLevel(logging.INFO)
        self.logger.handlers.clear()  # Clear any existing handlers
        
        # Console handler with colors
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_formatter = ColoredFormatter()
        console_handler.setFormatter(console_formatter)
        self.logger.addHandler(console_handler)
        
        # File handler
        file_handler = logging.FileHandler(self.log_file, mode='w')
        file_handler.setLevel(logging.DEBUG)
        file_formatter = logging.Formatter(
            '%(asctime)s | %(levelname)-8s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(file_formatter)
        self.logger.addHandler(file_handler)
        
        self.info(f"Logging initialized - File: {self.log_file}")
        
    def info(self, message: str):
        """Log info message"""
        self.logger.info(message)
        
    def warning(self, message: str):
        """Log warning message"""
        self.logger.warning(message)
        
    def error(self, message: str):
        """Log error message"""
        self.logger.error(message)
        
    def debug(self, message: str):
        """Log debug message"""
        self.logger.debug(message)
        
    def log_training_start(self, config: Dict[str, Any], model_info: Dict[str, Any]):
        """Log training start information"""
        self.info("=" * 80)
        self.info("🚀 STARTING TRAINING SESSION")
        self.info("=" * 80)
        
        self.info(f"📊 Experiment: {config.get('experiment_name', 'Unknown')}")
        self.info(f"📝 Description: {config.get('description', 'No description')}")
        self.info(f"🕒 Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        self.info("\n📋 CONFIGURATION:")
        self.info(f"  • Model: {model_info.get('hidden_size', 'Unknown')}d, {model_info.get('num_layers', 'Unknown')} layers")
        params = model_info.get('total_params', model_info.get('parameters', 'Unknown'))
        if isinstance(params, int):
            self.info(f"  • Parameters: {params:,}")
        else:
            self.info(f"  • Parameters: {params}")
        self.info(f"  • Optimizer: {config.get('optimizer', 'Unknown')}")
        self.info(f"  • Learning Rate: {config.get('learning_rate', 'Unknown')}")
        self.info(f"  • Batch Size: {config.get('batch_size', 'Unknown')}")
        max_steps = config.get('max_steps', 'Unknown')
        if isinstance(max_steps, int):
            self.info(f"  • Max Steps: {max_steps:,}")
        else:
            self.info(f"  • Max Steps: {max_steps}")
        self.info(f"  • Device: {config.get('device', 'Unknown')}")
        
        # Log to file with full config
        self.debug("Full configuration:")
        self.debug(json.dumps(config, indent=2, default=str))
        
    def log_epoch_start(self, epoch: int, total_epochs: Optional[int] = None):
        """Log epoch start"""
        if total_epochs:
            self.info(f"\n📈 EPOCH {epoch}/{total_epochs}")
        else:
            self.info(f"\n📈 EPOCH {epoch}")
        self.info("-" * 50)
        
    def log_step_metrics(self, step: int, metrics: Dict[str, Any], phase: str = "train"):
        """Log training step metrics with nice formatting"""
        
        # Store metrics
        metrics_entry = {
            'step': step,
            'phase': phase,
            'timestamp': datetime.now().isoformat(),
            **metrics
        }
        self.metrics_history.append(metrics_entry)
        
        # Save to JSONL file
        with open(self.metrics_file, 'a') as f:
            f.write(json.dumps(metrics_entry) + '\n')
        
        # Format for console
        if phase == "train":
            loss = metrics.get('loss', 0)
            lr = metrics.get('learning_rate', 0)
            self.info(f"Step {step:6d} | Loss: {loss:8.4f} | LR: {lr:.2e}")
            
        elif phase == "eval":
            eval_loss = metrics.get('loss', 0)
            action_loss = metrics.get('action_loss', 0)
            price_loss = metrics.get('price_loss', 0)
            
            self.info(f"📊 EVALUATION (Step {step})")
            self.info(f"   Eval Loss:   {eval_loss:8.4f}")
            self.info(f"   Action Loss: {action_loss:8.4f}")
            self.info(f"   Price Loss:  {price_loss:8.4f}")
            
            # Check if this is the best model
            if 'loss' in metrics:
                if 'best_eval_loss' not in self.best_metrics or metrics['loss'] < self.best_metrics['best_eval_loss']:
                    self.best_metrics['best_eval_loss'] = metrics['loss']
                    self.best_metrics['best_step'] = step
                    self.info(f"   ⭐ NEW BEST MODEL! (Loss: {metrics['loss']:.4f})")
    
    def log_epoch_summary(self, epoch: int, avg_loss: float, time_elapsed: float):
        """Log epoch summary"""
        self.info(f"✅ Epoch {epoch} completed")
        self.info(f"   Average Loss: {avg_loss:.4f}")
        self.info(f"   Time Elapsed: {time_elapsed:.1f}s")
        
    def log_checkpoint_saved(self, step: int, path: str):
        """Log checkpoint saving"""
        self.info(f"💾 Checkpoint saved at step {step}: {path}")
        
    def log_early_stopping(self, step: int, patience: int):
        """Log early stopping"""
        self.warning(f"⏹️ Early stopping triggered at step {step} (patience: {patience})")
        
    def log_training_complete(self, total_time: float, final_metrics: Dict[str, Any]):
        """Log training completion"""
        self.info("\n" + "=" * 80)
        self.info("🎉 TRAINING COMPLETED!")
        self.info("=" * 80)
        
        hours = int(total_time // 3600)
        minutes = int((total_time % 3600) // 60)
        seconds = int(total_time % 60)
        
        self.info(f"⏱️ Total Training Time: {hours:02d}:{minutes:02d}:{seconds:02d}")
        
        if self.best_metrics:
            self.info(f"⭐ Best Model:")
            self.info(f"   Step: {self.best_metrics.get('best_step', 'Unknown')}")
            self.info(f"   Loss: {self.best_metrics.get('best_eval_loss', 'Unknown'):.4f}")
        
        if final_metrics:
            self.info(f"📊 Final Metrics:")
            for key, value in final_metrics.items():
                if isinstance(value, float):
                    self.info(f"   {key}: {value:.4f}")
                else:
                    self.info(f"   {key}: {value}")
        
        self.info(f"📁 Logs saved to: {self.log_file}")
        self.info(f"📈 Metrics saved to: {self.metrics_file}")
        self.info("=" * 80)
    
    def log_error(self, error: Exception, step: Optional[int] = None):
        """Log training error"""
        if step:
            self.error(f"❌ Training failed at step {step}: {str(error)}")
        else:
            self.error(f"❌ Training failed: {str(error)}")
        
        # Log full traceback to file
        import traceback
        self.debug("Full traceback:")
        self.debug(traceback.format_exc())
    
    def log_resource_usage(self, gpu_memory: Optional[float] = None, cpu_percent: Optional[float] = None):
        """Log resource usage"""
        if gpu_memory is not None:
            self.debug(f"GPU Memory: {gpu_memory:.1f} MB")
        if cpu_percent is not None:
            self.debug(f"CPU Usage: {cpu_percent:.1f}%")
    
    def create_progress_bar_desc(self, step: int, loss: float, lr: float) -> str:
        """Create description for progress bar"""
        return f"Step {step} | Loss: {loss:.4f} | LR: {lr:.2e}"


def get_logger(log_dir: str, experiment_name: str = "training") -> TrainingLogger:
    """Factory function to create training logger"""
    return TrainingLogger(log_dir, experiment_name)


class MetricsTracker:
    """Track and analyze training metrics"""
    
    def __init__(self):
        self.metrics = []
        self.best_metrics = {}
        
    def add_metric(self, step: int, phase: str, **kwargs):
        """Add a metric entry"""
        entry = {
            'step': step,
            'phase': phase,
            'timestamp': datetime.now(),
            **kwargs
        }
        self.metrics.append(entry)
        
        # Update best metrics
        for key, value in kwargs.items():
            if isinstance(value, (int, float)):
                best_key = f"best_{key}"
                if best_key not in self.best_metrics or value < self.best_metrics[best_key]:
                    self.best_metrics[best_key] = value
                    self.best_metrics[f"{best_key}_step"] = step
    
    def get_recent_avg(self, metric: str, steps: int = 100) -> float:
        """Get recent average of a metric"""
        recent_metrics = [m for m in self.metrics[-steps:] if metric in m]
        if not recent_metrics:
            return 0.0
        return sum(m[metric] for m in recent_metrics) / len(recent_metrics)
    
    def get_smoothed_loss(self, window: int = 50) -> float:
        """Get smoothed loss over window"""
        recent_losses = [m['loss'] for m in self.metrics[-window:] if 'loss' in m]
        if not recent_losses:
            return 0.0
        return sum(recent_losses) / len(recent_losses)