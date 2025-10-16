#!/usr/bin/env python3
"""
Training Callbacks for Toto Training Pipeline
Provides early stopping, learning rate scheduling, and other training callbacks with comprehensive logging.
"""

import os
import json
import time
import math
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, List, Callable, Union
import logging
from dataclasses import dataclass, asdict
from abc import ABC, abstractmethod
import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.optim.lr_scheduler import _LRScheduler
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None


@dataclass
class CallbackState:
    """State information for callbacks"""
    epoch: int
    step: int
    train_loss: float
    val_loss: Optional[float] = None
    train_metrics: Optional[Dict[str, float]] = None
    val_metrics: Optional[Dict[str, float]] = None
    model_state_dict: Optional[Dict] = None
    optimizer_state_dict: Optional[Dict] = None
    timestamp: str = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now().isoformat()


class BaseCallback(ABC):
    """Base class for training callbacks"""
    
    def __init__(self, name: str):
        self.name = name
        self.logger = logging.getLogger(f"{__name__}.{name}")
        
    @abstractmethod
    def on_epoch_end(self, state: CallbackState) -> bool:
        """Called at the end of each epoch. Return True to stop training."""
        pass
    
    def on_training_start(self):
        """Called at the start of training"""
        pass
    
    def on_training_end(self):
        """Called at the end of training"""
        pass
    
    def on_batch_end(self, state: CallbackState):
        """Called at the end of each batch"""
        pass
    
    def get_state(self) -> Dict[str, Any]:
        """Get callback state for saving"""
        return {}
    
    def load_state(self, state: Dict[str, Any]):
        """Load callback state"""
        pass


class EarlyStopping(BaseCallback):
    """
    Early stopping callback with comprehensive logging.
    Monitors a metric and stops training when it stops improving.
    """
    
    def __init__(
        self,
        monitor: str = 'val_loss',
        patience: int = 10,
        min_delta: float = 0.0,
        mode: str = 'min',
        restore_best_weights: bool = True,
        verbose: bool = True,
        baseline: Optional[float] = None,
        save_best_model_path: Optional[str] = None
    ):
        super().__init__("EarlyStopping")
        
        self.monitor = monitor
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.restore_best_weights = restore_best_weights
        self.verbose = verbose
        self.baseline = baseline
        self.save_best_model_path = save_best_model_path
        
        # Internal state
        self.wait = 0
        self.stopped_epoch = 0
        self.best_weights = None
        self.best_epoch = 0
        self.best_step = 0
        
        if mode == 'min':
            self.monitor_op = np.less
            self.best = np.inf if baseline is None else baseline
        elif mode == 'max':
            self.monitor_op = np.greater
            self.best = -np.inf if baseline is None else baseline
        else:
            raise ValueError(f"Mode must be 'min' or 'max', got {mode}")
        
        # History
        self.history = []
        
        self.logger.info(f"Early stopping initialized:")
        self.logger.info(f"  Monitor: {monitor} ({mode})")
        self.logger.info(f"  Patience: {patience}")
        self.logger.info(f"  Min delta: {min_delta}")
        
    def on_training_start(self):
        """Reset state at training start"""
        self.wait = 0
        self.stopped_epoch = 0
        self.best_weights = None
        self.history = []
        self.logger.info("Early stopping monitoring started")
    
    def on_epoch_end(self, state: CallbackState) -> bool:
        """Check early stopping condition"""
        # Get monitored metric value
        current_value = None
        
        if state.val_metrics and self.monitor in state.val_metrics:
            current_value = state.val_metrics[self.monitor]
        elif state.train_metrics and self.monitor in state.train_metrics:
            current_value = state.train_metrics[self.monitor]
        elif self.monitor == 'val_loss' and state.val_loss is not None:
            current_value = state.val_loss
        elif self.monitor == 'train_loss':
            current_value = state.train_loss
        
        if current_value is None:
            self.logger.warning(f"Monitored metric '{self.monitor}' not found in state")
            return False
        
        # Check for improvement
        if self.monitor_op(current_value - self.min_delta, self.best):
            self.best = current_value
            self.wait = 0
            self.best_epoch = state.epoch
            self.best_step = state.step
            
            # Save best model weights
            if self.restore_best_weights and state.model_state_dict:
                self.best_weights = {k: v.clone() for k, v in state.model_state_dict.items()}
            
            # Save best model to file
            if self.save_best_model_path and state.model_state_dict:
                try:
                    torch.save({
                        'epoch': state.epoch,
                        'step': state.step,
                        'model_state_dict': state.model_state_dict,
                        'optimizer_state_dict': state.optimizer_state_dict,
                        'best_metric': current_value,
                        'monitor': self.monitor
                    }, self.save_best_model_path)
                    self.logger.info(f"Best model saved to {self.save_best_model_path}")
                except Exception as e:
                    self.logger.error(f"Failed to save best model: {e}")
            
            if self.verbose:
                self.logger.info(
                    f"🏆 Best {self.monitor}: {current_value:.6f} "
                    f"(epoch {state.epoch}, patience reset)"
                )
        else:
            self.wait += 1
            if self.verbose:
                self.logger.info(
                    f"Early stopping: {self.monitor}={current_value:.6f} "
                    f"(patience: {self.wait}/{self.patience})"
                )
        
        # Record history
        self.history.append({
            'epoch': state.epoch,
            'step': state.step,
            'monitored_value': current_value,
            'best_value': self.best,
            'wait': self.wait,
            'timestamp': state.timestamp
        })
        
        # Check if we should stop
        if self.wait >= self.patience:
            self.stopped_epoch = state.epoch
            if self.verbose:
                self.logger.info(
                    f"⏹️ Early stopping triggered at epoch {state.epoch}! "
                    f"Best {self.monitor}: {self.best:.6f} (epoch {self.best_epoch})"
                )
            return True
        
        return False
    
    def on_training_end(self):
        """Log final early stopping stats"""
        if self.stopped_epoch > 0:
            self.logger.info(f"Early stopping summary:")
            self.logger.info(f"  Stopped at epoch: {self.stopped_epoch}")
            self.logger.info(f"  Best {self.monitor}: {self.best:.6f} (epoch {self.best_epoch})")
            self.logger.info(f"  Total patience used: {self.patience}")
        else:
            self.logger.info("Training completed without early stopping")
    
    def get_best_weights(self):
        """Get the best model weights"""
        return self.best_weights
    
    def get_state(self) -> Dict[str, Any]:
        """Get callback state for saving"""
        return {
            'wait': self.wait,
            'best': self.best,
            'best_epoch': self.best_epoch,
            'best_step': self.best_step,
            'stopped_epoch': self.stopped_epoch,
            'history': self.history
        }
    
    def load_state(self, state: Dict[str, Any]):
        """Load callback state"""
        self.wait = state.get('wait', 0)
        self.best = state.get('best', np.inf if self.mode == 'min' else -np.inf)
        self.best_epoch = state.get('best_epoch', 0)
        self.best_step = state.get('best_step', 0)
        self.stopped_epoch = state.get('stopped_epoch', 0)
        self.history = state.get('history', [])


class ReduceLROnPlateau(BaseCallback):
    """
    Learning rate reduction callback with comprehensive logging.
    Reduces learning rate when a metric has stopped improving.
    """
    
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        monitor: str = 'val_loss',
        factor: float = 0.1,
        patience: int = 5,
        verbose: bool = True,
        mode: str = 'min',
        min_delta: float = 1e-4,
        cooldown: int = 0,
        min_lr: float = 0,
        eps: float = 1e-8
    ):
        super().__init__("ReduceLROnPlateau")
        
        self.optimizer = optimizer
        self.monitor = monitor
        self.factor = factor
        self.patience = patience
        self.verbose = verbose
        self.mode = mode
        self.min_delta = min_delta
        self.cooldown = cooldown
        self.min_lr = min_lr
        self.eps = eps
        
        # Internal state
        self.wait = 0
        self.cooldown_counter = 0
        self.num_bad_epochs = 0
        self.mode_worse = None
        
        if mode == 'min':
            self.monitor_op = lambda a, b: np.less(a, b - min_delta)
            self.best = np.inf
            self.mode_worse = np.inf
        elif mode == 'max':
            self.monitor_op = lambda a, b: np.greater(a, b + min_delta)
            self.best = -np.inf
            self.mode_worse = -np.inf
        else:
            raise ValueError(f"Mode must be 'min' or 'max', got {mode}")
        
        # History
        self.lr_history = []
        self.reductions = []
        
        self.logger.info(f"ReduceLROnPlateau initialized:")
        self.logger.info(f"  Monitor: {monitor} ({mode})")
        self.logger.info(f"  Factor: {factor}, Patience: {patience}")
        self.logger.info(f"  Min LR: {min_lr}, Min delta: {min_delta}")
    
    def on_training_start(self):
        """Reset state at training start"""
        self.wait = 0
        self.cooldown_counter = 0
        self.num_bad_epochs = 0
        self.best = np.inf if self.mode == 'min' else -np.inf
        self.lr_history = []
        self.reductions = []
        
        # Log initial learning rates
        current_lrs = [group['lr'] for group in self.optimizer.param_groups]
        self.logger.info(f"Initial learning rates: {current_lrs}")
    
    def on_epoch_end(self, state: CallbackState) -> bool:
        """Check if learning rate should be reduced"""
        # Get monitored metric value
        current_value = None
        
        if state.val_metrics and self.monitor in state.val_metrics:
            current_value = state.val_metrics[self.monitor]
        elif state.train_metrics and self.monitor in state.train_metrics:
            current_value = state.train_metrics[self.monitor]
        elif self.monitor == 'val_loss' and state.val_loss is not None:
            current_value = state.val_loss
        elif self.monitor == 'train_loss':
            current_value = state.train_loss
        
        if current_value is None:
            self.logger.warning(f"Monitored metric '{self.monitor}' not found in state")
            return False
        
        # Record current learning rates
        current_lrs = [group['lr'] for group in self.optimizer.param_groups]
        self.lr_history.append({
            'epoch': state.epoch,
            'learning_rates': current_lrs.copy(),
            'monitored_value': current_value,
            'timestamp': state.timestamp
        })
        
        if self.in_cooldown():
            self.cooldown_counter -= 1
            return False
        
        # Check for improvement
        if self.monitor_op(current_value, self.best):
            self.best = current_value
            self.num_bad_epochs = 0
        else:
            self.num_bad_epochs += 1
        
        if self.num_bad_epochs > self.patience:
            self.reduce_lr(state.epoch, current_value)
            self.cooldown_counter = self.cooldown
            self.num_bad_epochs = 0
        
        return False  # Never stop training
    
    def in_cooldown(self):
        """Check if we're in cooldown period"""
        return self.cooldown_counter > 0
    
    def reduce_lr(self, epoch: int, current_value: float):
        """Reduce learning rate"""
        old_lrs = [group['lr'] for group in self.optimizer.param_groups]
        new_lrs = []
        
        for group in self.optimizer.param_groups:
            old_lr = group['lr']
            new_lr = max(old_lr * self.factor, self.min_lr)
            if old_lr - new_lr > self.eps:
                group['lr'] = new_lr
            new_lrs.append(group['lr'])
        
        # Log the reduction
        reduction_info = {
            'epoch': epoch,
            'monitored_value': current_value,
            'old_lrs': old_lrs,
            'new_lrs': new_lrs,
            'factor': self.factor,
            'timestamp': datetime.now().isoformat()
        }
        
        self.reductions.append(reduction_info)
        
        if self.verbose:
            self.logger.info(
                f"📉 Learning rate reduced at epoch {epoch}:"
            )
            for i, (old_lr, new_lr) in enumerate(zip(old_lrs, new_lrs)):
                self.logger.info(f"  Group {i}: {old_lr:.2e} → {new_lr:.2e}")
            self.logger.info(f"  Reason: {self.monitor}={current_value:.6f} (no improvement for {self.patience} epochs)")
    
    def on_training_end(self):
        """Log final learning rate schedule summary"""
        self.logger.info("Learning rate schedule summary:")
        self.logger.info(f"  Total reductions: {len(self.reductions)}")
        
        if self.lr_history:
            initial_lrs = self.lr_history[0]['learning_rates']
            final_lrs = self.lr_history[-1]['learning_rates']
            
            self.logger.info(f"  Initial LRs: {initial_lrs}")
            self.logger.info(f"  Final LRs: {final_lrs}")
            
            for i, (init_lr, final_lr) in enumerate(zip(initial_lrs, final_lrs)):
                if init_lr > 0:
                    reduction_ratio = final_lr / init_lr
                    self.logger.info(f"  Group {i} reduction: {reduction_ratio:.6f}x")
    
    def get_lr_history(self) -> List[Dict[str, Any]]:
        """Get learning rate history"""
        return self.lr_history
    
    def get_reduction_history(self) -> List[Dict[str, Any]]:
        """Get learning rate reduction history"""
        return self.reductions
    
    def get_state(self) -> Dict[str, Any]:
        """Get callback state for saving"""
        return {
            'wait': self.wait,
            'cooldown_counter': self.cooldown_counter,
            'num_bad_epochs': self.num_bad_epochs,
            'best': self.best,
            'lr_history': self.lr_history,
            'reductions': self.reductions
        }
    
    def load_state(self, state: Dict[str, Any]):
        """Load callback state"""
        self.wait = state.get('wait', 0)
        self.cooldown_counter = state.get('cooldown_counter', 0)
        self.num_bad_epochs = state.get('num_bad_epochs', 0)
        self.best = state.get('best', np.inf if self.mode == 'min' else -np.inf)
        self.lr_history = state.get('lr_history', [])
        self.reductions = state.get('reductions', [])


class MetricTracker(BaseCallback):
    """
    Tracks and logs various training metrics over time.
    Provides statistical analysis and trend detection.
    """
    
    def __init__(
        self,
        metrics_to_track: Optional[List[str]] = None,
        window_size: int = 10,
        detect_plateaus: bool = True,
        plateau_threshold: float = 0.01,
        save_history: bool = True,
        history_file: Optional[str] = None
    ):
        super().__init__("MetricTracker")
        
        self.metrics_to_track = metrics_to_track or ['train_loss', 'val_loss']
        self.window_size = window_size
        self.detect_plateaus = detect_plateaus
        self.plateau_threshold = plateau_threshold
        self.save_history = save_history
        self.history_file = history_file or "metric_history.json"
        
        # Metric storage
        self.metric_history = {metric: [] for metric in self.metrics_to_track}
        self.epoch_stats = []
        self.plateau_warnings = []
        
        self.logger.info(f"Metric tracker initialized for: {self.metrics_to_track}")
    
    def on_epoch_end(self, state: CallbackState) -> bool:
        """Track metrics at epoch end"""
        current_metrics = {}
        
        # Collect metrics from state
        if 'train_loss' in self.metrics_to_track:
            current_metrics['train_loss'] = state.train_loss
        
        if 'val_loss' in self.metrics_to_track and state.val_loss is not None:
            current_metrics['val_loss'] = state.val_loss
        
        if state.train_metrics:
            for metric in self.metrics_to_track:
                if metric in state.train_metrics:
                    current_metrics[metric] = state.train_metrics[metric]
        
        if state.val_metrics:
            for metric in self.metrics_to_track:
                if metric in state.val_metrics:
                    current_metrics[metric] = state.val_metrics[metric]
        
        # Store metrics
        epoch_data = {
            'epoch': state.epoch,
            'step': state.step,
            'timestamp': state.timestamp,
            'metrics': current_metrics
        }
        
        self.epoch_stats.append(epoch_data)
        
        # Update metric history
        for metric, value in current_metrics.items():
            if metric in self.metric_history:
                self.metric_history[metric].append(value)
        
        # Detect plateaus
        if self.detect_plateaus:
            self._check_for_plateaus(state.epoch, current_metrics)
        
        # Log statistics periodically
        if state.epoch % 10 == 0:
            self._log_statistics(state.epoch)
        
        # Save history
        if self.save_history:
            self._save_history()
        
        return False
    
    def _check_for_plateaus(self, epoch: int, current_metrics: Dict[str, float]):
        """Check for metric plateaus"""
        for metric, history in self.metric_history.items():
            if len(history) >= self.window_size:
                recent_values = history[-self.window_size:]
                
                # Calculate coefficient of variation
                mean_val = np.mean(recent_values)
                std_val = np.std(recent_values)
                
                if mean_val != 0:
                    cv = std_val / abs(mean_val)
                    
                    if cv < self.plateau_threshold:
                        warning = {
                            'epoch': epoch,
                            'metric': metric,
                            'cv': cv,
                            'mean': mean_val,
                            'std': std_val,
                            'window_size': self.window_size,
                            'timestamp': datetime.now().isoformat()
                        }
                        
                        self.plateau_warnings.append(warning)
                        
                        self.logger.warning(
                            f"⚠️ Plateau detected for {metric} at epoch {epoch}: "
                            f"CV={cv:.6f} over last {self.window_size} epochs"
                        )
    
    def _log_statistics(self, epoch: int):
        """Log metric statistics"""
        self.logger.info(f"📊 Metric statistics at epoch {epoch}:")
        
        for metric, history in self.metric_history.items():
            if history:
                current = history[-1]
                mean_val = np.mean(history)
                std_val = np.std(history)
                min_val = np.min(history)
                max_val = np.max(history)
                
                # Trend over last 5 epochs
                if len(history) >= 5:
                    recent_trend = np.polyfit(range(5), history[-5:], 1)[0]
                    trend_str = "↗️" if recent_trend > 0 else "↘️" if recent_trend < 0 else "➡️"
                else:
                    trend_str = "—"
                
                self.logger.info(
                    f"  {metric}: {current:.6f} {trend_str} "
                    f"(μ={mean_val:.6f}, σ={std_val:.6f}, range=[{min_val:.6f}, {max_val:.6f}])"
                )
    
    def _save_history(self):
        """Save metric history to file"""
        try:
            history_data = {
                'metric_history': {k: v for k, v in self.metric_history.items()},
                'epoch_stats': self.epoch_stats,
                'plateau_warnings': self.plateau_warnings,
                'metadata': {
                    'window_size': self.window_size,
                    'plateau_threshold': self.plateau_threshold,
                    'last_updated': datetime.now().isoformat()
                }
            }
            
            with open(self.history_file, 'w') as f:
                json.dump(history_data, f, indent=2, default=str)
                
        except Exception as e:
            self.logger.error(f"Failed to save metric history: {e}")
    
    def get_metric_summary(self) -> Dict[str, Any]:
        """Get comprehensive metric summary"""
        summary = {
            'total_epochs': len(self.epoch_stats),
            'plateau_warnings': len(self.plateau_warnings),
            'metrics': {}
        }
        
        for metric, history in self.metric_history.items():
            if history:
                summary['metrics'][metric] = {
                    'count': len(history),
                    'current': history[-1],
                    'best': min(history) if 'loss' in metric else max(history),
                    'worst': max(history) if 'loss' in metric else min(history),
                    'mean': float(np.mean(history)),
                    'std': float(np.std(history)),
                    'trend': float(np.polyfit(range(len(history)), history, 1)[0]) if len(history) > 1 else 0.0
                }
        
        return summary
    
    def get_state(self) -> Dict[str, Any]:
        """Get callback state for saving"""
        return {
            'metric_history': self.metric_history,
            'epoch_stats': self.epoch_stats,
            'plateau_warnings': self.plateau_warnings
        }
    
    def load_state(self, state: Dict[str, Any]):
        """Load callback state"""
        self.metric_history = state.get('metric_history', {})
        self.epoch_stats = state.get('epoch_stats', [])
        self.plateau_warnings = state.get('plateau_warnings', [])


class CallbackManager:
    """
    Manages multiple training callbacks and coordinates their execution.
    """
    
    def __init__(self, callbacks: List[BaseCallback]):
        self.callbacks = callbacks
        self.logger = logging.getLogger(f"{__name__}.CallbackManager")
        
        self.logger.info(f"Callback manager initialized with {len(callbacks)} callbacks:")
        for cb in callbacks:
            self.logger.info(f"  - {cb.name}")
    
    def on_training_start(self):
        """Call on_training_start for all callbacks"""
        for callback in self.callbacks:
            try:
                callback.on_training_start()
            except Exception as e:
                self.logger.error(f"Error in {callback.name}.on_training_start(): {e}")
    
    def on_training_end(self):
        """Call on_training_end for all callbacks"""
        for callback in self.callbacks:
            try:
                callback.on_training_end()
            except Exception as e:
                self.logger.error(f"Error in {callback.name}.on_training_end(): {e}")
    
    def on_epoch_end(self, state: CallbackState) -> bool:
        """Call on_epoch_end for all callbacks. Return True if any callback wants to stop training."""
        should_stop = False
        
        for callback in self.callbacks:
            try:
                if callback.on_epoch_end(state):
                    should_stop = True
                    self.logger.info(f"Training stop requested by {callback.name}")
            except Exception as e:
                self.logger.error(f"Error in {callback.name}.on_epoch_end(): {e}")
        
        return should_stop
    
    def on_batch_end(self, state: CallbackState):
        """Call on_batch_end for all callbacks"""
        for callback in self.callbacks:
            try:
                callback.on_batch_end(state)
            except Exception as e:
                self.logger.error(f"Error in {callback.name}.on_batch_end(): {e}")
    
    def save_callbacks_state(self, filepath: str):
        """Save all callback states"""
        callback_states = {}
        
        for callback in self.callbacks:
            try:
                callback_states[callback.name] = callback.get_state()
            except Exception as e:
                self.logger.error(f"Error saving state for {callback.name}: {e}")
        
        try:
            with open(filepath, 'w') as f:
                json.dump(callback_states, f, indent=2, default=str)
            
            self.logger.info(f"Callback states saved to {filepath}")
        except Exception as e:
            self.logger.error(f"Failed to save callback states: {e}")
    
    def load_callbacks_state(self, filepath: str):
        """Load all callback states"""
        if not Path(filepath).exists():
            self.logger.warning(f"Callback state file not found: {filepath}")
            return
        
        try:
            with open(filepath, 'r') as f:
                callback_states = json.load(f)
            
            for callback in self.callbacks:
                if callback.name in callback_states:
                    try:
                        callback.load_state(callback_states[callback.name])
                        self.logger.info(f"Loaded state for {callback.name}")
                    except Exception as e:
                        self.logger.error(f"Error loading state for {callback.name}: {e}")
            
        except Exception as e:
            self.logger.error(f"Failed to load callback states: {e}")


# Convenience functions
def create_early_stopping(
    monitor: str = 'val_loss',
    patience: int = 10,
    mode: str = 'min',
    **kwargs
) -> EarlyStopping:
    """Create an early stopping callback with sensible defaults"""
    return EarlyStopping(
        monitor=monitor,
        patience=patience,
        mode=mode,
        **kwargs
    )


def create_lr_scheduler(
    optimizer: torch.optim.Optimizer,
    monitor: str = 'val_loss',
    patience: int = 5,
    factor: float = 0.5,
    **kwargs
) -> ReduceLROnPlateau:
    """Create a learning rate scheduler callback with sensible defaults"""
    return ReduceLROnPlateau(
        optimizer=optimizer,
        monitor=monitor,
        patience=patience,
        factor=factor,
        **kwargs
    )


def create_metric_tracker(
    metrics: Optional[List[str]] = None,
    **kwargs
) -> MetricTracker:
    """Create a metric tracker with sensible defaults"""
    return MetricTracker(
        metrics_to_track=metrics,
        **kwargs
    )


if __name__ == "__main__":
    # Example usage
    if TORCH_AVAILABLE:
        # Create a simple model and optimizer
        model = torch.nn.Linear(10, 1)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        
        # Create callbacks
        callbacks = [
            create_early_stopping(patience=3),
            create_lr_scheduler(optimizer, patience=2),
            create_metric_tracker(['train_loss', 'val_loss'])
        ]
        
        # Create callback manager
        manager = CallbackManager(callbacks)
        
        # Simulate training
        manager.on_training_start()
        
        for epoch in range(10):
            train_loss = 1.0 - epoch * 0.05
            val_loss = train_loss + 0.1 + (0.02 if epoch > 5 else 0)  # Simulate plateau
            
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
                print(f"Training stopped at epoch {epoch}")
                break
        
        manager.on_training_end()
        print("Example training completed!")
    else:
        print("PyTorch not available for example")