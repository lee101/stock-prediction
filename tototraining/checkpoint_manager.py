#!/usr/bin/env python3
"""
Model Checkpoint Management for Toto Training Pipeline
Provides automatic saving/loading of best models, checkpoint rotation, and recovery functionality.
"""

import os
import json
import shutil
import hashlib
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, List, Tuple, Callable
import logging
from dataclasses import dataclass, asdict
from collections import defaultdict
import numpy as np

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None


@dataclass
class CheckpointInfo:
    """Information about a model checkpoint"""
    path: str
    epoch: int
    step: int
    timestamp: str
    metrics: Dict[str, float]
    model_hash: str
    file_size_mb: float
    is_best: bool = False
    tags: Optional[Dict[str, str]] = None
    
    def __post_init__(self):
        if self.tags is None:
            self.tags = {}


class CheckpointManager:
    """
    Comprehensive checkpoint management system for model training.
    Handles automatic saving, best model tracking, checkpoint rotation, and recovery.
    """
    
    def __init__(
        self,
        checkpoint_dir: str = "checkpoints",
        max_checkpoints: int = 5,
        save_best_k: int = 3,
        monitor_metric: str = "val_loss",
        mode: str = "min",  # 'min' for loss, 'max' for accuracy
        save_frequency: int = 1,  # Save every N epochs
        save_on_train_end: bool = True,
        compress_checkpoints: bool = False,
        backup_best_models: bool = True
    ):
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch not available. Cannot use checkpoint manager.")
        
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(exist_ok=True)
        
        self.max_checkpoints = max_checkpoints
        self.save_best_k = save_best_k
        self.monitor_metric = monitor_metric
        self.mode = mode
        self.save_frequency = save_frequency
        self.save_on_train_end = save_on_train_end
        self.compress_checkpoints = compress_checkpoints
        self.backup_best_models = backup_best_models
        
        # Track checkpoints
        self.checkpoints = []  # List of CheckpointInfo
        self.best_checkpoints = []  # List of best CheckpointInfo
        self.best_metric_value = float('inf') if mode == 'min' else float('-inf')
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        
        # Create subdirectories
        (self.checkpoint_dir / "regular").mkdir(exist_ok=True)
        (self.checkpoint_dir / "best").mkdir(exist_ok=True)
        if self.backup_best_models:
            (self.checkpoint_dir / "backup").mkdir(exist_ok=True)
        
        # Load existing checkpoint info
        self._load_checkpoint_registry()
        
        print(f"Checkpoint manager initialized:")
        print(f"  Directory: {self.checkpoint_dir}")
        print(f"  Monitor metric: {self.monitor_metric} ({self.mode})")
        print(f"  Max checkpoints: {self.max_checkpoints}")
        print(f"  Save best K: {self.save_best_k}")
    
    def _is_better(self, current_value: float, best_value: float) -> bool:
        """Check if current metric is better than best"""
        if self.mode == 'min':
            return current_value < best_value
        else:
            return current_value > best_value
    
    def _calculate_file_hash(self, file_path: Path) -> str:
        """Calculate MD5 hash of a file"""
        hash_md5 = hashlib.md5()
        try:
            with open(file_path, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_md5.update(chunk)
            return hash_md5.hexdigest()
        except Exception:
            return "unknown"
    
    def _get_file_size_mb(self, file_path: Path) -> float:
        """Get file size in MB"""
        try:
            return file_path.stat().st_size / (1024 * 1024)
        except Exception:
            return 0.0
    
    def _save_checkpoint_registry(self):
        """Save checkpoint registry to disk"""
        registry_path = self.checkpoint_dir / "checkpoint_registry.json"
        
        registry_data = {
            'regular_checkpoints': [asdict(cp) for cp in self.checkpoints],
            'best_checkpoints': [asdict(cp) for cp in self.best_checkpoints],
            'best_metric_value': self.best_metric_value,
            'monitor_metric': self.monitor_metric,
            'mode': self.mode,
            'last_updated': datetime.now().isoformat()
        }
        
        try:
            with open(registry_path, 'w') as f:
                json.dump(registry_data, f, indent=2)
        except Exception as e:
            self.logger.error(f"Failed to save checkpoint registry: {e}")
    
    def _load_checkpoint_registry(self):
        """Load checkpoint registry from disk"""
        registry_path = self.checkpoint_dir / "checkpoint_registry.json"
        
        if not registry_path.exists():
            return
        
        try:
            with open(registry_path, 'r') as f:
                registry_data = json.load(f)
            
            # Load regular checkpoints
            self.checkpoints = [
                CheckpointInfo(**cp_data) 
                for cp_data in registry_data.get('regular_checkpoints', [])
            ]
            
            # Load best checkpoints
            self.best_checkpoints = [
                CheckpointInfo(**cp_data) 
                for cp_data in registry_data.get('best_checkpoints', [])
            ]
            
            # Load best metric value
            self.best_metric_value = registry_data.get(
                'best_metric_value', 
                float('inf') if self.mode == 'min' else float('-inf')
            )
            
            # Verify checkpoint files exist
            self._verify_checkpoints()
            
            print(f"Loaded checkpoint registry: {len(self.checkpoints)} regular, {len(self.best_checkpoints)} best")
            
        except Exception as e:
            self.logger.error(f"Failed to load checkpoint registry: {e}")
            self.checkpoints = []
            self.best_checkpoints = []
    
    def _verify_checkpoints(self):
        """Verify that checkpoint files exist and remove missing ones"""
        # Verify regular checkpoints
        valid_checkpoints = []
        for cp in self.checkpoints:
            if Path(cp.path).exists():
                valid_checkpoints.append(cp)
            else:
                self.logger.warning(f"Checkpoint file missing: {cp.path}")
        
        self.checkpoints = valid_checkpoints
        
        # Verify best checkpoints
        valid_best_checkpoints = []
        for cp in self.best_checkpoints:
            if Path(cp.path).exists():
                valid_best_checkpoints.append(cp)
            else:
                self.logger.warning(f"Best checkpoint file missing: {cp.path}")
        
        self.best_checkpoints = valid_best_checkpoints
    
    def save_checkpoint(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        epoch: int,
        step: int,
        metrics: Dict[str, float],
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        additional_state: Optional[Dict[str, Any]] = None,
        tags: Optional[Dict[str, str]] = None
    ) -> Optional[CheckpointInfo]:
        """Save a model checkpoint"""
        
        # Check if we should save based on frequency
        if epoch % self.save_frequency != 0 and not self.save_on_train_end:
            return None
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        checkpoint_name = f"checkpoint_epoch_{epoch}_step_{step}_{timestamp}.pth"
        checkpoint_path = self.checkpoint_dir / "regular" / checkpoint_name
        
        # Prepare state dict
        state = {
            'epoch': epoch,
            'step': step,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'metrics': metrics,
            'timestamp': timestamp,
            'monitor_metric': self.monitor_metric,
            'mode': self.mode
        }
        
        if scheduler is not None:
            state['scheduler_state_dict'] = scheduler.state_dict()
        
        if additional_state:
            state['additional_state'] = additional_state
        
        # Save checkpoint
        try:
            if self.compress_checkpoints:
                torch.save(state, checkpoint_path, _use_new_zipfile_serialization=True)
            else:
                torch.save(state, checkpoint_path)
            
            # Calculate file info
            file_hash = self._calculate_file_hash(checkpoint_path)
            file_size_mb = self._get_file_size_mb(checkpoint_path)
            
            # Create checkpoint info
            checkpoint_info = CheckpointInfo(
                path=str(checkpoint_path),
                epoch=epoch,
                step=step,
                timestamp=timestamp,
                metrics=metrics.copy(),
                model_hash=file_hash,
                file_size_mb=file_size_mb,
                is_best=False,
                tags=tags or {}
            )
            
            # Add to regular checkpoints
            self.checkpoints.append(checkpoint_info)
            
            # Handle checkpoint rotation
            self._rotate_checkpoints()
            
            # Check if this is a best checkpoint
            monitor_value = metrics.get(self.monitor_metric)
            if monitor_value is not None:
                self._check_and_save_best(checkpoint_info, monitor_value)
            
            # Save registry
            self._save_checkpoint_registry()
            
            self.logger.info(f"Saved checkpoint: {checkpoint_name}")
            self.logger.info(f"Metrics: {metrics}")
            
            return checkpoint_info
            
        except Exception as e:
            self.logger.error(f"Failed to save checkpoint: {e}")
            if checkpoint_path.exists():
                checkpoint_path.unlink()  # Clean up partial file
            return None
    
    def _rotate_checkpoints(self):
        """Remove old checkpoints to maintain max_checkpoints limit"""
        if len(self.checkpoints) <= self.max_checkpoints:
            return
        
        # Sort by epoch (keep most recent)
        self.checkpoints.sort(key=lambda x: x.epoch)
        
        # Remove oldest checkpoints
        while len(self.checkpoints) > self.max_checkpoints:
            old_checkpoint = self.checkpoints.pop(0)
            try:
                Path(old_checkpoint.path).unlink()
                self.logger.info(f"Removed old checkpoint: {Path(old_checkpoint.path).name}")
            except Exception as e:
                self.logger.error(f"Failed to remove checkpoint {old_checkpoint.path}: {e}")
    
    def _check_and_save_best(self, checkpoint_info: CheckpointInfo, monitor_value: float):
        """Check if checkpoint is among the best and save it"""
        if self._is_better(monitor_value, self.best_metric_value):
            self.best_metric_value = monitor_value
            
            # Create best checkpoint copy
            best_checkpoint_name = f"best_model_epoch_{checkpoint_info.epoch}_{self.monitor_metric}_{monitor_value:.6f}.pth"
            best_checkpoint_path = self.checkpoint_dir / "best" / best_checkpoint_name
            
            try:
                shutil.copy2(checkpoint_info.path, best_checkpoint_path)
                
                # Create best checkpoint info
                best_checkpoint_info = CheckpointInfo(
                    path=str(best_checkpoint_path),
                    epoch=checkpoint_info.epoch,
                    step=checkpoint_info.step,
                    timestamp=checkpoint_info.timestamp,
                    metrics=checkpoint_info.metrics.copy(),
                    model_hash=checkpoint_info.model_hash,
                    file_size_mb=self._get_file_size_mb(best_checkpoint_path),
                    is_best=True,
                    tags=checkpoint_info.tags.copy() if checkpoint_info.tags else {}
                )
                best_checkpoint_info.tags['is_best'] = 'true'
                best_checkpoint_info.tags['best_metric'] = self.monitor_metric
                
                self.best_checkpoints.append(best_checkpoint_info)
                
                # Rotate best checkpoints
                self._rotate_best_checkpoints()
                
                # Backup if enabled
                if self.backup_best_models:
                    self._backup_best_model(best_checkpoint_info)
                
                self.logger.info(f"🏆 NEW BEST MODEL! {self.monitor_metric}={monitor_value:.6f}")
                self.logger.info(f"Saved best model: {best_checkpoint_name}")
                
            except Exception as e:
                self.logger.error(f"Failed to save best checkpoint: {e}")
    
    def _rotate_best_checkpoints(self):
        """Remove old best checkpoints to maintain save_best_k limit"""
        if len(self.best_checkpoints) <= self.save_best_k:
            return
        
        # Sort by metric value (keep best ones)
        if self.mode == 'min':
            self.best_checkpoints.sort(key=lambda x: x.metrics.get(self.monitor_metric, float('inf')))
        else:
            self.best_checkpoints.sort(key=lambda x: x.metrics.get(self.monitor_metric, float('-inf')), reverse=True)
        
        # Remove worst checkpoints
        while len(self.best_checkpoints) > self.save_best_k:
            old_best = self.best_checkpoints.pop()
            try:
                Path(old_best.path).unlink()
                self.logger.info(f"Removed old best checkpoint: {Path(old_best.path).name}")
            except Exception as e:
                self.logger.error(f"Failed to remove best checkpoint {old_best.path}: {e}")
    
    def _backup_best_model(self, checkpoint_info: CheckpointInfo):
        """Create a backup copy of the best model"""
        backup_name = f"backup_{Path(checkpoint_info.path).name}"
        backup_path = self.checkpoint_dir / "backup" / backup_name
        
        try:
            shutil.copy2(checkpoint_info.path, backup_path)
            self.logger.info(f"Created backup: {backup_name}")
        except Exception as e:
            self.logger.error(f"Failed to create backup: {e}")
    
    def load_checkpoint(
        self, 
        checkpoint_path: str, 
        model: torch.nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        device: Optional[str] = None
    ) -> Dict[str, Any]:
        """Load a checkpoint"""
        
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        try:
            # Load checkpoint
            if device:
                checkpoint = torch.load(checkpoint_path, map_location=device)
            else:
                checkpoint = torch.load(checkpoint_path)
            
            # Load model state
            model.load_state_dict(checkpoint['model_state_dict'])
            
            # Load optimizer state
            if optimizer is not None and 'optimizer_state_dict' in checkpoint:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
            # Load scheduler state
            if scheduler is not None and 'scheduler_state_dict' in checkpoint:
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            
            self.logger.info(f"Loaded checkpoint: {checkpoint_path.name}")
            self.logger.info(f"Epoch: {checkpoint.get('epoch', 'unknown')}")
            self.logger.info(f"Metrics: {checkpoint.get('metrics', {})}")
            
            return checkpoint
            
        except Exception as e:
            self.logger.error(f"Failed to load checkpoint {checkpoint_path}: {e}")
            raise
    
    def load_best_checkpoint(
        self, 
        model: torch.nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        device: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """Load the best checkpoint"""
        
        if not self.best_checkpoints:
            self.logger.warning("No best checkpoints available")
            return None
        
        # Get the best checkpoint
        if self.mode == 'min':
            best_checkpoint = min(
                self.best_checkpoints, 
                key=lambda x: x.metrics.get(self.monitor_metric, float('inf'))
            )
        else:
            best_checkpoint = max(
                self.best_checkpoints, 
                key=lambda x: x.metrics.get(self.monitor_metric, float('-inf'))
            )
        
        return self.load_checkpoint(
            best_checkpoint.path, model, optimizer, scheduler, device
        )
    
    def get_checkpoint_summary(self) -> Dict[str, Any]:
        """Get summary of all checkpoints"""
        summary = {
            'total_checkpoints': len(self.checkpoints),
            'best_checkpoints': len(self.best_checkpoints),
            'monitor_metric': self.monitor_metric,
            'mode': self.mode,
            'best_metric_value': self.best_metric_value,
            'total_size_mb': sum(cp.file_size_mb for cp in self.checkpoints + self.best_checkpoints),
            'checkpoints': []
        }
        
        # Add checkpoint details
        all_checkpoints = self.checkpoints + self.best_checkpoints
        for cp in sorted(all_checkpoints, key=lambda x: x.epoch, reverse=True):
            summary['checkpoints'].append({
                'epoch': cp.epoch,
                'step': cp.step,
                'timestamp': cp.timestamp,
                'is_best': cp.is_best,
                'metrics': cp.metrics,
                'file_size_mb': cp.file_size_mb,
                'path': cp.path
            })
        
        return summary
    
    def cleanup_checkpoints(self, keep_best: bool = True, keep_latest: int = 1):
        """Clean up checkpoints (useful for disk space management)"""
        removed_count = 0
        
        # Keep only the latest N regular checkpoints
        if len(self.checkpoints) > keep_latest:
            self.checkpoints.sort(key=lambda x: x.epoch, reverse=True)
            checkpoints_to_remove = self.checkpoints[keep_latest:]
            self.checkpoints = self.checkpoints[:keep_latest]
            
            for cp in checkpoints_to_remove:
                try:
                    Path(cp.path).unlink()
                    removed_count += 1
                except Exception as e:
                    self.logger.error(f"Failed to remove checkpoint {cp.path}: {e}")
        
        # Optionally remove best checkpoints
        if not keep_best:
            for cp in self.best_checkpoints:
                try:
                    Path(cp.path).unlink()
                    removed_count += 1
                except Exception as e:
                    self.logger.error(f"Failed to remove best checkpoint {cp.path}: {e}")
            
            self.best_checkpoints = []
        
        self._save_checkpoint_registry()
        self.logger.info(f"Cleaned up {removed_count} checkpoints")
        
        return removed_count
    
    def export_checkpoint_info(self, output_path: str):
        """Export checkpoint information to JSON"""
        summary = self.get_checkpoint_summary()
        
        try:
            with open(output_path, 'w') as f:
                json.dump(summary, f, indent=2, default=str)
            
            self.logger.info(f"Exported checkpoint info to: {output_path}")
        except Exception as e:
            self.logger.error(f"Failed to export checkpoint info: {e}")


# Convenience function for quick checkpoint manager setup
def create_checkpoint_manager(
    checkpoint_dir: str = "checkpoints",
    monitor_metric: str = "val_loss",
    mode: str = "min",
    **kwargs
) -> CheckpointManager:
    """Create a checkpoint manager with sensible defaults"""
    return CheckpointManager(
        checkpoint_dir=checkpoint_dir,
        monitor_metric=monitor_metric,
        mode=mode,
        **kwargs
    )


if __name__ == "__main__":
    # Example usage
    if TORCH_AVAILABLE:
        # Create a simple model for testing
        model = torch.nn.Linear(10, 1)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        
        # Create checkpoint manager
        manager = create_checkpoint_manager("test_checkpoints")
        
        # Simulate training with checkpoints
        for epoch in range(5):
            train_loss = 1.0 - epoch * 0.1
            val_loss = train_loss + 0.05
            
            metrics = {
                'train_loss': train_loss,
                'val_loss': val_loss,
                'accuracy': 0.8 + epoch * 0.05
            }
            
            manager.save_checkpoint(
                model, optimizer, epoch, epoch * 100, metrics,
                tags={'experiment': 'test'}
            )
        
        # Print summary
        summary = manager.get_checkpoint_summary()
        print(json.dumps(summary, indent=2, default=str))
    else:
        print("PyTorch not available for example")