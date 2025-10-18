#!/usr/bin/env python3
"""
Comprehensive Toto Training Pipeline

This module provides a complete training framework for the Datadog Toto model with:
- Multi-GPU distributed training
- Mixed precision training
- Gradient clipping and memory optimization
- Checkpoint management and recovery
- Learning rate scheduling
- Validation metrics and evaluation
- Configuration management
- Integration with existing OHLC dataloader
"""

import os
import sys
import json
import shutil
import logging
import warnings
import contextlib
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Union, Any, Sequence
from dataclasses import dataclass, asdict
from collections import defaultdict
import random
import time
import math

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.cuda.amp import GradScaler
from torch.utils.data import DataLoader, Dataset
from torch.optim.lr_scheduler import ReduceLROnPlateau, OneCycleLR

from traininglib.compile_wrap import maybe_compile
from traininglib.optim_factory import make_optimizer
from traininglib.runtime_flags import bf16_supported, enable_fast_kernels
from traininglib.schedules import WarmupCosine
from traininglib.prof import maybe_profile
from traininglib.prefetch import CudaPrefetcher
from traininglib.ema import EMA
from traininglib.losses import huber_loss, heteroscedastic_gaussian_nll, pinball_loss
from hftraining.metrics import crps_from_quantiles, dm_test

# Add the toto directory to sys.path
toto_path = Path(__file__).parent.parent / "toto" / "toto"
sys.path.insert(0, str(toto_path))
# Also add the direct toto module path
sys.path.insert(0, str(Path(__file__).parent.parent / "toto"))

try:
    from toto.model.toto import Toto
    from toto.model.scaler import StdMeanScaler
    from toto.data.util.dataset import MaskedTimeseries
except ImportError as e:
    try:
        # Alternative import paths
        from model.toto import Toto
        from model.scaler import StdMeanScaler
        from data.util.dataset import MaskedTimeseries
    except ImportError as e2:
        warnings.warn(f"Failed to import Toto model components: {e}, {e2}")
        # Create minimal fallback for testing
        from typing import NamedTuple
        class Toto(nn.Module):
            def __init__(self, **kwargs):
                super().__init__()
                self.model = nn.Identity()

        class MaskedTimeseries(NamedTuple):
            series: torch.Tensor
            padding_mask: torch.Tensor
            id_mask: torch.Tensor
            timestamp_seconds: torch.Tensor
            time_interval_seconds: torch.Tensor

# Import our dataloader
try:
    from .toto_ohlc_dataloader import TotoOHLCDataLoader, DataLoaderConfig, TotoBatchSample
except ImportError:
    try:
        from toto_ohlc_dataloader import TotoOHLCDataLoader, DataLoaderConfig, TotoBatchSample  # type: ignore
    except ImportError:
        warnings.warn("TotoOHLCDataLoader not found, creating minimal fallback")
        class TotoOHLCDataLoader:
            def __init__(self, config):
                self.config = config
            def prepare_dataloaders(self):
                return {}

        @dataclass
        class DataLoaderConfig:
            pass

        class TotoBatchSample:  # type: ignore
            pass

try:
    from tensorboard_monitor import TensorBoardMonitor
except ImportError:
    TensorBoardMonitor = None


@dataclass
class TrainerConfig:
    """Configuration for TotoTrainer"""
    
    # Model parameters
    patch_size: int = 12
    stride: int = 6
    embed_dim: int = 256
    num_layers: int = 8
    num_heads: int = 8
    mlp_hidden_dim: int = 512
    dropout: float = 0.1
    spacewise_every_n_layers: int = 2
    scaler_cls: str = "model.scaler.StdMeanScaler"
    output_distribution_classes: List[str] = None
    
    # Training parameters
    learning_rate: float = 1e-4
    min_lr: float = 0.0
    weight_decay: float = 0.01
    batch_size: int = 32
    device_batch_size: Optional[int] = None
    global_batch_size: Optional[int] = None
    accumulation_steps: int = 1
    max_epochs: int = 100
    warmup_epochs: int = 10
    warmup_steps: Optional[int] = None
    
    # Optimization
    optimizer: str = "adamw"  # "adamw", "adam", "sgd"
    scheduler: str = "cosine"  # "cosine", "plateau", "onecycle", "none"
    optimizer_betas: Tuple[float, float] = (0.9, 0.95)
    optimizer_eps: float = 1e-8
    gradient_clip_val: float = 1.0
    use_mixed_precision: bool = True
    compile: bool = True
    require_gpu: bool = False
    use_cuda_graphs: bool = False
    cuda_graph_warmup: int = 3

    # Distributed training
    distributed: bool = False
    world_size: int = 1
    rank: int = 0
    local_rank: int = 0
    dist_backend: str = "nccl"
    dist_url: str = "env://"
    
    # Checkpointing
    save_dir: str = "checkpoints"
    save_every_n_epochs: int = 5
    keep_last_n_checkpoints: int = 3
    best_k_checkpoints: int = 1
    resume_from_checkpoint: Optional[str] = None
    pretrained_model_id: Optional[str] = None
    pretrained_checkpoint: Optional[str] = None
    pretrained_torch_dtype: Optional[str] = None
    
    # Validation and evaluation
    validation_frequency: int = 1  # Validate every N epochs
    early_stopping_patience: int = 10
    early_stopping_delta: float = 1e-4
    
    # Metrics
    compute_train_metrics: bool = True
    compute_val_metrics: bool = True
    metrics_log_frequency: int = 100  # Log metrics every N batches
    
    # Memory optimization
    gradient_checkpointing: bool = False
    memory_efficient_attention: bool = True
    pin_memory: bool = True
    freeze_backbone: bool = False
    trainable_param_substrings: Optional[List[str]] = None
    prefetch_to_device: bool = True

    # Logging
    log_level: str = "INFO"
    log_file: Optional[str] = "training.log"
    wandb_project: Optional[str] = None
    experiment_name: Optional[str] = None
    log_to_tensorboard: bool = True
    tensorboard_log_dir: str = "tensorboard_logs"
    
    # Export
    export_pretrained_dir: Optional[str] = None
    export_on_best: bool = True
    
    # Random seed
    random_seed: int = 42

    # Loss & EMA
    loss_type: str = "huber"  # "huber", "mse", "heteroscedastic", "quantile"
    huber_delta: float = 0.01
    quantile_levels: Optional[List[float]] = None
    ema_decay: Optional[float] = 0.999
    ema_eval: bool = True

    # Profiling
    profile: bool = False
    profile_log_dir: str = "runs/prof"
    
    def __post_init__(self):
        if self.output_distribution_classes is None:
            self.output_distribution_classes = ["model.distribution.StudentTOutput"]

        if self.experiment_name is None:
            self.experiment_name = f"toto_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        # Create save directory
        Path(self.save_dir).mkdir(parents=True, exist_ok=True)

        if self.log_to_tensorboard and self.tensorboard_log_dir:
            Path(self.tensorboard_log_dir).mkdir(parents=True, exist_ok=True)

        if self.device_batch_size is not None and self.device_batch_size <= 0:
            raise ValueError("device_batch_size must be positive when provided.")
        if self.global_batch_size is not None and self.global_batch_size <= 0:
            raise ValueError("global_batch_size must be positive when provided.")
        if self.ema_decay is not None and not (0.0 < self.ema_decay < 1.0):
            raise ValueError("ema_decay must lie in (0, 1) when enabled.")
        if self.cuda_graph_warmup < 0:
            raise ValueError("cuda_graph_warmup must be non-negative.")

        valid_losses = {"huber", "mse", "heteroscedastic", "quantile"}
        self.loss_type = self.loss_type.lower()
        if self.loss_type not in valid_losses:
            raise ValueError(f"Unsupported loss_type '{self.loss_type}'.")
        if self.quantile_levels is None:
            self.quantile_levels = [0.1, 0.5, 0.9]

        if self.export_pretrained_dir is None:
            self.export_pretrained_dir = str(Path(self.save_dir) / "hf_export")
        Path(self.export_pretrained_dir).mkdir(parents=True, exist_ok=True)

        self.best_k_checkpoints = max(1, int(self.best_k_checkpoints))

        if self.pretrained_model_id and self.pretrained_checkpoint:
            raise ValueError("Specify at most one of pretrained_model_id or pretrained_checkpoint.")

        if self.freeze_backbone and not self.trainable_param_substrings:
            self.trainable_param_substrings = [
                "output_distribution",
                "loc_proj",
                "scale_proj",
                "df",
            ]
    
    def save(self, path: str):
        """Save configuration to JSON file"""
        with open(path, 'w') as f:
            json.dump(asdict(self), f, indent=2)
    
    @classmethod
    def load(cls, path: str):
        """Load configuration from JSON file"""
        with open(path, 'r') as f:
            config_dict = json.load(f)
        return cls(**config_dict)


class MetricsTracker:
    """Tracks and computes training/validation metrics"""
    
    def __init__(self):
        self.reset()

    def reset(self):
        """Reset all metrics"""
        self.losses = []
        self.predictions = []  # percent predictions
        self.targets = []      # percent targets
        self.price_predictions = []
        self.price_targets = []
        self.batch_times = []
        self.learning_rates = []
        self.price_mae_samples: List[np.ndarray] = []
        self.naive_mae_samples: List[np.ndarray] = []
        self.crps_samples: List[float] = []
        self.quantile_levels: Optional[Sequence[float]] = None

    def update(
        self,
        loss: float,
        predictions: torch.Tensor | None = None,
        targets: torch.Tensor | None = None,
        price_predictions: torch.Tensor | None = None,
        price_targets: torch.Tensor | None = None,
        batch_time: float | None = None,
        learning_rate: float | None = None,
        prev_close: torch.Tensor | None = None,
        quantile_predictions: torch.Tensor | None = None,
        quantile_levels: Sequence[float] | None = None,
    ):
        """Update metrics with new batch data"""
        self.losses.append(loss)

        if predictions is not None and targets is not None:
            self.predictions.append(predictions.detach().cpu())
            self.targets.append(targets.detach().cpu())

        targets_cpu = None
        if price_predictions is not None and price_targets is not None:
            preds_cpu = price_predictions.detach().cpu()
            targets_cpu = price_targets.detach().cpu()
            if preds_cpu.ndim == 3 and preds_cpu.shape[1] == 1:
                preds_cpu = preds_cpu[:, 0, :]
            if targets_cpu.ndim == 3 and targets_cpu.shape[1] == 1:
                targets_cpu = targets_cpu[:, 0, :]
            self.price_predictions.append(preds_cpu)
            self.price_targets.append(targets_cpu)
            mae_batch = torch.mean(torch.abs(preds_cpu - targets_cpu), dim=1)
            self.price_mae_samples.append(mae_batch.numpy())
            if prev_close is not None:
                base = prev_close.detach().cpu()
                if base.ndim == 1:
                    base = base.unsqueeze(-1).expand_as(targets_cpu)
                elif base.ndim == 2 and base.shape[1] != targets_cpu.shape[1]:
                    base = base[:, -1:].expand_as(targets_cpu)
                elif base.ndim == 3 and base.shape[1] == 1:
                    base = base[:, 0, :]
                if base.ndim == 2:
                    naive_mae = torch.mean(torch.abs(base - targets_cpu), dim=1)
                    self.naive_mae_samples.append(naive_mae.numpy())

        if batch_time is not None:
            self.batch_times.append(batch_time)

        if learning_rate is not None:
            self.learning_rates.append(learning_rate)

        if (
            targets_cpu is not None
            and quantile_predictions is not None
            and quantile_levels is not None
        ):
            q_pred = quantile_predictions.detach().cpu()
            if q_pred.ndim == 4 and q_pred.shape[1] == 1:
                q_pred = q_pred[:, 0, :, :]
            if q_pred.ndim == 3 and q_pred.shape[1] != targets_cpu.shape[1] and q_pred.shape[2] == targets_cpu.shape[1]:
                q_pred = q_pred.transpose(1, 2)
            taus = torch.tensor(list(quantile_levels), dtype=targets_cpu.dtype)
            try:
                crps_val = crps_from_quantiles(targets_cpu, q_pred, taus)
                self.crps_samples.append(float(crps_val))
                self.quantile_levels = quantile_levels
            except Exception:
                # Ignore numerical issues; CRPS simply not logged for this batch.
                pass

    def compute_metrics(self) -> Dict[str, float]:
        """Compute and return all metrics"""
        metrics: Dict[str, float] = {}

        if self.losses:
            metrics['loss'] = float(np.mean(self.losses))
            metrics['loss_std'] = float(np.std(self.losses))

        if self.predictions and self.targets:
            all_preds = torch.cat(self.predictions, dim=0)
            all_targets = torch.cat(self.targets, dim=0)
            mse = F.mse_loss(all_preds, all_targets).item()
            mae = F.l1_loss(all_preds, all_targets).item()
            mape = torch.mean(torch.abs((all_targets - all_preds) / (all_targets.abs() + 1e-8))) * 100
            ss_res = torch.sum((all_targets - all_preds) ** 2)
            ss_tot = torch.sum((all_targets - torch.mean(all_targets)) ** 2)
            r2 = (1 - ss_res / ss_tot).item() if ss_tot > 0 else float('nan')
            metrics.update({
                'pct_mse': mse,
                'pct_rmse': math.sqrt(mse),
                'pct_mae': mae,
                'pct_mape': mape.item(),
                'pct_r2': r2,
            })

        if self.price_predictions and self.price_targets:
            price_preds = torch.cat(self.price_predictions, dim=0)
            price_targets = torch.cat(self.price_targets, dim=0)
            price_mse = F.mse_loss(price_preds, price_targets).item()
            price_mae = F.l1_loss(price_preds, price_targets).item()
            metrics.update({
                'price_mse': price_mse,
                'price_rmse': math.sqrt(price_mse),
                'price_mae': price_mae,
            })

        if self.price_mae_samples:
            mae_array = np.concatenate(self.price_mae_samples)
            metrics['price_mae'] = float(np.mean(mae_array))
            if self.naive_mae_samples:
                naive_array = np.concatenate(self.naive_mae_samples)
                metrics['naive_mae'] = float(np.mean(naive_array))
                dm_stat, dm_p = dm_test(mae_array, naive_array)
                metrics['dm_stat_vs_naive'] = float(dm_stat)
                metrics['dm_pvalue_vs_naive'] = float(dm_p)

        if self.crps_samples:
            metrics['price_crps'] = float(np.mean(self.crps_samples))

        if self.batch_times:
            metrics['batch_time_mean'] = float(np.mean(self.batch_times))
            metrics['batch_time_std'] = float(np.std(self.batch_times))
            metrics['steps_per_sec'] = len(self.batch_times) / sum(self.batch_times)

        if self.learning_rates:
            metrics['learning_rate'] = self.learning_rates[-1]

        return metrics


class CheckpointManager:
    """Manages model checkpoints with automatic cleanup"""
    
    def __init__(self, save_dir: str, keep_last_n: int = 3, best_k: int = 1):
        self.save_dir = Path(save_dir)
        self.keep_last_n = keep_last_n
        self.best_k = max(1, best_k)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.best_dir = self.save_dir / "best"
        self.best_dir.mkdir(parents=True, exist_ok=True)
        self.best_records_path = self.save_dir / "best_records.json"
    
    def save_checkpoint(self, 
                       model: nn.Module,
                       optimizer: torch.optim.Optimizer,
                       scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
                       scaler: Optional[GradScaler],
                       epoch: int,
                       best_val_loss: float,
                       metrics: Dict[str, float],
                       config: TrainerConfig,
                       dataloader_config: Optional[DataLoaderConfig] = None,
                       is_best: bool = False,
                       val_loss: Optional[float] = None):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.module.state_dict() if hasattr(model, 'module') else model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
            'scaler_state_dict': scaler.state_dict() if scaler else None,
            'best_val_loss': best_val_loss,
            'metrics': metrics,
            'config': asdict(config),
            'dataloader_config': asdict(dataloader_config) if dataloader_config else None,
            'timestamp': datetime.now().isoformat(),
            'val_loss': val_loss
        }
        
        # Save regular checkpoint
        checkpoint_path = self.save_dir / f"checkpoint_epoch_{epoch}.pt"
        torch.save(checkpoint, checkpoint_path)
        
        # Save best model (legacy single-best)
        if is_best:
            best_path = self.save_dir / "best_model.pt"
            torch.save(checkpoint, best_path)
        
        # Save latest
        latest_path = self.save_dir / "latest.pt"
        torch.save(checkpoint, latest_path)
        
        # Update best-k registry
        if val_loss is not None:
            self._update_best_checkpoints(checkpoint_path, float(val_loss))
        
        # Cleanup old checkpoints
        self._cleanup_checkpoints()
        
        return checkpoint_path
    
    def _load_best_records(self) -> List[Dict[str, Any]]:
        if self.best_records_path.exists():
            try:
                with self.best_records_path.open('r') as fp:
                    records = json.load(fp)
                if isinstance(records, list):
                    return records
            except Exception:
                pass
        return []
    
    def _save_best_records(self, records: List[Dict[str, Any]]) -> None:
        with self.best_records_path.open('w') as fp:
            json.dump(records, fp, indent=2)
    
    def _update_best_checkpoints(self, checkpoint_path: Path, val_loss: float) -> None:
        records = self._load_best_records()
        # Remove existing entry for this path if present
        records = [r for r in records if r.get("path") != str(checkpoint_path)]
        records.append({"path": str(checkpoint_path), "val_loss": val_loss})
        records.sort(key=lambda r: r["val_loss"])
        records = records[: self.best_k]
        self._save_best_records(records)
        
        # Refresh best directory contents
        for file in self.best_dir.glob("*.pt"):
            try:
                file.unlink()
            except FileNotFoundError:
                pass
        for rank, record in enumerate(records, start=1):
            src = Path(record["path"])
            if not src.exists():
                continue
            dest_name = f"rank{rank}_val{record['val_loss']:.6f}.pt"
            shutil.copy2(src, self.best_dir / dest_name)
    
    def _cleanup_checkpoints(self):
        """Remove old checkpoints, keeping only the last N"""
        checkpoint_files = list(self.save_dir.glob("checkpoint_epoch_*.pt"))
        if len(checkpoint_files) > self.keep_last_n:
            checkpoint_files.sort(key=lambda x: int(x.stem.split('_')[-1]))
            protected = {Path(record["path"]).resolve() for record in self._load_best_records()}
            remove_candidates = [
                f for f in checkpoint_files[:-self.keep_last_n] if f.resolve() not in protected
            ]
            for f in remove_candidates:
                try:
                    f.unlink()
                except FileNotFoundError:
                    pass
    
    def load_checkpoint(self, checkpoint_path: str) -> Dict[str, Any]:
        """Load checkpoint from file"""
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        return checkpoint
    
    def find_latest_checkpoint(self) -> Optional[str]:
        """Find the latest checkpoint file"""
        latest_path = self.save_dir / "latest.pt"
        if latest_path.exists():
            return str(latest_path)
        
        # Fallback to finding newest checkpoint file
        checkpoint_files = list(self.save_dir.glob("checkpoint_epoch_*.pt"))
        if checkpoint_files:
            latest_file = max(checkpoint_files, key=lambda x: int(x.stem.split('_')[-1]))
            return str(latest_file)
        
        return None


class TotoTrainer:
    """Comprehensive Toto model trainer with advanced features"""
    
    def __init__(self, 
                 config: TrainerConfig,
                 dataloader_config: DataLoaderConfig):
        self.config = config
        self.dataloader_config = dataloader_config
        
        # Set random seeds
        self._set_random_seeds()
        
        # Setup logging
        self._setup_logging()
        
        # Setup distributed training
        self._setup_distributed()
        self.device_batch_size: Optional[int] = None
        self._configure_batches()
        
        # Initialize components
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.autocast_dtype: Optional[torch.dtype] = None
        self.scaler: Optional[GradScaler] = None
        self._configure_precision()
        
        # Metrics and checkpointing
        self.metrics_tracker = MetricsTracker()
        self.checkpoint_manager = CheckpointManager(
            config.save_dir, 
            config.keep_last_n_checkpoints,
            best_k=config.best_k_checkpoints
        )
        
        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        self.best_export_metric = float('inf')
        self.training_start_time = None

        # Data loaders
        self.dataloaders = {}
        self.ema: Optional[EMA] = None
        self._ema_module: Optional[nn.Module] = None
        
        # Export directory for HuggingFace-compatible checkpoints
        self.export_dir = Path(self.config.export_pretrained_dir)
        self.export_dir.mkdir(parents=True, exist_ok=True)
        self.export_metadata_path = self.export_dir / "metadata.json"
        
        # Optional TensorBoard monitoring
        self.tensorboard_monitor = None
        if self.config.log_to_tensorboard and TensorBoardMonitor is not None:
            try:
                self.tensorboard_monitor = TensorBoardMonitor(
                    experiment_name=self.config.experiment_name,
                    log_dir=self.config.tensorboard_log_dir,
                    enable_model_graph=False,
                    enable_weight_histograms=False,
                    enable_gradient_histograms=False,
                    flush_secs=15
                )
            except Exception as e:
                self.logger.warning(f"TensorBoard monitor unavailable: {e}")
                self.tensorboard_monitor = None
        elif self.config.log_to_tensorboard and TensorBoardMonitor is None:
            self.logger.warning("TensorBoard not available. Install tensorboard to enable logging.")
        
        self.logger.info("TotoTrainer initialized")
    
    def _set_random_seeds(self):
        """Set random seeds for reproducibility"""
        random.seed(self.config.random_seed)
        np.random.seed(self.config.random_seed)
        torch.manual_seed(self.config.random_seed)
        torch.cuda.manual_seed_all(self.config.random_seed)
        
        # For deterministic training (slower but reproducible)
        if self.config.random_seed is not None:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    
    def _setup_logging(self):
        """Setup logging configuration"""
        log_level = getattr(logging, self.config.log_level.upper(), logging.INFO)

        handlers = [logging.StreamHandler(stream=sys.stdout)]
        if self.config.log_file:
            log_path = Path(self.config.log_file)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            handlers.append(logging.FileHandler(log_path))

        basic_config_kwargs = {
            "level": log_level,
            "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            "handlers": handlers,
        }

        try:
            logging.basicConfig(force=True, **basic_config_kwargs)
        except TypeError:
            root_logger = logging.getLogger()
            for handler in list(root_logger.handlers):
                root_logger.removeHandler(handler)
            logging.basicConfig(**basic_config_kwargs)

        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(log_level)
    
    def _setup_distributed(self):
        """Setup distributed training if enabled"""
        self.is_distributed = False
        self.is_main_process = True

        if self.config.distributed:
            if not torch.cuda.is_available():
                raise RuntimeError("Distributed training requires CUDA but no GPU is available.")
            if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
                self.config.rank = int(os.environ["RANK"])
                self.config.world_size = int(os.environ['WORLD_SIZE'])
                self.config.local_rank = int(os.environ['LOCAL_RANK'])
            
            torch.cuda.set_device(self.config.local_rank)
            dist.init_process_group(
                backend=self.config.dist_backend,
                init_method=self.config.dist_url,
                world_size=self.config.world_size,
                rank=self.config.rank
            )
            
            self.is_distributed = True
            self.is_main_process = self.config.rank == 0

            self.logger.info(f"Distributed training enabled: rank {self.config.rank}/{self.config.world_size}")

    def _configure_batches(self) -> None:
        per_device = self.config.device_batch_size
        if per_device is None:
            if hasattr(self.dataloader_config, "batch_size") and self.dataloader_config.batch_size:
                per_device = self.dataloader_config.batch_size
            else:
                per_device = self.config.batch_size

        if per_device <= 0:
            raise ValueError("Per-device batch size must be positive.")

        if hasattr(self.dataloader_config, "batch_size"):
            self.dataloader_config.batch_size = per_device

        world = self.config.world_size if self.is_distributed else 1
        if self.config.global_batch_size is not None:
            denom = per_device * world
            if denom == 0 or self.config.global_batch_size % denom != 0:
                raise ValueError(
                    "global_batch_size must be divisible by per-device batch size times world size."
                )
            self.config.accumulation_steps = max(1, self.config.global_batch_size // denom)

        self.device_batch_size = per_device
        effective_global = per_device * max(1, self.config.accumulation_steps) * world
        self.logger.info(
            "Effective batches -> per-device %d, grad_accum %d, world %d (global %d)",
            per_device,
            max(1, self.config.accumulation_steps),
            world,
            effective_global,
        )

    def _prefetch_loader(self, loader: DataLoader, device: torch.device):
        if self.config.prefetch_to_device and device.type == "cuda":
            return CudaPrefetcher(loader, device=device)
        return loader
    
    def _configure_precision(self) -> None:
        """Configure autocast dtype and gradient scaler based on hardware."""
        self.autocast_dtype = None
        self.scaler = None

        if not self.config.use_mixed_precision:
            return

        if torch.cuda.is_available():
            if bf16_supported():
                self.autocast_dtype = torch.bfloat16
                self.logger.info("Using bfloat16 autocast for CUDA training.")
            else:
                self.autocast_dtype = torch.float16
                self.scaler = GradScaler()
                self.logger.info("Using float16 autocast with GradScaler for CUDA training.")
        else:
            self.logger.info("Mixed precision requested but CUDA not available; defaulting to float32.")

    def _ema_target_module(self) -> nn.Module:
        if self.model is None:
            raise RuntimeError("Model not initialized before accessing EMA module.")
        return self.model.module if hasattr(self.model, "module") else self.model

    def _maybe_init_ema(self) -> None:
        if self.config.ema_decay is None:
            self.ema = None
            self._ema_module = None
            return

        module = self._ema_target_module()
        self.ema = EMA(module, decay=self.config.ema_decay)
        self._ema_module = module

    @contextlib.contextmanager
    def _ema_eval_context(self):
        if self.ema is None or not self.config.ema_eval:
            yield
            return
        target_module = self._ema_module or self._ema_target_module()
        self.ema.apply_to(target_module)
        try:
            yield
        finally:
            self.ema.restore(target_module)
    
    def _create_model(self, input_dim: int) -> nn.Module:
        """Create Toto model"""
        if self.config.require_gpu and not torch.cuda.is_available():
            raise RuntimeError("TrainerConfig.require_gpu is True but CUDA is not available.")

        pretrained_dtype: Optional[torch.dtype] = None
        if self.config.pretrained_torch_dtype:
            dtype_map = {
                "float32": torch.float32,
                "float16": torch.float16,
                "bfloat16": torch.bfloat16,
            }
            pretrained_dtype = dtype_map.get(self.config.pretrained_torch_dtype.lower())
            if pretrained_dtype is None:
                raise ValueError(
                    f"Unsupported pretrained_torch_dtype '{self.config.pretrained_torch_dtype}'."
                )

        device = torch.device(f'cuda:{self.config.local_rank}' if torch.cuda.is_available() else 'cpu')

        if self.config.pretrained_model_id:
            map_location = str(device)
            model = Toto.from_pretrained(
                self.config.pretrained_model_id,
                map_location=map_location,
            )
            if pretrained_dtype is not None:
                model = model.to(device=device, dtype=pretrained_dtype)
            else:
                model = model.to(device)
        else:
            model = Toto(
                patch_size=self.config.patch_size,
                stride=self.config.stride,
                embed_dim=self.config.embed_dim,
                num_layers=self.config.num_layers,
                num_heads=self.config.num_heads,
                mlp_hidden_dim=self.config.mlp_hidden_dim,
                dropout=self.config.dropout,
                spacewise_every_n_layers=self.config.spacewise_every_n_layers,
                scaler_cls=self.config.scaler_cls,
                output_distribution_classes=self.config.output_distribution_classes,
                use_memory_efficient_attention=self.config.memory_efficient_attention,
            )
            if pretrained_dtype is not None:
                model = model.to(dtype=pretrained_dtype)
            model = model.to(device)

            if self.config.pretrained_checkpoint:
                checkpoint = torch.load(
                    self.config.pretrained_checkpoint,
                    map_location=device,
                    weights_only=False,
                )
                state_dict = checkpoint.get("model_state_dict", checkpoint)
                missing, unexpected = model.load_state_dict(state_dict, strict=False)
                if missing:
                    self.logger.warning(
                        "Missing parameters when loading pretrained checkpoint: %s", missing
                    )
                if unexpected:
                    self.logger.warning(
                        "Unexpected parameters when loading pretrained checkpoint: %s", unexpected
                    )

        # Enable gradient checkpointing for memory efficiency
        if self.config.gradient_checkpointing and hasattr(model, "gradient_checkpointing_enable"):
            model.gradient_checkpointing_enable()

        if self.config.freeze_backbone:
            self._apply_parameter_freeze(model)

        if self.config.compile:
            self.logger.info(
                "torch.compile enabled; the first few batches may spend extra time compiling kernels."
            )
        model = maybe_compile(model, do_compile=self.config.compile)
        
        # Wrap with DDP if distributed
        if self.is_distributed:
            ddp_kwargs = dict(
                device_ids=[self.config.local_rank],
                output_device=self.config.local_rank,
                gradient_as_bucket_view=True,
                broadcast_buffers=False,
                find_unused_parameters=False,
            )
            if self.config.use_cuda_graphs:
                ddp_kwargs["static_graph"] = True
            try:
                model = DDP(model, **ddp_kwargs)
            except TypeError:
                ddp_kwargs.pop("static_graph", None)
                model = DDP(model, **ddp_kwargs)

        return model

    def _apply_parameter_freeze(self, model: nn.Module) -> None:
        substrings = self.config.trainable_param_substrings or []
        if not substrings:
            self.logger.warning(
                "freeze_backbone enabled but no trainable_param_substrings provided; freezing all parameters."
            )
        total_params = 0
        trainable_params = 0
        for name, param in model.named_parameters():
            total_params += param.numel()
            keep_trainable = any(sub in name for sub in substrings)
            param.requires_grad = keep_trainable
            if keep_trainable:
                trainable_params += param.numel()
        self.logger.info(
            "Backbone frozen. Trainable params: %s of %s (%.4f%%)",
            trainable_params,
            total_params,
            100.0 * trainable_params / max(total_params, 1),
        )
    
    def _create_optimizer(self) -> torch.optim.Optimizer:
        """Create optimizer"""
        if not any(p.requires_grad for p in self.model.parameters()):
            raise ValueError("No trainable parameters found for optimizer.")

        optimizer = make_optimizer(
            self.model,
            name=self.config.optimizer,
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
            betas=self.config.optimizer_betas,
            eps=self.config.optimizer_eps,
            fused=True,
        )
        return optimizer
    
    def _create_scheduler(self, steps_per_epoch: int) -> Optional[torch.optim.lr_scheduler._LRScheduler]:
        """Create learning rate scheduler"""
        schedule_name = self.config.scheduler.lower()
        if schedule_name == "none" or steps_per_epoch <= 0:
            return None

        total_steps = steps_per_epoch * self.config.max_epochs
        if total_steps <= 0:
            return None

        if self.config.warmup_steps is not None:
            warmup_steps = min(int(self.config.warmup_steps), max(total_steps - 1, 0))
        else:
            warmup_steps = int(self.config.warmup_epochs * steps_per_epoch)
            warmup_steps = min(warmup_steps, max(total_steps - 1, 0))
        warmup_steps = max(0, warmup_steps)

        if schedule_name == "cosine":
            return WarmupCosine(
                self.optimizer,
                warmup_steps=warmup_steps,
                total_steps=total_steps,
                min_lr=self.config.min_lr,
            )
        if schedule_name == "plateau":
            return ReduceLROnPlateau(
                self.optimizer,
                mode="min",
                factor=0.5,
                patience=5,
            )
        if schedule_name == "onecycle":
            pct_start = warmup_steps / total_steps if total_steps > 0 else 0.1
            return OneCycleLR(
                self.optimizer,
                max_lr=self.config.learning_rate,
                total_steps=total_steps,
                pct_start=pct_start,
            )
        raise ValueError(f"Unsupported scheduler: {self.config.scheduler}")

    def _forward_model(self, series: torch.Tensor, padding_mask: torch.Tensor, id_mask: torch.Tensor):
        module = self.model.module if hasattr(self.model, "module") else self.model
        if hasattr(module, "model"):
            return module.model(series, padding_mask, id_mask)
        return module(series, padding_mask, id_mask)

    @staticmethod
    def _ensure_tensor(value: Any, device: torch.device) -> Optional[torch.Tensor]:
        if value is None:
            return None
        if isinstance(value, torch.Tensor):
            return value.to(device)
        return torch.tensor(value, dtype=torch.float32, device=device)

    @staticmethod
    def _match_prediction_length(tensor: Optional[torch.Tensor], prediction_length: int) -> Optional[torch.Tensor]:
        if tensor is None:
            return None
        if tensor.ndim == 1:
            tensor = tensor.unsqueeze(-1)
        if tensor.ndim == 3 and tensor.shape[1] == 1:
            tensor = tensor[:, 0, :]
        elif tensor.ndim == 3:
            tensor = tensor[:, 0, :]
        if tensor.ndim == 2 and tensor.shape[-1] == prediction_length:
            return tensor
        if tensor.ndim != 2:
            raise RuntimeError(f"Unsupported tensor shape for match_prediction_length: {tensor.shape}")
        if tensor.shape[-1] > prediction_length:
            return tensor[:, -prediction_length:]
        pad_len = prediction_length - tensor.shape[-1]
        pad = tensor[:, -1:].expand(-1, pad_len)
        return torch.cat([tensor, pad], dim=-1)

    @staticmethod
    def _match_quantile_length(tensor: torch.Tensor, prediction_length: int) -> torch.Tensor:
        if tensor.shape[1] == prediction_length:
            return tensor
        if tensor.shape[1] > prediction_length:
            return tensor[:, -prediction_length:, :]
        pad_len = prediction_length - tensor.shape[1]
        pad = tensor[:, -1:, :].expand(-1, pad_len, -1)
        return torch.cat([tensor, pad], dim=1)

    def _get_quantile_predictions(
        self,
        output: Any,
        levels: Sequence[float],
        device: torch.device,
        dtype: torch.dtype,
        prediction_length: int,
    ) -> Optional[torch.Tensor]:
        if not levels:
            return None

        quantiles = None
        if isinstance(output, dict):
            for key in ("quantiles", "quantile_predictions", "quantile_outputs"):
                if key in output:
                    quantiles = output[key]
                    break

        if quantiles is None:
            return None

        q_tensor = quantiles.to(device=device, dtype=dtype)
        if q_tensor.ndim == 3:
            if q_tensor.shape[1] == len(levels):
                aligned = q_tensor.transpose(1, 2)  # [B, H, Q]
            elif q_tensor.shape[2] == len(levels):
                aligned = q_tensor  # [B, H, Q]
            else:
                return None
        else:
            return None

        aligned = self._match_quantile_length(aligned, prediction_length)
        return aligned

    def _ensure_prev_close(
        self,
        prev_close: Optional[torch.Tensor],
        series: torch.Tensor,
        prediction_length: int,
    ) -> torch.Tensor:
        if prev_close is None:
            prev_close = series[:, 0, -1]
        prev_close = prev_close.to(series.device, dtype=series.dtype)
        if prev_close.ndim == 0:
            prev_close = prev_close.unsqueeze(0)
        if prev_close.ndim == 1:
            prev_close = prev_close.unsqueeze(-1)
        if prev_close.ndim == 2 and prev_close.shape[-1] == prediction_length:
            return prev_close
        if prev_close.ndim == 2 and prev_close.shape[-1] == 1:
            return prev_close.expand(-1, prediction_length)
        if prev_close.ndim == 2:
            return prev_close[:, -1:].expand(-1, prediction_length)
        raise RuntimeError(f"Unsupported prev_close shape: {prev_close.shape}")

    @staticmethod
    def _infer_target_from_series(series: torch.Tensor, prediction_length: int) -> torch.Tensor:
        target_slice = series[:, 0, :]
        if target_slice.shape[-1] >= prediction_length:
            return target_slice[:, -prediction_length:]
        pad_len = prediction_length - target_slice.shape[-1]
        pad = target_slice[:, -1:].expand(-1, pad_len)
        return torch.cat([target_slice, pad], dim=-1)

    @staticmethod
    def _compute_pct_delta(values: torch.Tensor, baseline: torch.Tensor) -> torch.Tensor:
        denom = baseline.abs().clamp(min=1e-6)
        return (values - baseline) / denom

    @staticmethod
    def _reconstruct_price(prev_close: torch.Tensor, pct: torch.Tensor) -> torch.Tensor:
        denom = prev_close.abs().clamp(min=1e-6)
        return pct * denom + prev_close

    def _autocast_context(self, device: torch.device):
        if self.autocast_dtype is None or device.type != "cuda":
            return contextlib.nullcontext()
        return torch.autocast(device_type="cuda", dtype=self.autocast_dtype)

    def _extract_predictions(self, output: Any) -> torch.Tensor:
        if hasattr(output, "distribution"):
            return output.distribution.mean
        if hasattr(output, "loc"):
            return output.loc
        if isinstance(output, dict):
            for key in ("prediction", "predictions", "output"):
                if key in output:
                    return output[key]
        if isinstance(output, torch.Tensor):
            return output
        raise RuntimeError("Model output does not contain predictions tensor.")

    def _prepare_batch(
        self,
        batch: Union[MaskedTimeseries, Tuple[Any, Any], List[Any], Dict[str, Any]],
        device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor], Dict[str, Any]]:
        target_price: Optional[torch.Tensor] = None
        target_pct: Optional[torch.Tensor] = None
        prev_close: Optional[torch.Tensor] = None
        metadata: Dict[str, Any] = {}

        masked_field_names = {"series", "padding_mask", "id_mask", "timestamp_seconds", "time_interval_seconds"}
        toto_batch_type = globals().get("TotoBatchSample")

        if toto_batch_type is not None and isinstance(batch, toto_batch_type):
            candidate = batch.timeseries
            if hasattr(batch, "metadata"):
                extra = dict(batch.metadata())
            else:
                extra = {
                    "target_price": getattr(batch, "target_price", None),
                    "target_pct": getattr(batch, "target_pct", None),
                    "prev_close": getattr(batch, "prev_close", None),
                }
        else:
            candidate = batch
            extra = {}

            if hasattr(batch, "_fields"):
                field_names = getattr(batch, "_fields", ())
                if "timeseries" in field_names:
                    candidate = getattr(batch, "timeseries")
                    extra = {
                        name: getattr(batch, name)
                        for name in field_names
                        if name not in {"timeseries"} and name not in masked_field_names
                    }
                else:
                    candidate = batch
            elif isinstance(batch, (tuple, list)) and batch:
                candidate = batch[0]
                if len(batch) > 1 and isinstance(batch[1], dict):
                    extra = batch[1]
            elif isinstance(batch, dict) and "timeseries" in batch:
                candidate = batch["timeseries"]
                extra = {k: v for k, v in batch.items() if k != "timeseries"}

        if isinstance(candidate, MaskedTimeseries):
            masked = candidate.to(device)
            series = masked.series
            padding_mask = masked.padding_mask
            id_mask = masked.id_mask
        elif hasattr(candidate, "series") and hasattr(candidate, "padding_mask"):
            masked = candidate.to(device) if hasattr(candidate, "to") else candidate
            series = masked.series.to(device)
            padding_mask = masked.padding_mask.to(device)
            id_mask = masked.id_mask.to(device)
        elif isinstance(candidate, tuple) and len(candidate) == 2:
            x, y = candidate
            series = x.to(device).transpose(1, 2)
            batch_size, seq_len, features = x.shape
            padding_mask = torch.ones(batch_size, features, seq_len, dtype=torch.bool, device=device)
            id_mask = torch.zeros(batch_size, features, seq_len, dtype=torch.long, device=device)
            target_price = self._ensure_tensor(y, device)
        else:
            raise RuntimeError("Unsupported batch format encountered.")

        if isinstance(extra, dict):
            maybe_target_price = self._ensure_tensor(extra.get("target_price"), device)
            if maybe_target_price is not None:
                target_price = maybe_target_price
            target_pct = self._ensure_tensor(extra.get("target_pct"), device)
            prev_close = self._ensure_tensor(extra.get("prev_close"), device)
            metadata = {k: v for k, v in extra.items() if k not in {"target_price", "target_pct", "prev_close"}}

        return series, padding_mask, id_mask, target_price, target_pct, prev_close, metadata

    def _forward_batch(
        self,
        series: torch.Tensor,
        padding_mask: torch.Tensor,
        id_mask: torch.Tensor,
        target_price: Optional[torch.Tensor],
        target_pct: Optional[torch.Tensor],
        prev_close: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        device = series.device
        with self._autocast_context(device):
            output = self._forward_model(series, padding_mask, id_mask)
            predictions = self._extract_predictions(output)
            if predictions.ndim != 3:
                raise RuntimeError(f"Expected 3D predictions, got shape {predictions.shape}")

            price_predictions = predictions[:, 0, :].to(series.dtype)
            prediction_length = price_predictions.shape[-1]
            levels = self.config.quantile_levels or []
            quantile_tensor = (
                self._get_quantile_predictions(
                    output,
                    levels,
                    price_predictions.device,
                    price_predictions.dtype,
                    prediction_length,
                )
                if levels
                else None
            )

            target_pct = self._match_prediction_length(target_pct, prediction_length)
            prev_close_tensor = self._ensure_prev_close(prev_close, series, prediction_length)
            matched_target_price = self._match_prediction_length(target_price, prediction_length)
            if matched_target_price is None and target_pct is not None:
                matched_target_price = self._reconstruct_price(prev_close_tensor, target_pct)
            if matched_target_price is None:
                matched_target_price = self._infer_target_from_series(series, prediction_length)

            dtype = price_predictions.dtype
            if target_pct is not None:
                target_pct = target_pct.to(dtype)
            prev_close_tensor = prev_close_tensor.to(dtype)
            matched_target_price = matched_target_price.to(dtype)

            if target_pct is not None:
                targets_pct = target_pct
            else:
                targets_pct = self._compute_pct_delta(matched_target_price, prev_close_tensor)

            predictions_pct = self._compute_pct_delta(price_predictions, prev_close_tensor)
            loss = self._compute_loss(
                predictions_pct,
                targets_pct,
                price_predictions,
                matched_target_price,
                output,
                quantile_tensor,
            )

        return (
            loss,
            predictions_pct,
            targets_pct,
            price_predictions,
            matched_target_price,
            prev_close_tensor,
            quantile_tensor,
        )

    def _compute_loss(
        self,
        predictions_pct: torch.Tensor,
        targets_pct: torch.Tensor,
        price_predictions: torch.Tensor,
        matched_target_price: torch.Tensor,
        output: Any,
        quantile_tensor: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        loss_type = self.config.loss_type
        if targets_pct is None:
            raise RuntimeError("Targets required for loss computation.")

        if loss_type == "mse":
            return F.mse_loss(predictions_pct, targets_pct)
        if loss_type == "huber":
            return huber_loss(predictions_pct, targets_pct, delta=self.config.huber_delta)
        if loss_type == "heteroscedastic":
            log_sigma = None
            if isinstance(output, dict):
                if "log_sigma" in output:
                    log_sigma = output["log_sigma"]
                elif "sigma" in output:
                    sigma = output["sigma"]
                    log_sigma = sigma.clamp_min(1e-5).log()
            if log_sigma is None and hasattr(output, "distribution"):
                dist = output.distribution
                if hasattr(dist, "scale"):
                    scale = dist.scale
                    if torch.is_tensor(scale):
                        if scale.ndim == 3:
                            log_sigma = scale[:, 0, :].clamp_min(1e-5).log()
                        else:
                            log_sigma = scale.clamp_min(1e-5).log()
                if log_sigma is None and hasattr(dist, "log_scale"):
                    log_sigma = dist.log_scale
            if log_sigma is None:
                raise RuntimeError("heteroscedastic loss requires log_sigma or distribution scale outputs.")
            log_sigma = log_sigma.to(price_predictions.device, price_predictions.dtype)
            if log_sigma.ndim == 3:
                log_sigma = log_sigma[:, 0, :]
            log_sigma = self._match_prediction_length(log_sigma, price_predictions.shape[-1])
            return heteroscedastic_gaussian_nll(price_predictions, log_sigma, matched_target_price)
        if loss_type == "quantile":
            levels = self.config.quantile_levels or [0.1, 0.5, 0.9]
            aligned = quantile_tensor
            if aligned is None:
                aligned = self._get_quantile_predictions(
                    output,
                    levels,
                    price_predictions.device,
                    price_predictions.dtype,
                    price_predictions.shape[-1],
                )
            if aligned is not None:
                losses = [
                    pinball_loss(aligned[:, :, idx], matched_target_price, q, reduction="mean")
                    for idx, q in enumerate(levels)
                ]
                return sum(losses) / len(losses)
            if hasattr(output, "distribution") and hasattr(output.distribution, "icdf"):
                dist = output.distribution
                losses = []
                for q in levels:
                    prob = torch.full_like(price_predictions, float(q))
                    try:
                        quantile_vals = dist.icdf(prob.unsqueeze(1))
                    except Exception as exc:
                        raise RuntimeError("Distribution icdf evaluation failed for quantile loss.") from exc
                    if quantile_vals.ndim == 4:
                        quantile_vals = quantile_vals[:, 0, 0, :]
                    elif quantile_vals.ndim == 3:
                        quantile_vals = quantile_vals[:, 0, :]
                    losses.append(pinball_loss(quantile_vals, matched_target_price, q, reduction="mean"))
                return sum(losses) / len(losses)
            raise RuntimeError("Quantile loss requires model outputs with quantile predictions or icdf support.")

        raise AssertionError(f"Unhandled loss_type {loss_type}.")
    
    def prepare_data(self):
        """Prepare data loaders"""
        self.logger.info("Preparing data loaders...")
        
        # Create OHLC data loader
        dataloader = TotoOHLCDataLoader(self.dataloader_config)
        self.dataloaders = dataloader.prepare_dataloaders()
        
        if not self.dataloaders:
            raise ValueError("No data loaders created!")
        
        self.logger.info(f"Created data loaders: {list(self.dataloaders.keys())}")
        
        # Log dataset sizes
        for split, loader in self.dataloaders.items():
            self.logger.info(f"{split}: {len(loader.dataset)} samples, {len(loader)} batches")
    
    def setup_model(self):
        """Setup model, optimizer, and scheduler"""
        self.logger.info("Setting up model...")
        
        if not self.dataloaders:
            raise ValueError("Data loaders not prepared! Call prepare_data() first.")
        
        # Determine input dimension from data loader
        sample_batch = next(iter(self.dataloaders['train']))
        if isinstance(sample_batch, (tuple, list)):
            primary_sample = sample_batch[0]
        else:
            primary_sample = sample_batch

        if hasattr(primary_sample, 'series'):
            series_sample = primary_sample.series
            if series_sample.ndim == 3:
                # (batch, features, sequence)
                input_dim = series_sample.shape[1]
            elif series_sample.ndim == 2:
                # (features, sequence)
                input_dim = series_sample.shape[0]
            else:
                raise RuntimeError(f"Unexpected series shape: {series_sample.shape}")
        elif torch.is_tensor(primary_sample):
            input_dim = primary_sample.shape[-1]
        else:
            raise RuntimeError("Unable to infer input dimension from training batch.")
        
        self.logger.info(f"Input dimension: {input_dim}")
        
        # Create model
        self.model = self._create_model(input_dim)
        
        # Count parameters
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        self.logger.info(f"Model parameters: {total_params:,} total, {trainable_params:,} trainable")
        
        # Create optimizer
        self.optimizer = self._create_optimizer()
        
        # Create scheduler
        total_train_batches = len(self.dataloaders['train'])
        steps_per_epoch = max(1, math.ceil(total_train_batches / max(1, self.config.accumulation_steps)))
        self.scheduler = self._create_scheduler(steps_per_epoch)

        self.logger.info("Model setup completed")
        self._maybe_init_ema()
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load model from checkpoint"""
        self.logger.info(f"Loading checkpoint from {checkpoint_path}")
        
        checkpoint = self.checkpoint_manager.load_checkpoint(checkpoint_path)
        
        # Load model state
        if hasattr(self.model, 'module'):
            self.model.module.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        
        # Load optimizer state
        try:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        except (KeyError, ValueError) as exc:
            self.logger.warning(
                "Optimizer state in %s is incompatible with current configuration; proceeding with freshly initialized optimizer (%s)",
                checkpoint_path,
                exc,
            )

        # Load scheduler state
        if self.scheduler and checkpoint['scheduler_state_dict']:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        # Load scaler state
        if self.scaler and checkpoint['scaler_state_dict']:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        
        # Load training state
        self.current_epoch = checkpoint['epoch']
        self.best_val_loss = checkpoint['best_val_loss']

        self.logger.info(f"Checkpoint loaded: epoch {self.current_epoch}, best val loss: {self.best_val_loss:.6f}")
        if self.config.ema_decay is not None:
            self._maybe_init_ema()
    
    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch"""
        self.model.train()
        self.metrics_tracker.reset()
        
        device = next(self.model.parameters()).device
        accumulation = max(1, self.config.accumulation_steps)
        train_loader = self.dataloaders['train']
        iterable = self._prefetch_loader(train_loader, device)

        with enable_fast_kernels():
            for batch_idx, batch in enumerate(iterable):
                batch_start_time = time.time()

                (
                    series,
                    padding_mask,
                    id_mask,
                    target_price,
                    target_pct,
                    prev_close,
                    _,
                ) = self._prepare_batch(batch, device)

                (
                    loss,
                    predictions_pct,
                    targets_pct,
                    price_predictions,
                    matched_target_price,
                    prev_close_tensor,
                    quantile_tensor,
                ) = self._forward_batch(
                    series,
                    padding_mask,
                    id_mask,
                    target_price,
                    target_pct,
                    prev_close,
                )
                loss = loss / accumulation

                if self.scaler:
                    self.scaler.scale(loss).backward()
                else:
                    loss.backward()

                if (batch_idx + 1) % accumulation == 0:
                    if self.config.gradient_clip_val and self.config.gradient_clip_val > 0:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.gradient_clip_val)

                    if self.scaler:
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        self.optimizer.step()

                    self.optimizer.zero_grad(set_to_none=True)

                    if self.ema is not None:
                        target_module = self._ema_module or self._ema_target_module()
                        self.ema.update(target_module)

                    if self.scheduler and self.config.scheduler.lower() in {"cosine", "onecycle"}:
                        self.scheduler.step()

                    self.global_step += 1

                batch_time = time.time() - batch_start_time
                current_lr = self.optimizer.param_groups[0]["lr"]
                pct_mae = torch.mean(torch.abs(predictions_pct.detach() - targets_pct.detach())).item()
                price_mae = torch.mean(torch.abs(price_predictions.detach() - matched_target_price.detach())).item()

                self.metrics_tracker.update(
                    loss=loss.item() * accumulation,
                    predictions=predictions_pct.unsqueeze(1) if self.config.compute_train_metrics else None,
                    targets=targets_pct.unsqueeze(1) if self.config.compute_train_metrics else None,
                    price_predictions=price_predictions.unsqueeze(1) if self.config.compute_train_metrics else None,
                    price_targets=matched_target_price.unsqueeze(1) if self.config.compute_train_metrics else None,
                    batch_time=batch_time,
                    learning_rate=current_lr,
                    prev_close=prev_close_tensor if self.config.compute_train_metrics else None,
                    quantile_predictions=quantile_tensor if (self.config.compute_train_metrics and quantile_tensor is not None) else None,
                    quantile_levels=self.config.quantile_levels if (self.config.compute_train_metrics and quantile_tensor is not None) else None,
                )

                if batch_idx % self.config.metrics_log_frequency == 0:
                    self.logger.info(
                        "Epoch %d, Batch %d/%d, Loss %.6f, pct_mae %.6f, price_mae %.2f, LR %.8f",
                        self.current_epoch,
                        batch_idx,
                        len(train_loader),
                        loss.item(),
                        pct_mae,
                        price_mae,
                        current_lr,
                    )
        
        return self.metrics_tracker.compute_metrics()
    
    def validate_epoch(self) -> Dict[str, float]:
        """Validate for one epoch"""
        if 'val' not in self.dataloaders:
            return {}
        
        self.model.eval()
        self.metrics_tracker.reset()

        device = next(self.model.parameters()).device

        with torch.no_grad():
            val_loader = self.dataloaders['val']
            iterable = self._prefetch_loader(val_loader, device)
            with self._ema_eval_context():
                with enable_fast_kernels():
                    for batch_idx, batch in enumerate(iterable):
                        (
                            series,
                            padding_mask,
                            id_mask,
                            target_price,
                        target_pct,
                        prev_close,
                        _,
                    ) = self._prepare_batch(batch, device)

                    (
                        loss,
                        predictions_pct,
                        targets_pct,
                        price_predictions,
                        matched_target_price,
                        prev_close_tensor,
                        quantile_tensor,
                    ) = self._forward_batch(
                        series,
                        padding_mask,
                        id_mask,
                        target_price,
                        target_pct,
                        prev_close,
                    )

                    self.metrics_tracker.update(
                        loss=loss.item(),
                        predictions=predictions_pct.unsqueeze(1) if self.config.compute_val_metrics else None,
                        targets=targets_pct.unsqueeze(1) if self.config.compute_val_metrics else None,
                        price_predictions=price_predictions.unsqueeze(1) if self.config.compute_val_metrics else None,
                        price_targets=matched_target_price.unsqueeze(1) if self.config.compute_val_metrics else None,
                        prev_close=prev_close_tensor if self.config.compute_val_metrics else None,
                        quantile_predictions=quantile_tensor if (self.config.compute_val_metrics and quantile_tensor is not None) else None,
                        quantile_levels=self.config.quantile_levels if (self.config.compute_val_metrics and quantile_tensor is not None) else None,
                    )
        
        return self.metrics_tracker.compute_metrics()
    
    def train(self):
        """Main training loop"""
        self.logger.info("Starting training...")
        self.training_start_time = time.time()
        
        # Resume from checkpoint if specified
        if self.config.resume_from_checkpoint:
            self.load_checkpoint(self.config.resume_from_checkpoint)
        elif self.checkpoint_manager.find_latest_checkpoint():
            self.load_checkpoint(self.checkpoint_manager.find_latest_checkpoint())
        
        profile_ctx = maybe_profile(self.config.profile, self.config.profile_log_dir)
        with profile_ctx:
            # Training loop
            for epoch in range(self.current_epoch, self.config.max_epochs):
                self.current_epoch = epoch
                epoch_start_time = time.time()
                
                self.logger.info(f"Epoch {epoch + 1}/{self.config.max_epochs}")
                
                # Train epoch
                train_metrics = self.train_epoch()
                
                # Validation epoch
                val_metrics = {}
                if epoch % self.config.validation_frequency == 0:
                    val_metrics = self.validate_epoch()
                
                # Update scheduler
                if self.scheduler and self.config.scheduler.lower() == "plateau":
                    val_loss = val_metrics.get('loss', train_metrics['loss'])
                    self.scheduler.step(val_loss)
                
                epoch_time = time.time() - epoch_start_time
                current_lr = self.optimizer.param_groups[0]['lr'] if self.optimizer else 0.0
                
                # Log to monitoring systems
                self._log_epoch(epoch, train_metrics, val_metrics, epoch_time, current_lr)
                
                # Log metrics
                self._log_metrics(epoch, train_metrics, val_metrics)
                
                # Determine if this is the best model so far
                metric_for_patience = None
                if val_metrics and 'loss' in val_metrics:
                    metric_for_patience = val_metrics['loss']
                elif 'loss' in train_metrics:
                    metric_for_patience = train_metrics['loss']
            
                is_best = False
                if metric_for_patience is not None:
                    if metric_for_patience < self.best_val_loss - self.config.early_stopping_delta:
                        self.best_val_loss = metric_for_patience
                        self.patience_counter = 0
                        is_best = True
                    else:
                        self.patience_counter += 1
                
                # Save checkpoint
                if epoch % self.config.save_every_n_epochs == 0 or is_best:
                    val_loss_for_checkpoint = None
                    if val_metrics and 'loss' in val_metrics:
                        val_loss_for_checkpoint = float(val_metrics['loss'])
                    elif 'loss' in train_metrics:
                        val_loss_for_checkpoint = float(train_metrics['loss'])
                    self.checkpoint_manager.save_checkpoint(
                        model=self.model,
                        optimizer=self.optimizer,
                        scheduler=self.scheduler,
                        scaler=self.scaler,
                        epoch=epoch,
                        best_val_loss=self.best_val_loss,
                        metrics={**train_metrics, **val_metrics},
                        config=self.config,
                        dataloader_config=self.dataloader_config,
                        is_best=is_best,
                        val_loss=val_loss_for_checkpoint
                    )
                
                if is_best and self.config.export_on_best:
                    self._export_pretrained(epoch, train_metrics, val_metrics)
                
                # Early stopping
                if (self.config.early_stopping_patience > 0 and
                    metric_for_patience is not None and
                    self.patience_counter >= self.config.early_stopping_patience):
                    self.logger.info(f"Early stopping triggered after {self.patience_counter} epochs without improvement")
                    break
        
        total_time = time.time() - self.training_start_time if self.training_start_time else 0.0
        self.logger.info(f"Training completed! Total time: {total_time / 60:.2f} minutes.")
        self._finalize_logging(total_time)

    def _log_epoch(self,
                   epoch: int,
                   train_metrics: Dict[str, float],
                   val_metrics: Dict[str, float],
                   epoch_time: float,
                   learning_rate: float):
        """Log epoch-level metrics to auxiliary systems"""
        if self.tensorboard_monitor:
            try:
                self.tensorboard_monitor.log_training_metrics(
                    epoch=epoch + 1,
                    batch=0,
                    train_loss=train_metrics.get('loss', 0.0),
                    learning_rate=learning_rate
                )
                if val_metrics:
                    self.tensorboard_monitor.log_validation_metrics(
                        epoch=epoch + 1,
                        val_loss=val_metrics.get('loss', train_metrics.get('loss', 0.0))
                    )
                self.tensorboard_monitor.system_writer.add_scalar('Epoch/DurationSeconds', epoch_time, epoch)
            except Exception as e:
                self.logger.warning(f"Failed to log TensorBoard metrics: {e}")

    def _export_pretrained(self,
                           epoch: int,
                           train_metrics: Dict[str, float],
                           val_metrics: Dict[str, float]):
        """Export the current model weights in HuggingFace format"""
        metric_value = val_metrics.get('loss')
        if metric_value is None:
            metric_value = train_metrics.get('loss')
        if metric_value is None:
            return
        
        if metric_value >= self.best_export_metric - self.config.early_stopping_delta:
            return
        
        model_to_export = self.model.module if hasattr(self.model, 'module') else self.model
        
        # Clean export directory but keep parent
        for child in list(self.export_dir.iterdir()):
            if child.is_file():
                child.unlink()
            else:
                shutil.rmtree(child)
        
        model_to_export.eval()
        try:
            model_to_export.save_pretrained(str(self.export_dir))
        except Exception as e:
            self.logger.error(f"Failed to export model in HuggingFace format: {e}")
            return
        
        metadata = {
            "epoch": epoch + 1,
            "train_loss": float(train_metrics.get('loss', 0.0)),
            "val_loss": float(val_metrics.get('loss', train_metrics.get('loss', 0.0))),
            "exported_at": datetime.now().isoformat()
        }
        with open(self.export_metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        self.best_export_metric = metric_value
        self.logger.info(
            f"Exported HuggingFace checkpoint to {self.export_dir} "
            f"(epoch {epoch + 1}, val_loss={metadata['val_loss']:.6f})"
        )

    def _finalize_logging(self, total_time: float):
        """Close loggers and flush final metrics"""
        if self.tensorboard_monitor:
            try:
                self.tensorboard_monitor.system_writer.add_scalar(
                    'Training/TotalDurationSeconds',
                    total_time,
                    self.current_epoch
                )
                self.tensorboard_monitor.close()
            except Exception as e:
                self.logger.warning(f"Failed to finalize TensorBoard monitor: {e}")
    
    def _log_metrics(self, epoch: int, train_metrics: Dict[str, float], val_metrics: Dict[str, float]):
        """Log training metrics"""
        # Log to console
        log_msg = f"Epoch {epoch + 1} - Train Loss: {train_metrics.get('loss', 0):.6f}"
        if val_metrics:
            log_msg += f", Val Loss: {val_metrics.get('loss', 0):.6f}"
        
        if 'rmse' in train_metrics:
            log_msg += f", Train RMSE: {train_metrics['rmse']:.6f}"
        if 'rmse' in val_metrics:
            log_msg += f", Val RMSE: {val_metrics['rmse']:.6f}"
        
        self.logger.info(log_msg)
        
        # Log detailed metrics
        for metric_name, value in train_metrics.items():
            self.logger.debug(f"Train {metric_name}: {value}")
        
        for metric_name, value in val_metrics.items():
            self.logger.debug(f"Val {metric_name}: {value}")
    
    def evaluate(self, dataloader_name: str = 'test') -> Dict[str, float]:
        """Evaluate model on test data"""
        if dataloader_name not in self.dataloaders:
            self.logger.warning(f"No {dataloader_name} dataloader found")
            return {}
        
        self.logger.info(f"Evaluating on {dataloader_name} data...")
        
        self.model.eval()
        self.metrics_tracker.reset()
        
        device = next(self.model.parameters()).device
        
        with torch.no_grad():
            loader = self.dataloaders[dataloader_name]
            iterable = self._prefetch_loader(loader, device)
            with self._ema_eval_context():
                with enable_fast_kernels():
                    for batch in iterable:
                        batch_start_time = time.time()
                        (
                            series,
                            padding_mask,
                            id_mask,
                            target_price,
                        target_pct,
                        prev_close,
                        _,
                    ) = self._prepare_batch(batch, device)

                    (
                        loss,
                        predictions_pct,
                        targets_pct,
                        price_predictions,
                        matched_target_price,
                        prev_close_tensor,
                        quantile_tensor,
                    ) = self._forward_batch(
                        series,
                        padding_mask,
                        id_mask,
                        target_price,
                        target_pct,
                        prev_close,
                    )

                    self.metrics_tracker.update(
                        loss=loss.item(),
                        predictions=predictions_pct.unsqueeze(1) if self.config.compute_val_metrics else None,
                        targets=targets_pct.unsqueeze(1) if self.config.compute_val_metrics else None,
                        price_predictions=price_predictions.unsqueeze(1) if self.config.compute_val_metrics else None,
                        price_targets=matched_target_price.unsqueeze(1) if self.config.compute_val_metrics else None,
                        batch_time=time.time() - batch_start_time,
                        prev_close=prev_close_tensor if self.config.compute_val_metrics else None,
                        quantile_predictions=quantile_tensor if (self.config.compute_val_metrics and quantile_tensor is not None) else None,
                        quantile_levels=self.config.quantile_levels if (self.config.compute_val_metrics and quantile_tensor is not None) else None,
                    )
        
        metrics = self.metrics_tracker.compute_metrics()
        
        # Log evaluation results
        self.logger.info(f"Evaluation results on {dataloader_name}:")
        for metric_name, value in metrics.items():
            self.logger.info(f"  {metric_name}: {value}")
        
        return metrics


def main():
    """Example usage of TotoTrainer"""
    print("🚀 Toto Training Pipeline")
    
    # Configuration
    trainer_config = TrainerConfig(
        # Model config
        patch_size=12,
        stride=6,
        embed_dim=128,
        num_layers=6,
        num_heads=8,
        dropout=0.1,
        
        # Training config
        learning_rate=1e-4,
        weight_decay=0.01,
        batch_size=16,
        max_epochs=50,
        warmup_epochs=5,
        
        # Optimization
        optimizer="adamw",
        scheduler="cosine",
        gradient_clip_val=1.0,
        use_mixed_precision=True,
        require_gpu=True,
        
        # Validation
        validation_frequency=1,
        early_stopping_patience=10,
        
        # Checkpointing
        save_every_n_epochs=5,
        keep_last_n_checkpoints=3,
        
        # Logging
        log_level="INFO",
        log_file="training.log"
    )
    
    # Dataloader config
    dataloader_config = DataLoaderConfig(
        train_data_path="trainingdata/train",
        test_data_path="trainingdata/test",
        batch_size=16,
        sequence_length=96,
        prediction_length=24,
        validation_split=0.2,
        add_technical_indicators=True,
        normalization_method="robust"
    )
    
    # Create trainer
    trainer = TotoTrainer(trainer_config, dataloader_config)
    
    try:
        # Prepare data and setup model
        trainer.prepare_data()
        trainer.setup_model()
        
        # Start training
        trainer.train()
        
        # Evaluate on test set
        test_metrics = trainer.evaluate('test')
        print(f"✅ Training completed! Test metrics: {test_metrics}")
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
