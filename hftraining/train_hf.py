#!/usr/bin/env python3
"""
HuggingFace-style Training Script Entry Point
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from typing import Optional, List
from pathlib import Path
import matplotlib.pyplot as plt
from tqdm import tqdm
import json
from datetime import datetime
import warnings
import random
import contextlib
import math
from dataclasses import asdict
warnings.filterwarnings('ignore')
from torch.utils.data import DataLoader, Dataset
import sys
import os
import time
import shutil
import subprocess

# Add current directory to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)
# Add parent directory to path
sys.path.append(os.path.dirname(current_dir))

from hf_trainer import (
    HFTrainingConfig,
    TransformerTradingModel,
    MixedPrecisionTrainer,
    EarlyStopping,
    adaptive_clip_grad_
)
from modern_optimizers import get_optimizer as legacy_get_optimizer
from logging_utils import get_logger, MetricsTracker
try:
    import psutil  # Optional for CPU metrics
except Exception:
    psutil = None
from auto_tune import AutoBatchTuner
from differentiable_profit import compute_portfolio_pnl, sharpe_like_ratio

from wandboard import WandBoardLogger

try:
    from .asset_metadata import get_asset_class, get_asset_class_id, get_trading_fee
except ImportError:  # pragma: no cover - script execution
    from asset_metadata import get_asset_class, get_asset_class_id, get_trading_fee  # type: ignore

from loss_utils import TRADING_FEE

from traininglib import (
    bf16_supported,
    make_optimizer,
    WarmupCosine,
    write_report_markdown,
)
from hftraining.engine_speed import compile_model as compile_for_speed, fast_context
from hftraining.metrics import crps_from_quantiles, dm_test


class StockDataset(Dataset):
    """Dataset for stock trading data."""

    def __init__(
        self,
        data,
        sequence_length: int = 60,
        prediction_horizon: int = 5,
        processor: Optional['StockDataProcessor'] = None,  # quote to avoid circular import
        symbol: Optional[str] = None,
    ):
        self.data = data
        self.sequence_length = sequence_length
        self.prediction_horizon = prediction_horizon
        self.processor = processor
        self.symbol = symbol.upper() if symbol else None
        self.feature_names = getattr(processor, 'feature_names', None)
        if self.feature_names and 'close' in self.feature_names:
            self.close_index = self.feature_names.index('close')
        else:
            self.close_index = 3  # Fallback to traditional OHLC ordering

        asset_class = "equity"
        default_fee = TRADING_FEE
        class_id = 0
        if self.symbol:
            try:
                asset_class = get_asset_class(self.symbol)
                default_fee = get_trading_fee(self.symbol)
                class_id = get_asset_class_id(self.symbol)
            except FileNotFoundError:
                asset_class = "equity"
                default_fee = TRADING_FEE
                class_id = 0
        self.asset_class_name = asset_class
        self.asset_class_ids = torch.tensor([class_id], dtype=torch.long)
        self.per_asset_fees = torch.tensor([float(default_fee)], dtype=torch.float32)

        if len(data) < sequence_length + prediction_horizon:
            raise ValueError(f"Dataset too small: {len(data)} < {sequence_length + prediction_horizon}")

    def __len__(self):
        return len(self.data) - self.sequence_length - self.prediction_horizon + 1

    def _compute_future_return(self, sequence_np: np.ndarray, targets_np: np.ndarray) -> float:
        if self.processor is not None:
            raw_seq = self.processor.inverse_transform(sequence_np)
            raw_targets = self.processor.inverse_transform(targets_np)
            current_price = raw_seq[-1, self.close_index]
            next_price = raw_targets[0, self.close_index]
        else:
            current_price = sequence_np[-1, self.close_index]
            next_price = targets_np[0, self.close_index]

        return float((next_price - current_price) / (current_price + 1e-8))

    def __getitem__(self, idx):
        start_idx = idx
        end_idx = idx + self.sequence_length

        sequence_np = self.data[start_idx:end_idx]
        target_start = end_idx
        target_end = target_start + self.prediction_horizon
        targets_np = self.data[target_start:target_end]

        sequence = torch.from_numpy(sequence_np).float()
        targets = torch.from_numpy(targets_np).float()

        future_return = self._compute_future_return(sequence_np, targets_np)

        next_price = targets_np[0, self.close_index]
        current_price = sequence_np[-1, self.close_index]

        if next_price > current_price * 1.01:
            action_label = 0  # Buy
        elif next_price < current_price * 0.99:
            action_label = 2  # Sell
        else:
            action_label = 1  # Hold

        return {
            'input_ids': sequence,
            'labels': targets,
            'future_returns': torch.tensor(future_return, dtype=torch.float32).unsqueeze(0),
            'action_labels': torch.tensor(action_label, dtype=torch.long),
            'attention_mask': torch.ones(self.sequence_length),
            'last_close': torch.tensor(sequence_np[-1, self.close_index], dtype=torch.float32),
            'asset_class_ids': self.asset_class_ids.clone(),
            'per_asset_fees': self.per_asset_fees.clone(),
        }


class HFTrainer:
    """HuggingFace-style trainer for stock prediction"""
    
    def __init__(self, model, config: HFTrainingConfig, train_dataset, eval_dataset=None):
        self.model = model
        self.config = config
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.close_index = getattr(train_dataset, 'close_index', 3)
        
        # Setup device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self._fast_ctx = contextlib.ExitStack()
        self._fast_ctx.enter_context(fast_context())
        
        # Setup data parallel if available
        if torch.cuda.device_count() > 1 and config.use_data_parallel:
            self.model = nn.DataParallel(self.model)
        
        compile_requested = bool(getattr(self.config, 'use_compile', False))
        original_model_ref = self.model
        if compile_requested:
            self.model = compile_for_speed(self.model, dynamic=True)
        self._compile_enabled = compile_requested and self.model is not original_model_ref

        # Setup mixed precision (prefer BF16 on capable GPUs)
        mp_dtype = torch.bfloat16 if (bf16_supported() and getattr(self.config, 'use_bfloat16', True)) else None
        self.mp_trainer = MixedPrecisionTrainer(config.use_mixed_precision, dtype=mp_dtype)
        
        # Setup optimizer
        self.optimizer = self._create_optimizer()
        
        # Setup scheduler
        self.scheduler = self._create_scheduler()
        
        # Setup early stopping
        self.early_stopping = EarlyStopping(
            patience=config.early_stopping_patience,
            threshold=config.early_stopping_threshold,
            greater_is_better=config.greater_is_better
        )
        
        # Enhanced logging (initialize first)
        self.training_logger = get_logger(config.logging_dir, "training")
        self.metrics_tracker = MetricsTracker()
        
        if compile_requested:
            if self._compile_enabled:
                self.training_logger.info('Enabled torch.compile for the model')
            else:
                self.training_logger.logger.warning('torch.compile unavailable or disabled by runtime guards.')

        # Setup logging (TensorBoard)
        self.setup_logging()
        
        # Training state
        self.global_step = 0
        self.current_epoch = 0
        self.best_metric = None
        self.start_time = None
        self.last_step_time = None
        self.cum_return_train = 0.0
        self.train_return_steps = 0
        self.cum_return_equity = 0.0
        self.cum_return_crypto = 0.0
        self.train_equity_steps = 0
        self.train_crypto_steps = 0

    @property
    def step(self) -> int:
        """Backwards-compatible alias for ``global_step`` used by legacy tests."""
        return self.global_step

    @step.setter
    def step(self, value: int) -> None:
        self.global_step = int(value)

    def _get_gpu_metrics(self):
        """Safely collect GPU metrics if CUDA is available."""
        metrics = {}
        try:
            if torch.cuda.is_available():
                device_idx = 0
                metrics['gpu_memory_allocated_mb'] = torch.cuda.memory_allocated(device_idx) / (1024**2)
                metrics['gpu_memory_reserved_mb'] = torch.cuda.memory_reserved(device_idx) / (1024**2)
                metrics['gpu_max_memory_allocated_mb'] = torch.cuda.max_memory_allocated(device_idx) / (1024**2)
                # Optional: nvidia-smi utilization
                if shutil.which('nvidia-smi'):
                    try:
                        out = subprocess.check_output([
                            'nvidia-smi',
                            '--query-gpu=utilization.gpu,memory.used',
                            '--format=csv,noheader,nounits',
                            '-i', '0'
                        ], stderr=subprocess.DEVNULL, text=True).strip()
                        if out:
                            util_str, mem_used_str = out.split(',')
                            metrics['gpu_utilization_pct'] = float(util_str.strip())
                            metrics['gpu_memory_used_mb'] = float(mem_used_str.strip())
                    except Exception:
                        pass
        except Exception:
            # Never let metrics collection break training
            pass
        return metrics
        
    def _create_optimizer(self):
        """Create optimizer using the shared traininglib factory."""
        name = (self.config.optimizer_name or "adamw").lower()
        extra_kwargs = {}
        if name.startswith("muon"):
            extra_kwargs.update({
                "momentum": getattr(self.config, "muon_momentum", 0.95),
                "nesterov": getattr(self.config, "muon_nesterov", True),
                "ns_steps": getattr(self.config, "muon_ns_steps", 5),
            })

        try:
            return make_optimizer(
                self.model,
                name=name,
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay,
                betas=(self.config.adam_beta1, self.config.adam_beta2),
                eps=self.config.adam_epsilon,
                fused=True,
                **{k: v for k, v in extra_kwargs.items() if v is not None},
            )
        except ValueError:
            # Unknown optimizer name, fall back to legacy registry.
            try:
                return legacy_get_optimizer(
                    name,
                    self.model.parameters(),
                    lr=self.config.learning_rate,
                    betas=(self.config.adam_beta1, self.config.adam_beta2),
                    eps=self.config.adam_epsilon,
                    weight_decay=self.config.weight_decay,
                    **extra_kwargs,
                )
            except Exception as exc:
                if hasattr(self, 'training_logger'):
                    self.training_logger.logger.warning(f"Falling back to AdamW optimizer due to: {exc}")
                return torch.optim.AdamW(
                    self.model.parameters(),
                    lr=self.config.learning_rate,
                    betas=(self.config.adam_beta1, self.config.adam_beta2),
                    eps=self.config.adam_epsilon,
                    weight_decay=self.config.weight_decay,
                )
        except Exception as exc:  # pragma: no cover - runtime guard.
            if hasattr(self, 'training_logger'):
                self.training_logger.logger.warning(f"Optimizer setup failed ({name}), defaulting to AdamW: {exc}")
            return torch.optim.AdamW(
                self.model.parameters(),
                lr=self.config.learning_rate,
                betas=(self.config.adam_beta1, self.config.adam_beta2),
                eps=self.config.adam_epsilon,
                weight_decay=self.config.weight_decay,
            )

    def _create_scheduler(self):
        """Create learning rate scheduler"""
        total_steps = getattr(self.config, "max_steps", 0)
        if total_steps <= 0:
            dataset_size = len(self.train_dataset) if hasattr(self.train_dataset, "__len__") else 0
            if dataset_size and getattr(self.config, "batch_size", 0):
                steps_per_epoch = math.ceil(dataset_size / max(1, self.config.batch_size))
                total_steps = steps_per_epoch * max(1, getattr(self.config, "num_train_epochs", 1))
            else:
                total_steps = 1
        min_lr = float(getattr(self.config, "min_learning_rate", 0.0))
        return WarmupCosine(
            self.optimizer,
            warmup_steps=self.config.warmup_steps,
            total_steps=total_steps,
            min_lr=min_lr,
        )
    
    def setup_logging(self):
        """Setup logging directories and tensorboard"""
        # Resolve dirs relative to this file to avoid hftraining/hftraining nesting
        base_dir = Path(__file__).parent
        def _resolve_dir(path_str: str) -> Path:
            p = Path(path_str)
            if p.is_absolute():
                return p
            parts = p.parts
            if parts and parts[0].lower() == 'hftraining':
                p = Path(*parts[1:]) if len(parts) > 1 else Path('.')
            return base_dir / p

        # Normalize paths and write back to config for consistency downstream
        self.config.output_dir = str(_resolve_dir(self.config.output_dir))
        self.config.logging_dir = str(_resolve_dir(self.config.logging_dir))
        self.config.cache_dir = str(_resolve_dir(self.config.cache_dir))

        # Create directories
        Path(self.config.output_dir).mkdir(parents=True, exist_ok=True)
        Path(self.config.logging_dir).mkdir(parents=True, exist_ok=True)
        Path(self.config.cache_dir).mkdir(parents=True, exist_ok=True)

        # Setup experiment tracking
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        default_run_name = f"hf_training_{timestamp}"
        run_name = getattr(self.config, "wandb_run_name", None) or default_run_name
        tb_subdir = getattr(self.config, "tensorboard_subdir", None) or run_name
        wandb_config = (
            asdict(self.config)
            if hasattr(self.config, "__dataclass_fields__")
            else dict(self.config.__dict__)
        )
        self.metrics_logger = WandBoardLogger(
            run_name=run_name,
            project=getattr(self.config, "wandb_project", None),
            entity=getattr(self.config, "wandb_entity", None),
            tags=tuple(getattr(self.config, "wandb_tags", ()) or ()),
            group=getattr(self.config, "wandb_group", None),
            notes=getattr(self.config, "wandb_notes", None),
            mode=getattr(self.config, "wandb_mode", "auto"),
            enable_wandb=bool(getattr(self.config, "use_wandb", True)),
            log_dir=self.config.logging_dir,
            tensorboard_subdir=tb_subdir,
            config=wandb_config,
            settings=getattr(self.config, "wandb_settings", None),
        )
        self.tb_log_dir = str(self.metrics_logger.tensorboard_log_dir)
        # Preserve backwards compatibility for code paths that still expect a raw writer
        self.writer = self.metrics_logger.tensorboard_writer
        if self.metrics_logger.wandb_enabled:
            self.training_logger.info(
                f"W&B logging enabled - project: {self.metrics_logger.project}, run: {self.metrics_logger.run_name}"
            )
        else:
            msg = "W&B logging disabled; mirroring metrics to TensorBoard only."
            if self.metrics_logger.last_error is not None:
                msg += f" Last error: {self.metrics_logger.last_error}"
            self.training_logger.info(msg)
        # Perf CSV file
        self.perf_csv_path = Path(self.tb_log_dir) / 'perf_metrics.csv'
        try:
            with open(self.perf_csv_path, 'w') as f:
                f.write('step,epoch,step_time_s,samples_per_sec,tokens_per_sec,lr,gpu_mem_alloc_mb\n')
        except Exception:
            pass
        
        self.training_logger.info(f"TensorBoard logging to: {self.tb_log_dir}")
        self.training_logger.info(f"Output directory: {self.config.output_dir}")
    
    def train(self):
        """Main training loop"""
        
        # Start timing
        self.start_time = time.time()

        # Optional auto-tuning (batch size / accumulation)
        auto_env = os.environ.get('AUTO_TUNE', '0') == '1'
        if getattr(self.config, 'auto_tune', False) or auto_env:
            try:
                pin_mem = bool(torch.cuda.is_available())
                tuner = AutoBatchTuner(self.device, steps=int(getattr(self.config, 'tuning_steps', 10)), num_workers=0, pin_memory=pin_mem)
                best = tuner.tune(self, self.train_dataset, self.config.batch_size)
                old_bs = self.config.batch_size
                self.config.batch_size = int(best.get('batch_size', old_bs))
                # Adjust accumulation to reach target effective batch if requested
                target_eff = getattr(self.config, 'target_effective_batch_size', None)
                if target_eff and self.config.batch_size > 0:
                    from math import ceil
                    acc = max(1, ceil(target_eff / self.config.batch_size))
                    acc = min(acc, int(getattr(self.config, 'max_gradient_accumulation', 16)))
                    self.config.gradient_accumulation_steps = acc
                self.training_logger.info(
                    f"Auto-tune: batch_size {old_bs} -> {self.config.batch_size}; grad_accum={self.config.gradient_accumulation_steps}"
                )
            except Exception as e:
                self.training_logger.warning(f"Auto-tune skipped due to error: {e}")
        
        # Log training start
        model_info = {
            'hidden_size': self.config.hidden_size,
            'num_layers': self.config.num_layers,
            'total_params': sum(p.numel() for p in self.model.parameters())
        }
        
        config_dict = {
            'experiment_name': 'hf_training',
            'description': 'HuggingFace-style stock prediction training',
            'optimizer': self.config.optimizer_name,
            'learning_rate': self.config.learning_rate,
            'batch_size': self.config.batch_size,
            'max_steps': self.config.max_steps,
            'device': str(self.device)
        }
        
        self.training_logger.log_training_start(config_dict, model_info)
        
        # Create data loaders
        pin_mem = bool(torch.cuda.is_available())
        num_workers = max(0, int(getattr(self.config, 'dataloader_num_workers', 0)))
        persistent_workers = bool(getattr(self.config, 'persistent_workers', True) and num_workers > 0)
        prefetch_factor = int(getattr(self.config, 'prefetch_factor', 2)) if num_workers > 0 else None
        train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_mem,
            persistent_workers=persistent_workers,
            **({"prefetch_factor": prefetch_factor} if prefetch_factor else {})
        )
        
        eval_loader = None
        if self.eval_dataset:
            eval_loader = DataLoader(
                self.eval_dataset,
                batch_size=self.config.batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=pin_mem,
                persistent_workers=persistent_workers,
                **({"prefetch_factor": prefetch_factor} if prefetch_factor else {})
            )
        
        # Training loop
        self.model.train()
        pbar = tqdm(total=self.config.max_steps, desc="Training", ncols=120)
        
        epoch = 0
        while self.global_step < self.config.max_steps:
            epoch += 1
            self.current_epoch = epoch
            
            # Log epoch start
            if self.config.max_steps // len(train_loader) > 1:
                total_epochs = self.config.max_steps // len(train_loader)
                self.training_logger.log_epoch_start(epoch, total_epochs)
            else:
                self.training_logger.log_epoch_start(epoch)
            
            epoch_start_time = time.time()
            epoch_loss = 0
            epoch_steps = 0
            
            for batch_idx, batch in enumerate(train_loader):
                if self.global_step >= self.config.max_steps:
                    break
                
                # Move batch to device
                non_block = bool(torch.cuda.is_available())
                batch = {k: v.to(self.device, non_blocking=non_block) for k, v in batch.items()}
                
                # Training step
                step_start = time.time()
                loss = self.training_step(batch)
                if torch.cuda.is_available():
                    try:
                        torch.cuda.synchronize()
                    except Exception:
                        pass
                self.last_step_time = max(1e-9, time.time() - step_start)
                epoch_loss += loss
                epoch_steps += 1
                
                # Enhanced Logging
                if self.global_step % self.config.logging_steps == 0:
                    # Log to TensorBoard
                    batch_size = self.config.batch_size
                    samples_per_sec = float(batch_size) / self.last_step_time if self.last_step_time else 0.0
                    # Estimate tokens/sec akin to nanochat logging
                    seq_len = getattr(self.config, 'sequence_length', None)
                    tokens_per_sec = float(samples_per_sec * seq_len) if seq_len else None
                    sys_metrics = self._get_gpu_metrics()
                    avg_total_return = self.cum_return_train / max(1, self.train_return_steps)
                    equity_steps = max(1, self.train_equity_steps)
                    crypto_steps = max(1, self.train_crypto_steps)
                    avg_equity_return = self.cum_return_equity / equity_steps
                    avg_crypto_return = self.cum_return_crypto / crypto_steps
                    tb_metrics = {
                        'train/loss': loss,
                        'train/learning_rate': self.scheduler.get_last_lr()[0],
                        'train/epoch': epoch,
                        'train/step_time_s': self.last_step_time,
                        'train/samples_per_sec': samples_per_sec,
                        'train/avg_return': avg_total_return,
                        'train/cum_return': self.cum_return_train,
                        'train/avg_return_equity': avg_equity_return,
                        'train/avg_return_crypto': avg_crypto_return,
                    }
                    if tokens_per_sec is not None:
                        tb_metrics['train/tokens_per_sec'] = tokens_per_sec
                    # Add GPU metrics to TB if present
                    for k, v in sys_metrics.items():
                        tb_metrics[f'system/{k}'] = v
                    # CPU utilization (optional)
                    if psutil is not None:
                        try:
                            tb_metrics['system/cpu_percent'] = psutil.cpu_percent(interval=None)
                        except Exception:
                            pass
                    self.log_metrics(tb_metrics)
                    
                    # Log to file and console
                    metrics = {
                        'loss': loss,
                        'learning_rate': self.scheduler.get_last_lr()[0],
                        'epoch': epoch,
                        'avg_return': avg_total_return,
                        'avg_return_equity': avg_equity_return,
                        'avg_return_crypto': avg_crypto_return,
                    }
                    if tokens_per_sec is not None:
                        metrics['tokens_per_sec'] = tokens_per_sec
                    # Also report GPU mem to file logger if available
                    cpu_pct = None
                    if psutil is not None:
                        try:
                            cpu_pct = psutil.cpu_percent(interval=None)
                        except Exception:
                            cpu_pct = None
                    self.training_logger.log_resource_usage(
                        gpu_memory=sys_metrics.get('gpu_memory_allocated_mb'),
                        cpu_percent=cpu_pct
                    )
                    self.training_logger.log_step_metrics(self.global_step, metrics, "train")
                    self.metrics_tracker.add_metric(self.global_step, "train", **metrics)
                    # Write perf CSV
                    try:
                        with open(self.perf_csv_path, 'a') as f:
                            tok_str = f"{tokens_per_sec:.1f}" if tokens_per_sec is not None else "0.0"
                            f.write(
                                f"{self.global_step},{epoch},{self.last_step_time:.6f},{samples_per_sec:.3f},{tok_str},{self.scheduler.get_last_lr()[0]:.6e},{sys_metrics.get('gpu_memory_allocated_mb', 0):.1f}\n"
                            )
                    except Exception:
                        pass
                    
                    # Update progress bar
                    pbar.set_description(
                        self.training_logger.create_progress_bar_desc(
                            self.global_step, loss, self.scheduler.get_last_lr()[0]
                        )
                    )
                
                # Evaluation
                if (self.eval_dataset and 
                    self.global_step % self.config.eval_steps == 0 and 
                    self.global_step > 0):
                    eval_metrics = self.evaluate(eval_loader)
                    
                    # Log to TensorBoard
                    self.log_metrics(eval_metrics, prefix='eval')
                    
                    # Enhanced evaluation logging
                    self.training_logger.log_step_metrics(self.global_step, eval_metrics, "eval")
                    self.metrics_tracker.add_metric(self.global_step, "eval", **eval_metrics)

                    # If profit tracker provided recent metrics, print concise profit summary too
                    try:
                        if hasattr(self, 'last_profit_metrics') and self.last_profit_metrics is not None:
                            pm = self.last_profit_metrics
                            self.training_logger.info(
                                f"   Profit: Return {pm.total_return:.2%} | Sharpe {pm.sharpe_ratio:.2f} | MaxDD {pm.max_drawdown:.2%}"
                            )
                    except Exception:
                        pass
                    
                    # Early stopping check
                    metric_value = eval_metrics.get('loss', loss)
                    self.early_stopping(metric_value)
                    
                    if self.early_stopping.should_stop:
                        self.training_logger.log_early_stopping(
                            self.global_step, 
                            self.config.early_stopping_patience
                        )
                        break
                
                # Save checkpoint
                if self.global_step % self.config.save_steps == 0 and self.global_step > 0:
                    checkpoint_path = self.save_checkpoint()
                    self.training_logger.log_checkpoint_saved(self.global_step, checkpoint_path)
                
                pbar.update(1)
            
            if self.early_stopping.should_stop:
                break
            
            # Log epoch summary
            epoch_time = time.time() - epoch_start_time
            avg_epoch_loss = epoch_loss / epoch_steps if epoch_steps > 0 else 0
            self.training_logger.log_epoch_summary(epoch, avg_epoch_loss, epoch_time)
        
        pbar.close()
        
        # Calculate total training time
        total_training_time = time.time() - self.start_time
        
        # Save final model
        final_checkpoint_path = self.save_checkpoint(is_final=True)
        
        # Get final metrics
        final_metrics = {
            'final_loss': float(self.metrics_tracker.get_recent_avg('loss', 10)),
            'total_steps': int(self.global_step),
            'total_epochs': int(epoch),
        }
        avg_total_return = self.cum_return_train / max(1, self.train_return_steps)
        avg_equity_return = self.cum_return_equity / max(1, self.train_equity_steps)
        avg_crypto_return = self.cum_return_crypto / max(1, self.train_crypto_steps)
        final_metrics.update(
            {
                'avg_train_return': float(avg_total_return),
                'avg_equity_return': float(avg_equity_return),
                'avg_crypto_return': float(avg_crypto_return),
                'train_return_samples': int(self.train_return_steps),
            }
        )
        
        # Log training completion
        self.training_logger.log_training_complete(total_training_time, final_metrics)

        report_args = asdict(self.config) if hasattr(self.config, "__dataclass_fields__") else dict(self.config.__dict__)
        eval_summary = dict(getattr(self.training_logger, "best_metrics", {}))
        notes = None
        if eval_summary.get('best_eval_loss') is not None:
            best_step = eval_summary.get('best_step')
            notes = f"Best eval loss {eval_summary['best_eval_loss']:.6f} at global step {best_step}."
        report_path = Path(self.config.output_dir) / "run_report.md"
        try:
            write_report_markdown(
                str(report_path),
                title="HF Training Run",
                args=report_args,
                train_metrics=final_metrics,
                eval_metrics=eval_summary or None,
                notes=notes,
            )
            self.training_logger.info(f"Run report written to {report_path}")
        except Exception as exc:
            self.training_logger.logger.debug(f"Failed to write report.md: {exc}")

        if hasattr(self, "metrics_logger"):
            final_prefixed = {f"final/{k}": v for k, v in final_metrics.items()}
            final_prefixed["final/training_time_s"] = float(total_training_time)
            try:
                self.metrics_logger.log(final_prefixed, step=self.global_step, commit=True)
            except Exception:
                pass
            try:
                self.metrics_logger.finish()
            except Exception:
                pass

        self._fast_ctx.close()
        return self.model

    def evaluation_step(self):
        """Execute a validation pass and return aggregated metrics."""
        if self.eval_dataset is None:
            return {}
        eval_loader = DataLoader(
            self.eval_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.dataloader_num_workers,
            pin_memory=torch.cuda.is_available(),
        )
        return self.evaluate(eval_loader)

    def training_step(self, batch):
        """Single training step with gradient accumulation"""
        # Only zero grad on accumulation boundary
        if self.global_step % self.config.gradient_accumulation_steps == 0:
            self.optimizer.zero_grad(set_to_none=True)

        batch_return = None

        with self.mp_trainer.autocast():
            # Forward pass
            inputs = batch['input_ids']
            # Tiny input jitter augmentation on normalized inputs
            if self.model.training and getattr(self.config, 'input_noise_std', 0.0) > 0:
                if random.random() < getattr(self.config, 'input_noise_prob', 0.0):
                    noise = torch.randn_like(inputs) * float(self.config.input_noise_std)
                    max_mag = float(getattr(self.config, 'input_noise_clip', 0.02))
                    noise = torch.clamp(noise, -max_mag, max_mag)
                    inputs = inputs + noise

            outputs = self.model(
                inputs,
                attention_mask=batch['attention_mask']
            )
            
            # Calculate losses with label smoothing for stability
            action_loss = F.cross_entropy(
                outputs['action_logits'],
                batch['action_labels'],
                label_smoothing=0.1
            )

            price_loss = F.mse_loss(
                outputs['price_predictions'],
                batch['labels'][:, :self.config.prediction_horizon, self.close_index]
            )

            future_returns = batch.get('future_returns')
            if future_returns is None:
                current_close = batch['input_ids'][:, -1, self.close_index]
                next_close = batch['labels'][:, 0, self.close_index]
                future_returns = ((next_close - current_close) / (current_close + 1e-8)).unsqueeze(-1)

            device = outputs['action_logits'].device
            future_returns = future_returns.to(device)
            allocations = outputs.get('allocations')
            transaction_cost = float(getattr(self.config, 'transaction_cost_bps', 0.0)) / 10000.0

            asset_class_ids = batch.get('asset_class_ids')
            if asset_class_ids is not None:
                asset_class_ids = asset_class_ids.to(device)
            per_asset_fees = batch.get('per_asset_fees')
            if per_asset_fees is not None:
                per_asset_fees = per_asset_fees.to(device)

            pnl_result = compute_portfolio_pnl(
                allocations,
                future_returns,
                transaction_cost,
                per_asset_costs=per_asset_fees,
                return_per_asset=True,
            )
            if isinstance(pnl_result, tuple):
                pnl, per_asset_net = pnl_result
            else:
                pnl = pnl_result
                per_asset_net = None
            profit_loss = -pnl.mean()
            sharpe_penalty = -sharpe_like_ratio(pnl)

            profit_weight = float(getattr(self.config, 'profit_loss_weight', 0.0))
            if profit_weight > 0.0:
                warmup = int(getattr(self.config, 'profit_curriculum_warmup_steps', 0))
                schedule = int(getattr(self.config, 'profit_curriculum_steps', 0))
                if self.global_step < warmup:
                    effective_profit_weight = 0.0
                else:
                    if schedule > 0:
                        progress = min(1.0, (self.global_step - warmup) / float(schedule))
                    else:
                        progress = 1.0
                    effective_profit_weight = profit_weight * progress
            else:
                effective_profit_weight = 0.0

            profit_component = effective_profit_weight * (profit_loss + 0.1 * sharpe_penalty)

            batch_return = pnl.mean().item()
            batch_equity_return = None
            batch_crypto_return = None
            if per_asset_net is not None and asset_class_ids is not None:
                if asset_class_ids.dim() == 1:
                    asset_class_ids = asset_class_ids.unsqueeze(0).expand_as(per_asset_net)
                equity_mask = (asset_class_ids == 0).to(per_asset_net.dtype)
                crypto_mask = (asset_class_ids == 1).to(per_asset_net.dtype)
                if torch.count_nonzero(equity_mask).item() > 0:
                    equity_returns = (per_asset_net * equity_mask).sum(dim=-1)
                    batch_equity_return = equity_returns.mean().item()
                if torch.count_nonzero(crypto_mask).item() > 0:
                    crypto_returns = (per_asset_net * crypto_mask).sum(dim=-1)
                    batch_crypto_return = crypto_returns.mean().item()

            total_loss = (action_loss + 0.5 * price_loss + profit_component) / self.config.gradient_accumulation_steps

        # Backward pass
        if not torch.isfinite(total_loss):
            # Skip step on NaN/Inf loss
            self.training_logger.logger.warning(f"Non-finite loss at step {self.global_step}: {total_loss.item()}")
            self.optimizer.zero_grad(set_to_none=True)
            self.global_step += 1
            return float('inf')

        scaled_loss = self.mp_trainer.scale_loss(total_loss)
        scaled_loss.backward()
        
        should_step = ((self.global_step + 1) % self.config.gradient_accumulation_steps == 0)

        # Gradient post-processing and safety checks only when stepping
        grad_norm = None
        if should_step:
            if self.mp_trainer.enabled:
                self.mp_trainer.scaler.unscale_(self.optimizer)

            if getattr(self.config, 'use_adaptive_grad_clip', False):
                adaptive_clip_grad_(
                    self.model.parameters(),
                    clip_factor=getattr(self.config, 'agc_clip_factor', 0.01),
                    eps=getattr(self.config, 'agc_eps', 1e-3)
                )

            if self.config.max_grad_norm > 0:
                grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)

            if getattr(self.config, 'skip_non_finite_grads', True):
                if grad_norm is not None and not torch.isfinite(grad_norm):
                    self.training_logger.logger.warning(
                        f"Non-finite grad norm at step {self.global_step}: {grad_norm} — skipping optimizer step"
                    )
                    self.optimizer.zero_grad(set_to_none=True)
                    self.global_step += 1
                    return total_loss.item() * self.config.gradient_accumulation_steps
        
        # Optimizer step only on accumulation boundary
        if should_step:
            self.mp_trainer.step_optimizer(self.optimizer)
            self.scheduler.step()
        
        self.global_step += 1
        
        # Update running training return if available
        try:
            if batch_return is not None:
                self.cum_return_train += float(batch_return)
                self.train_return_steps += 1
            if batch_equity_return is not None:
                self.cum_return_equity += float(batch_equity_return)
                self.train_equity_steps += 1
            if batch_crypto_return is not None:
                self.cum_return_crypto += float(batch_crypto_return)
                self.train_crypto_steps += 1
        except Exception:
            pass

        # Optional: log grad norm to TensorBoard
        if grad_norm is not None and self.global_step % max(1, self.config.logging_steps) == 0:
            try:
                self.metrics_logger.add_scalar('train/grad_norm', float(grad_norm), self.global_step)
            except Exception:
                pass

        return total_loss.item() * self.config.gradient_accumulation_steps  # Rescale for logging

    def benchmark_step(self, batch) -> float:
        """Run a forward+backward pass that does not step optimizers or advance state.

        - Does not modify global_step, scheduler, or optimizer state (other than transient grads).
        - Zeroes gradients before and after to avoid accumulation.
        Returns the scalar total loss value for reference.
        """
        was_training = self.model.training
        self.model.train()
        try:
            self.optimizer.zero_grad(set_to_none=True)
            with self.mp_trainer.autocast():
                inputs = batch['input_ids']
                outputs = self.model(inputs, attention_mask=batch.get('attention_mask'))
                action_loss = F.cross_entropy(
                    outputs['action_logits'],
                    batch['action_labels'],
                    label_smoothing=0.1
                )
                price_loss = F.mse_loss(
                    outputs['price_predictions'],
                    batch['labels'][:, :self.config.prediction_horizon, self.close_index]
                )
                future_returns = batch.get('future_returns')
                if future_returns is None:
                    current_close = batch['input_ids'][:, -1, self.close_index]
                    next_close = batch['labels'][:, 0, self.close_index]
                    future_returns = ((next_close - current_close) / (current_close + 1e-8)).unsqueeze(-1)
                device = outputs['action_logits'].device
                future_returns = future_returns.to(device)
                per_asset_fees = batch.get('per_asset_fees')
                if per_asset_fees is not None:
                    per_asset_fees = per_asset_fees.to(device)
                pnl = compute_portfolio_pnl(
                    outputs['allocations'],
                    future_returns,
                    float(getattr(self.config, 'transaction_cost_bps', 0.0)) / 10000.0,
                    per_asset_costs=per_asset_fees,
                )
                profit_loss = -pnl.mean()
                sharpe_penalty = -sharpe_like_ratio(pnl)
                profit_weight = float(getattr(self.config, 'profit_loss_weight', 0.0))
                total_loss = action_loss + 0.5 * price_loss + profit_weight * (profit_loss + 0.1 * sharpe_penalty)
            if self.mp_trainer.enabled:
                self.mp_trainer.scaler.scale(total_loss).backward()
                self.mp_trainer.scaler.unscale_(self.optimizer)
            else:
                total_loss.backward()
            # no optimizer or scheduler step here
            loss_val = float(total_loss.detach().item())
        finally:
            try:
                self.optimizer.zero_grad(set_to_none=True)
            except Exception:
                pass
            if not was_training:
                self.model.eval()
        return loss_val
    
    def evaluate(self, eval_loader):
        """Evaluation loop."""
        self.model.eval()

        total_loss = 0.0
        total_action_loss = 0.0
        total_price_loss = 0.0
        total_steps = 0
        total_return = 0.0
        total_return_equity = 0.0
        total_return_crypto = 0.0
        mae_samples: List[np.ndarray] = []
        baseline_mae_samples: List[np.ndarray] = []
        sum_squared_error = 0.0
        total_points = 0
        crps_values: List[float] = []
        quantile_levels = getattr(self.config, "quantile_levels", None)
        tau_tensor = None

        with torch.no_grad():
            for batch in tqdm(eval_loader, desc="Evaluating", leave=False):
                non_block = bool(torch.cuda.is_available())
                batch = {k: v.to(self.device, non_blocking=non_block) for k, v in batch.items()}

                outputs = self.model(
                    batch['input_ids'],
                    attention_mask=batch['attention_mask']
                )

                action_loss = F.cross_entropy(
                    outputs['action_logits'],
                    batch['action_labels']
                )

                targets = batch['labels'][:, :self.config.prediction_horizon, self.close_index]
                preds = outputs['price_predictions']

                price_loss = F.mse_loss(preds, targets)

                total_loss += action_loss.item() + 0.5 * price_loss.item()
                total_action_loss += action_loss.item()
                total_price_loss += price_loss.item()
                total_steps += 1

                mae_batch = torch.mean(torch.abs(preds - targets), dim=1)
                mae_samples.append(mae_batch.detach().cpu().numpy())

                baseline = batch['input_ids'][:, -1, self.close_index].unsqueeze(1).expand_as(targets)
                baseline_mae = torch.mean(torch.abs(baseline - targets), dim=1)
                baseline_mae_samples.append(baseline_mae.detach().cpu().numpy())

                sum_squared_error += torch.sum((preds - targets) ** 2).item()
                total_points += int(preds.numel())

                price_quantiles = outputs.get('price_quantiles')
                if price_quantiles is not None and quantile_levels:
                    if tau_tensor is None:
                        tau_tensor = torch.tensor(
                            quantile_levels,
                            device=price_quantiles.device,
                            dtype=price_quantiles.dtype,
                        )
                    crps_val = crps_from_quantiles(targets, price_quantiles, tau_tensor)
                    crps_values.append(float(crps_val.detach().cpu()))

                try:
                    future_returns = batch.get('future_returns')
                    if future_returns is None:
                        current_close = batch['input_ids'][:, -1, self.close_index]
                        next_close = batch['labels'][:, 0, self.close_index]
                        future_returns = ((next_close - current_close) / (current_close + 1e-8)).unsqueeze(-1)
                    per_asset_fees = batch.get('per_asset_fees')
                    pnl_result = compute_portfolio_pnl(
                        outputs['allocations'],
                        future_returns,
                        float(getattr(self.config, 'transaction_cost_bps', 0.0)) / 10000.0,
                        per_asset_costs=per_asset_fees,
                        return_per_asset=True,
                    )
                    if isinstance(pnl_result, tuple):
                        pnl, per_asset_net = pnl_result
                    else:
                        pnl = pnl_result
                        per_asset_net = None
                    total_return += float(pnl.mean().item())
                    if per_asset_net is not None:
                        asset_class_ids = batch.get('asset_class_ids')
                        if asset_class_ids is not None:
                            if asset_class_ids.dim() == 1:
                                asset_class_ids = asset_class_ids.unsqueeze(0).expand_as(per_asset_net)
                            equity_mask = (asset_class_ids == 0).to(per_asset_net.dtype)
                            crypto_mask = (asset_class_ids == 1).to(per_asset_net.dtype)
                            if torch.count_nonzero(equity_mask).item() > 0:
                                total_return_equity += float((per_asset_net * equity_mask).sum(dim=-1).mean().item())
                            if torch.count_nonzero(crypto_mask).item() > 0:
                                total_return_crypto += float((per_asset_net * crypto_mask).sum(dim=-1).mean().item())
                except Exception:
                    pass

        self.model.train()

        denom = max(1, total_steps)
        metrics = {
            'loss': total_loss / max(total_steps, 1),
            'action_loss': total_action_loss / max(total_steps, 1),
            'price_loss': total_price_loss / max(total_steps, 1),
            'avg_return': (total_return / denom),
            'cum_return': total_return,
            'avg_return_equity': (total_return_equity / denom),
            'avg_return_crypto': (total_return_crypto / denom),
            'cum_return_equity': total_return_equity,
            'cum_return_crypto': total_return_crypto,
        }

        if mae_samples:
            mae_array = np.concatenate(mae_samples)
            baseline_mae_array = np.concatenate(baseline_mae_samples)
            metrics['mae'] = float(mae_array.mean())
            metrics['baseline_mae'] = float(baseline_mae_array.mean())
            if total_points > 0:
                metrics['rmse'] = float(math.sqrt(sum_squared_error / total_points))
            dm_stat, dm_p = dm_test(mae_array, baseline_mae_array)
            metrics['dm_stat_vs_naive'] = float(dm_stat)
            metrics['dm_pvalue_vs_naive'] = float(dm_p)

        if crps_values:
            metrics['crps'] = float(np.mean(crps_values))

        return metrics
    
    def log_metrics(self, metrics, prefix='train', step=None, commit=None):
        """Log metrics to the unified experiment tracker."""
        if not hasattr(self, "metrics_logger"):
            return
        if step is None:
            step = self.global_step
        formatted = {}
        for key, value in metrics.items():
            metric_key = key if key.startswith(prefix) else f"{prefix}/{key}"
            formatted[metric_key] = value
        self.metrics_logger.log(formatted, step=step, commit=commit)
    
    def save_checkpoint(self, is_final=False):
        """Save model checkpoint"""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'global_step': self.global_step,
            'epoch': self.current_epoch,
            'config': self.config
        }
        
        if is_final:
            checkpoint_path = Path(self.config.output_dir) / "final_model.pth"
        else:
            checkpoint_path = Path(self.config.output_dir) / f"checkpoint_step_{self.global_step}.pth"
        
        torch.save(checkpoint, checkpoint_path)
        return str(checkpoint_path)


def load_data():
    """Load and prepare training data"""
    # This should be adapted to load your specific stock data
    # For now, we'll create dummy data
    print("Loading stock data...")
    
    # Try to load real data first
    data_path = Path("../trainingdata")
    if data_path.exists():
        # Look for CSV files in train directory
        train_path = data_path / "train"
        if train_path.exists():
            csv_files = list(train_path.glob("*.csv"))
            if csv_files:
                print(f"Found {len(csv_files)} CSV files in train directory")
                
                # Load and combine multiple stock files for better training
                all_data = []
                target_columns = 4  # OHLC only, skip volume for consistency
                
                for csv_file in csv_files[:50]:  # Load up to 50 stocks for more diverse data
                    try:
                        df = pd.read_csv(csv_file)
                        # Expected columns: timestamp, symbol, Open, High, Low, Close
                        if 'Open' in df.columns:
                            # Select OHLC columns only for consistency
                            cols = ['Open', 'High', 'Low', 'Close']
                            if all(col in df.columns for col in cols):
                                stock_data = df[cols].values
                                # Convert to float and handle any non-numeric values
                                stock_data = pd.DataFrame(stock_data).apply(pd.to_numeric, errors='coerce').ffill().fillna(0).values
                                if len(stock_data) > 100:  # Only use stocks with enough data
                                    all_data.append(stock_data)
                                    print(f"  Loaded {csv_file.stem}: {stock_data.shape}")
                    except Exception as e:
                        print(f"  Error loading {csv_file.stem}: {e}")
                        continue
                
                if all_data:
                    # Concatenate all stock data
                    data = np.vstack(all_data)
                    print(f"Loaded combined real data: {data.shape}")
                    return data
        
        # Fallback to root directory CSV files  
        csv_files = list(data_path.glob("*.csv"))
        if csv_files and csv_files[0].stem != 'data_summary':
            print(f"Found {len(csv_files)} CSV files")
            # Load first CSV as example
            df = pd.read_csv(csv_files[0])
            
            # Assume columns: [timestamp, open, high, low, close, volume, ...]
            if len(df.columns) >= 5:
                # Take OHLCV columns
                data = df.iloc[:, 1:6].values  # Skip timestamp
                print(f"Loaded real data: {data.shape}")
                return data
    
    # Generate synthetic data if no real data found
    print("Generating synthetic stock data...")
    np.random.seed(42)
    
    # Generate realistic stock price movements
    length = 10000
    initial_price = 100.0
    
    # Random walk with drift
    returns = np.random.normal(0.0005, 0.02, length)  # 0.05% daily return, 2% volatility
    prices = [initial_price]
    
    for i in range(1, length):
        new_price = prices[-1] * (1 + returns[i])
        prices.append(max(new_price, 0.01))  # Prevent negative prices
    
    prices = np.array(prices)
    
    # Generate OHLCV data
    data = []
    for i in range(len(prices)):
        price = prices[i]
        
        # Generate realistic OHLC from close price
        volatility = abs(np.random.normal(0, 0.01))
        high = price * (1 + volatility)
        low = price * (1 - volatility)
        open_price = np.random.uniform(low, high)
        
        # Volume (random but realistic)
        volume = np.random.exponential(1000000)
        
        data.append([open_price, high, low, price, volume])
    
    data = np.array(data)
    print(f"Generated synthetic data: {data.shape}")
    
    return data


def main():
    """Main training function"""
    # Check if GPU is available
    import os
    os.environ['LD_LIBRARY_PATH'] = '/home/lee/.pyenv/versions/3.12.7/lib/python3.12/site-packages/nvidia/nvjitlink/lib:' + os.environ.get('LD_LIBRARY_PATH', '')
    
    # Test GPU availability
    try:
        gpu_available = torch.cuda.is_available()
        if gpu_available:
            print(f"✅ GPU Available: {torch.cuda.get_device_name(0)}")
            print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
            device_str = "cuda"
            batch_size = 128  # Larger batch for GPU
        else:
            print("⚠️  No GPU detected, using CPU")
            device_str = "cpu"
            batch_size = 64
    except Exception as e:
        print(f"⚠️  GPU detection failed: {e}")
        device_str = "cpu"
        batch_size = 64
    
    # Configuration
    config = HFTrainingConfig(
        # Model
        hidden_size=256 if device_str == "cuda" else 128,  # Larger model for GPU
        num_layers=6 if device_str == "cuda" else 3,       # More layers for GPU
        num_heads=8 if device_str == "cuda" else 4,        # More heads for GPU
        
        # Training
        learning_rate=3e-4,  # Optimal LR
        warmup_steps=100,    # Reasonable warmup
        max_steps=5000 if device_str == "cuda" else 1000,  # More steps for GPU
        batch_size=batch_size,
        
        # Optimizer
        optimizer_name="lion" if device_str == "cuda" else "adamw",  # Lion for GPU, AdamW for CPU
        weight_decay=0.01,
        adam_beta1=0.9,
        adam_beta2=0.999,
        
        # Evaluation
        eval_steps=100,      # Regular evaluation
        save_steps=500,      # Save periodically
        logging_steps=20,    # Frequent logging
        
        # Training stability
        max_grad_norm=1.0,   # Global gradient clipping
        use_adaptive_grad_clip=True,  # Per-parameter AGC for stability
        agc_clip_factor=0.01,
        agc_eps=1e-3,
        skip_non_finite_grads=True,
        gradient_accumulation_steps=1 if device_str == "cuda" else 2,  # Less accumulation for GPU

        # Mixed precision for GPU
        use_mixed_precision=(device_str == "cuda"),
        use_bfloat16=True,
        use_compile=(device_str == "cuda"),
        allow_tf32=True,

        # DataLoader perf
        dataloader_num_workers=4 if device_str == "cuda" else 0,
        persistent_workers=True,
        prefetch_factor=2,

        # Micro augmentation (normalized inputs)
        input_noise_std=0.001,
        input_noise_prob=0.5,
        input_noise_clip=0.02,
        
        # Early stopping
        early_stopping_patience=15,
        early_stopping_threshold=0.001,
        
        # Output
        output_dir="hftraining/output",
        logging_dir="hftraining/logs"
    )
    
    # Load data
    data = load_data()
    
    # Normalize data
    data_mean = data.mean(axis=0)
    data_std = data.std(axis=0)
    data_normalized = (data - data_mean) / (data_std + 1e-8)
    
    # Split data
    train_size = int(0.8 * len(data_normalized))
    val_size = int(0.1 * len(data_normalized))
    
    train_data = data_normalized[:train_size]
    val_data = data_normalized[train_size:train_size + val_size]
    test_data = data_normalized[train_size + val_size:]
    
    print(f"Data splits - Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")
    
    # Create datasets
    train_dataset = StockDataset(
        train_data,
        sequence_length=config.sequence_length,
        prediction_horizon=config.prediction_horizon
    )
    
    val_dataset = StockDataset(
        val_data,
        sequence_length=config.sequence_length,
        prediction_horizon=config.prediction_horizon
    ) if len(val_data) > config.sequence_length + config.prediction_horizon else None
    
    print(f"Dataset sizes - Train: {len(train_dataset)}, Val: {len(val_dataset) if val_dataset else 0}")
    
    # Create model
    model = TransformerTradingModel(config, input_dim=data.shape[1])
    
    # Create trainer
    trainer = HFTrainer(
        model=model,
        config=config,
        train_dataset=train_dataset,
        eval_dataset=val_dataset
    )
    
    # Train model with error handling
    try:
        trained_model = trainer.train()
        return trained_model
    except KeyboardInterrupt:
        trainer.training_logger.warning("Training interrupted by user")
        checkpoint_path = trainer.save_checkpoint()
        trainer.training_logger.log_checkpoint_saved(trainer.global_step, checkpoint_path)
        return None
    except Exception as e:
        trainer.training_logger.log_error(e, trainer.global_step)
        return None


if __name__ == "__main__":
    main()
