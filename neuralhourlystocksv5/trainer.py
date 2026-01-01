"""V5 Trainer for hourly crypto trading with temperature annealing.

Key features:
- Dual optimizer (Muon + AdamW)
- Temperature annealing for differentiable fills
- Early stopping based on Sortino ratio
- Checkpoint management
"""
from __future__ import annotations

import json
import math
import os
import random
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader

from neuralhourlystocksv5.config import (
    DatasetConfigStocksV5,
    PolicyConfigStocksV5,
    SimulationConfigStocksV5,
    TrainingConfigStocksV5,
)
from neuralhourlystocksv5.data import (
    HOURLY_FEATURES_STOCKS_V5,
    StockFeatureNormalizer,
    HourlyStockDataModuleV5,
    MultiSymbolStockDataModuleV5,
)
from neuralhourlystocksv5.model import HourlyStockPolicyV5
from neuralhourlystocksv5.simulation import compute_v5_loss, simulate_batch


class HourlyStockTrainerV5:
    """V5 trainer with temperature annealing and position-length simulation."""

    def __init__(
        self,
        config: TrainingConfigStocksV5,
        data_module: HourlyStockDataModuleV5 | MultiSymbolStockDataModuleV5,
    ) -> None:
        self.config = config
        self.data_module = data_module

        # Setup device
        if config.device:
            self.device = torch.device(config.device)
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        # Enable TF32 for faster matmul on Ampere GPUs
        if config.use_tf32 and torch.cuda.is_available():
            torch.set_float32_matmul_precision("high")
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

        # Create model
        input_dim = len(data_module.feature_columns)
        self.policy_config = config.get_policy_config(input_dim)
        self.model = HourlyStockPolicyV5(self.policy_config).to(self.device)

        # Compile if enabled
        if config.use_compile and hasattr(torch, "compile"):
            self.model = torch.compile(self.model, mode=config.compile_mode)

        # Setup simulation config
        self.sim_config = config.get_simulation_config()

        # Temperature schedule
        self.temp_schedule = config.get_temperature_schedule()

        # Setup optimizers
        self._setup_optimizers()

        # AMP (mixed precision)
        self.use_amp = config.use_amp and self.device.type == "cuda"
        self.amp_dtype = (
            torch.bfloat16 if config.amp_dtype == "bfloat16" else torch.float16
        )
        self.scaler = GradScaler(enabled=self.use_amp)

        # Checkpoint directory
        self.checkpoint_dir = Path(config.checkpoint_root)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Training state
        self.epoch = 0
        self.best_sortino = float("-inf")
        self.patience_counter = 0
        self.history: List[Dict[str, float]] = []

    def _setup_optimizers(self) -> None:
        """Setup dual optimizer (Muon for matrix params, AdamW for others)."""
        config = self.config

        # Separate parameters
        matrix_params = []
        embed_params = []
        head_params = []

        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            if "embed" in name or "patch_embed" in name:
                embed_params.append(param)
            elif "head" in name:
                head_params.append(param)
            else:
                matrix_params.append(param)

        # Try to use Muon optimizer for matrix params
        try:
            from muon import Muon

            self.optimizer = Muon(
                [
                    {"params": matrix_params, "lr": config.matrix_lr},
                    {
                        "params": embed_params,
                        "lr": config.embed_lr,
                        "muon": False,
                    },  # AdamW for embeds
                    {
                        "params": head_params,
                        "lr": config.head_lr,
                        "muon": False,
                    },  # AdamW for heads
                ],
                lr=config.matrix_lr,
                momentum=config.muon_momentum,
                weight_decay=config.weight_decay,
            )
            self.using_muon = True
        except ImportError:
            # Fallback to AdamW
            self.optimizer = torch.optim.AdamW(
                [
                    {"params": matrix_params, "lr": config.learning_rate},
                    {"params": embed_params, "lr": config.learning_rate * 2},
                    {"params": head_params, "lr": config.learning_rate},
                ],
                lr=config.learning_rate,
                betas=config.adamw_betas,
                eps=config.adamw_eps,
                weight_decay=config.weight_decay,
            )
            self.using_muon = False

        # Learning rate scheduler
        if config.use_cosine_schedule:
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=config.epochs, eta_min=config.learning_rate * 0.01
            )
        else:
            self.scheduler = None

    def train(self) -> Dict[str, Any]:
        """Run full training loop."""
        config = self.config

        train_loader = self.data_module.train_dataloader(
            batch_size=config.batch_size, num_workers=config.num_workers
        )
        val_loader = self.data_module.val_dataloader(
            batch_size=config.batch_size, num_workers=config.num_workers
        )

        print(f"Training on {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")
        print(f"Using Muon optimizer: {self.using_muon}")

        for epoch in range(config.epochs):
            self.epoch = epoch
            temperature = self.temp_schedule.get_temperature(epoch, config.epochs)

            # Training epoch
            train_metrics = self._run_epoch(
                train_loader, training=True, temperature=temperature
            )

            # Validation epoch
            with torch.no_grad():
                val_metrics = self._run_epoch(
                    val_loader, training=False, temperature=0.001  # Near-hard for val
                )

            # Update scheduler
            if self.scheduler is not None:
                self.scheduler.step()

            # Record history
            epoch_record = {
                "epoch": epoch,
                "temperature": temperature,
                "train_loss": train_metrics["loss"],
                "train_sortino": train_metrics["sortino"],
                "train_return": train_metrics["mean_return"],
                "val_loss": val_metrics["loss"],
                "val_sortino": val_metrics["sortino"],
                "val_return": val_metrics["mean_return"],
                "val_tp_rate": val_metrics["tp_rate"],
                "val_forced_rate": val_metrics["forced_exit_rate"],
                "val_hold_hours": val_metrics["avg_hold_hours"],
                "val_position_length": val_metrics["avg_position_length"],
            }
            self.history.append(epoch_record)

            # Log progress
            if epoch % 10 == 0 or epoch == config.epochs - 1:
                print(
                    f"Epoch {epoch:3d} | "
                    f"Temp {temperature:.4f} | "
                    f"Train Loss {train_metrics['loss']:.4f} | "
                    f"Val Sortino {val_metrics['sortino']:.2f} | "
                    f"Val Return {val_metrics['mean_return']*100:.2f}% | "
                    f"TP Rate {val_metrics['tp_rate']*100:.1f}% | "
                    f"Avg Hold {val_metrics['avg_hold_hours']:.1f}h"
                )

            # Save checkpoint if best
            if val_metrics["sortino"] > self.best_sortino:
                self.best_sortino = val_metrics["sortino"]
                self.patience_counter = 0
                self._save_checkpoint(epoch, val_metrics, is_best=True)
            else:
                self.patience_counter += 1

            # Periodic checkpoint
            if epoch % 20 == 0:
                self._save_checkpoint(epoch, val_metrics, is_best=False)

        # Final save
        self._save_checkpoint(config.epochs - 1, val_metrics, is_best=False)

        return {
            "best_sortino": self.best_sortino,
            "final_epoch": config.epochs - 1,
            "history": self.history,
        }

    def _run_epoch(
        self,
        dataloader: DataLoader,
        training: bool,
        temperature: float,
    ) -> Dict[str, float]:
        """Run a single epoch."""
        if training:
            self.model.train()
        else:
            self.model.eval()

        metrics_accum = defaultdict(list)

        for batch in dataloader:
            # Move batch to device
            batch = {k: v.to(self.device) for k, v in batch.items()}

            if training:
                self.optimizer.zero_grad()

            with autocast(
                device_type=self.device.type,
                dtype=self.amp_dtype,
                enabled=self.use_amp,
            ):
                # Forward pass
                outputs = self.model(batch["features"])

                # Decode actions
                actions = self.model.decode_actions(
                    outputs,
                    reference_close=batch["current_close"],
                    temperature=temperature,
                )

                # Simulate trades
                result = simulate_batch(
                    batch=batch,
                    actions=actions,
                    config=self.sim_config,
                    temperature=temperature if training else 0.001,
                )

                # Compute loss
                loss_dict = compute_v5_loss(
                    result=result,
                    length_probs=actions["length_probs"],
                    position_length=actions["position_length"],
                    buy_offset=actions["buy_offset"],
                    sell_offset=actions["sell_offset"],
                    config=self.config,
                )

                loss = loss_dict["loss"]

            if training:
                # Backward pass
                self.scaler.scale(loss).backward()

                # Gradient clipping
                if self.config.grad_clip > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.config.grad_clip
                    )

                self.scaler.step(self.optimizer)
                self.scaler.update()

            # Record metrics
            for key, value in loss_dict.items():
                if isinstance(value, torch.Tensor):
                    metrics_accum[key].append(value.item())
                else:
                    metrics_accum[key].append(value)

        # Average metrics
        return {k: sum(v) / len(v) for k, v in metrics_accum.items()}

    def _save_checkpoint(
        self,
        epoch: int,
        metrics: Dict[str, float],
        is_best: bool,
    ) -> None:
        """Save model checkpoint."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        name = f"hourlyv5_epoch{epoch:03d}_{timestamp}"
        if is_best:
            name = f"best_{name}"

        checkpoint_path = self.checkpoint_dir / f"{name}.pt"

        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "config": {
                "training": self.config,
                "policy": self.policy_config,
                "simulation": self.sim_config,
            },
            "metrics": metrics,
            "normalizer": self.data_module.normalizer.to_dict(),
            "feature_columns": list(self.data_module.feature_columns),
            "best_sortino": self.best_sortino,
            "history": self.history,
        }

        torch.save(checkpoint, checkpoint_path)

        # Clean up old checkpoints
        self._cleanup_checkpoints()

    def _cleanup_checkpoints(self) -> None:
        """Keep only top K checkpoints."""
        checkpoints = list(self.checkpoint_dir.glob("hourlyv5_*.pt"))

        # Exclude best checkpoints from cleanup
        regular_checkpoints = [c for c in checkpoints if not c.name.startswith("best_")]

        if len(regular_checkpoints) > self.config.top_k_checkpoints:
            # Sort by modification time
            regular_checkpoints.sort(key=lambda p: p.stat().st_mtime)
            # Remove oldest
            for checkpoint in regular_checkpoints[: -self.config.top_k_checkpoints]:
                checkpoint.unlink()

    @classmethod
    def load_checkpoint(
        cls,
        checkpoint_path: str | Path,
        device: Optional[str] = None,
    ) -> Tuple["HourlyStockTrainerV5", Dict[str, Any]]:
        """Load trainer from checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location="cpu")

        # Reconstruct configs
        training_config = checkpoint["config"]["training"]
        if device:
            training_config.device = device

        # Create dummy data module (for normalizer)
        # In practice, you'd reload the actual data
        normalizer = StockFeatureNormalizer.from_dict(checkpoint["normalizer"])

        # Create model
        policy_config = checkpoint["config"]["policy"]
        model = HourlyStockPolicyV5(policy_config)
        model.load_state_dict(checkpoint["model_state_dict"])

        return model, {
            "normalizer": normalizer,
            "feature_columns": checkpoint["feature_columns"],
            "metrics": checkpoint["metrics"],
            "epoch": checkpoint["epoch"],
        }


def set_seed(seed: int) -> None:
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
