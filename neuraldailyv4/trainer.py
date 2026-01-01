"""V4 Trainer with multi-window training and aggregation.

Key features:
- Multi-window loss computation with trimmed mean
- Quantile calibration loss
- Same aggregation process at train and validation
"""
from __future__ import annotations

import json
import time
from collections import defaultdict
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
from loguru import logger
from torch.cuda.amp import GradScaler
from torch.amp import autocast
from tqdm import tqdm
from wandboard import WandBoardLogger

from neuraldailyv4.config import DailyTrainingConfigV4, PolicyConfigV4
from neuraldailyv4.data import DailyDataModuleV4
from neuraldailyv4.model import MultiSymbolPolicyV4, create_group_mask
from neuraldailyv4.simulation import (
    MultiWindowResult,
    compute_v4_loss,
    simulate_multi_window,
)


class NeuralDailyTrainerV4:
    """V4 Trainer with multi-window architecture."""

    def __init__(
        self,
        config: DailyTrainingConfigV4,
        data_module: DailyDataModuleV4,
    ):
        self.config = config
        self.data_module = data_module

        # Setup device
        if config.device:
            self.device = torch.device(config.device)
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        logger.info(f"Using device: {self.device}")

        # Enable TF32 if requested
        if config.use_tf32 and self.device.type == "cuda":
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            logger.info("Enabled TF32 for matmul and cuDNN")

        # Setup data
        logger.info("Setting up data module...")
        self.data_module.setup()

        # Create model
        policy_config = config.get_policy_config(len(self.data_module.feature_columns))
        self.model = MultiSymbolPolicyV4(policy_config).to(self.device)

        # Create simulation config
        self.sim_config = config.get_simulation_config()

        # Temperature schedule
        self.temp_schedule = config.get_temperature_schedule()

        # Setup optimizer
        self._setup_optimizer()

        # Setup AMP
        self.use_amp = config.use_amp and self.device.type == "cuda"
        if self.use_amp:
            self.amp_dtype = getattr(torch, config.amp_dtype)
            self.scaler = GradScaler()
            logger.info(f"Using AMP with dtype: {config.amp_dtype}")
        else:
            self.amp_dtype = torch.float32
            self.scaler = None

        # Checkpoint directory
        self.checkpoint_dir = Path(config.checkpoint_root) / config.run_name
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Best metrics tracking
        self.best_val_sharpe = float("-inf")
        self.best_epoch = 0

        # Setup WandB/TensorBoard logging
        self.tracker: Optional[WandBoardLogger] = None

    def _setup_optimizer(self):
        """Setup optimizer with parameter groups."""
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
            elif "head" in name or "multiwindow_head" in name:
                head_params.append(param)
            else:
                matrix_params.append(param)

        if config.optimizer_name == "adamw":
            self.optimizer = torch.optim.AdamW([
                {"params": matrix_params, "lr": config.learning_rate},
                {"params": embed_params, "lr": config.learning_rate * 2},
                {"params": head_params, "lr": config.learning_rate},
            ], betas=config.adamw_betas, eps=config.adamw_eps)
        else:
            # Dual optimizer: try Muon for matrix params, AdamW for others
            try:
                from muon import Muon

                self.matrix_optimizer = Muon(
                    matrix_params,
                    lr=config.matrix_lr,
                    momentum=config.muon_momentum,
                )
                self.other_optimizer = torch.optim.AdamW(
                    [
                        {"params": embed_params, "lr": config.embed_lr},
                        {"params": head_params, "lr": config.head_lr},
                    ],
                    betas=config.adamw_betas,
                    eps=config.adamw_eps,
                )
                self.optimizer = None
                logger.info("Using dual optimizer: Muon + AdamW")
            except ImportError:
                logger.warning("Muon not available, falling back to AdamW")
                self.optimizer = torch.optim.AdamW([
                    {"params": matrix_params, "lr": config.learning_rate},
                    {"params": embed_params, "lr": config.learning_rate * 2},
                    {"params": head_params, "lr": config.learning_rate},
                ], betas=config.adamw_betas, eps=config.adamw_eps)
                self.matrix_optimizer = None
                self.other_optimizer = None

    def _step_optimizer(self):
        """Step optimizer(s)."""
        if self.optimizer is not None:
            if self.scaler is not None:
                self.scaler.step(self.optimizer)
            else:
                self.optimizer.step()
        else:
            if self.scaler is not None:
                self.scaler.step(self.matrix_optimizer)
                self.scaler.step(self.other_optimizer)
            else:
                self.matrix_optimizer.step()
                self.other_optimizer.step()

    def _zero_grad(self):
        """Zero gradients."""
        if self.optimizer is not None:
            self.optimizer.zero_grad()
        else:
            self.matrix_optimizer.zero_grad()
            self.other_optimizer.zero_grad()

    def _run_epoch(
        self,
        dataloader,
        training: bool,
        temperature: float,
        quiet: bool = False,
    ) -> Dict[str, float]:
        """Run one epoch."""
        self.model.train(training)

        metrics = defaultdict(list)
        num_batches = 0

        if quiet:
            progress = dataloader
        else:
            progress = tqdm(dataloader, desc="Train" if training else "Val", leave=False)

        for batch in progress:
            if self.config.dry_train_steps and num_batches >= self.config.dry_train_steps:
                break

            # Move to device
            features = batch["features"].to(self.device)
            future_highs = batch["future_highs"].to(self.device)
            future_lows = batch["future_lows"].to(self.device)
            future_closes = batch["future_closes"].to(self.device)
            reference_close = batch["reference_close"].to(self.device)
            chronos_high = batch["chronos_high"].to(self.device)
            chronos_low = batch["chronos_low"].to(self.device)
            asset_class = batch["asset_class"].to(self.device)
            group_ids = batch["group_id"].to(self.device)

            # Create group mask
            group_mask = create_group_mask(group_ids)

            if training:
                self._zero_grad()

            # Forward pass
            with autocast(device_type=self.device.type, dtype=self.amp_dtype, enabled=self.use_amp):
                outputs = self.model(features, group_mask)

                # Get last timestep for reference prices (V3 data has shape [batch, seq_len])
                ref_close_last = reference_close[:, -1] if reference_close.dim() > 1 else reference_close
                ch_high_last = chronos_high[:, -1] if chronos_high.dim() > 1 else chronos_high
                ch_low_last = chronos_low[:, -1] if chronos_low.dim() > 1 else chronos_low

                # Decode actions
                actions = self.model.decode_actions(
                    outputs,
                    reference_close=ref_close_last,
                    chronos_high=ch_high_last,
                    chronos_low=ch_low_last,
                    asset_class=asset_class,
                )

                # Simulate multi-window
                result = simulate_multi_window(
                    future_highs=future_highs,
                    future_lows=future_lows,
                    future_closes=future_closes,
                    buy_quantiles=actions["buy_quantiles"],
                    sell_quantiles=actions["sell_quantiles"],
                    position_size=actions["position_size"],
                    confidence=actions["confidence"],
                    exit_days=actions["exit_days"],
                    reference_price=ref_close_last,
                    config=self.sim_config,
                    temperature=temperature if training else 0.0,
                )

                # Compute loss
                loss = compute_v4_loss(
                    result,
                    actions["buy_quantiles"],
                    actions["sell_quantiles"],
                    future_lows,
                    future_highs,
                    self.config.quantile_levels,
                    self.sim_config,
                    return_weight=self.config.return_loss_weight,
                    sharpe_weight=self.config.sharpe_loss_weight,
                    forced_exit_penalty=self.config.forced_exit_penalty,
                    quantile_calibration_weight=self.config.quantile_calibration_weight,
                    position_regularization=self.config.position_regularization,
                    quantile_ordering_weight=self.config.quantile_ordering_weight,
                    exit_days_penalty_weight=self.config.exit_days_penalty_weight,
                    exit_days=actions["exit_days"],
                    position_size=actions["position_size"],
                    utilization_loss_weight=getattr(self.config, 'utilization_loss_weight', 0.0),
                    utilization_target=getattr(self.config, 'utilization_target', 0.5),
                )

            if training:
                if self.scaler is not None:
                    self.scaler.scale(loss).backward()
                    self.scaler.unscale_(self.optimizer if self.optimizer else self.matrix_optimizer)
                    nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)
                    self._step_optimizer()
                    self.scaler.update()
                else:
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)
                    self._step_optimizer()

            # Record metrics
            metrics["loss"].append(loss.item())
            metrics["return"].append(result.aggregated_return.mean().item())
            metrics["sharpe"].append(result.sharpe.item())
            metrics["tp_rate"].append(result.avg_tp_rate.mean().item())
            metrics["forced_exit_rate"].append(result.forced_exit_rate.item())
            metrics["avg_hold"].append(result.avg_hold_days.mean().item())
            metrics["avg_confidence"].append(result.avg_confidence.item())

            num_batches += 1

            if not quiet:
                progress.set_postfix({
                    "loss": f"{loss.item():.4f}",
                    "sharpe": f"{result.sharpe.item():.3f}",
                    "tp": f"{result.avg_tp_rate.mean().item():.1%}",
                })

        # Average metrics
        return {k: sum(v) / len(v) if v else 0.0 for k, v in metrics.items()}

    def train(self, log_every: int = 10, early_stopping_patience: int = 20) -> Dict[str, List[float]]:
        """Run full training loop.

        Args:
            log_every: Log metrics every N epochs (default 10)
            early_stopping_patience: Stop training if val Sharpe doesn't improve for N epochs (0 to disable)
        """
        config = self.config

        # Get dataloaders
        train_loader = self.data_module.train_dataloader(config.batch_size, config.num_workers)
        val_loader = self.data_module.val_dataloader(config.batch_size, config.num_workers)

        logger.info(f"Training batches: {len(train_loader)}, Validation batches: {len(val_loader)}")
        if early_stopping_patience > 0:
            logger.info(f"Early stopping enabled: patience={early_stopping_patience} epochs")

        # Initialize WandB/TensorBoard tracker
        self.tracker = WandBoardLogger(
            run_name=config.run_name,
            project=config.wandb_project,
            entity=config.wandb_entity,
            log_dir=config.log_dir,
            tensorboard_subdir=config.run_name,
            config={
                "epochs": config.epochs,
                "batch_size": config.batch_size,
                "learning_rate": config.learning_rate,
                "weight_decay": config.weight_decay,
                "transformer_dim": config.transformer_dim,
                "transformer_layers": config.transformer_layers,
                "transformer_heads": config.transformer_heads,
                "patch_size": config.patch_size,
                "num_windows": config.num_windows,
                "window_size": config.window_size,
                "num_quantiles": config.num_quantiles,
                "max_hold_days": config.max_hold_days,
                "maker_fee": config.maker_fee,
                "return_loss_weight": config.return_loss_weight,
                "sharpe_loss_weight": config.sharpe_loss_weight,
                "forced_exit_penalty": config.forced_exit_penalty,
            },
            log_metrics=True,
        )
        self.tracker.watch(self.model)

        # History tracking
        history = defaultdict(list)

        # Early stopping tracking
        epochs_without_improvement = 0
        best_sharpe_for_stopping = float("-inf")

        start_time = time.time()

        for epoch in range(config.epochs):
            epoch_start = time.time()

            # Get temperature
            temperature = self.temp_schedule.get_temperature(epoch, config.epochs)

            # Only show progress bars on logging epochs
            quiet = ((epoch + 1) % log_every != 0) and (epoch + 1 != config.epochs)

            # Training
            train_metrics = self._run_epoch(train_loader, training=True, temperature=temperature, quiet=quiet)

            # Validation
            with torch.no_grad():
                val_metrics = self._run_epoch(val_loader, training=False, temperature=0.0, quiet=quiet)

            epoch_time = time.time() - epoch_start

            # Log metrics only every N epochs (or last epoch)
            if (epoch + 1) % log_every == 0 or epoch + 1 == config.epochs:
                logger.info(
                    f"Epoch {epoch + 1}/{config.epochs} ({epoch_time:.1f}s) | "
                    f"Train Loss: {train_metrics['loss']:.4f}, Sharpe: {train_metrics['sharpe']:.3f} | "
                    f"Val Loss: {val_metrics['loss']:.4f}, Sharpe: {val_metrics['sharpe']:.3f}, "
                    f"TP: {val_metrics['tp_rate']:.1%}, Hold: {val_metrics['avg_hold']:.1f}d"
                )

            # Track history
            history["train_loss"].append(train_metrics["loss"])
            history["train_sharpe"].append(train_metrics["sharpe"])
            history["val_loss"].append(val_metrics["loss"])
            history["val_sharpe"].append(val_metrics["sharpe"])
            history["val_tp_rate"].append(val_metrics["tp_rate"])
            history["val_avg_hold"].append(val_metrics["avg_hold"])

            # Log to WandB/TensorBoard
            if self.tracker is not None:
                self.tracker.log({
                    "loss/train": train_metrics["loss"],
                    "loss/val": val_metrics["loss"],
                    "sharpe/train": train_metrics["sharpe"],
                    "sharpe/val": val_metrics["sharpe"],
                    "return/train": train_metrics["return"],
                    "return/val": val_metrics["return"],
                    "tp_rate/train": train_metrics["tp_rate"],
                    "tp_rate/val": val_metrics["tp_rate"],
                    "forced_exit_rate/train": train_metrics["forced_exit_rate"],
                    "forced_exit_rate/val": val_metrics["forced_exit_rate"],
                    "avg_hold/train": train_metrics["avg_hold"],
                    "avg_hold/val": val_metrics["avg_hold"],
                    "confidence/train": train_metrics["avg_confidence"],
                    "confidence/val": val_metrics["avg_confidence"],
                    "temperature": temperature,
                    "epoch": epoch + 1,
                }, step=epoch + 1)

            # Save checkpoint if best (handle NaN by treating as -inf)
            import math
            val_sharpe = val_metrics["sharpe"]
            if math.isnan(val_sharpe) or (math.isinf(val_sharpe) and val_sharpe < 0):
                val_sharpe_for_comparison = float("-inf")
            else:
                val_sharpe_for_comparison = val_sharpe

            # Always save first epoch as fallback and set best_epoch to 1
            if epoch == 0:
                self._save_checkpoint(1, val_metrics)
                # Always set best_epoch to 1 as fallback, update sharpe if better
                self.best_epoch = 1
                if val_sharpe_for_comparison > self.best_val_sharpe:
                    self.best_val_sharpe = val_sharpe_for_comparison
            elif val_sharpe_for_comparison > self.best_val_sharpe:
                self.best_val_sharpe = val_sharpe_for_comparison
                self.best_epoch = epoch + 1
                self._save_checkpoint(epoch + 1, val_metrics)

            # Early stopping check
            if early_stopping_patience > 0:
                if val_metrics["sharpe"] > best_sharpe_for_stopping:
                    best_sharpe_for_stopping = val_metrics["sharpe"]
                    epochs_without_improvement = 0
                else:
                    epochs_without_improvement += 1

                if epochs_without_improvement >= early_stopping_patience:
                    logger.warning(
                        f"Early stopping triggered at epoch {epoch + 1}: "
                        f"no improvement for {early_stopping_patience} epochs"
                    )
                    break

        total_time = time.time() - start_time
        logger.info(f"Training complete in {total_time / 60:.1f} minutes")
        logger.info(f"Best validation Sharpe: {self.best_val_sharpe:.4f} at epoch {self.best_epoch}")

        # Close tracker
        if self.tracker is not None:
            self.tracker.finish()
            self.tracker = None

        return dict(history)

    def _save_checkpoint(self, epoch: int, metrics: Dict[str, float]) -> None:
        """Save model checkpoint."""
        checkpoint_path = self.checkpoint_dir / f"epoch_{epoch:04d}.pt"

        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "metrics": metrics,
            "config": {
                "policy": self.model.config.__dict__,
                "simulation": self.sim_config.__dict__,
                "training": {
                    k: v for k, v in self.config.__dict__.items()
                    if not k.startswith("_") and k != "dataset"
                },
            },
            "normalizer": self.data_module.normalizer,
            "feature_columns": self.data_module.feature_columns,
        }

        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Saved checkpoint: {checkpoint_path}")

        # Save manifest
        manifest = {
            "run_name": self.config.run_name,
            "best_epoch": self.best_epoch,
            "best_val_sharpe": self.best_val_sharpe,
            "checkpoints": sorted([p.name for p in self.checkpoint_dir.glob("epoch_*.pt")]),
            "wandb_project": self.config.wandb_project,
            "wandb_entity": self.config.wandb_entity,
        }

        with open(self.checkpoint_dir / "manifest.json", "w") as f:
            json.dump(manifest, f, indent=2)
