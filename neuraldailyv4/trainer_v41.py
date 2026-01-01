"""V4.1 Trainer with fixed train=inference alignment.

Key fixes from V4:
1. Single-trade simulation matching inference
2. Aggregate predictions BEFORE simulation
3. Minimum spread enforcement
4. Position size regularization
"""
from __future__ import annotations

import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import torch
import torch.nn as nn
from loguru import logger
from torch.cuda.amp import GradScaler
from torch.amp import autocast

from neuraldailyv4.config import DailyTrainingConfigV4, PolicyConfigV4
from neuraldailyv4.data import DailyDataModuleV4
from neuraldailyv4.model import MultiSymbolPolicyV4, create_group_mask
from neuraldailyv4.simulation_v41 import (
    TradeResult,
    aggregate_predictions,
    compute_v41_loss,
    simulate_single_trade,
)


class NeuralDailyTrainerV41:
    """V4.1 Trainer with exact train=inference simulation."""

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

        # Enable TF32
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

        # Simulation config
        self.sim_config = config.get_simulation_config()

        # Temperature schedule
        self.temp_schedule = config.get_temperature_schedule()

        # Setup optimizer (simpler - just AdamW)
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            betas=config.adamw_betas,
            eps=config.adamw_eps,
            weight_decay=0.01,  # Add weight decay for regularization
        )

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

        # Best metrics
        self.best_val_sharpe = float("-inf")
        self.best_epoch = 0

    def _run_epoch(
        self,
        dataloader,
        training: bool,
        temperature: float,
        quiet: bool = False,
    ) -> Dict[str, float]:
        """Run one epoch with V4.1 simulation."""
        self.model.train(training)

        metrics = defaultdict(list)
        num_batches = 0

        for batch in dataloader:
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
                self.optimizer.zero_grad()

            # Forward pass
            with autocast(device_type=self.device.type, dtype=self.amp_dtype, enabled=self.use_amp):
                outputs = self.model(features, group_mask)

                # Get reference prices (last timestep)
                ref_close = reference_close[:, -1] if reference_close.dim() > 1 else reference_close
                ch_high = chronos_high[:, -1] if chronos_high.dim() > 1 else chronos_high
                ch_low = chronos_low[:, -1] if chronos_low.dim() > 1 else chronos_low

                # Decode actions (multi-window outputs)
                actions = self.model.decode_actions(
                    outputs,
                    reference_close=ref_close,
                    chronos_high=ch_high,
                    chronos_low=ch_low,
                    asset_class=asset_class,
                )

                # V4.1: Aggregate predictions FIRST (matching inference)
                agg = aggregate_predictions(
                    buy_quantiles=actions["buy_quantiles"],
                    sell_quantiles=actions["sell_quantiles"],
                    exit_days=actions["exit_days"],
                    position_size=actions["position_size"],
                    confidence=actions["confidence"],
                    trim_fraction=self.config.trim_fraction,
                    min_spread=0.02,  # 2% minimum spread
                )

                # V4.1: Simulate SINGLE trade with aggregated predictions
                result = simulate_single_trade(
                    future_highs=future_highs,
                    future_lows=future_lows,
                    future_closes=future_closes,
                    buy_price=agg["buy_price"],
                    sell_price=agg["sell_price"],
                    exit_days=agg["exit_days"],
                    position_size=agg["position_size"],
                    reference_price=ref_close,
                    config=self.sim_config,
                    temperature=temperature if training else 0.0,
                )

                # V4.1: Compute loss with better regularization
                loss = compute_v41_loss(
                    result,
                    buy_price=agg["buy_price"],
                    sell_price=agg["sell_price"],
                    position_size=agg["position_size"],
                    reference_price=ref_close,
                    return_weight=1.0,
                    sharpe_weight=0.5,
                    forced_exit_penalty=0.5,
                    position_reg_weight=0.1,
                    spread_bonus_weight=0.1,
                    target_position=0.3,
                )

            if training:
                if self.scaler is not None:
                    self.scaler.scale(loss).backward()
                    self.scaler.unscale_(self.optimizer)
                    nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)
                    self.optimizer.step()

            # Record metrics
            metrics["loss"].append(loss.item())
            metrics["return"].append(result.mean_return.item())
            metrics["sharpe"].append(result.sharpe.item())
            metrics["tp_rate"].append(result.tp_hit.mean().item())
            metrics["forced_exit_rate"].append(result.forced_exit_rate.item())
            metrics["avg_hold"].append(result.actual_hold_days.mean().item())
            metrics["position_size"].append(agg["position_size"].mean().item())

            # Compute spread
            spread = (agg["sell_price"] - agg["buy_price"]) / (ref_close + 1e-8)
            metrics["spread"].append(spread.mean().item())

            num_batches += 1

        return {k: sum(v) / len(v) if v else 0.0 for k, v in metrics.items()}

    def train(self, log_every: int = 10) -> Dict[str, List[float]]:
        """Run full training loop."""
        config = self.config

        train_loader = self.data_module.train_dataloader(config.batch_size, config.num_workers)
        val_loader = self.data_module.val_dataloader(config.batch_size, config.num_workers)

        logger.info(f"Training batches: {len(train_loader)}, Validation batches: {len(val_loader)}")

        history = defaultdict(list)
        start_time = time.time()

        for epoch in range(config.epochs):
            epoch_start = time.time()
            temperature = self.temp_schedule.get_temperature(epoch, config.epochs)

            quiet = ((epoch + 1) % log_every != 0) and (epoch + 1 != config.epochs)

            # Training
            train_metrics = self._run_epoch(train_loader, training=True, temperature=temperature, quiet=quiet)

            # Validation
            with torch.no_grad():
                val_metrics = self._run_epoch(val_loader, training=False, temperature=0.0, quiet=quiet)

            epoch_time = time.time() - epoch_start

            # Log every N epochs
            if (epoch + 1) % log_every == 0 or epoch + 1 == config.epochs:
                logger.info(
                    f"Epoch {epoch + 1}/{config.epochs} ({epoch_time:.1f}s) | "
                    f"Train Loss: {train_metrics['loss']:.4f}, Sharpe: {train_metrics['sharpe']:.3f} | "
                    f"Val Sharpe: {val_metrics['sharpe']:.3f}, TP: {val_metrics['tp_rate']:.1%}, "
                    f"Spread: {val_metrics['spread']:.2%}, Pos: {val_metrics['position_size']:.2f}"
                )

            # Track history
            for key in train_metrics:
                history[f"train_{key}"].append(train_metrics[key])
            for key in val_metrics:
                history[f"val_{key}"].append(val_metrics[key])

            # Save best checkpoint (handle NaN by treating as -inf)
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

        total_time = time.time() - start_time
        logger.info(f"Training complete in {total_time / 60:.1f} minutes")
        logger.info(f"Best validation Sharpe: {self.best_val_sharpe:.4f} at epoch {self.best_epoch}")

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
            "version": "v4.1",
        }

        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Saved checkpoint: {checkpoint_path}")

        # Save manifest
        manifest = {
            "run_name": self.config.run_name,
            "best_epoch": self.best_epoch,
            "best_val_sharpe": self.best_val_sharpe,
            "version": "v4.1",
            "checkpoints": sorted([p.name for p in self.checkpoint_dir.glob("epoch_*.pt")]),
        }

        with open(self.checkpoint_dir / "manifest.json", "w") as f:
            json.dump(manifest, f, indent=2)
