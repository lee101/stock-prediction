"""Training loop for ChronosPnL Trader.

Key innovation: Uses Chronos2 PnL forecast as a differentiable
training signal. The model is optimized to generate trades that
Chronos2 predicts will be profitable.
"""
from __future__ import annotations

import logging
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from chronospnltrader.config import (
    DataConfig,
    ForecastConfig,
    PolicyConfig,
    SimulationConfig,
    TemperatureSchedule,
    TrainingConfig,
)
from chronospnltrader.data import ChronosPnLDataModule, MultiSymbolDataModule, get_all_stock_symbols
from chronospnltrader.forecaster import Chronos2Forecaster
from chronospnltrader.model import ChronosPnLPolicy, create_model
from chronospnltrader.simulator import (
    SimulationHistory,
    compute_loss,
    run_simulation_30_days,
    simulate_batch,
)

logger = logging.getLogger(__name__)


@dataclass
class TrainingMetrics:
    """Metrics from a training epoch."""

    epoch: int
    train_loss: float
    val_loss: float
    train_sortino: float
    val_sortino: float
    train_pnl: float
    val_pnl: float
    temperature: float
    learning_rate: float


class ChronosPnLTrainer:
    """Trainer for ChronosPnL trading model.

    Key features:
    - Differentiable simulation with temperature annealing
    - Chronos2 PnL forecast as training signal
    - Periodic 30-day PnL simulation for evaluation
    - Checkpoint management
    """

    def __init__(
        self,
        config: TrainingConfig,
        data_module: ChronosPnLDataModule,
        forecaster: Optional[Chronos2Forecaster] = None,
    ) -> None:
        self.config = config
        self.data_module = data_module
        self.forecaster = forecaster

        # Device
        if config.device:
            self.device = torch.device(config.device)
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Precision
        if config.use_tf32 and torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

        # Create model
        input_dim = len(data_module.feature_columns)
        policy_config = config.get_policy_config(input_dim)
        self.model = create_model(policy_config).to(self.device)

        # Load checkpoint if specified
        if config.preload_checkpoint_path and Path(config.preload_checkpoint_path).exists():
            self._load_checkpoint(config.preload_checkpoint_path)

        # Compile if requested
        if config.use_compile:
            self.model = torch.compile(self.model, mode="reduce-overhead")

        # Optimizer
        self.optimizer = self._create_optimizer()
        self.scheduler = self._create_scheduler()

        # Temperature schedule
        self.temp_schedule = config.get_temperature_schedule()
        self.sim_config = config.get_simulation_config()

        # Tracking
        self.epoch = 0
        self.best_val_sortino = float("-inf")
        self.history = SimulationHistory.create(self.device)
        self.metrics_history: List[TrainingMetrics] = []

        # Checkpoints
        self.checkpoint_dir = Path(config.checkpoint_root)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def _create_optimizer(self) -> optim.Optimizer:
        """Create optimizer."""
        return optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            betas=self.config.adamw_betas,
            eps=self.config.adamw_eps,
            weight_decay=self.config.weight_decay,
        )

    def _create_scheduler(self) -> optim.lr_scheduler.LRScheduler:
        """Create learning rate scheduler."""
        if self.config.use_cosine_schedule:
            return optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.config.epochs,
                eta_min=self.config.learning_rate * 0.1,
            )
        else:
            return optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=30,
                gamma=0.5,
            )

    def train_epoch(self, temperature: float) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        train_loader = self.data_module.train_dataloader(
            batch_size=self.config.batch_size,
            num_workers=self.config.num_workers,
        )

        total_loss = 0.0
        total_sortino = 0.0
        total_pnl = 0.0
        n_batches = 0

        for batch in train_loader:
            # Move to device
            batch = {k: v.to(self.device) for k, v in batch.items()}

            # Forward pass
            outputs = self.model(batch["features"], batch.get("pnl_history"))
            actions = self.model.decode_actions(
                outputs,
                batch["current_close"],
                temperature=temperature,
            )

            # Simulate
            result = simulate_batch(
                batch=batch,
                actions=actions,
                config=self.sim_config,
                temperature=temperature,
            )

            # Get PnL forecast (differentiable approximation)
            pnl_forecast = self._get_pnl_forecast(batch["pnl_history"])

            # Compute loss
            losses = compute_loss(
                result=result,
                pnl_history=batch["pnl_history"],
                length_probs=actions["length_probs"],
                position_length=actions["position_length"],
                buy_offset=actions["buy_offset"],
                sell_offset=actions["sell_offset"],
                pnl_forecast=pnl_forecast,
                config=self.config,
            )

            loss = losses["loss"]

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()

            if self.config.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.grad_clip,
                )

            self.optimizer.step()

            # Track
            total_loss += loss.item()
            total_sortino += losses["sortino"].item()
            total_pnl += losses["mean_return"].item()
            n_batches += 1

            # Update history
            self.history.update(result.returns.detach())

        return {
            "loss": total_loss / max(1, n_batches),
            "sortino": total_sortino / max(1, n_batches),
            "pnl": total_pnl / max(1, n_batches),
        }

    def validate(self, temperature: float) -> Dict[str, float]:
        """Validate on held-out data."""
        self.model.eval()
        val_loader = self.data_module.val_dataloader(
            batch_size=self.config.batch_size,
            num_workers=self.config.num_workers,
        )

        total_loss = 0.0
        total_sortino = 0.0
        total_pnl = 0.0
        n_batches = 0

        with torch.no_grad():
            for batch in val_loader:
                batch = {k: v.to(self.device) for k, v in batch.items()}

                outputs = self.model(batch["features"], batch.get("pnl_history"))
                actions = self.model.decode_actions(
                    outputs,
                    batch["current_close"],
                    temperature=temperature,
                )

                result = simulate_batch(
                    batch=batch,
                    actions=actions,
                    config=self.sim_config,
                    temperature=temperature,
                )

                pnl_forecast = self._get_pnl_forecast(batch["pnl_history"])

                losses = compute_loss(
                    result=result,
                    pnl_history=batch["pnl_history"],
                    length_probs=actions["length_probs"],
                    position_length=actions["position_length"],
                    buy_offset=actions["buy_offset"],
                    sell_offset=actions["sell_offset"],
                    pnl_forecast=pnl_forecast,
                    config=self.config,
                )

                total_loss += losses["loss"].item()
                total_sortino += losses["sortino"].item()
                total_pnl += losses["mean_return"].item()
                n_batches += 1

        return {
            "loss": total_loss / max(1, n_batches),
            "sortino": total_sortino / max(1, n_batches),
            "pnl": total_pnl / max(1, n_batches),
        }

    def _get_pnl_forecast(self, pnl_history: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Get differentiable PnL forecast for training."""
        if self.forecaster is not None:
            return self.forecaster.forecast_pnl_differentiable(pnl_history, self.device)
        else:
            # Fallback: simple exponential weighted prediction
            history_len = pnl_history.size(1)
            weights = torch.exp(torch.linspace(-2, 0, history_len, device=self.device))
            weights = weights / weights.sum()

            weighted_avg = (pnl_history * weights).sum(dim=1)
            recent = pnl_history[:, -7:].mean(dim=1) if history_len >= 7 else pnl_history.mean(dim=1)
            pnl_std = pnl_history.std(dim=1) + 1e-8

            predicted_pnl = 0.6 * weighted_avg + 0.4 * recent
            confidence = torch.sigmoid(-pnl_std * 10)

            return {
                "predicted_pnl": predicted_pnl,
                "confidence": confidence,
                "momentum": recent,
                "volatility": pnl_std,
                "cumulative": pnl_history.sum(dim=1),
            }

    def train(self) -> TrainingMetrics:
        """Run full training loop."""
        logger.info(f"Starting training for {self.config.epochs} epochs")
        logger.info(f"Device: {self.device}")
        logger.info(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")

        for epoch in range(self.config.epochs):
            self.epoch = epoch
            temperature = self.temp_schedule.get_temperature(epoch, self.config.epochs)
            current_lr = self.optimizer.param_groups[0]["lr"]

            # Train
            train_metrics = self.train_epoch(temperature)

            # Validate
            val_metrics = self.validate(temperature)

            # Step scheduler
            self.scheduler.step()

            # Log
            logger.info(
                f"Epoch {epoch + 1}/{self.config.epochs} | "
                f"Train Loss: {train_metrics['loss']:.4f} | "
                f"Val Loss: {val_metrics['loss']:.4f} | "
                f"Train Sortino: {train_metrics['sortino']:.2f} | "
                f"Val Sortino: {val_metrics['sortino']:.2f} | "
                f"Temp: {temperature:.4f} | "
                f"LR: {current_lr:.2e}"
            )

            # Track metrics
            metrics = TrainingMetrics(
                epoch=epoch,
                train_loss=train_metrics["loss"],
                val_loss=val_metrics["loss"],
                train_sortino=train_metrics["sortino"],
                val_sortino=val_metrics["sortino"],
                train_pnl=train_metrics["pnl"],
                val_pnl=val_metrics["pnl"],
                temperature=temperature,
                learning_rate=current_lr,
            )
            self.metrics_history.append(metrics)

            # Save best checkpoint
            if val_metrics["sortino"] > self.best_val_sortino:
                self.best_val_sortino = val_metrics["sortino"]
                self._save_checkpoint("best.pt")
                logger.info(f"New best Sortino: {self.best_val_sortino:.2f}")

            # Periodic checkpoints
            if (epoch + 1) % 10 == 0:
                self._save_checkpoint(f"epoch_{epoch + 1}.pt")

        # Final checkpoint
        self._save_checkpoint("final.pt")

        return self.metrics_history[-1]

    def evaluate_30_days(self) -> Dict[str, float]:
        """Run 30-day simulation for final evaluation."""
        return run_simulation_30_days(
            data_module=self.data_module,
            model=self.model,
            config=self.sim_config,
            device=self.device,
            use_simple_algo=False,
        )

    def _save_checkpoint(self, filename: str) -> None:
        """Save model checkpoint."""
        path = self.checkpoint_dir / filename
        torch.save({
            "epoch": self.epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "best_val_sortino": self.best_val_sortino,
            "config": self.config,
        }, path)
        logger.info(f"Saved checkpoint to {path}")

    def _load_checkpoint(self, path: str) -> None:
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        if "optimizer_state_dict" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if "scheduler_state_dict" in checkpoint:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        if "epoch" in checkpoint:
            self.epoch = checkpoint["epoch"]
        if "best_val_sortino" in checkpoint:
            self.best_val_sortino = checkpoint["best_val_sortino"]
        logger.info(f"Loaded checkpoint from {path}")


def train_single_symbol(
    symbol: str,
    config: Optional[TrainingConfig] = None,
) -> TrainingMetrics:
    """Train on a single symbol."""
    if config is None:
        config = TrainingConfig()

    data_config = DataConfig(symbols=(symbol,))
    data_module = ChronosPnLDataModule(data_config)

    trainer = ChronosPnLTrainer(config, data_module)
    return trainer.train()


def train_multi_symbol(
    symbols: Optional[List[str]] = None,
    config: Optional[TrainingConfig] = None,
) -> TrainingMetrics:
    """Train on multiple symbols."""
    if config is None:
        config = TrainingConfig()

    if symbols is None:
        symbols = get_all_stock_symbols()[:50]  # Top 50 by default

    data_config = DataConfig()
    data_module = MultiSymbolDataModule(symbols, data_config)

    trainer = ChronosPnLTrainer(config, data_module)
    return trainer.train()
