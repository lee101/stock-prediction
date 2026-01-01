"""V5 Trainer with NEPA-style portfolio training.

Key features:
- Dual optimizer (Muon + AdamW) from nanochat
- Temperature annealing for soft rebalancing
- NEPA loss for sequence coherence
- Sortino-based optimization
- Multi-asset portfolio training
"""
from __future__ import annotations

import math
import os
from datetime import datetime
from functools import partial
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from loguru import logger

from neuraldailyv5.config import DailyTrainingConfigV5, PolicyConfigV5
from neuraldailyv5.data import DailyDataModuleV5
from neuraldailyv5.model import PortfolioPolicyV5
from neuraldailyv5.simulation import PortfolioSimulatorV5, compute_v5_loss


class Muon(torch.optim.Optimizer):
    """
    Muon optimizer for linear layers (from nanochat).
    Uses Newton-Schulz orthogonalization for momentum.
    """

    def __init__(self, params, lr=0.02, momentum=0.95):
        defaults = dict(lr=lr, momentum=momentum)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            lr = group['lr']
            momentum = group['momentum']

            for p in group['params']:
                if p.grad is None:
                    continue

                g = p.grad
                state = self.state[p]

                if 'momentum_buffer' not in state:
                    state['momentum_buffer'] = torch.zeros_like(g)

                buf = state['momentum_buffer']
                buf.mul_(momentum).add_(g)

                # Newton-Schulz orthogonalization (simplified)
                if g.ndim >= 2:
                    # For 2D weight matrices, apply orthogonalization
                    g_flat = buf.view(buf.size(0), -1)
                    if g_flat.size(0) <= g_flat.size(1):
                        # Use transpose for efficiency
                        u = g_flat @ g_flat.T
                        # Newton-Schulz iteration
                        for _ in range(5):
                            u = 1.5 * u - 0.5 * u @ u @ u
                        update = u @ g_flat
                    else:
                        update = g_flat
                    update = update.view_as(buf)
                else:
                    update = buf

                p.add_(update, alpha=-lr)


def setup_dual_optimizer(
    model: nn.Module,
    config: DailyTrainingConfigV5,
) -> Tuple[List[torch.optim.Optimizer], List]:
    """
    Set up dual optimizer (Muon for linear, AdamW for embeddings).

    Returns:
        optimizers: List of optimizer instances
        schedulers: List of LR schedulers
    """
    # Separate parameters
    matrix_params = []
    embed_params = []
    head_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        if 'embed' in name or 'patch_embed' in name:
            embed_params.append(param)
        elif 'head' in name or 'to_latent' in name or 'predict_next' in name:
            head_params.append(param)
        else:
            matrix_params.append(param)

    # Scale learning rates by model dimension (from nanochat)
    model_dim = config.transformer_dim
    dmodel_lr_scale = (model_dim / 768) ** -0.5

    # AdamW for embeddings and heads
    adamw_groups = [
        {"params": embed_params, "lr": config.embed_lr * dmodel_lr_scale},
        {"params": head_params, "lr": config.head_lr * dmodel_lr_scale},
    ]
    adamw = torch.optim.AdamW(
        adamw_groups,
        betas=config.adamw_betas,
        eps=config.adamw_eps,
        weight_decay=config.weight_decay,
    )

    # Muon for transformer blocks
    muon = Muon(
        matrix_params,
        lr=config.matrix_lr,
        momentum=config.muon_momentum,
    )

    optimizers = [adamw, muon]

    # Store initial LRs for scheduling
    for opt in optimizers:
        for group in opt.param_groups:
            group["initial_lr"] = group["lr"]

    # Cosine schedule
    def cosine_schedule(step: int, total_steps: int, warmup_steps: int) -> float:
        if step < warmup_steps:
            return step / warmup_steps
        progress = (step - warmup_steps) / (total_steps - warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * progress))

    return optimizers, [cosine_schedule]


class NeuralDailyTrainerV5:
    """V5 Trainer with portfolio-centric training."""

    def __init__(
        self,
        config: DailyTrainingConfigV5,
        data_module: DailyDataModuleV5,
    ):
        self.config = config
        self.data_module = data_module

        # Set up device
        if config.device:
            self.device = torch.device(config.device)
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Enable TF32 for faster training
        if config.use_tf32 and self.device.type == "cuda":
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

        # Set up data
        data_module.setup()
        self.num_assets = data_module.num_assets
        input_dim = len(data_module.feature_columns)

        # Create model
        policy_config = config.get_policy_config(input_dim, self.num_assets)
        self.model = PortfolioPolicyV5(policy_config).to(self.device)

        # Create simulator
        sim_config = config.get_simulation_config()
        self.simulator = PortfolioSimulatorV5(sim_config, self.num_assets)

        # Set up optimizers
        self.optimizers, self.schedule_fns = setup_dual_optimizer(self.model, config)

        # AMP
        self.scaler = GradScaler() if config.use_amp else None
        self.amp_dtype = getattr(torch, config.amp_dtype) if config.use_amp else torch.float32

        # Temperature schedule
        self.temp_schedule = config.get_temperature_schedule()

        # Compile model if requested
        if config.use_compile and hasattr(torch, "compile"):
            logger.info("Compiling model with torch.compile...")
            self.model = torch.compile(self.model, mode=config.compile_mode)

        # Checkpoint directory
        self.checkpoint_dir = Path(config.checkpoint_root) / config.run_name
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Asset class tensor
        self.asset_class = data_module.get_asset_class_tensor(self.device)

        # Training state
        self.global_step = 0
        self.best_sortino = float("-inf")

    def _forward_step(
        self,
        batch: Dict[str, torch.Tensor],
        temperature: float,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Forward step with portfolio simulation.

        Args:
            batch: Dict with features, daily_returns, asset_class
            temperature: Current temperature for soft simulation

        Returns:
            loss: Scalar loss
            metrics: Dict of metrics
        """
        # Move batch to device
        features = batch["features"].to(self.device)  # (batch, num_assets, seq_len, num_features)
        daily_returns = batch["daily_returns"].to(self.device)  # (batch, num_assets, lookahead_days)
        asset_class = batch["asset_class"].to(self.device)  # (batch, num_assets)

        # Reshape features: (batch, num_assets, seq_len, features) -> (batch*num_assets, seq_len, features)
        batch_size, num_assets, seq_len, num_features = features.shape
        features_flat = features.view(batch_size * num_assets, seq_len, num_features)

        # Forward pass
        with autocast(enabled=self.config.use_amp, dtype=self.amp_dtype):
            outputs = self.model(features_flat, return_latents=True)

            # Reshape weights back: (batch*num_assets, num_assets) -> use first batch's weights
            # Note: Each sample in the flat batch predicts weights for all assets
            # We need to aggregate or use a different approach

            # Simpler approach: Use the mean features across assets for portfolio decision
            features_mean = features.mean(dim=1)  # (batch, seq_len, num_features)
            outputs = self.model(features_mean, return_latents=True)

            # Get portfolio weights
            weights = outputs["weights"]  # (batch, num_assets)

            # Transpose returns for simulation: (batch, num_assets, days) -> (batch, days, num_assets)
            daily_returns_t = daily_returns.transpose(1, 2)

            # Simulate portfolio
            result = self.simulator.simulate(
                target_weights=weights,
                daily_returns=daily_returns_t,
                asset_class=self.asset_class,
                temperature=temperature,
            )

            # Compute loss
            loss, loss_components = compute_v5_loss(
                result=result,
                outputs=outputs,
                daily_volatility=daily_returns_t,
                config=self.config.get_simulation_config(),
                return_weight=self.config.return_loss_weight,
                sortino_weight=self.config.sortino_loss_weight,
                nepa_weight=self.config.nepa_loss_weight,
                turnover_penalty=self.config.turnover_penalty,
                concentration_penalty=self.config.concentration_penalty,
                volatility_calibration_weight=self.config.volatility_calibration_weight,
            )

        # Collect metrics
        metrics = {
            "loss": loss.item(),
            "mean_return": result.mean_return.item(),
            "sortino": result.sortino_ratio.item(),
            "sharpe": result.sharpe_ratio.item(),
            "turnover": result.total_turnover.mean().item(),
            "max_drawdown": result.max_drawdown.item(),
            **{k: v.item() for k, v in loss_components.items() if isinstance(v, torch.Tensor)},
        }

        return loss, metrics

    def _backward_step(self, loss: torch.Tensor) -> None:
        """Backward pass with gradient scaling."""
        if self.scaler is not None:
            self.scaler.scale(loss).backward()
            for opt in self.optimizers:
                self.scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)
            for opt in self.optimizers:
                self.scaler.step(opt)
            self.scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)
            for opt in self.optimizers:
                opt.step()

        for opt in self.optimizers:
            opt.zero_grad(set_to_none=True)

    def _update_lr(self, step: int, total_steps: int) -> None:
        """Update learning rate based on schedule."""
        for schedule_fn in self.schedule_fns:
            lr_mult = schedule_fn(step, total_steps, self.config.warmup_steps)
            for opt in self.optimizers:
                for group in opt.param_groups:
                    group["lr"] = group["initial_lr"] * lr_mult

    def train_epoch(
        self,
        epoch: int,
        train_loader,
        total_epochs: int,
    ) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        temperature = self.temp_schedule.get_temperature(epoch, total_epochs)

        epoch_metrics = {
            "loss": 0.0,
            "mean_return": 0.0,
            "sortino": 0.0,
            "sharpe": 0.0,
            "turnover": 0.0,
        }
        num_batches = 0

        for batch in train_loader:
            # Forward
            loss, metrics = self._forward_step(batch, temperature)

            # Backward
            self._backward_step(loss)

            # Update LR
            self._update_lr(self.global_step, total_epochs * len(train_loader))
            self.global_step += 1

            # Accumulate metrics
            for k in epoch_metrics:
                if k in metrics:
                    epoch_metrics[k] += metrics[k]
            num_batches += 1

            # Dry run check
            if self.config.dry_train_steps and self.global_step >= self.config.dry_train_steps:
                break

        # Average
        for k in epoch_metrics:
            epoch_metrics[k] /= max(1, num_batches)

        epoch_metrics["temperature"] = temperature
        return epoch_metrics

    @torch.no_grad()
    def validate(self, val_loader) -> Dict[str, float]:
        """Validate model."""
        self.model.eval()

        epoch_metrics = {
            "loss": 0.0,
            "mean_return": 0.0,
            "sortino": 0.0,
            "sharpe": 0.0,
            "turnover": 0.0,
        }
        num_batches = 0

        for batch in val_loader:
            loss, metrics = self._forward_step(batch, temperature=0.0)

            for k in epoch_metrics:
                if k in metrics:
                    epoch_metrics[k] += metrics[k]
            num_batches += 1

        for k in epoch_metrics:
            epoch_metrics[k] /= max(1, num_batches)

        return epoch_metrics

    def save_checkpoint(self, epoch: int, metrics: Dict[str, float]) -> None:
        """Save model checkpoint."""
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_states": [opt.state_dict() for opt in self.optimizers],
            "global_step": self.global_step,
            "best_sortino": self.best_sortino,
            "metrics": metrics,
            "config": self.config,
        }

        # Save latest
        latest_path = self.checkpoint_dir / "latest.pt"
        torch.save(checkpoint, latest_path)

        # Save best if improved
        if metrics.get("sortino", float("-inf")) > self.best_sortino:
            self.best_sortino = metrics["sortino"]
            best_path = self.checkpoint_dir / "best.pt"
            torch.save(checkpoint, best_path)
            logger.info(f"New best Sortino: {self.best_sortino:.4f}")

    def load_checkpoint(self, path: str) -> int:
        """Load checkpoint and return starting epoch."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        for opt, state in zip(self.optimizers, checkpoint["optimizer_states"]):
            opt.load_state_dict(state)
        self.global_step = checkpoint["global_step"]
        self.best_sortino = checkpoint.get("best_sortino", float("-inf"))
        return checkpoint["epoch"] + 1

    def train(self, log_every: int = 10) -> Dict[str, List[float]]:
        """
        Full training loop.

        Returns:
            history: Dict of metric lists over epochs
        """
        # Create data loaders
        train_loader = self.data_module.train_dataloader(
            self.config.batch_size,
            self.config.num_workers,
        )
        val_loader = self.data_module.val_dataloader(
            self.config.batch_size,
            self.config.num_workers,
        )

        # Resume from checkpoint if specified
        start_epoch = 0
        if self.config.preload_checkpoint_path:
            start_epoch = self.load_checkpoint(self.config.preload_checkpoint_path)
            logger.info(f"Resumed from epoch {start_epoch}")

        history = {
            "train_loss": [],
            "train_sortino": [],
            "val_loss": [],
            "val_sortino": [],
            "val_sharpe": [],
            "val_return": [],
            "val_turnover": [],
        }

        for epoch in range(start_epoch, self.config.epochs):
            # Train
            train_metrics = self.train_epoch(epoch, train_loader, self.config.epochs)
            history["train_loss"].append(train_metrics["loss"])
            history["train_sortino"].append(train_metrics["sortino"])

            # Validate
            val_metrics = self.validate(val_loader)
            history["val_loss"].append(val_metrics["loss"])
            history["val_sortino"].append(val_metrics["sortino"])
            history["val_sharpe"].append(val_metrics["sharpe"])
            history["val_return"].append(val_metrics["mean_return"])
            history["val_turnover"].append(val_metrics["turnover"])

            # Log
            if (epoch + 1) % log_every == 0 or epoch == 0:
                logger.info(
                    f"Epoch {epoch + 1}/{self.config.epochs} | "
                    f"Train Loss: {train_metrics['loss']:.4f} | "
                    f"Val Sortino: {val_metrics['sortino']:.4f} | "
                    f"Val Sharpe: {val_metrics['sharpe']:.4f} | "
                    f"Val Return: {val_metrics['mean_return']:.4%} | "
                    f"Turnover: {val_metrics['turnover']:.2%}"
                )

            # Save checkpoint
            self.save_checkpoint(epoch, val_metrics)

            # Dry run check
            if self.config.dry_train_steps and self.global_step >= self.config.dry_train_steps:
                logger.info(f"Dry run completed after {self.global_step} steps")
                break

        return history
