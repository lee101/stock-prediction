#!/usr/bin/env python3
"""
Differentiable portfolio allocation trainer that optimises allocations across
stock pairs using Amazon Toto forecasts and real profit as the objective.

The trainer consumes `PairStockDataset` examples and learns a continuous
allocation policy that is rebalanced at market open and market close.
"""

from __future__ import annotations

import contextlib
import os
import time
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from traininglib import (
    enable_fast_kernels,
    bf16_supported,
    maybe_compile,
    make_optimizer,
    WarmupCosine,
)

from wandboard import WandBoardLogger

try:
    from .differentiable_profit import compute_portfolio_pnl, sharpe_like_ratio
except ImportError:  # Allow script-style execution
    from differentiable_profit import compute_portfolio_pnl, sharpe_like_ratio  # type: ignore


@dataclass
class PortfolioRLConfig:
    hidden_size: int = 256
    num_layers: int = 4
    num_heads: int = 4
    dropout: float = 0.1
    learning_rate: float = 3e-4
    batch_size: int = 64
    epochs: int = 20
    transaction_cost_bps: float = 10.0
    risk_penalty: float = 0.1
    entropy_coef: float = 0.01
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    leverage_limit: float = 2.0
    borrowing_cost: float = 0.0675
    trading_days_per_year: int = 252
    optimizer: str = "adamw"
    weight_decay: float = 0.01
    compile: bool = True
    grad_clip: float = 1.0
    warmup_steps: int = 500
    min_learning_rate: float = 0.0
    logging_dir: str = "hftraining/portfolio_logs"
    use_wandb: bool = field(
        default_factory=lambda: os.getenv("WANDB_DISABLED", "0").lower() not in {"1", "true", "yes"}
    )
    wandb_project: Optional[str] = None
    wandb_entity: Optional[str] = None
    wandb_run_name: Optional[str] = None
    wandb_group: Optional[str] = None
    wandb_tags: Tuple[str, ...] = field(default_factory=tuple)
    wandb_mode: str = "auto"
    wandb_settings: Optional[Dict[str, Any]] = None
    tensorboard_subdir: Optional[str] = None


class PortfolioAllocationModel(nn.Module):
    """
    Lightweight transformer that emits allocation weights for a pair of stocks.
    """

    def __init__(self, input_dim: int, config: PortfolioRLConfig, num_assets: int):
        super().__init__()
        self.config = config
        self.num_assets = num_assets
        self.input_proj = nn.Linear(input_dim, config.hidden_size)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.hidden_size,
            nhead=config.num_heads,
            dim_feedforward=config.hidden_size * 4,
            dropout=config.dropout,
            activation='gelu',
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=config.num_layers)
        self.norm = nn.LayerNorm(config.hidden_size)
        self.head = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size // 2),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_size // 2, num_assets),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x = self.input_proj(inputs)
        x = self.encoder(x)
        pooled = self.norm(x[:, -1, :])
        logits = self.head(pooled)
        raw = torch.tanh(logits)
        scaled = raw * self.config.leverage_limit
        gross = scaled.abs().sum(dim=-1, keepdim=True)
        scale = torch.clamp(gross / self.config.leverage_limit, min=1.0)
        weights = scaled / scale
        return weights


class DifferentiablePortfolioTrainer:
    """Trains a portfolio allocation policy with differentiable profit."""

    def __init__(
        self,
        model: PortfolioAllocationModel,
        config: PortfolioRLConfig,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        metrics_logger: Optional[WandBoardLogger] = None,
    ):
        self.config = config
        self.model = model.to(config.device)
        self._stack = contextlib.ExitStack()
        self._stack.enter_context(enable_fast_kernels())
        self.model = maybe_compile(self.model, do_compile=config.compile)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = make_optimizer(
            self.model,
            name=config.optimizer,
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
        total_steps = max(1, len(self.train_loader) * config.epochs)
        self.scheduler = WarmupCosine(
            self.optimizer,
            warmup_steps=config.warmup_steps,
            total_steps=total_steps,
            min_lr=config.min_learning_rate,
        )
        self.autocast_dtype = torch.bfloat16 if bf16_supported() else None
        self._best_val_profit: float = float("-inf")
        self._best_epoch: int = -1
        self._best_state: Optional[Dict[str, torch.Tensor]] = None
        self._owns_logger = metrics_logger is None
        if metrics_logger is None:
            run_name = config.wandb_run_name or f"portfolio_rl_{int(time.time())}"
            tb_subdir = config.tensorboard_subdir or run_name
            self.metrics_logger = WandBoardLogger(
                run_name=run_name,
                project=config.wandb_project,
                entity=config.wandb_entity,
                tags=tuple(config.wandb_tags or ()),
                group=config.wandb_group,
                mode=config.wandb_mode,
                enable_wandb=bool(config.use_wandb),
                log_dir=config.logging_dir,
                tensorboard_subdir=tb_subdir,
                config=asdict(config),
                settings=config.wandb_settings,
            )
        else:
            self.metrics_logger = metrics_logger

    def _step(self, batch: Dict[str, torch.Tensor], training: bool = True) -> Dict[str, float]:
        inputs = batch['input_ids'].to(self.config.device)
        future_returns = batch['future_returns'].to(self.config.device)

        autocast_ctx = (
            torch.autocast("cuda", dtype=self.autocast_dtype)
            if self.autocast_dtype and self.config.device.startswith("cuda")
            else contextlib.nullcontext()
        )
        with autocast_ctx:
            weights = self.model(inputs)
        transaction_cost = self.config.transaction_cost_bps / 10000.0
        per_asset_fees = batch.get('per_asset_fees')
        if per_asset_fees is not None:
            per_asset_fees = per_asset_fees.to(self.config.device)
        asset_class_ids = batch.get('asset_class_ids')
        if asset_class_ids is not None:
            asset_class_ids = asset_class_ids.to(self.config.device)
        pnl_result = compute_portfolio_pnl(
            weights,
            future_returns,
            transaction_cost=transaction_cost,
            leverage_limit=self.config.leverage_limit,
            borrowing_cost=self.config.borrowing_cost,
            trading_days=self.config.trading_days_per_year,
            per_asset_costs=per_asset_fees,
            return_per_asset=True,
        )
        if isinstance(pnl_result, tuple):
            pnl, per_asset_net = pnl_result
        else:
            pnl = pnl_result
            per_asset_net = None
        profit = pnl.mean()
        profit_equity = None
        profit_crypto = None
        if per_asset_net is not None and asset_class_ids is not None:
            if asset_class_ids.dim() == 1:
                asset_class_ids = asset_class_ids.unsqueeze(0).expand_as(per_asset_net)
            equity_mask = (asset_class_ids == 0).to(per_asset_net.dtype)
            crypto_mask = (asset_class_ids == 1).to(per_asset_net.dtype)
            if torch.count_nonzero(equity_mask).item() > 0:
                profit_equity = (per_asset_net * equity_mask).sum(dim=-1).mean()
            if torch.count_nonzero(crypto_mask).item() > 0:
                profit_crypto = (per_asset_net * crypto_mask).sum(dim=-1).mean()
        sharpe = sharpe_like_ratio(pnl)
        entropy = -(weights.clamp_min(1e-8) * weights.clamp_min(1e-8).log()).sum(dim=-1).mean()
        loss = -(profit - self.config.risk_penalty * sharpe) + self.config.entropy_coef * entropy

        if training:
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if self.config.grad_clip and self.config.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)
            self.optimizer.step()
            self.scheduler.step()

        return {
            'loss': float(loss.detach().cpu().item()),
            'profit': float(profit.detach().cpu().item()),
            'sharpe': float(sharpe.detach().cpu().item()),
            'profit_equity': float(profit_equity.detach().cpu().item()) if profit_equity is not None else 0.0,
            'profit_crypto': float(profit_crypto.detach().cpu().item()) if profit_crypto is not None else 0.0,
        }

    def train(self) -> Dict[str, float]:
        metrics: Dict[str, float] = {}
        for epoch in range(self.config.epochs):
            self.model.train()
            epoch_loss = 0.0
            epoch_profit = 0.0
            epoch_profit_equity = 0.0
            epoch_profit_crypto = 0.0
            for batch in self.train_loader:
                stats = self._step(batch, training=True)
                epoch_loss += stats['loss']
                epoch_profit += stats['profit']
                epoch_profit_equity += stats.get('profit_equity', 0.0)
                epoch_profit_crypto += stats.get('profit_crypto', 0.0)

            epoch_loss /= max(1, len(self.train_loader))
            epoch_profit /= max(1, len(self.train_loader))
            epoch_profit_equity /= max(1, len(self.train_loader))
            epoch_profit_crypto /= max(1, len(self.train_loader))
            metrics[f'train/loss_epoch_{epoch}'] = epoch_loss
            metrics[f'train/profit_epoch_{epoch}'] = epoch_profit
            metrics[f'train/profit_equity_epoch_{epoch}'] = epoch_profit_equity
            metrics[f'train/profit_crypto_epoch_{epoch}'] = epoch_profit_crypto
            current_lr = self.scheduler.get_last_lr()[0] if self.scheduler.get_last_lr() else self.config.learning_rate
            self.metrics_logger.log(
                {
                    'train/loss': epoch_loss,
                    'train/profit': epoch_profit,
                    'train/profit_equity': epoch_profit_equity,
                    'train/profit_crypto': epoch_profit_crypto,
                    'train/learning_rate': current_lr,
                },
                step=epoch,
            )

            if self.val_loader is not None:
                self.model.eval()
                with torch.no_grad():
                    val_loss = 0.0
                    val_profit = 0.0
                    val_sharpe = 0.0
                    val_profit_equity = 0.0
                    val_profit_crypto = 0.0
                    for batch in self.val_loader:
                        stats = self._step(batch, training=False)
                        val_loss += stats['loss']
                        val_profit += stats['profit']
                        val_sharpe += stats['sharpe']
                        val_profit_equity += stats.get('profit_equity', 0.0)
                        val_profit_crypto += stats.get('profit_crypto', 0.0)
                val_loss /= max(1, len(self.val_loader))
                val_profit /= max(1, len(self.val_loader))
                val_sharpe /= max(1, len(self.val_loader))
                val_profit_equity /= max(1, len(self.val_loader))
                val_profit_crypto /= max(1, len(self.val_loader))
                metrics[f'val/loss_epoch_{epoch}'] = val_loss
                metrics[f'val/profit_epoch_{epoch}'] = val_profit
                metrics[f'val/sharpe_epoch_{epoch}'] = val_sharpe
                metrics[f'val/profit_equity_epoch_{epoch}'] = val_profit_equity
                metrics[f'val/profit_crypto_epoch_{epoch}'] = val_profit_crypto
                self.metrics_logger.log(
                    {
                        'val/loss': val_loss,
                        'val/profit': val_profit,
                        'val/sharpe': val_sharpe,
                        'val/profit_equity': val_profit_equity,
                        'val/profit_crypto': val_profit_crypto,
                    },
                    step=epoch,
                )
                if val_profit > self._best_val_profit:
                    self._best_val_profit = val_profit
                    self._best_epoch = epoch
                    module = self.model._orig_mod if hasattr(self.model, "_orig_mod") else self.model
                    self._best_state = {
                        k: v.detach().cpu().clone()
                        for k, v in module.state_dict().items()
                    }

        self._stack.close()
        summary_metrics = {
            'best/val_profit': self._best_val_profit,
            'best/epoch': float(self._best_epoch),
        }
        self.metrics_logger.log(summary_metrics, step=self.config.epochs, commit=True)
        if self._owns_logger:
            self.metrics_logger.finish()
        metrics["best_val_profit"] = self._best_val_profit
        metrics["best_epoch"] = self._best_epoch
        return metrics

    def best_state_dict(self) -> Optional[Dict[str, torch.Tensor]]:
        return self._best_state

    def export_state_dict(self) -> Dict[str, torch.Tensor]:
        module = self.model._orig_mod if hasattr(self.model, "_orig_mod") else self.model
        return {
            k: v.detach().cpu().clone()
            for k, v in module.state_dict().items()
        }
