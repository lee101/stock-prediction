#!/usr/bin/env python3
"""
Differentiable portfolio allocation trainer that optimises allocations across
stock pairs using Amazon Toto forecasts and real profit as the objective.

The trainer consumes `PairStockDataset` examples and learns a continuous
allocation policy that is rebalanced at market open and market close.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

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
    ):
        self.model = model.to(config.device)
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=0.01)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=config.epochs)

    def _step(self, batch: Dict[str, torch.Tensor], training: bool = True) -> Dict[str, float]:
        inputs = batch['input_ids'].to(self.config.device)
        future_returns = batch['future_returns'].to(self.config.device)

        weights = self.model(inputs)
        transaction_cost = self.config.transaction_cost_bps / 10000.0
        pnl = compute_portfolio_pnl(
            weights,
            future_returns,
            transaction_cost=transaction_cost,
            leverage_limit=self.config.leverage_limit,
            borrowing_cost=self.config.borrowing_cost,
            trading_days=self.config.trading_days_per_year,
        )
        profit = pnl.mean()
        sharpe = sharpe_like_ratio(pnl)
        entropy = -(weights.clamp_min(1e-8) * weights.clamp_min(1e-8).log()).sum(dim=-1).mean()
        loss = -(profit - self.config.risk_penalty * sharpe) + self.config.entropy_coef * entropy

        if training:
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()

        return {
            'loss': float(loss.detach().cpu().item()),
            'profit': float(profit.detach().cpu().item()),
            'sharpe': float(sharpe.detach().cpu().item()),
        }

    def train(self) -> Dict[str, float]:
        metrics: Dict[str, float] = {}
        for epoch in range(self.config.epochs):
            self.model.train()
            epoch_loss = 0.0
            epoch_profit = 0.0
            for batch in self.train_loader:
                stats = self._step(batch, training=True)
                epoch_loss += stats['loss']
                epoch_profit += stats['profit']

            epoch_loss /= max(1, len(self.train_loader))
            epoch_profit /= max(1, len(self.train_loader))
            metrics[f'train/loss_epoch_{epoch}'] = epoch_loss
            metrics[f'train/profit_epoch_{epoch}'] = epoch_profit

            if self.val_loader is not None:
                self.model.eval()
                with torch.no_grad():
                    val_loss = 0.0
                    val_profit = 0.0
                    val_sharpe = 0.0
                    for batch in self.val_loader:
                        stats = self._step(batch, training=False)
                        val_loss += stats['loss']
                        val_profit += stats['profit']
                        val_sharpe += stats['sharpe']
                val_loss /= max(1, len(self.val_loader))
                val_profit /= max(1, len(self.val_loader))
                val_sharpe /= max(1, len(self.val_loader))
                metrics[f'val/loss_epoch_{epoch}'] = val_loss
                metrics[f'val/profit_epoch_{epoch}'] = val_profit
                metrics[f'val/sharpe_epoch_{epoch}'] = val_sharpe

            self.scheduler.step()

        return metrics
