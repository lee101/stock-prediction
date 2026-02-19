#!/usr/bin/env python3
"""Multi-asset RL policy that learns cross-stock allocation."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class MultiAssetConfig:
    num_assets: int = 18
    feature_dim: int = 16
    hidden_dim: int = 512
    num_heads: int = 8
    num_layers: int = 4
    max_len: int = 32
    dropout: float = 0.3


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]


class CrossAssetAttention(nn.Module):
    """Attention layer that lets assets attend to each other."""

    def __init__(self, hidden_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, num_assets, hidden_dim)
        attn_out, _ = self.attn(x, x, x)
        return self.norm(x + self.dropout(attn_out))


class AssetEncoder(nn.Module):
    """Encode single asset's time series features."""

    def __init__(self, config: MultiAssetConfig):
        super().__init__()
        self.input_proj = nn.Linear(config.feature_dim, config.hidden_dim)
        self.pos_enc = PositionalEncoding(config.hidden_dim, config.max_len)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.hidden_dim,
            nhead=config.num_heads,
            dim_feedforward=config.hidden_dim * 4,
            dropout=config.dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=config.num_layers // 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, feature_dim)
        x = self.input_proj(x)
        x = self.pos_enc(x)
        x = self.encoder(x)
        return x[:, -1]  # Return last timestep


class MultiAssetPolicy(nn.Module):
    """Policy network for multi-asset allocation.

    Takes features for all assets and outputs allocation weights.
    Uses cross-asset attention to model relationships between stocks.
    """

    def __init__(self, config: MultiAssetConfig):
        super().__init__()
        self.config = config

        # Per-asset encoder (shared weights)
        self.asset_encoder = AssetEncoder(config)

        # Portfolio state embedding (current holdings)
        self.portfolio_embed = nn.Linear(config.num_assets, config.hidden_dim)

        # Cross-asset attention layers
        self.cross_attn_layers = nn.ModuleList([
            CrossAssetAttention(config.hidden_dim, config.num_heads, config.dropout)
            for _ in range(config.num_layers // 2)
        ])

        # Output heads
        self.allocation_head = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.GELU(),
            nn.Linear(config.hidden_dim, 1),
        )

        # Value head for advantage estimation
        self.value_head = nn.Sequential(
            nn.Linear(config.hidden_dim * config.num_assets, config.hidden_dim),
            nn.GELU(),
            nn.Linear(config.hidden_dim, 1),
        )

    def forward(
        self,
        asset_features: torch.Tensor,  # (batch, num_assets, seq_len, feature_dim)
        portfolio_state: Optional[torch.Tensor] = None,  # (batch, num_assets)
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size = asset_features.size(0)
        num_assets = asset_features.size(1)

        # Encode each asset independently
        asset_encodings = []
        for i in range(num_assets):
            enc = self.asset_encoder(asset_features[:, i])  # (batch, hidden_dim)
            asset_encodings.append(enc)

        # Stack: (batch, num_assets, hidden_dim)
        x = torch.stack(asset_encodings, dim=1)

        # Add portfolio state embedding if provided
        if portfolio_state is not None:
            port_emb = self.portfolio_embed(portfolio_state).unsqueeze(1)  # (batch, 1, hidden_dim)
            x = x + port_emb.expand(-1, num_assets, -1)

        # Cross-asset attention
        for layer in self.cross_attn_layers:
            x = layer(x)

        # Allocation weights (unnormalized logits)
        allocation_logits = self.allocation_head(x).squeeze(-1)  # (batch, num_assets)

        # Softmax for allocation weights (sum to 1)
        allocation_weights = F.softmax(allocation_logits, dim=-1)

        # Value estimate
        value = self.value_head(x.flatten(1))  # (batch, 1)

        return allocation_weights, value

    def get_action(
        self,
        asset_features: torch.Tensor,
        portfolio_state: Optional[torch.Tensor] = None,
        temperature: float = 1.0,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Get action with exploration noise."""
        allocation_weights, value = self(asset_features, portfolio_state)

        # Add Gumbel noise for exploration
        if temperature > 0:
            noise = -torch.log(-torch.log(torch.rand_like(allocation_weights) + 1e-8) + 1e-8)
            noisy_logits = torch.log(allocation_weights + 1e-8) + noise * temperature
            action = F.softmax(noisy_logits, dim=-1)
        else:
            action = allocation_weights

        # Log probability of action
        log_prob = torch.sum(allocation_weights * torch.log(action + 1e-8), dim=-1)

        return action, log_prob, value


class DifferentiablePortfolioSim(nn.Module):
    """Differentiable portfolio simulation for end-to-end training."""

    def __init__(self, num_assets: int, transaction_cost: float = 0.001):
        super().__init__()
        self.num_assets = num_assets
        self.transaction_cost = transaction_cost

    def forward(
        self,
        allocations: torch.Tensor,  # (batch, time, num_assets)
        returns: torch.Tensor,  # (batch, time, num_assets)
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Simulate portfolio with given allocations.
        Returns equity curve and per-step returns.
        """
        batch_size, time_steps, _ = allocations.shape

        equity = torch.ones(batch_size, device=allocations.device)
        equity_curve = [equity]
        portfolio_returns = []

        prev_alloc = torch.zeros(batch_size, self.num_assets, device=allocations.device)

        for t in range(time_steps):
            # Current allocation
            curr_alloc = allocations[:, t]

            # Transaction costs from rebalancing
            turnover = torch.abs(curr_alloc - prev_alloc).sum(dim=-1)
            tx_cost = turnover * self.transaction_cost

            # Portfolio return (weighted sum of asset returns)
            port_return = (curr_alloc * returns[:, t]).sum(dim=-1)

            # Net return after costs
            net_return = port_return - tx_cost

            # Update equity
            equity = equity * (1 + net_return)
            equity_curve.append(equity)
            portfolio_returns.append(net_return)

            prev_alloc = curr_alloc

        equity_curve = torch.stack(equity_curve, dim=1)
        portfolio_returns = torch.stack(portfolio_returns, dim=1)

        return equity_curve, portfolio_returns

    def compute_sortino(self, returns: torch.Tensor, target: float = 0.0) -> torch.Tensor:
        """Compute Sortino ratio from returns."""
        excess_returns = returns - target
        mean_return = excess_returns.mean(dim=-1)

        # Downside deviation
        downside = torch.clamp(target - returns, min=0)
        downside_std = torch.sqrt((downside ** 2).mean(dim=-1) + 1e-8)

        sortino = mean_return / (downside_std + 1e-8)
        return sortino


def build_multiasset_policy(config: MultiAssetConfig) -> MultiAssetPolicy:
    """Build multi-asset policy network."""
    return MultiAssetPolicy(config)


if __name__ == "__main__":
    # Test the model
    config = MultiAssetConfig(num_assets=18, feature_dim=16, hidden_dim=256, num_layers=4)
    model = build_multiasset_policy(config)

    batch_size = 4
    seq_len = 32

    # Random inputs
    features = torch.randn(batch_size, config.num_assets, seq_len, config.feature_dim)
    portfolio = torch.softmax(torch.randn(batch_size, config.num_assets), dim=-1)

    # Forward pass
    alloc, value = model(features, portfolio)
    print(f"Allocation shape: {alloc.shape}")  # (batch, num_assets)
    print(f"Value shape: {value.shape}")  # (batch, 1)
    print(f"Allocation sum: {alloc.sum(dim=-1)}")  # Should be ~1.0

    # Test action sampling
    action, log_prob, _ = model.get_action(features, portfolio, temperature=0.5)
    print(f"Action shape: {action.shape}")
    print(f"Log prob shape: {log_prob.shape}")
