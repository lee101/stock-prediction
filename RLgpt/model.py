from __future__ import annotations

import torch
from torch import nn

from RLgpt.config import PlannerConfig


class CrossAssetDailyPlanner(nn.Module):
    def __init__(self, input_dim: int, config: PlannerConfig | None = None) -> None:
        super().__init__()
        if input_dim <= 0:
            raise ValueError("input_dim must be positive")
        self.config = config or PlannerConfig()
        self.input_proj = nn.Linear(input_dim, self.config.hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.config.hidden_dim,
            nhead=self.config.heads,
            dim_feedforward=self.config.hidden_dim * 4,
            dropout=self.config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=self.config.depth)
        self.asset_head = nn.Linear(self.config.hidden_dim, 6)
        self.global_head = nn.Sequential(
            nn.LayerNorm(self.config.hidden_dim),
            nn.Linear(self.config.hidden_dim, self.config.hidden_dim // 2),
            nn.GELU(),
            nn.Linear(self.config.hidden_dim // 2, 1),
        )

    def forward(self, features: torch.Tensor) -> dict[str, torch.Tensor]:
        if features.ndim != 3:
            raise ValueError(f"Expected [batch, assets, features], got shape {tuple(features.shape)}")
        hidden = self.input_proj(features)
        hidden = self.encoder(hidden)
        pooled = hidden.mean(dim=1)
        raw = self.asset_head(hidden)

        spread_span = self.config.max_half_spread_bps - self.config.min_half_spread_bps
        budget_scale = 0.25 + 0.75 * torch.sigmoid(self.global_head(pooled))

        return {
            "allocation_logits": raw[..., 0],
            "center_offset_bps": self.config.max_center_offset_bps * torch.tanh(raw[..., 1]),
            "half_spread_bps": self.config.min_half_spread_bps + spread_span * torch.sigmoid(raw[..., 2]),
            "max_long_fraction": torch.sigmoid(raw[..., 3]),
            "max_short_fraction": torch.sigmoid(raw[..., 4]),
            "trade_fraction": torch.sigmoid(raw[..., 5]),
            "budget_scale": budget_scale,
        }
