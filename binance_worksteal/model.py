"""Neural daily work-steal policy network.

Learns per-symbol dip-buying parameters (buy offset, sell offset, intensity)
from daily OHLCV + technical indicators, replacing fixed rule-based hyperparameters.
"""
from __future__ import annotations

import math
from typing import Dict

import torch
from torch import nn


class PositionalEncoding(nn.Module):
    def __init__(self, dim: int, dropout: float = 0.1, max_len: int = 512) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2) * (-math.log(10000.0) / dim))
        pe = torch.zeros(max_len, dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term[: pe[:, 1::2].shape[1]])
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[: x.size(1)].unsqueeze(0)
        return self.dropout(x)


def _init_policy_weights(module: nn.Module, head: nn.Linear) -> None:
    for m in module.modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
    nn.init.zeros_(head.weight)
    nn.init.zeros_(head.bias)


def _logits_to_actions(
    logits: torch.Tensor, max_buy: float, max_sell: float
) -> Dict[str, torch.Tensor]:
    return {
        "buy_offset": torch.sigmoid(logits[:, :, 0]) * max_buy,
        "sell_offset": torch.sigmoid(logits[:, :, 1]) * max_sell,
        "intensity": torch.sigmoid(logits[:, :, 2]),
    }


class DailyWorkStealPolicy(nn.Module):
    """Original transformer policy (flattened symbols). Kept for backward compat."""

    def __init__(
        self,
        n_features: int = 11,
        n_symbols: int = 30,
        hidden_dim: int = 256,
        num_layers: int = 4,
        num_heads: int = 4,
        seq_len: int = 30,
        dropout: float = 0.1,
        max_buy_offset: float = 0.30,
        max_sell_offset: float = 0.30,
    ) -> None:
        super().__init__()
        self.n_features = n_features
        self.n_symbols = n_symbols
        self.hidden_dim = hidden_dim
        self.max_buy_offset = max_buy_offset
        self.max_sell_offset = max_sell_offset

        input_dim = n_symbols * n_features
        self.embed = nn.Linear(input_dim, hidden_dim)
        self.pos_encoding = PositionalEncoding(hidden_dim, dropout=dropout, max_len=max(seq_len * 2, 64))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(hidden_dim)
        self.head = nn.Linear(hidden_dim, n_symbols * 3)
        _init_policy_weights(self, self.head)

    def forward(self, features: torch.Tensor) -> Dict[str, torch.Tensor]:
        B, T, S, F = features.shape
        x = features.reshape(B, T, S * F)

        h = self.embed(x)
        h = self.pos_encoding(h)
        h = self.encoder(h)
        h = self.norm(h)

        h_last = h[:, -1, :]
        logits = self.head(h_last).view(B, self.n_symbols, 3)
        return _logits_to_actions(logits, self.max_buy_offset, self.max_sell_offset)

    def get_actions_for_step(self, features: torch.Tensor) -> Dict[str, torch.Tensor]:
        return self.forward(features)


class PerSymbolWorkStealPolicy(nn.Module):
    """Per-symbol encoder + cross-symbol attention policy.

    Each symbol gets [seq_len, n_features] through a shared temporal encoder,
    then cross-symbol attention captures market-wide patterns.
    Output: per-symbol [buy_offset, sell_offset, intensity].
    """

    def __init__(
        self,
        n_features: int = 11,
        n_symbols: int = 30,
        hidden_dim: int = 128,
        num_temporal_layers: int = 2,
        num_cross_layers: int = 2,
        num_heads: int = 4,
        seq_len: int = 30,
        dropout: float = 0.1,
        max_buy_offset: float = 0.30,
        max_sell_offset: float = 0.30,
    ) -> None:
        super().__init__()
        self.n_features = n_features
        self.n_symbols = n_symbols
        self.hidden_dim = hidden_dim
        self.max_buy_offset = max_buy_offset
        self.max_sell_offset = max_sell_offset

        self.feature_embed = nn.Linear(n_features, hidden_dim)
        self.pos_encoding = PositionalEncoding(hidden_dim, dropout=dropout, max_len=max(seq_len * 2, 64))

        temporal_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=num_heads,
            dim_feedforward=hidden_dim * 4, dropout=dropout,
            batch_first=True, activation="gelu",
        )
        self.temporal_encoder = nn.TransformerEncoder(temporal_layer, num_layers=num_temporal_layers)
        self.temporal_norm = nn.LayerNorm(hidden_dim)

        cross_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=num_heads,
            dim_feedforward=hidden_dim * 4, dropout=dropout,
            batch_first=True, activation="gelu",
        )
        self.cross_encoder = nn.TransformerEncoder(cross_layer, num_layers=num_cross_layers)
        self.cross_norm = nn.LayerNorm(hidden_dim)

        self.head = nn.Linear(hidden_dim, 3)
        _init_policy_weights(self, self.head)

    def forward(self, features: torch.Tensor) -> Dict[str, torch.Tensor]:
        B, T, S, F = features.shape

        x = features.permute(0, 2, 1, 3).reshape(B * S, T, F)
        h = self.feature_embed(x)
        h = self.pos_encoding(h)
        h = self.temporal_encoder(h)
        h = self.temporal_norm(h)
        sym_embeds = h[:, -1, :].view(B, S, self.hidden_dim)

        cross = self.cross_encoder(sym_embeds)
        cross = self.cross_norm(cross)

        logits = self.head(cross)
        return _logits_to_actions(logits, self.max_buy_offset, self.max_sell_offset)

    def get_actions_for_step(self, features: torch.Tensor) -> Dict[str, torch.Tensor]:
        return self.forward(features)
