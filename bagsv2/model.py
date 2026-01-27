"""Neural model v2 for Bags.fm - LSTM with Attention."""

from __future__ import annotations

from typing import Tuple

import torch
from torch import nn
import torch.nn.functional as F


class TemporalAttention(nn.Module):
    """Attention mechanism over time steps."""

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, lstm_out: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            lstm_out: (batch, seq_len, hidden_dim)
        Returns:
            context: (batch, hidden_dim) - attention-weighted representation
            weights: (batch, seq_len) - attention weights
        """
        # Compute attention scores
        scores = self.attention(lstm_out).squeeze(-1)  # (batch, seq_len)
        weights = F.softmax(scores, dim=1)  # (batch, seq_len)

        # Weighted sum
        context = torch.bmm(weights.unsqueeze(1), lstm_out).squeeze(1)  # (batch, hidden_dim)
        return context, weights


class BagsNeuralModelV2(nn.Module):
    """LSTM with Attention for trading signals.

    Architecture:
    1. Per-bar feature embedding
    2. Bidirectional LSTM for temporal patterns
    3. Temporal attention to weight important bars
    4. Dual heads for signal (buy/sell) and position size
    """

    def __init__(
        self,
        features_per_bar: int = 5,
        context_bars: int = 32,
        lstm_hidden: int = 64,
        lstm_layers: int = 2,
        fc_hidden: int = 64,
        dropout: float = 0.2,
        bidirectional: bool = True,
    ) -> None:
        super().__init__()
        self.features_per_bar = features_per_bar
        self.context_bars = context_bars

        # Feature embedding per bar
        self.feature_embed = nn.Sequential(
            nn.Linear(features_per_bar, lstm_hidden // 2),
            nn.LayerNorm(lstm_hidden // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # Bidirectional LSTM
        self.lstm = nn.LSTM(
            input_size=lstm_hidden // 2,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0,
            bidirectional=bidirectional,
        )

        lstm_out_dim = lstm_hidden * (2 if bidirectional else 1)

        # Temporal attention
        self.attention = TemporalAttention(lstm_out_dim)

        # Aggregate features processing (volatility, momentum, etc.)
        self.agg_embed = nn.Sequential(
            nn.Linear(7, 32),  # 7 aggregate features
            nn.LayerNorm(32),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # Combined head
        combined_dim = lstm_out_dim + 32  # attention output + agg features

        self.fc = nn.Sequential(
            nn.Linear(combined_dim, fc_hidden),
            nn.LayerNorm(fc_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fc_hidden, fc_hidden // 2),
            nn.ReLU(),
        )

        # Output heads
        self.signal_head = nn.Linear(fc_hidden // 2, 1)
        self.size_head = nn.Linear(fc_hidden // 2, 1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (batch, total_features) where total_features = context_bars * features_per_bar + 7 agg features
        Returns:
            signal_logit: (batch,)
            size_logit: (batch,)
        """
        batch_size = x.shape[0]

        # Split per-bar features and aggregate features
        per_bar_dim = self.context_bars * self.features_per_bar
        per_bar_features = x[:, :per_bar_dim]
        agg_features = x[:, per_bar_dim:]

        # Reshape per-bar features to (batch, context_bars, features_per_bar)
        per_bar_features = per_bar_features.view(batch_size, self.context_bars, self.features_per_bar)

        # Embed each bar
        embedded = self.feature_embed(per_bar_features)  # (batch, context_bars, lstm_hidden//2)

        # LSTM
        lstm_out, _ = self.lstm(embedded)  # (batch, context_bars, lstm_out_dim)

        # Attention
        context, _ = self.attention(lstm_out)  # (batch, lstm_out_dim)

        # Process aggregate features
        agg_embedded = self.agg_embed(agg_features)  # (batch, 32)

        # Combine
        combined = torch.cat([context, agg_embedded], dim=1)

        # FC layers
        fc_out = self.fc(combined)

        # Output heads
        signal_logit = self.signal_head(fc_out).squeeze(-1)
        size_logit = self.size_head(fc_out).squeeze(-1)

        return signal_logit, size_logit


class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance.

    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    """

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, reduction: str = "mean"):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            inputs: logits (batch,)
            targets: binary labels (batch,)
        """
        p = torch.sigmoid(inputs)
        ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")

        p_t = p * targets + (1 - p) * (1 - targets)
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        focal_weight = alpha_t * (1 - p_t) ** self.gamma

        loss = focal_weight * ce_loss

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


# Simple MLP alternative for comparison
class BagsNeuralModelV2Simple(nn.Module):
    """Enhanced MLP with residual connections."""

    def __init__(
        self,
        input_dim: int,
        hidden_dims: Tuple[int, ...] = (128, 64, 32),
        dropout: float = 0.2,
    ) -> None:
        super().__init__()

        layers = []
        in_dim = input_dim

        for i, out_dim in enumerate(hidden_dims):
            layers.append(nn.Linear(in_dim, out_dim))
            layers.append(nn.LayerNorm(out_dim))
            layers.append(nn.GELU())  # GELU often works better than ReLU
            layers.append(nn.Dropout(dropout))
            in_dim = out_dim

        self.backbone = nn.Sequential(*layers)
        self.signal_head = nn.Linear(hidden_dims[-1], 1)
        self.size_head = nn.Linear(hidden_dims[-1], 1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        features = self.backbone(x)
        signal_logit = self.signal_head(features).squeeze(-1)
        size_logit = self.size_head(features).squeeze(-1)
        return signal_logit, size_logit
