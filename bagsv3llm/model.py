"""Transformer model for trading signals (BagsV3LLM).

Modern transformer architecture adapted for continuous financial time series:
- Rotary Position Embeddings (RoPE) for position encoding
- QK LayerNorm for stable training
- ReLU^2 activation in MLP (following recent research)
- RMSNorm (no learnable params) for efficiency
- Causal attention for autoregressive pattern learning
- Dual output heads: signal (buy/no-buy) and position size
- Chronos2 forecast features integration

Architecture is inspired by nanochat/GPT but adapted for:
- Continuous inputs (price features) instead of discrete tokens
- Binary classification + regression outputs
- Integration of external forecasts
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class BagsV3Config:
    """Configuration for BagsV3LLM transformer."""

    context_length: int = 256  # Number of bars in context
    features_per_bar: int = 5  # Base features: returns, range_pct, oc_return, upper_wick, lower_wick
    chronos_features: int = 12  # Chronos2 forecast features (3 horizons x 4 values)
    agg_features: int = 7  # Aggregate features like volatility, momentum

    n_layer: int = 6  # Number of transformer layers
    n_head: int = 8  # Number of attention heads
    n_embd: int = 128  # Embedding dimension

    dropout: float = 0.1
    bias: bool = False  # No bias in linear layers (modern practice)

    # For pre-training
    pretrain_mask_ratio: float = 0.15  # Ratio of bars to mask for reconstruction

    @property
    def head_dim(self) -> int:
        return self.n_embd // self.n_head

    @property
    def total_bar_features(self) -> int:
        """Total features per bar including Chronos forecasts."""
        return self.features_per_bar + self.chronos_features

    @property
    def input_dim(self) -> int:
        """Total input dimension for the model."""
        return self.context_length * self.features_per_bar + self.agg_features


def rms_norm(x: torch.Tensor) -> torch.Tensor:
    """RMSNorm without learnable parameters."""
    return F.rms_norm(x, (x.size(-1),))


def apply_rotary_emb(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Apply rotary position embeddings to queries/keys."""
    assert x.ndim == 4  # (batch, heads, seq_len, head_dim)
    d = x.shape[3] // 2
    x1, x2 = x[..., :d], x[..., d:]
    y1 = x1 * cos + x2 * sin
    y2 = x1 * (-sin) + x2 * cos
    return torch.cat([y1, y2], dim=3).to(x.dtype)


class CausalSelfAttention(nn.Module):
    """Multi-head causal self-attention with RoPE and QK norm."""

    def __init__(self, config: BagsV3Config, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = config.head_dim

        assert self.n_embd % self.n_head == 0

        # Separate Q, K, V projections (cleaner than combined)
        self.c_q = nn.Linear(self.n_embd, self.n_head * self.head_dim, bias=config.bias)
        self.c_k = nn.Linear(self.n_embd, self.n_head * self.head_dim, bias=config.bias)
        self.c_v = nn.Linear(self.n_embd, self.n_head * self.head_dim, bias=config.bias)
        self.c_proj = nn.Linear(self.n_embd, self.n_embd, bias=config.bias)

        self.dropout = nn.Dropout(config.dropout)

    def forward(
        self,
        x: torch.Tensor,
        cos_sin: Tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, T, C = x.size()

        # Project to Q, K, V
        q = self.c_q(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = self.c_k(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = self.c_v(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        # Apply RoPE to Q and K
        cos, sin = cos_sin
        q = apply_rotary_emb(q, cos, sin)
        k = apply_rotary_emb(k, cos, sin)

        # QK LayerNorm for stability
        q = rms_norm(q)
        k = rms_norm(k)

        # Scaled dot-product attention with causal mask
        y = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attention_mask,
            is_causal=(attention_mask is None),
            dropout_p=self.dropout.p if self.training else 0.0,
        )

        # Reshape and project
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.c_proj(y)
        y = self.dropout(y)

        return y


class MLP(nn.Module):
    """MLP with ReLU^2 activation (following recent research)."""

    def __init__(self, config: BagsV3Config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.c_fc(x)
        x = F.relu(x).square()  # ReLU^2 activation
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class TransformerBlock(nn.Module):
    """Single transformer block with pre-norm architecture."""

    def __init__(self, config: BagsV3Config, layer_idx: int):
        super().__init__()
        self.attn = CausalSelfAttention(config, layer_idx)
        self.mlp = MLP(config)

    def forward(
        self,
        x: torch.Tensor,
        cos_sin: Tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Pre-norm architecture
        x = x + self.attn(rms_norm(x), cos_sin, attention_mask)
        x = x + self.mlp(rms_norm(x))
        return x


class BagsV3Transformer(nn.Module):
    """Transformer for trading signal prediction.

    Takes OHLC bar features + Chronos2 forecasts as input,
    outputs buy signal probability and position size.
    """

    def __init__(self, config: BagsV3Config):
        super().__init__()
        self.config = config

        # Input projection: per-bar features to embedding dimension
        self.bar_embed = nn.Sequential(
            nn.Linear(config.features_per_bar, config.n_embd),
            nn.LayerNorm(config.n_embd),
            nn.GELU(),
            nn.Dropout(config.dropout),
        )

        # Chronos forecast embedding (separate pathway)
        self.chronos_embed = nn.Sequential(
            nn.Linear(config.chronos_features, config.n_embd // 2),
            nn.LayerNorm(config.n_embd // 2),
            nn.GELU(),
            nn.Dropout(config.dropout),
        )

        # Combine bar features and chronos features
        self.feature_combine = nn.Linear(
            config.n_embd + config.n_embd // 2,
            config.n_embd,
            bias=config.bias
        )

        # Transformer layers
        self.layers = nn.ModuleList([
            TransformerBlock(config, layer_idx)
            for layer_idx in range(config.n_layer)
        ])

        # Aggregate features processing (volatility, momentum, etc.)
        self.agg_embed = nn.Sequential(
            nn.Linear(config.agg_features, config.n_embd // 2),
            nn.LayerNorm(config.n_embd // 2),
            nn.GELU(),
            nn.Dropout(config.dropout),
        )

        # Output head combining transformer output and aggregate features
        self.head_combine = nn.Sequential(
            nn.Linear(config.n_embd + config.n_embd // 2, config.n_embd),
            nn.LayerNorm(config.n_embd),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.n_embd, config.n_embd // 2),
            nn.GELU(),
        )

        # Dual output heads
        self.signal_head = nn.Linear(config.n_embd // 2, 1)
        self.size_head = nn.Linear(config.n_embd // 2, 1)

        # Pre-compute rotary embeddings
        self.rotary_seq_len = config.context_length * 2  # Over-allocate
        cos, sin = self._precompute_rotary_embeddings(
            self.rotary_seq_len, config.head_dim
        )
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

        # Initialize weights
        self.apply(self._init_weights)

        # Zero-init certain projections for better training dynamics
        for block in self.layers:
            nn.init.zeros_(block.attn.c_proj.weight)
            nn.init.zeros_(block.mlp.c_proj.weight)

    def _init_weights(self, module: nn.Module):
        if isinstance(module, nn.Linear):
            fan_out, fan_in = module.weight.shape
            std = 1.0 / math.sqrt(fan_in) * min(1.0, math.sqrt(fan_out / fan_in))
            nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=1.0)

    def _precompute_rotary_embeddings(
        self,
        seq_len: int,
        head_dim: int,
        base: float = 10000.0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Precompute rotary position embeddings."""
        channel_range = torch.arange(0, head_dim, 2, dtype=torch.float32)
        inv_freq = 1.0 / (base ** (channel_range / head_dim))
        t = torch.arange(seq_len, dtype=torch.float32)
        freqs = torch.outer(t, inv_freq)
        cos, sin = freqs.cos(), freqs.sin()
        # Shape: (1, 1, seq_len, head_dim//2) for broadcasting with (B, n_head, T, head_dim//2)
        cos = cos[None, None, :, :].bfloat16()
        sin = sin[None, None, :, :].bfloat16()
        return cos, sin

    def forward(
        self,
        bar_features: torch.Tensor,  # (batch, context_length, features_per_bar)
        chronos_features: torch.Tensor,  # (batch, context_length, chronos_features)
        agg_features: torch.Tensor,  # (batch, agg_features)
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Returns:
            signal_logit: (batch,) - logit for buy signal
            size_logit: (batch,) - logit for position size
        """
        B, T, _ = bar_features.shape

        # Embed bar features
        bar_emb = self.bar_embed(bar_features)  # (B, T, n_embd)

        # Embed Chronos features
        chronos_emb = self.chronos_embed(chronos_features)  # (B, T, n_embd//2)

        # Combine bar and chronos features
        combined = torch.cat([bar_emb, chronos_emb], dim=-1)
        x = self.feature_combine(combined)  # (B, T, n_embd)

        # Apply pre-embedding norm
        x = rms_norm(x)

        # Get rotary embeddings for current sequence length
        # cos/sin have shape (1, 1, seq_len, head_dim//2)
        assert T <= self.cos.size(2), f"Sequence length {T} exceeds rotary cache"
        cos_sin = (
            self.cos[:, :, :T, :].to(x.device),
            self.sin[:, :, :T, :].to(x.device)
        )

        # Transformer layers
        for layer in self.layers:
            x = layer(x, cos_sin, attention_mask)

        # Final norm
        x = rms_norm(x)

        # Use the last position's output (like GPT for next token prediction)
        x_last = x[:, -1, :]  # (B, n_embd)

        # Process aggregate features
        agg_emb = self.agg_embed(agg_features)  # (B, n_embd//2)

        # Combine transformer output and aggregate features
        combined_out = torch.cat([x_last, agg_emb], dim=-1)
        head_out = self.head_combine(combined_out)

        # Output heads
        signal_logit = self.signal_head(head_out).squeeze(-1)
        size_logit = self.size_head(head_out).squeeze(-1)

        return signal_logit, size_logit

    def forward_flat(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass with flat input tensor for compatibility with v2.

        Args:
            x: (batch, total_features) where:
               - First context_length * features_per_bar are per-bar features
               - Next context_length * chronos_features are Chronos predictions (optional, zeros if not available)
               - Last agg_features are aggregate features
        """
        batch_size = x.shape[0]

        # Calculate dimensions
        per_bar_dim = self.config.context_length * self.config.features_per_bar
        chronos_dim = self.config.context_length * self.config.chronos_features

        # Split features
        per_bar_flat = x[:, :per_bar_dim]

        # Check if Chronos features are included
        if x.shape[1] > per_bar_dim + self.config.agg_features:
            chronos_flat = x[:, per_bar_dim:per_bar_dim + chronos_dim]
            agg_features = x[:, per_bar_dim + chronos_dim:]
        else:
            # No Chronos features, use zeros
            chronos_flat = torch.zeros(
                batch_size, chronos_dim,
                dtype=x.dtype, device=x.device
            )
            agg_features = x[:, per_bar_dim:]

        # Reshape to (batch, context_length, features_per_bar)
        bar_features = per_bar_flat.view(
            batch_size, self.config.context_length, self.config.features_per_bar
        )
        chronos_features = chronos_flat.view(
            batch_size, self.config.context_length, self.config.chronos_features
        )

        return self.forward(bar_features, chronos_features, agg_features)


class BagsV3PretrainModel(nn.Module):
    """Pre-training model with masked reconstruction objective.

    Uses the transformer backbone with an additional reconstruction head
    for self-supervised pre-training on stock data.
    """

    def __init__(self, config: BagsV3Config):
        super().__init__()
        self.config = config

        # Core transformer (same architecture)
        self.transformer = BagsV3Transformer(config)

        # Reconstruction head for masked bars
        self.reconstruction_head = nn.Sequential(
            nn.Linear(config.n_embd, config.n_embd * 2),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.n_embd * 2, config.features_per_bar),
        )

    def forward(
        self,
        bar_features: torch.Tensor,  # (batch, context_length, features_per_bar)
        chronos_features: torch.Tensor,  # (batch, context_length, chronos_features)
        agg_features: torch.Tensor,  # (batch, agg_features)
        mask: Optional[torch.Tensor] = None,  # (batch, context_length) - True where masked
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Pre-training forward pass.

        Returns:
            reconstructed: (batch, context_length, features_per_bar) - reconstructed bar features
            signal_logit: (batch,) - buy signal logit
            size_logit: (batch,) - size logit
        """
        B, T, _ = bar_features.shape

        # Store original for reconstruction loss
        original_bars = bar_features.clone()

        # Apply masking if provided
        if mask is not None:
            # Replace masked positions with learnable mask token or zeros
            bar_features = bar_features.clone()
            bar_features[mask] = 0.0

        # Run through transformer
        bar_emb = self.transformer.bar_embed(bar_features)
        chronos_emb = self.transformer.chronos_embed(chronos_features)
        combined = torch.cat([bar_emb, chronos_emb], dim=-1)
        x = self.transformer.feature_combine(combined)
        x = rms_norm(x)

        cos_sin = (
            self.transformer.cos[:, :, :T, :].to(x.device),
            self.transformer.sin[:, :, :T, :].to(x.device)
        )

        for layer in self.transformer.layers:
            x = layer(x, cos_sin)

        x = rms_norm(x)

        # Reconstruction of all positions
        reconstructed = self.reconstruction_head(x)  # (B, T, features_per_bar)

        # Signal and size outputs (using last position)
        x_last = x[:, -1, :]
        agg_emb = self.transformer.agg_embed(agg_features)
        combined_out = torch.cat([x_last, agg_emb], dim=-1)
        head_out = self.transformer.head_combine(combined_out)

        signal_logit = self.transformer.signal_head(head_out).squeeze(-1)
        size_logit = self.transformer.size_head(head_out).squeeze(-1)

        return reconstructed, signal_logit, size_logit


class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance in trading signals."""

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, reduction: str = "mean"):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
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


class PretrainLoss(nn.Module):
    """Combined loss for pre-training: reconstruction + optional signal prediction."""

    def __init__(
        self,
        recon_weight: float = 1.0,
        signal_weight: float = 0.1,
        size_weight: float = 0.05,
    ):
        super().__init__()
        self.recon_weight = recon_weight
        self.signal_weight = signal_weight
        self.size_weight = size_weight
        self.focal_loss = FocalLoss()
        self.mse_loss = nn.MSELoss()

    def forward(
        self,
        reconstructed: torch.Tensor,
        original: torch.Tensor,
        mask: Optional[torch.Tensor],
        signal_logit: torch.Tensor,
        signal_target: torch.Tensor,
        size_logit: torch.Tensor,
        size_target: torch.Tensor,
    ) -> Tuple[torch.Tensor, dict]:
        # Reconstruction loss (only on masked positions if mask provided)
        if mask is not None and mask.any():
            recon_loss = self.mse_loss(
                reconstructed[mask],
                original[mask]
            )
        else:
            recon_loss = self.mse_loss(reconstructed, original)

        # Signal and size losses
        signal_loss = self.focal_loss(signal_logit, signal_target)
        size_loss = self.mse_loss(torch.sigmoid(size_logit), size_target)

        total_loss = (
            self.recon_weight * recon_loss +
            self.signal_weight * signal_loss +
            self.size_weight * size_loss
        )

        metrics = {
            "recon_loss": recon_loss.item(),
            "signal_loss": signal_loss.item(),
            "size_loss": size_loss.item(),
            "total_loss": total_loss.item(),
        }

        return total_loss, metrics
