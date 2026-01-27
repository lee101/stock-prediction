"""BagsV5 - State-of-the-art transformer for trading signals.

Cutting-edge architecture:
- Muon optimizer for weight matrices (2-3x faster convergence)
- RMSNorm (faster than LayerNorm, same quality)
- SwiGLU activation (better than ReLU/GELU)
- Rotary Position Embeddings (RoPE)
- Flash Attention via PyTorch SDPA
- Gradient checkpointing for memory efficiency
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


@dataclass
class BagsV5Config:
    """Configuration for BagsV5 model."""
    # Input dimensions
    context_length: int = 96
    features_per_bar: int = 5  # OHLC returns + volume
    agg_features: int = 7      # Aggregate features

    # Architecture
    n_layer: int = 4
    n_head: int = 8
    n_embd: int = 64
    dropout: float = 0.1
    bias: bool = False

    # Modern features
    use_flash_attn: bool = True
    use_gradient_checkpointing: bool = False
    rope_theta: float = 10000.0

    # SwiGLU hidden multiplier (2/3 * 4 = 8/3 for efficiency)
    ffn_multiplier: float = 2.667


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization.

    More efficient than LayerNorm - no mean subtraction, half the parameters.
    """
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # RMS = sqrt(mean(x^2))
        rms = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x * rms * self.weight


class RotaryEmbedding(nn.Module):
    """Rotary Position Embeddings (RoPE)."""

    def __init__(self, dim: int, max_seq_len: int = 512, theta: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.theta = theta

        # Precompute frequencies
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # Precompute cos/sin cache
        self._build_cache(max_seq_len)

    def _build_cache(self, seq_len: int):
        t = torch.arange(seq_len, device=self.inv_freq.device)
        freqs = torch.outer(t, self.inv_freq)
        # Shape: (seq_len, dim/2) -> (1, 1, seq_len, dim/2)
        cos = freqs.cos()[None, None, :, :]
        sin = freqs.sin()[None, None, :, :]
        self.register_buffer("cos_cached", cos, persistent=False)
        self.register_buffer("sin_cached", sin, persistent=False)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        seq_len = x.shape[2]
        if seq_len > self.max_seq_len:
            self._build_cache(seq_len)
        return self.cos_cached[:, :, :seq_len, :], self.sin_cached[:, :, :seq_len, :]


def apply_rotary_emb(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Apply rotary embeddings to input tensor."""
    # x: (B, n_head, T, head_dim)
    # Split into pairs for rotation
    x1 = x[..., ::2]
    x2 = x[..., 1::2]

    # Apply rotation
    rotated = torch.stack([-x2, x1], dim=-1).flatten(-2)

    # Expand cos/sin for full head_dim
    cos = cos.repeat_interleave(2, dim=-1)
    sin = sin.repeat_interleave(2, dim=-1)

    return x * cos + rotated * sin


class SwiGLU(nn.Module):
    """SwiGLU activation: Swish(xW) * (xV)

    Better than ReLU/GELU for transformers.
    """
    def __init__(self, in_features: int, hidden_features: int, bias: bool = False):
        super().__init__()
        self.w1 = nn.Linear(in_features, hidden_features, bias=bias)  # Gate
        self.w2 = nn.Linear(in_features, hidden_features, bias=bias)  # Up
        self.w3 = nn.Linear(hidden_features, in_features, bias=bias)  # Down

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w3(F.silu(self.w1(x)) * self.w2(x))


class MultiHeadAttention(nn.Module):
    """Multi-head attention with RoPE and Flash Attention."""

    def __init__(self, config: BagsV5Config):
        super().__init__()
        self.n_head = config.n_head
        self.head_dim = config.n_embd // config.n_head
        self.use_flash = config.use_flash_attn

        self.qkv = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.out_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

        # RoPE
        self.rotary = RotaryEmbedding(
            self.head_dim,
            max_seq_len=config.context_length * 2,
            theta=config.rope_theta
        )

    def forward(self, x: torch.Tensor, causal: bool = True) -> torch.Tensor:
        B, T, C = x.shape

        # QKV projection
        qkv = self.qkv(x).reshape(B, T, 3, self.n_head, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, n_head, T, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Apply RoPE to Q and K
        cos, sin = self.rotary(q)
        q = apply_rotary_emb(q, cos, sin)
        k = apply_rotary_emb(k, cos, sin)

        # Attention
        if self.use_flash:
            # Use PyTorch's optimized SDPA (includes Flash Attention)
            attn_out = F.scaled_dot_product_attention(
                q, k, v,
                dropout_p=self.dropout.p if self.training else 0.0,
                is_causal=causal,
            )
        else:
            # Manual attention
            scale = 1.0 / math.sqrt(self.head_dim)
            attn = (q @ k.transpose(-2, -1)) * scale
            if causal:
                mask = torch.triu(torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=1)
                attn = attn.masked_fill(mask, float('-inf'))
            attn = F.softmax(attn, dim=-1)
            attn = self.dropout(attn)
            attn_out = attn @ v

        # Reshape and project
        attn_out = attn_out.transpose(1, 2).reshape(B, T, C)
        return self.out_proj(attn_out)


class TransformerBlock(nn.Module):
    """Transformer block with pre-norm, RoPE attention, and SwiGLU FFN."""

    def __init__(self, config: BagsV5Config):
        super().__init__()
        self.norm1 = RMSNorm(config.n_embd)
        self.attn = MultiHeadAttention(config)
        self.norm2 = RMSNorm(config.n_embd)

        # SwiGLU FFN
        hidden_dim = int(config.n_embd * config.ffn_multiplier)
        self.ffn = SwiGLU(config.n_embd, hidden_dim, bias=config.bias)

        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pre-norm attention with residual
        x = x + self.dropout(self.attn(self.norm1(x)))
        # Pre-norm FFN with residual
        x = x + self.dropout(self.ffn(self.norm2(x)))
        return x


class BagsV5Transformer(nn.Module):
    """BagsV5 Transformer for trading signal prediction.

    State-of-the-art architecture with:
    - RMSNorm
    - SwiGLU
    - RoPE
    - Flash Attention
    - Gradient checkpointing
    """

    def __init__(self, config: BagsV5Config):
        super().__init__()
        self.config = config

        # Input projections
        self.bar_embed = nn.Linear(config.features_per_bar, config.n_embd)
        self.agg_embed = nn.Linear(config.agg_features, config.n_embd)

        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(config) for _ in range(config.n_layer)
        ])

        # Output normalization
        self.norm = RMSNorm(config.n_embd)

        # Prediction heads
        self.signal_head = nn.Linear(config.n_embd, 1)
        self.size_head = nn.Linear(config.n_embd, 1)

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)

    def forward(
        self,
        bar_features: torch.Tensor,  # (B, T, features_per_bar)
        agg_features: torch.Tensor,  # (B, agg_features)
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B, T, _ = bar_features.shape

        # Embed bar features
        x = self.bar_embed(bar_features)  # (B, T, n_embd)

        # Add aggregate context to all positions
        agg_emb = self.agg_embed(agg_features).unsqueeze(1)  # (B, 1, n_embd)
        x = x + agg_emb

        # Transformer blocks
        for block in self.blocks:
            if self.config.use_gradient_checkpointing and self.training:
                x = checkpoint(block, x, use_reentrant=False)
            else:
                x = block(x)

        # Final norm and pooling (use last position for causal)
        x = self.norm(x)
        x_last = x[:, -1, :]  # (B, n_embd)

        # Predictions
        signal_logit = self.signal_head(x_last).squeeze(-1)  # (B,)
        size_logit = self.size_head(x_last).squeeze(-1)      # (B,)

        return signal_logit, size_logit

    def get_weight_params(self):
        """Get 2D weight parameters for Muon optimizer."""
        params = []
        for name, p in self.named_parameters():
            if p.ndim == 2 and 'embed' not in name and 'head' not in name:
                params.append(p)
        return params

    def get_other_params(self):
        """Get other parameters for AdamW optimizer."""
        weight_params = set(id(p) for p in self.get_weight_params())
        return [p for p in self.parameters() if id(p) not in weight_params]


class FocalLoss(nn.Module):
    """Focal Loss for class imbalance."""

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        ce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')

        p_t = probs * targets + (1 - probs) * (1 - targets)
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        focal_weight = alpha_t * (1 - p_t) ** self.gamma

        return (focal_weight * ce_loss).mean()
