"""Neural model for price optimization with PnL forecasting.

Key features:
- Transformer architecture with patch embedding
- Outputs: position_length, position_size, buy_offset, sell_offset
- Fee-aware price clamping
- PnL history integration for Chronos2-style optimization
"""
from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from chronospnltrader.config import PolicyConfig


def rms_norm(x: torch.Tensor) -> torch.Tensor:
    """RMSNorm without learnable parameters."""
    return F.rms_norm(x, (x.size(-1),))


def apply_rotary_emb(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    """Apply rotary position embeddings."""
    x_r = x[..., ::2]
    x_i = x[..., 1::2]
    out_r = x_r * cos - x_i * sin
    out_i = x_r * sin + x_i * cos
    out = torch.stack([out_r, out_i], dim=-1)
    return out.flatten(-2)


class RotaryEmbedding(nn.Module):
    """Rotary Position Embeddings (RoPE)."""

    def __init__(self, dim: int, max_len: int = 512, base: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.max_len = max_len
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._build_cache(max_len)

    def _build_cache(self, seq_len: int) -> None:
        t = torch.arange(seq_len, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq)
        cos = freqs.cos()[None, :, None, :]
        sin = freqs.sin()[None, :, None, :]
        self.register_buffer("cos_cached", cos, persistent=False)
        self.register_buffer("sin_cached", sin, persistent=False)
        self.cached_len = seq_len

    def forward(self, seq_len: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if seq_len > self.cached_len:
            self._build_cache(seq_len)
        return self.cos_cached[:, :seq_len], self.sin_cached[:, :seq_len]


class PatchEmbedding(nn.Module):
    """Embed time series patches (4-hour patches for hourly data)."""

    def __init__(self, config: PolicyConfig):
        super().__init__()
        self.patch_size = config.patch_size
        self.input_dim = config.input_dim

        patch_dim = config.patch_size * config.input_dim
        self.embed = nn.Linear(patch_dim, config.hidden_dim, bias=False)

        if config.patch_residual:
            self.residual = nn.Sequential(
                nn.Linear(config.hidden_dim, config.hidden_dim * 2, bias=False),
                nn.GELU(),
                nn.Linear(config.hidden_dim * 2, config.hidden_dim, bias=False),
            )
        else:
            self.residual = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        num_patches = T // self.patch_size

        x = x[:, :num_patches * self.patch_size].contiguous()
        x = x.reshape(B, num_patches, self.patch_size * D)

        h = self.embed(x)
        if self.residual is not None:
            h = h + self.residual(h)

        return h


class MultiQueryAttention(nn.Module):
    """Multi-Query Attention with QK normalization."""

    def __init__(self, config: PolicyConfig, layer_idx: int):
        super().__init__()
        self.n_head = config.num_heads
        self.n_kv_head = config.num_kv_heads
        self.head_dim = config.head_dim

        self.q_proj = nn.Linear(config.hidden_dim, config.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(config.hidden_dim, config.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.hidden_dim, config.num_kv_heads * self.head_dim, bias=False)
        self.out_proj = nn.Linear(config.hidden_dim, config.hidden_dim, bias=False)
        self.dropout = nn.Dropout(config.dropout)

    def forward(
        self,
        x: torch.Tensor,
        cos_sin: Tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        B, T, _ = x.size()
        cos, sin = cos_sin

        q = self.q_proj(x).view(B, T, self.n_head, self.head_dim)
        k = self.k_proj(x).view(B, T, self.n_kv_head, self.head_dim)
        v = self.v_proj(x).view(B, T, self.n_kv_head, self.head_dim)

        q = apply_rotary_emb(q, cos, sin)
        k = apply_rotary_emb(k, cos, sin)

        q = F.rms_norm(q, (q.size(-1),))
        k = F.rms_norm(k, (k.size(-1),))

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        if self.n_kv_head != self.n_head:
            n_rep = self.n_head // self.n_kv_head
            k = k.repeat_interleave(n_rep, dim=1)
            v = v.repeat_interleave(n_rep, dim=1)

        y = F.scaled_dot_product_attention(
            q, k, v,
            is_causal=True,
            dropout_p=self.dropout.p if self.training else 0.0,
        )

        y = y.transpose(1, 2).contiguous().view(B, T, -1)
        return self.out_proj(y)


class MLP(nn.Module):
    """MLP with ReLU^2 activation."""

    def __init__(self, config: PolicyConfig):
        super().__init__()
        self.fc = nn.Linear(config.hidden_dim, 4 * config.hidden_dim, bias=False)
        self.proj = nn.Linear(4 * config.hidden_dim, config.hidden_dim, bias=False)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc(x)
        x = F.relu(x).square()
        x = self.dropout(x)
        return self.proj(x)


class TransformerBlock(nn.Module):
    """Transformer block with attention and MLP."""

    def __init__(self, config: PolicyConfig, layer_idx: int):
        super().__init__()
        self.attn = MultiQueryAttention(config, layer_idx)
        self.mlp = MLP(config)

    def forward(
        self,
        x: torch.Tensor,
        cos_sin: Tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        x = x + self.attn(rms_norm(x), cos_sin)
        x = x + self.mlp(rms_norm(x))
        return x


class PnLEncoder(nn.Module):
    """Encodes PnL history for integration with price features.

    Learns to extract patterns from historical PnL that predict
    future profitability.
    """

    def __init__(self, hidden_dim: int, max_history: int = 210):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.max_history = max_history

        # Project PnL history to embedding
        self.pnl_embed = nn.Sequential(
            nn.Linear(max_history, hidden_dim, bias=False),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim, bias=False),
        )

        # Learnable statistics extraction
        self.stats_embed = nn.Linear(5, hidden_dim, bias=False)

    def forward(self, pnl_history: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pnl_history: (batch, history_len)

        Returns:
            (batch, hidden_dim) PnL embedding
        """
        B = pnl_history.size(0)
        device = pnl_history.device

        # Pad/truncate to max_history
        if pnl_history.size(1) < self.max_history:
            pad = torch.zeros(B, self.max_history - pnl_history.size(1), device=device)
            pnl_history = torch.cat([pad, pnl_history], dim=1)
        else:
            pnl_history = pnl_history[:, -self.max_history:]

        # Direct embedding
        pnl_embed = self.pnl_embed(pnl_history)

        # Extract statistics
        mean = pnl_history.mean(dim=1, keepdim=True)
        std = pnl_history.std(dim=1, keepdim=True) + 1e-8
        cumsum = pnl_history.sum(dim=1, keepdim=True)
        recent = pnl_history[:, -7:].mean(dim=1, keepdim=True)
        trend = (pnl_history[:, -7:].mean(dim=1) - pnl_history[:, -35:-7].mean(dim=1)).unsqueeze(1)

        stats = torch.cat([mean, std, cumsum, recent, trend], dim=1)
        stats_embed = self.stats_embed(stats)

        return pnl_embed + stats_embed


class ChronosPnLPolicy(nn.Module):
    """Policy network for ChronosPnL trading.

    Key features:
    - Patch embedding for price features
    - PnL history encoding
    - Transformer backbone
    - Fee-aware price output heads

    The model learns to optimize trades such that the Chronos2
    PnL forecaster predicts profitability.
    """

    def __init__(self, config: PolicyConfig):
        super().__init__()
        self.config = config

        # Price feature embedding
        if config.use_patch_embedding:
            self.patch_embed = PatchEmbedding(config)
        else:
            self.patch_embed = None
            self.embed = nn.Linear(config.input_dim, config.hidden_dim, bias=False)

        # PnL history encoding
        self.pnl_encoder = PnLEncoder(config.hidden_dim)

        # RoPE
        self.rope = RotaryEmbedding(config.head_dim, max_len=config.max_len)

        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(config, i) for i in range(config.num_layers)
        ])

        # Output heads
        self.position_length_head = nn.Linear(config.hidden_dim, config.num_position_classes, bias=False)
        self.position_size_head = nn.Linear(config.hidden_dim, 1, bias=False)
        self.buy_offset_head = nn.Linear(config.hidden_dim, 1, bias=False)
        self.sell_offset_head = nn.Linear(config.hidden_dim, 1, bias=False)

        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                fan_in = module.weight.size(1)
                std = 1.0 / math.sqrt(fan_in)
                nn.init.normal_(module.weight, mean=0.0, std=std)

        # Small init for output heads
        for head in [self.position_length_head, self.position_size_head,
                     self.buy_offset_head, self.sell_offset_head]:
            nn.init.normal_(head.weight, mean=0.0, std=0.02)

        # Zero init for residual projections
        for block in self.blocks:
            nn.init.zeros_(block.attn.out_proj.weight)
            nn.init.zeros_(block.mlp.proj.weight)

    def forward(
        self,
        features: torch.Tensor,
        pnl_history: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            features: (batch, seq_len, input_dim)
            pnl_history: (batch, history_len) optional PnL history

        Returns:
            Dict with logits for each output
        """
        # Patch embedding
        if self.patch_embed is not None:
            h = self.patch_embed(features)
        else:
            h = self.embed(features)

        # Add PnL embedding if available
        if pnl_history is not None:
            pnl_embed = self.pnl_encoder(pnl_history)
            # Add to last position (most recent)
            h[:, -1] = h[:, -1] + pnl_embed

        h = rms_norm(h)

        T = h.size(1)
        cos_sin = self.rope(T)

        for block in self.blocks:
            h = block(h, cos_sin)

        h = rms_norm(h)

        # Pool: last patch
        h_pooled = h[:, -1, :]

        outputs = {
            "position_length_logits": self.position_length_head(h_pooled),
            "position_size_logits": self.position_size_head(h_pooled),
            "buy_offset_logits": self.buy_offset_head(h_pooled),
            "sell_offset_logits": self.sell_offset_head(h_pooled),
        }

        # Softcap
        softcap = self.config.logits_softcap
        for key in outputs:
            outputs[key] = softcap * torch.tanh(outputs[key] / softcap)

        return outputs

    def decode_actions(
        self,
        outputs: Dict[str, torch.Tensor],
        reference_close: torch.Tensor,
        temperature: float = 1.0,
    ) -> Dict[str, torch.Tensor]:
        """Decode outputs to trading actions with fee-aware safety."""
        config = self.config
        device = reference_close.device

        # Position length
        length_probs = F.softmax(outputs["position_length_logits"] / temperature, dim=-1)
        hours = torch.arange(config.num_position_classes, device=device, dtype=torch.float32)
        position_length = (length_probs * hours).sum(dim=-1)

        # Position size
        position_size = torch.sigmoid(outputs["position_size_logits"]).squeeze(-1)

        # Fee-aware prices
        buy_raw = torch.sigmoid(outputs["buy_offset_logits"]).squeeze(-1)
        sell_raw = torch.sigmoid(outputs["sell_offset_logits"]).squeeze(-1)

        offset_range = config.max_price_offset_pct - config.min_price_offset_pct
        buy_offset = config.min_price_offset_pct + buy_raw * offset_range
        sell_offset = config.min_price_offset_pct + sell_raw * offset_range

        buy_price = reference_close * (1.0 - buy_offset)
        sell_price = reference_close * (1.0 + sell_offset)

        return {
            "position_length": position_length,
            "position_size": position_size,
            "buy_price": buy_price,
            "sell_price": sell_price,
            "buy_offset": buy_offset,
            "sell_offset": sell_offset,
            "length_probs": length_probs,
        }

    def get_hard_actions(
        self,
        outputs: Dict[str, torch.Tensor],
        reference_close: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Get hard (discrete) actions for inference."""
        config = self.config

        position_length = outputs["position_length_logits"].argmax(dim=-1).float()
        position_size = torch.sigmoid(outputs["position_size_logits"]).squeeze(-1)

        buy_raw = torch.sigmoid(outputs["buy_offset_logits"]).squeeze(-1)
        sell_raw = torch.sigmoid(outputs["sell_offset_logits"]).squeeze(-1)

        offset_range = config.max_price_offset_pct - config.min_price_offset_pct
        buy_offset = config.min_price_offset_pct + buy_raw * offset_range
        sell_offset = config.min_price_offset_pct + sell_raw * offset_range

        buy_price = reference_close * (1.0 - buy_offset)
        sell_price = reference_close * (1.0 + sell_offset)

        return {
            "position_length": position_length,
            "position_size": position_size,
            "buy_price": buy_price,
            "sell_price": sell_price,
            "buy_offset": buy_offset,
            "sell_offset": sell_offset,
        }


def create_model(config: PolicyConfig) -> ChronosPnLPolicy:
    """Factory function to create the model."""
    return ChronosPnLPolicy(config)
