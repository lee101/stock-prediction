"""V3 Timed Transformer model with explicit exit timing output.

Key changes from V2:
- 5 outputs: buy_price, sell_price, trade_amount, confidence, exit_days
- exit_days is learned (1-10 days) to control maximum hold duration
- Model learns when to exit, not just at what price

Original V2 improvements retained:
- Rotary Position Embeddings (RoPE)
- Multi-Query Attention (MQA) with fewer KV heads
- QK Normalization for training stability
- RMSNorm without learnable parameters
- ReLU^2 activation in MLP
- Logits softcap to prevent extreme outputs
"""
from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from neuraldailyv3timed.config import PolicyConfigV3


def rms_norm(x: torch.Tensor) -> torch.Tensor:
    """RMSNorm without learnable parameters."""
    return F.rms_norm(x, (x.size(-1),))


def apply_rotary_emb(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    """Apply rotary position embeddings to input tensor.

    Args:
        x: (batch, seq_len, num_heads, head_dim)
        cos: (1, seq_len, 1, head_dim // 2)
        sin: (1, seq_len, 1, head_dim // 2)

    Returns:
        Tensor with rotary embeddings applied
    """
    # Split into even and odd dimensions
    x_r = x[..., ::2]  # Even indices
    x_i = x[..., 1::2]  # Odd indices

    # Apply rotation
    out_r = x_r * cos - x_i * sin
    out_i = x_r * sin + x_i * cos

    # Interleave back
    out = torch.stack([out_r, out_i], dim=-1)
    return out.flatten(-2)


class RotaryEmbedding(nn.Module):
    """Rotary Position Embeddings (RoPE) from nanochat."""

    def __init__(self, dim: int, max_len: int = 4096, base: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.max_len = max_len
        self.base = base

        # Precompute inverse frequencies
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # Build initial cache
        self._build_cache(max_len)

    def _build_cache(self, seq_len: int) -> None:
        """Build cos/sin cache for given sequence length."""
        t = torch.arange(seq_len, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq)

        # Shape: (1, seq_len, 1, dim // 2) for broadcasting
        cos = freqs.cos()[None, :, None, :]
        sin = freqs.sin()[None, :, None, :]

        self.register_buffer("cos_cached", cos, persistent=False)
        self.register_buffer("sin_cached", sin, persistent=False)
        self.cached_len = seq_len

    def forward(self, seq_len: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get cos/sin for given sequence length."""
        if seq_len > self.cached_len:
            self._build_cache(seq_len)
        return self.cos_cached[:, :seq_len], self.sin_cached[:, :seq_len]


class MultiQueryAttention(nn.Module):
    """
    Multi-Query Attention with QK normalization.

    From nanochat: Uses fewer KV heads than query heads for efficiency.
    QK normalization improves training stability.
    """

    def __init__(self, config: PolicyConfigV3, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.n_head = config.num_heads
        self.n_kv_head = config.num_kv_heads
        self.head_dim = config.head_dim

        # Projections
        self.q_proj = nn.Linear(config.hidden_dim, config.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(config.hidden_dim, config.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.hidden_dim, config.num_kv_heads * self.head_dim, bias=False)
        self.out_proj = nn.Linear(config.hidden_dim, config.hidden_dim, bias=False)

        self.dropout = nn.Dropout(config.dropout)

    def forward(
        self,
        x: torch.Tensor,
        cos_sin: Tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, hidden_dim)
            cos_sin: Tuple of (cos, sin) from RotaryEmbedding
            attention_mask: Optional mask for attention

        Returns:
            (batch, seq_len, hidden_dim)
        """
        B, T, _ = x.size()
        cos, sin = cos_sin

        # Project Q, K, V
        q = self.q_proj(x).view(B, T, self.n_head, self.head_dim)
        k = self.k_proj(x).view(B, T, self.n_kv_head, self.head_dim)
        v = self.v_proj(x).view(B, T, self.n_kv_head, self.head_dim)

        # Apply RoPE
        q = apply_rotary_emb(q, cos, sin)
        k = apply_rotary_emb(k, cos, sin)

        # QK normalization (from nanochat)
        q = F.rms_norm(q, (q.size(-1),))
        k = F.rms_norm(k, (k.size(-1),))

        # Transpose for attention: (B, H, T, D)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Scaled dot-product attention
        # Handle MQA by expanding KV heads to match query heads
        if self.n_kv_head != self.n_head:
            # Expand KV for MQA: repeat each KV head for multiple query heads
            n_rep = self.n_head // self.n_kv_head
            k = k.repeat_interleave(n_rep, dim=1)
            v = v.repeat_interleave(n_rep, dim=1)

        y = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attention_mask,
            is_causal=True,
            dropout_p=self.dropout.p if self.training else 0.0,
        )

        # Reshape and project output
        y = y.transpose(1, 2).contiguous().view(B, T, -1)
        return self.out_proj(y)


class CrossSymbolAttention(nn.Module):
    """
    Cross-symbol attention for learning inter-asset relationships.

    Allows symbols in the same batch to share information during
    processing. Uses group masking to control which symbols can
    attend to each other.
    """

    def __init__(self, config: PolicyConfigV3):
        super().__init__()
        self.config = config
        self.n_head = config.num_heads
        self.head_dim = config.head_dim

        # Use same number of heads for Q/K/V in cross-symbol attention
        self.q_proj = nn.Linear(config.hidden_dim, config.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(config.hidden_dim, config.num_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.hidden_dim, config.num_heads * self.head_dim, bias=False)
        self.out_proj = nn.Linear(config.hidden_dim, config.hidden_dim, bias=False)

        self.dropout = nn.Dropout(config.dropout)

    def forward(
        self,
        x: torch.Tensor,
        group_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Cross-symbol attention across the batch dimension.

        Args:
            x: (batch, seq_len, hidden_dim)
            group_mask: (batch, batch) boolean mask - True where symbols can attend

        Returns:
            (batch, seq_len, hidden_dim)
        """
        B, T, D = x.shape

        # Transpose to (T, B, D) for cross-symbol attention
        x_t = x.transpose(0, 1)  # (T, B, D)

        # Project and reshape
        q = self.q_proj(x_t).view(T, B, self.n_head, self.head_dim)
        k = self.k_proj(x_t).view(T, B, self.n_head, self.head_dim)
        v = self.v_proj(x_t).view(T, B, self.n_head, self.head_dim)

        # QK normalization
        q = F.rms_norm(q, (q.size(-1),))
        k = F.rms_norm(k, (k.size(-1),))

        # Reshape for attention: (T, H, B, D) - attending across B (symbols)
        q = q.permute(0, 2, 1, 3)  # (T, H, B, D)
        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)

        # Compute attention scores
        scale = 1.0 / math.sqrt(self.head_dim)
        attn = torch.einsum("thbd,thcd->thbc", q, k) * scale  # (T, H, B, B)

        # Apply group mask if provided
        if group_mask is not None:
            # Ensure self-attention (diagonal) is always allowed
            mask = group_mask | torch.eye(B, device=group_mask.device, dtype=torch.bool)
            # Expand for time and heads: (1, 1, B, B)
            mask = mask[None, None, :, :]
            attn = attn.masked_fill(~mask, float("-inf"))

        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        # Apply attention to values
        y = torch.einsum("thbc,thcd->thbd", attn, v)  # (T, H, B, D)

        # Reshape and project
        y = y.permute(2, 0, 1, 3).contiguous()  # (B, T, H, D)
        y = y.view(B, T, -1)
        return self.out_proj(y)


class MLP(nn.Module):
    """MLP with ReLU^2 activation (from nanochat)."""

    def __init__(self, config: PolicyConfigV3):
        super().__init__()
        self.fc = nn.Linear(config.hidden_dim, 4 * config.hidden_dim, bias=False)
        self.proj = nn.Linear(4 * config.hidden_dim, config.hidden_dim, bias=False)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc(x)
        x = F.relu(x).square()  # ReLU^2 activation
        x = self.dropout(x)
        return self.proj(x)


class TransformerBlock(nn.Module):
    """
    Transformer block with:
    - Pre-norm architecture (RMSNorm before each sub-layer)
    - Multi-Query Attention with RoPE and QK-norm
    - Optional cross-symbol attention
    - MLP with ReLU^2
    """

    def __init__(self, config: PolicyConfigV3, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx

        self.time_attn = MultiQueryAttention(config, layer_idx)
        self.mlp = MLP(config)
        self.cross_attn = CrossSymbolAttention(config) if config.use_cross_attention else None

    def forward(
        self,
        x: torch.Tensor,
        cos_sin: Tuple[torch.Tensor, torch.Tensor],
        group_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, hidden_dim)
            cos_sin: Rotary embeddings
            group_mask: Optional cross-symbol attention mask

        Returns:
            (batch, seq_len, hidden_dim)
        """
        # Time attention with pre-norm
        x = x + self.time_attn(rms_norm(x), cos_sin)

        # Cross-symbol attention with pre-norm (if enabled)
        if self.cross_attn is not None:
            x = x + self.cross_attn(rms_norm(x), group_mask)

        # MLP with pre-norm
        x = x + self.mlp(rms_norm(x))

        return x


class MultiSymbolPolicyV3(nn.Module):
    """
    V3 Timed Policy network with explicit exit timing.

    Key changes from V2:
    - 5 outputs instead of 4: adds exit_days (1-10 days)
    - Model learns when to exit, not just at what price
    - Exit timing is enforced at both training and inference

    Retained from V2:
    - RoPE position encoding
    - Multi-Query Attention for efficiency
    - QK normalization for stability
    - RMSNorm without learnable parameters
    - ReLU^2 activation in MLP
    - Logits softcap
    """

    def __init__(self, config: PolicyConfigV3):
        super().__init__()
        self.config = config

        # Input embedding
        self.embed = nn.Linear(config.input_dim, config.hidden_dim, bias=False)

        # Rotary embeddings
        self.rope = RotaryEmbedding(
            config.head_dim,
            max_len=config.max_len,
        )

        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(config, i) for i in range(config.num_layers)
        ])

        # Output head: buy_price, sell_price, trade_amount, confidence, exit_days
        # V3: 5 outputs instead of 4
        self.head = nn.Linear(config.hidden_dim, 5, bias=False)

        # Initialize weights
        self._init_weights()

    def _init_weights(self) -> None:
        """Weight initialization scaled by model dimensions (from nanochat)."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                fan_out = module.weight.size(0)
                fan_in = module.weight.size(1)
                std = 1.0 / math.sqrt(fan_in) * min(1.0, math.sqrt(fan_out / fan_in))
                nn.init.normal_(module.weight, mean=0.0, std=std)

        # Small random init for head to enable differentiation between samples
        # (Zero init causes all predictions to be identical)
        nn.init.normal_(self.head.weight, mean=0.0, std=0.1)
        for block in self.blocks:
            nn.init.zeros_(block.time_attn.out_proj.weight)
            nn.init.zeros_(block.mlp.proj.weight)
            if block.cross_attn is not None:
                nn.init.zeros_(block.cross_attn.out_proj.weight)

    def forward(
        self,
        features: torch.Tensor,
        group_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            features: (batch, seq_len, input_dim) - normalized features
            group_mask: (batch, batch) - cross-symbol attention mask

        Returns:
            Dict with logits for buy_price, sell_price, trade_amount, confidence, exit_days
        """
        B, T, _ = features.shape

        # Embed input
        h = self.embed(features)
        h = rms_norm(h)  # Normalize after embedding

        # Get rotary embeddings
        cos_sin = self.rope(T)

        # Transformer blocks
        for block in self.blocks:
            h = block(h, cos_sin, group_mask)

        # Final norm and head
        h = rms_norm(h)
        logits = self.head(h)

        # Logits softcap (from nanochat) - prevents extreme values
        softcap = self.config.logits_softcap
        logits = softcap * torch.tanh(logits / softcap)

        return {
            "buy_price_logits": logits[..., 0:1],
            "sell_price_logits": logits[..., 1:2],
            "trade_amount_logits": logits[..., 2:3],
            "confidence_logits": logits[..., 3:4],
            "exit_days_logits": logits[..., 4:5],  # NEW in V3
        }

    def decode_actions(
        self,
        outputs: Dict[str, torch.Tensor],
        *,
        reference_close: torch.Tensor,
        chronos_high: torch.Tensor,
        chronos_low: torch.Tensor,
        asset_class: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Decode model outputs to trading actions.

        Args:
            outputs: Dict from forward() with logits
            reference_close: (batch, seq_len) - close prices
            chronos_high: (batch, seq_len) - forecast high
            chronos_low: (batch, seq_len) - forecast low
            asset_class: (batch,) - 0 for stocks, 1 for crypto

        Returns:
            Dict with buy_price, sell_price, trade_amount, confidence, exit_days
        """
        config = self.config

        # Buy price: sigmoid gives 0-1 offset below reference
        buy_offset = config.price_offset_pct * torch.sigmoid(
            outputs["buy_price_logits"]
        ).squeeze(-1)
        buy_price = reference_close * (1.0 - buy_offset)

        # Clamp to [chronos_low, reference_close]
        buy_price = torch.clamp(buy_price, min=chronos_low, max=reference_close)

        # Sell price: sigmoid gives 0-1 offset above reference
        sell_offset = config.price_offset_pct * torch.sigmoid(
            outputs["sell_price_logits"]
        ).squeeze(-1)
        sell_price = reference_close * (1.0 + sell_offset)

        # Clamp to [reference_close, chronos_high]
        sell_price = torch.clamp(sell_price, min=reference_close, max=chronos_high)

        # Ensure minimum gap between buy and sell
        min_gap = reference_close * config.min_price_gap_pct
        sell_price = torch.maximum(sell_price, buy_price + min_gap)

        # Trade amount: sigmoid gives 0-1, then scale by leverage
        base_amount = torch.sigmoid(outputs["trade_amount_logits"]).squeeze(-1)

        if asset_class is not None:
            # Scale by asset-specific leverage limits
            # asset_class: 0 = stock (2x), 1 = crypto (1x)
            leverage_limit = torch.where(
                asset_class.unsqueeze(-1) > 0.5,
                torch.tensor(config.crypto_max_leverage, device=base_amount.device),
                torch.tensor(config.equity_max_leverage, device=base_amount.device),
            )
            trade_amount = base_amount * leverage_limit.expand_as(base_amount)
        else:
            trade_amount = base_amount * config.equity_max_leverage

        # Confidence: just sigmoid
        confidence = torch.sigmoid(outputs["confidence_logits"]).squeeze(-1)

        # Exit days: sigmoid scaled to [min_exit_days, max_exit_days]
        # V3 NEW: Explicit exit timing
        exit_days_raw = torch.sigmoid(outputs["exit_days_logits"]).squeeze(-1)
        exit_days_range = config.max_exit_days - config.min_exit_days
        exit_days = config.min_exit_days + exit_days_raw * exit_days_range

        return {
            "buy_price": buy_price,
            "sell_price": sell_price,
            "trade_amount": trade_amount,
            "confidence": confidence,
            "exit_days": exit_days,  # NEW in V3: 1-10 days
        }


def create_group_mask(group_ids: torch.Tensor) -> torch.Tensor:
    """
    Create cross-symbol attention mask from group IDs.

    Args:
        group_ids: (batch,) - integer group ID per symbol

    Returns:
        (batch, batch) boolean mask - True where symbols can attend
    """
    # Symbols in the same group can attend to each other
    return group_ids.unsqueeze(0) == group_ids.unsqueeze(1)
