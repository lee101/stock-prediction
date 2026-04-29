"""Cross-attention transformer for daily stock top-1 prediction.

Two-stage attention:
  1. Temporal self-attention within each symbol over a 256-day rolling
     context of (15 daily features). Uses RoPE positional encoding +
     RMSNorm + SwiGLU.
  2. Cross-sectional self-attention across ALL active symbols on day D
     (no positional embedding — symbols are exchangeable).

Output: per-symbol logit for P(target_oc_up) — same binary label XGB uses.

Train one day-batch at a time: pick a date D, gather 256-day windows for
every active symbol on that day, single forward pass produces a logit per
symbol. Loss = mean BCE.

Designed for ~30M params at d_model=256, n_layers=6, n_heads=8.

This file is self-contained; no external imports beyond torch + nn.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ── Modern building blocks ───────────────────────────────────────────────────

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Cast to fp32 for the norm reduction (stability under bf16)
        out = x * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + self.eps)
        return (out * self.weight).to(x.dtype)


def precompute_rope(seq_len: int, head_dim: int, base: float = 10_000.0,
                    device: torch.device | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (cos, sin) of shape (seq_len, head_dim) suitable for rotary."""
    half = head_dim // 2
    freqs = base ** (-torch.arange(0, half, device=device).float() / half)
    t = torch.arange(seq_len, device=device).float()
    angles = torch.outer(t, freqs)             # (seq_len, half)
    cos = torch.cos(angles).repeat_interleave(2, dim=-1)  # (seq_len, head_dim)
    sin = torch.sin(angles).repeat_interleave(2, dim=-1)
    return cos, sin


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Rotate x along its last (head_dim) axis using RoPE.

    x: (B, T, H, hd) — rotary applied along T using head_dim hd.
    cos/sin: (T, hd).
    """
    # Pair-rotate every two channels: x_even, x_odd → (x_even*cos - x_odd*sin, x_even*sin + x_odd*cos)
    x1 = x[..., 0::2]
    x2 = x[..., 1::2]
    rot = torch.stack((-x2, x1), dim=-1).flatten(-2)
    # Broadcast cos/sin from (T, hd) to (1, T, 1, hd) for (B, T, H, hd) input
    cos_b = cos.unsqueeze(0).unsqueeze(2)   # (1, T, 1, hd)
    sin_b = sin.unsqueeze(0).unsqueeze(2)
    return (x * cos_b + rot * sin_b).to(x.dtype)


class SwiGLU(nn.Module):
    """Modern feed-forward block: SiLU(W1 x) * (W2 x), then W3 down-projects."""
    def __init__(self, dim: int, hidden_mult: float = 8.0 / 3.0) -> None:
        super().__init__()
        hidden = int(round(dim * hidden_mult / 64.0)) * 64  # multiple of 64
        self.w1 = nn.Linear(dim, hidden, bias=False)
        self.w2 = nn.Linear(dim, hidden, bias=False)
        self.w3 = nn.Linear(hidden, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w3(F.silu(self.w1(x)) * self.w2(x))


# ── Attention blocks ─────────────────────────────────────────────────────────

class TemporalAttention(nn.Module):
    """Causal self-attention with RoPE over the 256-step time axis."""

    def __init__(self, dim: int, n_heads: int) -> None:
        super().__init__()
        assert dim % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.out = nn.Linear(dim, dim, bias=False)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        # x: (B, T, D)
        B, T, D = x.shape
        qkv = self.qkv(x).view(B, T, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)              # each (B, T, H, hd)
        q = apply_rope(q, cos[:T], sin[:T])
        k = apply_rope(k, cos[:T], sin[:T])
        # → (B, H, T, hd) for SDPA
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        att = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        att = att.transpose(1, 2).contiguous().view(B, T, D)
        return self.out(att)


class CrossSectionalAttention(nn.Module):
    """Plain (non-causal, no positional encoding) self-attention across symbols.

    Input is one day's (n_syms, D); the attention treats the n_syms axis as a
    permutation-invariant set. Output keeps the same shape.
    """
    def __init__(self, dim: int, n_heads: int) -> None:
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.out = nn.Linear(dim, dim, bias=False)

    def forward(self, x: torch.Tensor, key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # x: (B=1, S, D)
        B, S, D = x.shape
        qkv = self.qkv(x).view(B, S, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)              # each (B, S, H, hd)
        q = q.transpose(1, 2)                    # (B, H, S, hd)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        attn_mask = None
        if key_padding_mask is not None:
            # mask is (B, S), True where INVALID (we want to mask those keys).
            attn_mask = key_padding_mask[:, None, None, :]   # broadcast over H, query S
            # attn_mask=True ⇒ position is masked OUT
            attn_mask = ~attn_mask  # SDPA accepts bool where True means KEEP/attend
            # Actually: torch.nn.functional.scaled_dot_product_attention(attn_mask=...) where
            # bool True means "attend" (matches torch.nn.MultiheadAttention's attn_mask convention
            # changed in newer torch — use additive instead for safety).
            additive = torch.zeros_like(attn_mask, dtype=q.dtype)
            additive.masked_fill_(~attn_mask, float("-inf"))
            attn_mask = additive
        att = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, is_causal=False)
        att = att.transpose(1, 2).contiguous().view(B, S, D)
        return self.out(att)


# ── Transformer blocks ───────────────────────────────────────────────────────

class TemporalBlock(nn.Module):
    def __init__(self, dim: int, n_heads: int) -> None:
        super().__init__()
        self.norm1 = RMSNorm(dim)
        self.attn = TemporalAttention(dim, n_heads)
        self.norm2 = RMSNorm(dim)
        self.ff = SwiGLU(dim)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), cos, sin)
        x = x + self.ff(self.norm2(x))
        return x


class CrossSectionalBlock(nn.Module):
    def __init__(self, dim: int, n_heads: int) -> None:
        super().__init__()
        self.norm1 = RMSNorm(dim)
        self.attn = CrossSectionalAttention(dim, n_heads)
        self.norm2 = RMSNorm(dim)
        self.ff = SwiGLU(dim)

    def forward(self, x: torch.Tensor, key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), key_padding_mask=key_padding_mask)
        x = x + self.ff(self.norm2(x))
        return x


# ── Full model ───────────────────────────────────────────────────────────────

@dataclass
class CrossAttnConfig:
    n_features: int = 15
    seq_len: int = 256
    d_model: int = 256
    n_temporal_layers: int = 4
    n_cross_layers: int = 2
    n_heads: int = 8
    dropout: float = 0.0
    grad_checkpoint: bool = True


class CrossAttnTransformerV1(nn.Module):
    def __init__(self, cfg: CrossAttnConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.feat_proj = nn.Linear(cfg.n_features, cfg.d_model, bias=False)
        self.in_norm = RMSNorm(cfg.d_model)
        self.temporal_blocks = nn.ModuleList(
            [TemporalBlock(cfg.d_model, cfg.n_heads) for _ in range(cfg.n_temporal_layers)]
        )
        self.cross_blocks = nn.ModuleList(
            [CrossSectionalBlock(cfg.d_model, cfg.n_heads) for _ in range(cfg.n_cross_layers)]
        )
        self.final_norm = RMSNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, 1, bias=True)

        # Buffer-cached RoPE
        cos, sin = precompute_rope(cfg.seq_len, cfg.d_model // cfg.n_heads)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(
        self,
        x: torch.Tensor,                                    # (S, T, F)  one day's panel
        valid_mask: Optional[torch.Tensor] = None,          # (S,) bool — True = valid symbol
    ) -> torch.Tensor:
        """Forward for ONE day. x has shape (S, T, F). Returns logits of shape (S,)."""
        S, T, _ = x.shape
        # Temporal stage: treat S as the batch.
        h = self.feat_proj(x)                # (S, T, D)
        h = self.in_norm(h)
        cos = self.rope_cos.to(dtype=h.dtype, device=h.device)
        sin = self.rope_sin.to(dtype=h.dtype, device=h.device)
        for blk in self.temporal_blocks:
            if self.cfg.grad_checkpoint and self.training:
                h = torch.utils.checkpoint.checkpoint(
                    blk, h, cos, sin, use_reentrant=False
                )
            else:
                h = blk(h, cos, sin)
        # Pool: take the last time step (no-lookahead — features at index T-1 are
        # the most recent end-of-day-D-1 view).
        sym_emb = h[:, -1, :]               # (S, D)

        # Cross-sectional stage: add a fake batch dim of 1.
        h2 = sym_emb.unsqueeze(0)            # (1, S, D)
        kpm = None
        if valid_mask is not None:
            kpm = (~valid_mask).unsqueeze(0)  # True = invalid (mask out)
        for blk in self.cross_blocks:
            if self.cfg.grad_checkpoint and self.training:
                h2 = torch.utils.checkpoint.checkpoint(
                    blk, h2, kpm, use_reentrant=False
                )
            else:
                h2 = blk(h2, kpm)
        h2 = self.final_norm(h2)             # (1, S, D)
        logits = self.head(h2).squeeze(-1).squeeze(0)  # (S,)
        return logits

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


__all__ = ["CrossAttnTransformerV1", "CrossAttnConfig"]
