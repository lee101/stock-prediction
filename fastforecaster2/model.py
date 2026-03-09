from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    """RMSNorm with a learned per-channel scale."""

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.rsqrt(torch.mean(x * x, dim=-1, keepdim=True) + self.eps)
        return x * rms * self.weight


class FastForecastBlock(nn.Module):
    """Hybrid local-conv + causal-attention + ReLU^2 feed-forward block."""

    def __init__(
        self,
        *,
        dim: int,
        num_heads: int,
        ff_multiplier: int,
        dropout: float,
        qk_norm: bool,
        qk_norm_eps: float,
    ) -> None:
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError("dim must be divisible by num_heads")

        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.qk_norm = qk_norm
        self.qk_norm_eps = qk_norm_eps

        self.norm_conv = RMSNorm(dim)
        self.depthwise = nn.Conv1d(dim, dim, kernel_size=3, padding=1, groups=dim, bias=False)
        self.conv_proj = nn.Linear(dim, dim, bias=False)

        self.norm_attn = RMSNorm(dim)
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.attn_out = nn.Linear(dim, dim, bias=False)
        self.q_scale = nn.Parameter(torch.ones(num_heads))
        self.k_scale = nn.Parameter(torch.ones(num_heads))

        ff_dim = dim * ff_multiplier
        self.norm_ffn = RMSNorm(dim)
        self.ff_in = nn.Linear(dim, ff_dim * 2, bias=False)
        self.ff_out = nn.Linear(ff_dim, dim, bias=False)

        self.dropout = nn.Dropout(dropout)

    def _causal_attention(self, x: torch.Tensor) -> torch.Tensor:
        bsz, seqlen, _ = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)

        q = q.view(bsz, seqlen, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(bsz, seqlen, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(bsz, seqlen, self.num_heads, self.head_dim).transpose(1, 2)
        if self.qk_norm:
            q = q * torch.rsqrt(torch.mean(q * q, dim=-1, keepdim=True) + self.qk_norm_eps)
            k = k * torch.rsqrt(torch.mean(k * k, dim=-1, keepdim=True) + self.qk_norm_eps)
            q = q * self.q_scale.view(1, self.num_heads, 1, 1)
            k = k * self.k_scale.view(1, self.num_heads, 1, 1)

        # Keep attention dropout at 0 here for maximum kernel compatibility.
        # Regularization is still applied via residual-path dropout.
        try:
            attn = F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=None,
                dropout_p=0.0,
                is_causal=True,
            )
        except RuntimeError:
            # Some global settings disable the math SDP kernel; force-enable it as a fallback.
            if q.is_cuda and hasattr(torch.backends.cuda, "sdp_kernel"):
                with torch.backends.cuda.sdp_kernel(enable_flash=False, enable_mem_efficient=True, enable_math=True):
                    attn = F.scaled_dot_product_attention(
                        q,
                        k,
                        v,
                        attn_mask=None,
                        dropout_p=0.0,
                        is_causal=True,
                    )
            else:
                raise
        attn = attn.transpose(1, 2).contiguous().view(bsz, seqlen, self.dim)
        return self.attn_out(attn)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        conv_input = self.norm_conv(x).transpose(1, 2)
        conv_out = self.depthwise(conv_input).transpose(1, 2)
        x = x + self.dropout(self.conv_proj(conv_out))

        attn_out = self._causal_attention(self.norm_attn(x))
        x = x + self.dropout(attn_out)

        ffn_input = self.norm_ffn(x)
        gate, value = self.ff_in(ffn_input).chunk(2, dim=-1)
        # ReLU^2 is one of the robust activation choices in recent speedrun-derived models.
        ffn_hidden = torch.relu(gate).pow(2) * value
        ffn_out = self.ff_out(self.dropout(ffn_hidden))
        x = x + self.dropout(ffn_out)
        return x


class FastForecaster2Model(nn.Module):
    """High-throughput forecasting backbone optimized for low MAE objectives."""

    def __init__(
        self,
        *,
        input_dim: int,
        horizon: int,
        hidden_dim: int,
        num_layers: int,
        num_heads: int,
        ff_multiplier: int,
        dropout: float,
        max_symbols: int,
        qk_norm: bool,
        qk_norm_eps: float,
    ) -> None:
        super().__init__()
        if input_dim <= 0:
            raise ValueError("input_dim must be > 0")
        if horizon <= 0:
            raise ValueError("horizon must be > 0")
        if num_layers <= 0:
            raise ValueError("num_layers must be > 0")

        self.horizon = horizon
        self.input_proj = nn.Linear(input_dim, hidden_dim, bias=False)
        self.symbol_embedding = nn.Embedding(max(1, max_symbols), hidden_dim)
        self.blocks = nn.ModuleList(
            [
                FastForecastBlock(
                    dim=hidden_dim,
                    num_heads=num_heads,
                    ff_multiplier=ff_multiplier,
                    dropout=dropout,
                    qk_norm=qk_norm,
                    qk_norm_eps=qk_norm_eps,
                )
                for _ in range(num_layers)
            ]
        )
        self.final_norm = RMSNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim, bias=False),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, horizon, bias=True),
        )

        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Conv1d):
                nn.init.kaiming_uniform_(module.weight, a=math.sqrt(5))
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)

        # Start from near-identity return prediction (zero return), stabilizing early epochs.
        last_linear = self.head[-1]
        assert isinstance(last_linear, nn.Linear)
        nn.init.zeros_(last_linear.weight)
        nn.init.zeros_(last_linear.bias)

    def forward(self, x: torch.Tensor, symbol_idx: torch.Tensor | None = None) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError("Input must be [batch, sequence, features].")

        hidden = self.input_proj(x)
        if symbol_idx is not None:
            symbol_emb = self.symbol_embedding(symbol_idx)
            hidden = hidden + symbol_emb.unsqueeze(1)

        hidden = self.dropout(hidden)
        for block in self.blocks:
            hidden = block(hidden)

        hidden = self.final_norm(hidden)
        last_state = hidden[:, -1, :]
        return self.head(last_state)
