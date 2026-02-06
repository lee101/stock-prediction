from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Mapping, Tuple

import torch
from torch import nn
import torch.nn.functional as F

from .config import PolicyConfig


class PositionalEncoding(nn.Module):
    def __init__(self, dim: int, dropout: float = 0.0, max_len: int = 2048) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2) * (-math.log(10000.0) / dim))
        pe = torch.zeros(max_len, dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.size(1)
        x = x + self.pe[:seq_len].unsqueeze(0)
        return self.dropout(x)


class BinancePolicyBase(nn.Module):
    """Shared decode logic for Binance policies."""

    def __init__(
        self,
        *,
        price_offset_pct: float,
        min_price_gap_pct: float,
        trade_amount_scale: float,
        use_midpoint_offsets: bool,
    ) -> None:
        super().__init__()
        self.price_offset_pct = price_offset_pct
        self.min_gap_pct = min_price_gap_pct
        self.trade_amount_scale = trade_amount_scale
        self.use_midpoint_offsets = use_midpoint_offsets

    def decode_actions(
        self,
        outputs: Dict[str, torch.Tensor],
        *,
        reference_close: torch.Tensor,
        chronos_high: torch.Tensor,
        chronos_low: torch.Tensor,
        dynamic_offset_pct: torch.Tensor | Tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> Dict[str, torch.Tensor]:
        if dynamic_offset_pct is None:
            buy_offset_pct = sell_offset_pct = torch.full_like(reference_close, self.price_offset_pct)
        elif isinstance(dynamic_offset_pct, (tuple, list)) and len(dynamic_offset_pct) == 2:
            buy_offset_pct, sell_offset_pct = dynamic_offset_pct
            buy_offset_pct = buy_offset_pct.to(reference_close.device, reference_close.dtype)
            sell_offset_pct = sell_offset_pct.to(reference_close.device, reference_close.dtype)
        else:
            offset_pct = dynamic_offset_pct.to(reference_close.device, reference_close.dtype)  # type: ignore
            buy_offset_pct = sell_offset_pct = offset_pct

        ref = torch.clamp(reference_close, min=1e-8)
        low_anchor = torch.minimum(chronos_low, ref)
        high_anchor = torch.maximum(chronos_high, ref)

        buy_unit = torch.sigmoid(outputs["buy_price_logits"]).squeeze(-1)
        sell_unit = torch.sigmoid(outputs["sell_price_logits"]).squeeze(-1)

        if self.use_midpoint_offsets:
            buy_price = low_anchor + buy_unit * (ref - low_anchor)
            sell_price = ref + sell_unit * (high_anchor - ref)
        else:
            buy_offset = buy_offset_pct * buy_unit
            sell_offset = sell_offset_pct * sell_unit
            buy_price = ref * (1.0 - buy_offset)
            sell_price = ref * (1.0 + sell_offset)

        gap = torch.clamp(ref * self.min_gap_pct, min=1e-8)
        sell_price = torch.maximum(sell_price, buy_price + gap)
        buy_amount = torch.sigmoid(outputs["buy_amount_logits"]).squeeze(-1) * self.trade_amount_scale
        sell_amount = torch.sigmoid(outputs["sell_amount_logits"]).squeeze(-1) * self.trade_amount_scale
        trade_amount = torch.maximum(buy_amount, sell_amount)
        return {
            "buy_price": buy_price,
            "sell_price": sell_price,
            "trade_amount": trade_amount,
            "buy_amount": buy_amount,
            "sell_amount": sell_amount,
        }


class BinanceHourlyPolicy(BinancePolicyBase):
    """Transformer encoder that outputs limit prices and trade intensities."""

    def __init__(self, config: PolicyConfig) -> None:
        super().__init__(
            price_offset_pct=config.price_offset_pct,
            min_price_gap_pct=config.min_price_gap_pct,
            trade_amount_scale=config.trade_amount_scale,
            use_midpoint_offsets=config.use_midpoint_offsets,
        )
        self.embed = nn.Linear(config.input_dim, config.hidden_dim)
        self.pos_encoding = PositionalEncoding(config.hidden_dim, dropout=config.dropout, max_len=config.max_len)
        with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.FLASH_ATTENTION):
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=config.hidden_dim,
                nhead=config.num_heads,
                dim_feedforward=config.hidden_dim * 4,
                dropout=config.dropout,
                batch_first=True,
                activation="gelu",
            )
            self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=config.num_layers)
        self.norm = nn.LayerNorm(config.hidden_dim)
        self.head = nn.Linear(config.hidden_dim, 4)

    def forward(self, features: torch.Tensor) -> Dict[str, torch.Tensor]:
        h = self.embed(features)
        h = self.pos_encoding(h)
        h = self.encoder(h)
        h = self.norm(h)
        logits = self.head(h)
        return {
            "buy_price_logits": logits[..., 0:1],
            "sell_price_logits": logits[..., 1:2],
            "buy_amount_logits": logits[..., 2:3],
            "sell_amount_logits": logits[..., 3:4],
        }

def _rms_norm(x: torch.Tensor, eps: float) -> torch.Tensor:
    if hasattr(F, "rms_norm"):
        return F.rms_norm(x, (x.size(-1),), eps=eps)
    variance = x.pow(2).mean(dim=-1, keepdim=True)
    return x * torch.rsqrt(variance + eps)


def _apply_rotary_emb(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    x_r = x[..., ::2]
    x_i = x[..., 1::2]
    out_r = x_r * cos - x_i * sin
    out_i = x_r * sin + x_i * cos
    out = torch.stack([out_r, out_i], dim=-1)
    return out.flatten(-2)


class RotaryEmbedding(nn.Module):
    def __init__(self, dim: int, max_len: int = 4096, base: float = 10000.0) -> None:
        super().__init__()
        self.dim = dim
        self.max_len = max_len
        self.base = base
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
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


class MultiQueryAttention(nn.Module):
    def __init__(self, config: PolicyConfig, layer_idx: int) -> None:
        super().__init__()
        self.layer_idx = int(layer_idx)
        self.n_head = int(config.num_heads)
        self.head_dim = int(config.hidden_dim // config.num_heads)
        kv_heads = config.num_kv_heads or max(1, self.n_head // 2)
        if self.n_head % kv_heads != 0:
            raise ValueError("num_kv_heads must divide num_heads for multi-query attention.")
        self.n_kv_head = int(kv_heads)
        if config.hidden_dim % self.n_head != 0:
            raise ValueError("hidden_dim must be divisible by num_heads.")

        self.q_proj = nn.Linear(config.hidden_dim, self.n_head * self.head_dim, bias=False)
        self.k_proj = nn.Linear(config.hidden_dim, self.n_kv_head * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.hidden_dim, self.n_kv_head * self.head_dim, bias=False)
        self.out_proj = nn.Linear(config.hidden_dim, config.hidden_dim, bias=False)
        self.dropout = nn.Dropout(config.dropout)
        self.use_qk_norm = config.use_qk_norm
        self.causal = config.use_causal_attention
        self.rms_eps = config.rms_norm_eps
        self.attention_window = config.attention_window if config.attention_window and config.attention_window > 0 else None
        self._window_mask: torch.Tensor | None = None
        self._window_mask_len = 0
        self._window_mask_device: torch.device | None = None
        self.use_value_embedding = bool(config.use_value_embedding) and int(config.value_embedding_every) > 0
        if self.use_value_embedding:
            every = int(config.value_embedding_every)
            self.use_value_embedding = (self.layer_idx % every == 0)
        self.value_embedding_scale = float(config.value_embedding_scale)
        if self.use_value_embedding:
            self.value_embedding = nn.Parameter(torch.zeros(config.max_len, config.hidden_dim))
        else:
            self.register_parameter("value_embedding", None)

    def forward(self, x: torch.Tensor, cos_sin: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        B, T, _ = x.shape
        cos, sin = cos_sin
        q = self.q_proj(x).view(B, T, self.n_head, self.head_dim)
        k = self.k_proj(x).view(B, T, self.n_kv_head, self.head_dim)
        v_input = x
        if self.value_embedding is not None:
            if T > self.value_embedding.size(0):
                raise ValueError(
                    f"Sequence length {T} exceeds value embedding max_len {self.value_embedding.size(0)}."
                )
            v_input = v_input + self.value_embedding[:T].unsqueeze(0) * self.value_embedding_scale
        v = self.v_proj(v_input).view(B, T, self.n_kv_head, self.head_dim)

        q = _apply_rotary_emb(q, cos, sin)
        k = _apply_rotary_emb(k, cos, sin)
        if self.use_qk_norm:
            q = _rms_norm(q, eps=self.rms_eps)
            k = _rms_norm(k, eps=self.rms_eps)

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        if self.n_kv_head != self.n_head:
            n_rep = self.n_head // self.n_kv_head
            k = k.repeat_interleave(n_rep, dim=1)
            v = v.repeat_interleave(n_rep, dim=1)
        attn_mask = self._sliding_window_mask(T, x.device)
        y = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attn_mask,
            is_causal=self.causal if attn_mask is None else False,
            dropout_p=self.dropout.p if self.training else 0.0,
        )
        y = y.transpose(1, 2).contiguous().view(B, T, -1)
        return self.out_proj(y)

    def _sliding_window_mask(self, seq_len: int, device: torch.device) -> torch.Tensor | None:
        if self.attention_window is None:
            return None
        # NOTE: Do not cache this mask on the module during torch.compile() runs.
        # Torch Inductor may wrap forward in CUDAGraphs and reuse output buffers
        # between steps; holding onto an output tensor (e.g. via self._window_mask)
        # can trigger "overwritten by a subsequent run" errors.
        idx = torch.arange(seq_len, device=device)
        dist = idx[:, None] - idx[None, :]
        if self.causal:
            return (dist >= 0) & (dist < self.attention_window)
        return dist.abs() < self.attention_window


class FeedForward(nn.Module):
    def __init__(self, config: PolicyConfig) -> None:
        super().__init__()
        inner_dim = int(config.hidden_dim * config.mlp_ratio)
        self.fc1 = nn.Linear(config.hidden_dim, inner_dim, bias=False)
        self.fc2 = nn.Linear(inner_dim, config.hidden_dim, bias=False)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(x)).square()
        x = self.dropout(x)
        return self.fc2(x)


class NanoTransformerBlock(nn.Module):
    def __init__(self, config: PolicyConfig, layer_idx: int) -> None:
        super().__init__()
        self.attn = MultiQueryAttention(config, layer_idx)
        self.mlp = FeedForward(config)
        self.dropout = nn.Dropout(config.dropout)
        self.rms_eps = config.rms_norm_eps
        self.use_residual_scalars = bool(config.use_residual_scalars)
        if self.use_residual_scalars:
            self.attn_resid_scale = nn.Parameter(torch.tensor(float(config.residual_scale_init)))
            self.attn_skip_scale = nn.Parameter(torch.tensor(float(config.skip_scale_init)))
            self.mlp_resid_scale = nn.Parameter(torch.tensor(float(config.residual_scale_init)))
            self.mlp_skip_scale = nn.Parameter(torch.tensor(float(config.skip_scale_init)))
        else:
            self.register_parameter("attn_resid_scale", None)
            self.register_parameter("attn_skip_scale", None)
            self.register_parameter("mlp_resid_scale", None)
            self.register_parameter("mlp_skip_scale", None)

    def forward(self, x: torch.Tensor, cos_sin: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        x0 = x
        x = x + self.dropout(self.attn(_rms_norm(x, self.rms_eps), cos_sin))
        if self.use_residual_scalars:
            x = self.attn_resid_scale * x + self.attn_skip_scale * x0
        x0 = x
        x = x + self.dropout(self.mlp(_rms_norm(x, self.rms_eps)))
        if self.use_residual_scalars:
            x = self.mlp_resid_scale * x + self.mlp_skip_scale * x0
        return x


class BinanceHourlyPolicyNano(BinancePolicyBase):
    """Nanochat-style transformer policy."""

    def __init__(self, config: PolicyConfig) -> None:
        super().__init__(
            price_offset_pct=config.price_offset_pct,
            min_price_gap_pct=config.min_price_gap_pct,
            trade_amount_scale=config.trade_amount_scale,
            use_midpoint_offsets=config.use_midpoint_offsets,
        )
        self.config = config
        self.embed = nn.Linear(config.input_dim, config.hidden_dim, bias=False)
        self.rope = RotaryEmbedding(
            config.hidden_dim // config.num_heads,
            max_len=config.max_len,
            base=config.rope_base,
        )
        self.blocks = nn.ModuleList(
            [NanoTransformerBlock(config, layer_idx) for layer_idx in range(config.num_layers)]
        )
        self.norm_eps = config.rms_norm_eps
        self.head = nn.Linear(config.hidden_dim, 4, bias=False)
        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                fan_out = module.weight.size(0)
                fan_in = module.weight.size(1)
                std = 1.0 / math.sqrt(fan_in) * min(1.0, math.sqrt(fan_out / fan_in))
                nn.init.normal_(module.weight, mean=0.0, std=std)
        nn.init.zeros_(self.head.weight)
        for block in self.blocks:
            nn.init.zeros_(block.attn.out_proj.weight)
            nn.init.zeros_(block.mlp.fc2.weight)

    def forward(self, features: torch.Tensor) -> Dict[str, torch.Tensor]:
        _, seq_len, _ = features.shape
        h = _rms_norm(self.embed(features), self.norm_eps)
        cos_sin = self.rope(seq_len)
        for block in self.blocks:
            h = block(h, cos_sin)
        h = _rms_norm(h, self.norm_eps)
        logits = self.head(h)
        softcap = float(self.config.logits_softcap)
        if softcap > 0:
            logits = softcap * torch.tanh(logits / softcap)
        return {
            "buy_price_logits": logits[..., 0:1],
            "sell_price_logits": logits[..., 1:2],
            "buy_amount_logits": logits[..., 2:3],
            "sell_amount_logits": logits[..., 3:4],
        }


def build_policy(policy_cfg: PolicyConfig) -> BinancePolicyBase:
    arch = (policy_cfg.model_arch or "classic").lower()
    if arch in {"nano", "nanochat", "modern"}:
        return BinanceHourlyPolicyNano(policy_cfg)
    return BinanceHourlyPolicy(policy_cfg)


def policy_config_from_payload(
    payload: Mapping[str, object],
    *,
    input_dim: int,
    state_dict: Mapping[str, object] | None = None,
) -> PolicyConfig:
    def _maybe(value: object, cast_fn, default):
        if value is None:
            return default
        try:
            return cast_fn(value)
        except (TypeError, ValueError):
            return default

    max_len = _maybe(payload.get("max_len"), int, None)
    if max_len is None and state_dict is not None:
        pe = state_dict.get("pos_encoding.pe")
        if isinstance(pe, torch.Tensor) and pe.ndim >= 2:
            max_len = int(pe.shape[0])
        if max_len is None:
            for key, tensor in state_dict.items():
                if key.endswith("value_embedding") and isinstance(tensor, torch.Tensor) and tensor.ndim == 2:
                    max_len = int(tensor.shape[0])
                    break
    if max_len is None:
        max_len = 2048

    return PolicyConfig(
        input_dim=input_dim,
        hidden_dim=_maybe(payload.get("transformer_dim", payload.get("hidden_dim")), int, 256),
        dropout=_maybe(payload.get("transformer_dropout", payload.get("dropout")), float, 0.1),
        price_offset_pct=_maybe(payload.get("price_offset_pct"), float, 0.0003),
        min_price_gap_pct=_maybe(payload.get("min_price_gap_pct"), float, 0.0003),
        trade_amount_scale=_maybe(payload.get("trade_amount_scale"), float, 100.0),
        num_heads=_maybe(payload.get("transformer_heads", payload.get("num_heads")), int, 8),
        num_layers=_maybe(payload.get("transformer_layers", payload.get("num_layers")), int, 4),
        max_len=max_len,
        use_midpoint_offsets=bool(payload.get("use_midpoint_offsets", True)),
        model_arch=str(payload.get("model_arch", "classic")),
        num_kv_heads=_maybe(payload.get("num_kv_heads"), int, None),
        mlp_ratio=_maybe(payload.get("mlp_ratio"), float, 4.0),
        logits_softcap=_maybe(payload.get("logits_softcap"), float, 12.0),
        rope_base=_maybe(payload.get("rope_base"), float, 10000.0),
        use_qk_norm=bool(payload.get("use_qk_norm", True)),
        use_causal_attention=bool(payload.get("use_causal_attention", True)),
        rms_norm_eps=_maybe(payload.get("rms_norm_eps"), float, 1e-5),
        attention_window=_maybe(payload.get("attention_window"), int, None),
        use_residual_scalars=bool(payload.get("use_residual_scalars", False)),
        residual_scale_init=_maybe(payload.get("residual_scale_init"), float, 1.0),
        skip_scale_init=_maybe(payload.get("skip_scale_init"), float, 0.0),
        use_value_embedding=bool(payload.get("use_value_embedding", False)),
        value_embedding_every=_maybe(payload.get("value_embedding_every"), int, 2),
        value_embedding_scale=_maybe(payload.get("value_embedding_scale"), float, 1.0),
    )


def align_state_dict_input_dim(
    state_dict: Mapping[str, object],
    *,
    input_dim: int,
) -> Dict[str, torch.Tensor]:
    """Pad/trim embedding weights so checkpoints remain loadable after feature changes."""
    if "embed.weight" not in state_dict:
        return dict(state_dict)
    weight = state_dict.get("embed.weight")
    if not isinstance(weight, torch.Tensor) or weight.ndim != 2:
        return dict(state_dict)
    if weight.shape[1] == int(input_dim):
        return dict(state_dict)

    target = int(input_dim)
    if weight.shape[1] < target:
        pad = torch.zeros(weight.shape[0], target - weight.shape[1], dtype=weight.dtype)
        updated = torch.cat([weight, pad], dim=1)
    else:
        updated = weight[:, :target]

    patched = dict(state_dict)
    patched["embed.weight"] = updated
    return patched


__all__ = [
    "BinanceHourlyPolicy",
    "BinanceHourlyPolicyNano",
    "BinancePolicyBase",
    "PolicyConfig",
    "build_policy",
    "align_state_dict_input_dim",
    "policy_config_from_payload",
]
