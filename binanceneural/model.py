from __future__ import annotations

import logging
import math
from contextlib import ExitStack, contextmanager
from typing import Dict, Mapping, Tuple

import torch
from torch import nn
import torch.nn.functional as F

from .config import PolicyConfig
from .kernels.attention import HAS_TRITON, multi_query_attention as _triton_mqa

try:
    from torch.nn.attention.flex_attention import flex_attention, create_block_mask
    HAS_FLEX_ATTENTION = True
except ImportError:
    HAS_FLEX_ATTENTION = False

logger = logging.getLogger(__name__)

_flex_block_mask_cache: dict[tuple[int, str], object] = {}


def _get_flex_causal_mask(seq_len: int, device: torch.device) -> object:
    key = (seq_len, str(device))
    if key not in _flex_block_mask_cache:
        def causal_mask(b, h, q_idx, kv_idx):
            return q_idx >= kv_idx
        _flex_block_mask_cache[key] = create_block_mask(
            causal_mask, B=None, H=None, Q_LEN=seq_len, KV_LEN=seq_len, device=device,
        )
    return _flex_block_mask_cache[key]


def _is_attention_backend_error(exc: BaseException) -> bool:
    message = str(exc).lower()
    indicators = (
        "no available kernel",
        "not implemented",
        "not available",
        "flash attention",
        "scaled_dot_product_attention",
        "sdpa",
    )
    return any(indicator in message for indicator in indicators)


@contextmanager
def _mha_fastpath_context(enabled: bool):
    mha_backend = getattr(torch.backends, "mha", None)
    setter = getattr(mha_backend, "set_fastpath_enabled", None) if mha_backend is not None else None
    getter = getattr(mha_backend, "get_fastpath_enabled", None) if mha_backend is not None else None
    if not callable(setter):
        yield
        return

    previous = getter() if callable(getter) else None
    setter(bool(enabled))
    try:
        yield
    finally:
        if previous is not None:
            setter(bool(previous))


def _scaled_dot_product_attention_reference(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    attn_mask: torch.Tensor | None = None,
    dropout_p: float = 0.0,
    is_causal: bool = False,
) -> torch.Tensor:
    """Reference attention implementation used when SDPA kernels are unavailable."""

    d_k = q.size(-1)
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)

    if attn_mask is not None:
        if attn_mask.dtype == torch.bool:
            scores = scores.masked_fill(~attn_mask, float("-inf"))
        else:
            scores = scores + attn_mask

    if is_causal:
        causal_mask = torch.triu(
            torch.ones(
                scores.size(-2),
                scores.size(-1),
                dtype=torch.bool,
                device=scores.device,
            ),
            diagonal=1,
        )
        scores = scores.masked_fill(causal_mask, float("-inf"))

    attn = torch.softmax(scores, dim=-1)
    if dropout_p > 0.0:
        attn = F.dropout(attn, p=dropout_p, training=torch.is_grad_enabled())
    return torch.matmul(attn, v)


def _scaled_dot_product_attention_with_fallback(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    attn_mask: torch.Tensor | None = None,
    dropout_p: float = 0.0,
    is_causal: bool = False,
) -> torch.Tensor:
    """Call torch SDPA when available, otherwise fall back to the reference path."""

    native_fn = getattr(F, "scaled_dot_product_attention", None)
    if native_fn is not None:
        try:
            return native_fn(
                q,
                k,
                v,
                attn_mask=attn_mask,
                dropout_p=dropout_p,
                is_causal=is_causal,
            )
        except (RuntimeError, NotImplementedError) as exc:
            message = str(exc).lower()
            fallback_indicators = (
                "not implemented",
                "not available",
                "no available kernel",
                "only available",
                "does not support",
            )
            if not any(indicator in message for indicator in fallback_indicators):
                raise

    return _scaled_dot_product_attention_reference(
        q,
        k,
        v,
        attn_mask=attn_mask,
        dropout_p=dropout_p,
        is_causal=is_causal,
    )


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
        max_hold_hours: float = 24.0,
    ) -> None:
        super().__init__()
        self.price_offset_pct = price_offset_pct
        self.min_gap_pct = min_price_gap_pct
        self.trade_amount_scale = trade_amount_scale
        self.use_midpoint_offsets = use_midpoint_offsets
        self.max_hold_hours = max_hold_hours

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
        result = {
            "buy_price": buy_price,
            "sell_price": sell_price,
            "trade_amount": trade_amount,
            "buy_amount": buy_amount,
            "sell_amount": sell_amount,
        }
        if "hold_hours_logits" in outputs:
            result["hold_hours"] = torch.sigmoid(outputs["hold_hours_logits"]).squeeze(-1) * self.max_hold_hours
        if "allocation_logits" in outputs:
            result["allocation_fraction"] = torch.sigmoid(outputs["allocation_logits"]).squeeze(-1)
        return result


class BinanceHourlyPolicy(BinancePolicyBase):
    """Transformer encoder that outputs limit prices and trade intensities."""

    def __init__(self, config: PolicyConfig) -> None:
        super().__init__(
            price_offset_pct=config.price_offset_pct,
            min_price_gap_pct=config.min_price_gap_pct,
            trade_amount_scale=config.trade_amount_scale,
            use_midpoint_offsets=config.use_midpoint_offsets,
            max_hold_hours=config.max_hold_hours,
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
            self.encoder = nn.TransformerEncoder(
                encoder_layer, num_layers=config.num_layers, enable_nested_tensor=False,
            )
        self._attention_backend_fallback = False
        self.norm = nn.LayerNorm(config.hidden_dim)
        self.head = nn.Linear(config.hidden_dim, config.num_outputs)

    def _encode_once(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.encoder(hidden)

    def _encode_math_fallback(self, hidden: torch.Tensor) -> torch.Tensor:
        with ExitStack() as stack:
            stack.enter_context(_mha_fastpath_context(enabled=False))
            sdpa_kernel = getattr(torch.nn.attention, "sdpa_kernel", None)
            sdp_backend = getattr(torch.nn.attention, "SDPBackend", None)
            if callable(sdpa_kernel) and sdp_backend is not None and hasattr(sdp_backend, "MATH"):
                stack.enter_context(sdpa_kernel(sdp_backend.MATH))
            return self.encoder(hidden)

    def _encode_with_backend_fallback(self, hidden: torch.Tensor) -> torch.Tensor:
        if self._attention_backend_fallback:
            return self._encode_math_fallback(hidden)

        try:
            return self._encode_once(hidden)
        except (RuntimeError, NotImplementedError) as exc:
            if not _is_attention_backend_error(exc):
                raise
            self._attention_backend_fallback = True
            logger.warning(
                "Classic transformer attention backend unavailable; disabling fastpath and "
                "falling back to math SDPA for stability: %s",
                exc,
            )
            return self._encode_math_fallback(hidden)

    def forward(self, features: torch.Tensor) -> Dict[str, torch.Tensor]:
        h = self.embed(features)
        h = self.pos_encoding(h)
        h = self._encode_with_backend_fallback(h)
        h = self.norm(h)
        logits = self.head(h)
        out = {
            "buy_price_logits": logits[..., 0:1],
            "sell_price_logits": logits[..., 1:2],
            "buy_amount_logits": logits[..., 2:3],
            "sell_amount_logits": logits[..., 3:4],
        }
        if logits.shape[-1] > 4:
            out["hold_hours_logits"] = logits[..., 4:5]
        if logits.shape[-1] > 5:
            out["allocation_logits"] = logits[..., 5:6]
        return out

try:
    from binanceneural.kernels.rope import apply_rope as _triton_apply_rope
    from binanceneural.kernels.norm import rms_norm as _triton_rms_norm, fused_rms_norm_qkv as _triton_fused_rms_norm_qkv
    _HAS_TRITON_KERNELS = True
except Exception:
    _HAS_TRITON_KERNELS = False


def _rms_norm(x: torch.Tensor, eps: float) -> torch.Tensor:
    """Unweighted T5-style RMS norm. Uses Triton kernel when available on CUDA."""
    if _HAS_TRITON_KERNELS and x.is_cuda:
        return _triton_rms_norm(x, weight=None, eps=eps)
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
        self.use_value_embedding = bool(config.use_value_embedding) and int(config.value_embedding_every) > 0
        if self.use_value_embedding:
            every = int(config.value_embedding_every)
            self.use_value_embedding = (self.layer_idx % every == 0)
        self.value_embedding_scale = float(config.value_embedding_scale)
        if self.use_value_embedding:
            ve_len = config.max_len + (int(config.num_memory_tokens) if config.num_memory_tokens else 0)
            self.value_embedding = nn.Parameter(torch.zeros(ve_len, config.hidden_dim))
        else:
            self.register_parameter("value_embedding", None)
        # Dilated attention: per-head-group strides for multi-scale temporal coverage
        self.dilated_strides: list[int] = []
        if config.dilated_strides:
            self.dilated_strides = [int(s) for s in config.dilated_strides.split(",")]
        # Number of memory tokens (needed for mask construction)
        self.num_memory_tokens = int(config.num_memory_tokens) if config.num_memory_tokens else 0
        self.use_flex_attention = bool(config.use_flex_attention) and HAS_FLEX_ATTENTION

    def forward(
        self,
        x: torch.Tensor,
        cos_sin: Tuple[torch.Tensor, torch.Tensor],
        x_pre_norm: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass for multi-query attention.

        Args:
            x: Input tensor (B, T, hidden_dim). When x_pre_norm is None this is
               the already-normalized residual (legacy path). When x_pre_norm is
               provided this is the raw residual used only for the value branch;
               the fused norm+QKV kernel normalizes x_pre_norm.
            cos_sin: Rotary embeddings (cos, sin).
            x_pre_norm: Optional raw (un-normalized) input (B, T, hidden_dim).
               When provided the fused_rms_norm_qkv kernel is used to compute
               norm(x_pre_norm) -> Q, K, V in a single kernel launch instead of
               three separate ops.
        """
        B, T, _ = x.shape
        cos, sin = cos_sin

        if x_pre_norm is not None and _HAS_TRITON_KERNELS and x_pre_norm.is_cuda:
            # Fused path: one kernel call normalizes x_pre_norm and computes
            # Q, K, V projections simultaneously (saving two redundant norm passes).
            q_flat, k_flat, v_flat = _triton_fused_rms_norm_qkv(
                x_pre_norm,
                None,  # unweighted RMS norm — no learnable scale (matches _rms_norm)
                self.q_proj.weight,
                self.k_proj.weight,
                self.v_proj.weight,
                eps=self.rms_eps,
            )
            q = q_flat.view(B, T, self.n_head, self.head_dim)
            k = k_flat.view(B, T, self.n_kv_head, self.head_dim)
            if self.value_embedding is not None:
                # value_embedding is added to normalized x before V projection.
                # x (first argument) is already normalized in the fused path, so
                # use it to apply the additive embedding then re-project.
                if T > self.value_embedding.size(0):
                    raise ValueError(
                        f"Sequence length {T} exceeds value embedding max_len {self.value_embedding.size(0)}."
                    )
                v_input = x + self.value_embedding[:T].unsqueeze(0) * self.value_embedding_scale
                v = self.v_proj(v_input).view(B, T, self.n_kv_head, self.head_dim)
            else:
                v = v_flat.view(B, T, self.n_kv_head, self.head_dim)
        else:
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

        # Apply RoPE to Q and K. Use fused Triton kernel (both Q+K in one launch)
        # when available; fall back to sequential PyTorch otherwise.
        if _HAS_TRITON_KERNELS and q.is_cuda:
            q, k = _triton_apply_rope(q, k, cos, sin)
        else:
            q = _apply_rotary_emb(q, cos, sin)
            k = _apply_rotary_emb(k, cos, sin)
        if self.use_qk_norm:
            q = _rms_norm(q, eps=self.rms_eps)
            k = _rms_norm(k, eps=self.rms_eps)

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        attn_mask = self._build_attention_mask(T, x.device)
        dropout_p = self.dropout.p if self.training else 0.0

        # Use the Triton fused MQA kernel when:
        #   1. Triton is available
        #   2. Not training (kernel has no backward pass)
        #   3. Inputs are on CUDA
        #   4. No dropout (kernel doesn't support dropout)
        #   5. head_dim is in supported set {16, 32, 64, 128}
        use_triton = (
            HAS_TRITON
            and not self.training
            and q.is_cuda
            and dropout_p == 0.0
            and self.head_dim in (16, 32, 64, 128)
        )

        use_flex = (
            self.use_flex_attention
            and q.is_cuda
            and attn_mask is None
            and self.causal
            and dropout_p == 0.0
        )

        if use_triton:
            triton_mask: torch.Tensor | None = None
            if attn_mask is not None:
                if attn_mask.dtype == torch.bool:
                    triton_mask = torch.zeros(
                        attn_mask.shape, dtype=q.dtype, device=q.device
                    ).masked_fill_(~attn_mask, float("-inf"))
                else:
                    triton_mask = attn_mask.to(q.dtype)
                while triton_mask.dim() < 4:
                    triton_mask = triton_mask.unsqueeze(0)
            y = _triton_mqa(
                q,
                k,
                v,
                causal=self.causal if attn_mask is None else False,
                mask=triton_mask,
            )
        elif use_flex:
            block_mask = _get_flex_causal_mask(T, q.device)
            y = flex_attention(q, k, v, block_mask=block_mask, enable_gqa=True)
        else:
            if self.n_kv_head != self.n_head:
                n_rep = self.n_head // self.n_kv_head
                k = k.repeat_interleave(n_rep, dim=1)
                v = v.repeat_interleave(n_rep, dim=1)
            y = _scaled_dot_product_attention_with_fallback(
                q,
                k,
                v,
                attn_mask=attn_mask,
                is_causal=self.causal if attn_mask is None else False,
                dropout_p=dropout_p,
            )
        y = y.transpose(1, 2).contiguous().view(B, T, -1)
        return self.out_proj(y)

    def _build_attention_mask(self, seq_len: int, device: torch.device) -> torch.Tensor | None:
        """Build attention mask combining sliding window, dilated strides, and memory tokens."""
        has_dilated = len(self.dilated_strides) > 0
        has_window = self.attention_window is not None
        has_memory = self.num_memory_tokens > 0

        if not has_dilated and not has_window and not has_memory:
            return None

        # For dilated attention: build per-head mask with different strides
        if has_dilated:
            return self._dilated_mask(seq_len, device)

        # For sliding window (possibly with memory tokens)
        return self._sliding_window_mask(seq_len, device)

    def _sliding_window_mask(self, seq_len: int, device: torch.device) -> torch.Tensor | None:
        if self.attention_window is None and self.num_memory_tokens == 0:
            return None
        # NOTE: Do not cache this mask on the module during torch.compile() runs.
        # Torch Inductor may wrap forward in CUDAGraphs and reuse output buffers
        # between steps; holding onto an output tensor (e.g. via self._window_mask)
        # can trigger "overwritten by a subsequent run" errors.
        idx = torch.arange(seq_len, device=device)
        M = self.num_memory_tokens

        if self.attention_window is not None:
            dist = idx[:, None] - idx[None, :]
            if self.causal:
                mask = (dist >= 0) & (dist < self.attention_window)
            else:
                mask = dist.abs() < self.attention_window
        else:
            # No window but has memory tokens - start with full causal or full mask
            if self.causal:
                mask = idx[:, None] >= idx[None, :]
            else:
                mask = torch.ones(seq_len, seq_len, dtype=torch.bool, device=device)

        # Memory tokens (first M positions) can attend to / be attended by everything
        if M > 0:
            mask[:M, :] = True  # memory tokens attend to all
            mask[:, :M] = True  # all tokens attend to memory tokens

        return mask

    def _dilated_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """Build per-head dilated attention mask.

        Each head group gets a different stride. Head group i attends to
        positions where (position_distance % stride_i == 0). This gives
        multi-scale temporal coverage: stride=1 for local, stride=24 for
        daily patterns, etc.

        Returns mask of shape (num_heads, seq_len, seq_len).
        """
        strides = self.dilated_strides
        M = self.num_memory_tokens
        idx = torch.arange(seq_len, device=device)
        dist = idx[:, None] - idx[None, :]

        # Divide heads into groups, one per stride
        n_groups = len(strides)
        heads_per_group = self.n_head // n_groups
        remainder = self.n_head % n_groups

        mask = torch.zeros(self.n_head, seq_len, seq_len, dtype=torch.bool, device=device)

        head_offset = 0
        for g, stride in enumerate(strides):
            n_h = heads_per_group + (1 if g < remainder else 0)
            # This group attends to positions at multiples of stride
            if self.causal:
                group_mask = (dist >= 0) & (dist % stride == 0)
            else:
                group_mask = dist.abs() % stride == 0

            # Apply sliding window on top if configured
            if self.attention_window is not None:
                effective_window = self.attention_window * stride
                if self.causal:
                    group_mask = group_mask & (dist < effective_window)
                else:
                    group_mask = group_mask & (dist.abs() < effective_window)

            mask[head_offset:head_offset + n_h] = group_mask.unsqueeze(0)
            head_offset += n_h

        # Memory tokens always fully connected
        if M > 0:
            mask[:, :M, :] = True
            mask[:, :, :M] = True

        return mask


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
        # x_normed is always computed: it feeds attn() on the CPU/no-Triton path,
        # and the fused Triton path re-derives it internally from x_pre_norm=x0
        # to avoid a separate norm kernel launch for Q/K/V projections.
        x_normed = _rms_norm(x, self.rms_eps)
        if _HAS_TRITON_KERNELS and x.is_cuda:
            x = x + self.dropout(self.attn(x_normed, cos_sin, x_pre_norm=x0))
        else:
            x = x + self.dropout(self.attn(x_normed, cos_sin))
        if self.use_residual_scalars:
            x = self.attn_resid_scale * x + self.attn_skip_scale * x0
        x0 = x
        x = x + self.dropout(self.mlp(_rms_norm(x, self.rms_eps)))
        if self.use_residual_scalars:
            x = self.mlp_resid_scale * x + self.mlp_skip_scale * x0
        return x


class BinanceHourlyPolicyNano(BinancePolicyBase):
    """Nanochat-style transformer policy with optional memory tokens and dilated attention."""

    def __init__(self, config: PolicyConfig) -> None:
        super().__init__(
            price_offset_pct=config.price_offset_pct,
            min_price_gap_pct=config.min_price_gap_pct,
            trade_amount_scale=config.trade_amount_scale,
            use_midpoint_offsets=config.use_midpoint_offsets,
            max_hold_hours=config.max_hold_hours,
        )
        self.config = config
        self.embed = nn.Linear(config.input_dim, config.hidden_dim, bias=False)
        self.num_memory_tokens = int(config.num_memory_tokens) if config.num_memory_tokens else 0
        # Learnable memory tokens that participate in attention for global context
        if self.num_memory_tokens > 0:
            self.memory_tokens = nn.Parameter(
                torch.randn(1, self.num_memory_tokens, config.hidden_dim) * 0.02
            )
        else:
            self.register_parameter("memory_tokens", None)
        # RoPE max_len accounts for memory tokens prepended to sequence
        rope_max = config.max_len + self.num_memory_tokens
        self.rope = RotaryEmbedding(
            config.hidden_dim // config.num_heads,
            max_len=rope_max,
            base=config.rope_base,
        )
        self.blocks = nn.ModuleList(
            [NanoTransformerBlock(config, layer_idx) for layer_idx in range(config.num_layers)]
        )
        self.norm_eps = config.rms_norm_eps
        self.head = nn.Linear(config.hidden_dim, config.num_outputs, bias=False)
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
        B, seq_len, _ = features.shape
        h = _rms_norm(self.embed(features), self.norm_eps)

        # Prepend memory tokens if configured
        M = self.num_memory_tokens
        if M > 0 and self.memory_tokens is not None:
            mem = self.memory_tokens.expand(B, -1, -1)
            h = torch.cat([mem, h], dim=1)  # (B, M+seq_len, hidden_dim)

        total_len = h.size(1)
        cos_sin = self.rope(total_len)
        for block in self.blocks:
            h = block(h, cos_sin)
        h = _rms_norm(h, self.norm_eps)

        # Strip memory tokens - only decode from real sequence positions
        if M > 0:
            h = h[:, M:, :]

        logits = self.head(h)
        softcap = float(self.config.logits_softcap)
        if softcap > 0:
            logits = softcap * torch.tanh(logits / softcap)
        out = {
            "buy_price_logits": logits[..., 0:1],
            "sell_price_logits": logits[..., 1:2],
            "buy_amount_logits": logits[..., 2:3],
            "sell_amount_logits": logits[..., 3:4],
        }
        if logits.shape[-1] > 4:
            out["hold_hours_logits"] = logits[..., 4:5]
        if logits.shape[-1] > 5:
            out["allocation_logits"] = logits[..., 5:6]
        return out


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
    if max_len is None:
        seq_len = _maybe(payload.get("sequence_length"), int, None)
        if seq_len is not None:
            max_len = max(seq_len, 32)
    if max_len is None and state_dict is not None:
        pe = state_dict.get("pos_encoding.pe")
        if isinstance(pe, torch.Tensor) and pe.ndim >= 2:
            max_len = int(pe.shape[0])
        if max_len is None:
            n_mem = _maybe(payload.get("num_memory_tokens"), int, 0)
            for key, tensor in state_dict.items():
                if key.endswith("value_embedding") and isinstance(tensor, torch.Tensor) and tensor.ndim == 2:
                    max_len = int(tensor.shape[0]) - n_mem
                    break
    if max_len is None:
        max_len = 2048

    if state_dict is not None:
        embed_w = state_dict.get("embed.weight")
        if isinstance(embed_w, torch.Tensor) and embed_w.ndim == 2:
            input_dim = int(embed_w.shape[1])

    return PolicyConfig(
        input_dim=input_dim,
        hidden_dim=_maybe(payload.get("transformer_dim") or payload.get("hidden_dim"), int, 256),
        dropout=_maybe(payload.get("transformer_dropout") or payload.get("dropout"), float, 0.1),
        price_offset_pct=_maybe(payload.get("price_offset_pct"), float, 0.0003),
        min_price_gap_pct=_maybe(payload.get("min_price_gap_pct"), float, 0.0003),
        trade_amount_scale=_maybe(payload.get("trade_amount_scale"), float, 100.0),
        num_heads=_maybe(payload.get("transformer_heads") or payload.get("num_heads"), int, 8),
        num_layers=_maybe(payload.get("transformer_layers") or payload.get("num_layers"), int, 4),
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
        num_memory_tokens=_maybe(payload.get("num_memory_tokens"), int, 0),
        dilated_strides=str(payload.get("dilated_strides", "")),
        use_flex_attention=bool(payload.get("use_flex_attention", True)),
        num_outputs=_maybe(payload.get("num_outputs"), int, 4),
        max_hold_hours=_maybe(payload.get("max_hold_hours"), float, 24.0),
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
        pad = torch.zeros(weight.shape[0], target - weight.shape[1], dtype=weight.dtype, device=weight.device)
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
    "HAS_FLEX_ATTENTION",
    "PolicyConfig",
    "PositionalEncoding",
    "align_state_dict_input_dim",
    "build_policy",
    "policy_config_from_payload",
]
