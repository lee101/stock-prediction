from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np

from .config import PolicyConfig


Array = jax.Array


def _strip_orig_mod_prefix(state_dict: Mapping[str, Any]) -> dict[str, Any]:
    if any(str(key).startswith("_orig_mod.") for key in state_dict):
        return {str(key).replace("_orig_mod.", "", 1): value for key, value in state_dict.items()}
    return {str(key): value for key, value in state_dict.items()}


def sinusoidal_positional_encoding(seq_len: int, dim: int, *, dtype: Any = jnp.float32) -> Array:
    position = jnp.arange(seq_len, dtype=dtype)[:, None]
    div_term = jnp.exp(jnp.arange(0, dim, 2, dtype=dtype) * (-jnp.log(10000.0) / dim))
    pe = jnp.zeros((seq_len, dim), dtype=dtype)
    pe = pe.at[:, 0::2].set(jnp.sin(position * div_term))
    pe = pe.at[:, 1::2].set(jnp.cos(position * div_term))
    return pe


class ClassicSelfAttention(nn.Module):
    hidden_dim: int
    num_heads: int
    dropout: float = 0.0

    @nn.compact
    def __call__(self, x: Array, *, deterministic: bool) -> Array:
        if self.hidden_dim % self.num_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_heads")
        head_dim = self.hidden_dim // self.num_heads

        q = nn.Dense(self.hidden_dim, name="q_proj")(x)
        k = nn.Dense(self.hidden_dim, name="k_proj")(x)
        v = nn.Dense(self.hidden_dim, name="v_proj")(x)

        def _reshape_heads(tensor: Array) -> Array:
            batch, steps, _ = tensor.shape
            tensor = tensor.reshape(batch, steps, self.num_heads, head_dim)
            return jnp.transpose(tensor, (0, 2, 1, 3))

        q = _reshape_heads(q)
        k = _reshape_heads(k)
        v = _reshape_heads(v)

        scale = 1.0 / jnp.sqrt(jnp.asarray(head_dim, dtype=x.dtype))
        scores = jnp.einsum("bhtd,bhsd->bhts", q, k) * scale
        attn = jax.nn.softmax(scores, axis=-1)
        attn = nn.Dropout(rate=self.dropout, name="attn_dropout")(attn, deterministic=deterministic)
        context = jnp.einsum("bhts,bhsd->bhtd", attn, v)
        context = jnp.transpose(context, (0, 2, 1, 3)).reshape(x.shape)
        return nn.Dense(self.hidden_dim, name="out_proj")(context)


class ClassicTransformerLayer(nn.Module):
    hidden_dim: int
    num_heads: int
    dropout: float = 0.0

    @nn.compact
    def __call__(self, x: Array, *, deterministic: bool) -> Array:
        attn_out = ClassicSelfAttention(
            hidden_dim=self.hidden_dim,
            num_heads=self.num_heads,
            dropout=self.dropout,
            name="self_attn",
        )(x, deterministic=deterministic)
        attn_out = nn.Dropout(rate=self.dropout, name="dropout1")(attn_out, deterministic=deterministic)
        x = nn.LayerNorm(name="norm1")(x + attn_out)

        ff = nn.Dense(self.hidden_dim * 4, name="linear1")(x)
        ff = jax.nn.gelu(ff, approximate=False)
        ff = nn.Dropout(rate=self.dropout, name="dropout_ff")(ff, deterministic=deterministic)
        ff = nn.Dense(self.hidden_dim, name="linear2")(ff)
        ff = nn.Dropout(rate=self.dropout, name="dropout2")(ff, deterministic=deterministic)
        return nn.LayerNorm(name="norm2")(x + ff)


class JaxClassicPolicy(nn.Module):
    config: PolicyConfig

    @nn.compact
    def __call__(self, features: Array, *, deterministic: bool = True) -> dict[str, Array]:
        hidden = nn.Dense(self.config.hidden_dim, name="embed")(features)
        seq_len = hidden.shape[1]
        pe = sinusoidal_positional_encoding(seq_len, self.config.hidden_dim, dtype=hidden.dtype)
        hidden = hidden + pe[None, :, :]
        hidden = nn.Dropout(rate=self.config.dropout, name="pos_dropout")(hidden, deterministic=deterministic)

        for idx in range(self.config.num_layers):
            hidden = ClassicTransformerLayer(
                hidden_dim=self.config.hidden_dim,
                num_heads=self.config.num_heads,
                dropout=self.config.dropout,
                name=f"layers_{idx}",
            )(hidden, deterministic=deterministic)

        hidden = nn.LayerNorm(name="norm")(hidden)
        logits = nn.Dense(self.config.num_outputs, name="head")(hidden)
        outputs = {
            "buy_price_logits": logits[..., 0:1],
            "sell_price_logits": logits[..., 1:2],
            "buy_amount_logits": logits[..., 2:3],
            "sell_amount_logits": logits[..., 3:4],
        }
        if logits.shape[-1] > 4:
            outputs["hold_hours_logits"] = logits[..., 4:5]
        if logits.shape[-1] > 5:
            outputs["allocation_logits"] = logits[..., 5:6]
        return outputs


def decode_actions_jax(
    outputs: Mapping[str, Array],
    *,
    reference_close: Array,
    chronos_high: Array,
    chronos_low: Array,
    price_offset_pct: float,
    min_price_gap_pct: float,
    trade_amount_scale: float,
    use_midpoint_offsets: bool,
    max_hold_hours: float = 24.0,
) -> dict[str, Array]:
    reference_close = jnp.asarray(reference_close)
    chronos_high = jnp.asarray(chronos_high)
    chronos_low = jnp.asarray(chronos_low)

    ref = jnp.clip(reference_close, min=1e-8)
    low_anchor = jnp.minimum(chronos_low, ref)
    high_anchor = jnp.maximum(chronos_high, ref)

    buy_unit = jax.nn.sigmoid(jnp.asarray(outputs["buy_price_logits"])).squeeze(-1)
    sell_unit = jax.nn.sigmoid(jnp.asarray(outputs["sell_price_logits"])).squeeze(-1)

    if use_midpoint_offsets:
        buy_price = low_anchor + buy_unit * (ref - low_anchor)
        sell_price = ref + sell_unit * (high_anchor - ref)
    else:
        buy_price = ref * (1.0 - price_offset_pct * buy_unit)
        sell_price = ref * (1.0 + price_offset_pct * sell_unit)

    gap = jnp.clip(ref * min_price_gap_pct, min=1e-8)
    sell_price = jnp.maximum(sell_price, buy_price + gap)
    buy_amount = jax.nn.sigmoid(jnp.asarray(outputs["buy_amount_logits"])).squeeze(-1) * trade_amount_scale
    sell_amount = jax.nn.sigmoid(jnp.asarray(outputs["sell_amount_logits"])).squeeze(-1) * trade_amount_scale
    trade_amount = jnp.maximum(buy_amount, sell_amount)

    decoded = {
        "buy_price": buy_price,
        "sell_price": sell_price,
        "trade_amount": trade_amount,
        "buy_amount": buy_amount,
        "sell_amount": sell_amount,
    }
    if "hold_hours_logits" in outputs:
        decoded["hold_hours"] = jax.nn.sigmoid(jnp.asarray(outputs["hold_hours_logits"])).squeeze(-1) * max_hold_hours
    if "allocation_logits" in outputs:
        decoded["allocation_fraction"] = jax.nn.sigmoid(jnp.asarray(outputs["allocation_logits"])).squeeze(-1)
    return decoded


def build_classic_policy_config(
    payload: Mapping[str, Any],
    *,
    input_dim: int,
    state_dict: Mapping[str, Any] | None = None,
) -> PolicyConfig:
    from .model import policy_config_from_payload

    cfg = policy_config_from_payload(payload, input_dim=input_dim, state_dict=state_dict)
    if (cfg.model_arch or "classic").lower() != "classic":
        raise ValueError(f"JAX policy currently supports only classic arch, got {cfg.model_arch!r}")
    if cfg.num_outputs < 4:
        raise ValueError(f"classic policy requires at least 4 outputs, got {cfg.num_outputs}")
    return cfg


def convert_torch_classic_state_dict(
    state_dict: Mapping[str, Any],
    *,
    config: PolicyConfig,
) -> dict[str, Any]:
    import torch

    stripped = _strip_orig_mod_prefix(state_dict)

    def _to_numpy(value: Any) -> np.ndarray:
        if isinstance(value, np.ndarray):
            return value.astype(np.float32, copy=False)
        if isinstance(value, torch.Tensor):
            return value.detach().cpu().numpy().astype(np.float32, copy=False)
        return np.asarray(value, dtype=np.float32)

    params: dict[str, Any] = {
        "embed": {
            "kernel": _to_numpy(stripped["embed.weight"]).T,
            "bias": _to_numpy(stripped["embed.bias"]),
        },
        "norm": {
            "scale": _to_numpy(stripped["norm.weight"]),
            "bias": _to_numpy(stripped["norm.bias"]),
        },
        "head": {
            "kernel": _to_numpy(stripped["head.weight"]).T,
            "bias": _to_numpy(stripped["head.bias"]),
        },
    }

    for idx in range(config.num_layers):
        prefix = f"encoder.layers.{idx}"
        in_proj_weight = _to_numpy(stripped[f"{prefix}.self_attn.in_proj_weight"])
        in_proj_bias = _to_numpy(stripped[f"{prefix}.self_attn.in_proj_bias"])
        q_w, k_w, v_w = np.split(in_proj_weight, 3, axis=0)
        q_b, k_b, v_b = np.split(in_proj_bias, 3, axis=0)
        params[f"layers_{idx}"] = {
            "self_attn": {
                "q_proj": {"kernel": q_w.T, "bias": q_b},
                "k_proj": {"kernel": k_w.T, "bias": k_b},
                "v_proj": {"kernel": v_w.T, "bias": v_b},
                "out_proj": {
                    "kernel": _to_numpy(stripped[f"{prefix}.self_attn.out_proj.weight"]).T,
                    "bias": _to_numpy(stripped[f"{prefix}.self_attn.out_proj.bias"]),
                },
            },
            "norm1": {
                "scale": _to_numpy(stripped[f"{prefix}.norm1.weight"]),
                "bias": _to_numpy(stripped[f"{prefix}.norm1.bias"]),
            },
            "norm2": {
                "scale": _to_numpy(stripped[f"{prefix}.norm2.weight"]),
                "bias": _to_numpy(stripped[f"{prefix}.norm2.bias"]),
            },
            "linear1": {
                "kernel": _to_numpy(stripped[f"{prefix}.linear1.weight"]).T,
                "bias": _to_numpy(stripped[f"{prefix}.linear1.bias"]),
            },
            "linear2": {
                "kernel": _to_numpy(stripped[f"{prefix}.linear2.weight"]).T,
                "bias": _to_numpy(stripped[f"{prefix}.linear2.bias"]),
            },
        }
    return params


def init_classic_params(
    *,
    config: PolicyConfig,
    rng: Array,
    sequence_length: int,
) -> dict[str, Any]:
    model = JaxClassicPolicy(config)
    dummy = jnp.zeros((1, sequence_length, config.input_dim), dtype=jnp.float32)
    variables = model.init(rng, dummy, deterministic=True)
    return variables["params"]

