from __future__ import annotations

import json
from pathlib import Path

import pytest

try:
    import jax.numpy as jnp  # noqa: F401
    import flax  # noqa: F401
except ImportError:
    pytest.skip("jax/flax not installed", allow_module_level=True)

import jax.numpy as jnp
import numpy as np
import torch

from binanceneural.jax_policy import (
    JaxClassicPolicy,
    build_classic_policy_config,
    convert_torch_classic_state_dict,
    decode_actions_jax,
)
from binanceneural.model import build_policy, policy_config_from_payload
from src.torch_load_utils import torch_load_compat


CHECKPOINT_DIR = Path("unified_hourly_experiment/checkpoints/wd_0.06_s42")
CHECKPOINT_PATH = CHECKPOINT_DIR / "epoch_008.pt"


def test_jax_classic_policy_matches_torch_checkpoint_logits() -> None:
    payload = json.loads((CHECKPOINT_DIR / "training_meta.json").read_text())
    checkpoint = torch_load_compat(CHECKPOINT_PATH, map_location="cpu", weights_only=False)
    state_dict = checkpoint.get("state_dict", checkpoint)

    torch_cfg = policy_config_from_payload(payload, input_dim=len(payload["feature_columns"]), state_dict=state_dict)
    torch_model = build_policy(torch_cfg)
    torch_model.load_state_dict(state_dict, strict=False)
    torch_model.eval()

    jax_cfg = build_classic_policy_config(
        payload,
        input_dim=len(payload["feature_columns"]),
        state_dict=state_dict,
    )
    jax_model = JaxClassicPolicy(jax_cfg)
    jax_params = convert_torch_classic_state_dict(state_dict, config=jax_cfg)

    rng = np.random.default_rng(42)
    features = rng.normal(size=(2, payload["sequence_length"], len(payload["feature_columns"]))).astype(np.float32)
    with torch.inference_mode():
        torch_outputs = torch_model(torch.from_numpy(features))
    jax_outputs = jax_model.apply({"params": jax_params}, jnp.asarray(features), deterministic=True)

    for key in ("buy_price_logits", "sell_price_logits", "buy_amount_logits", "sell_amount_logits"):
        np.testing.assert_allclose(
            np.asarray(torch_outputs[key]),
            np.asarray(jax_outputs[key]),
            rtol=2e-2,
            atol=2e-2,
        )


def test_decode_actions_jax_matches_torch_decode() -> None:
    payload = json.loads((CHECKPOINT_DIR / "training_meta.json").read_text())
    checkpoint = torch_load_compat(CHECKPOINT_PATH, map_location="cpu", weights_only=False)
    state_dict = checkpoint.get("state_dict", checkpoint)
    torch_cfg = policy_config_from_payload(payload, input_dim=len(payload["feature_columns"]), state_dict=state_dict)
    torch_model = build_policy(torch_cfg)
    torch_model.load_state_dict(state_dict, strict=False)
    torch_model.eval()

    rng = np.random.default_rng(7)
    logits = {
        "buy_price_logits": torch.from_numpy(rng.normal(size=(2, 4, 1)).astype(np.float32)),
        "sell_price_logits": torch.from_numpy(rng.normal(size=(2, 4, 1)).astype(np.float32)),
        "buy_amount_logits": torch.from_numpy(rng.normal(size=(2, 4, 1)).astype(np.float32)),
        "sell_amount_logits": torch.from_numpy(rng.normal(size=(2, 4, 1)).astype(np.float32)),
    }
    reference_close = torch.from_numpy(rng.uniform(10.0, 20.0, size=(2, 4)).astype(np.float32))
    chronos_high = reference_close + 1.5
    chronos_low = reference_close - 1.5
    torch_decoded = torch_model.decode_actions(
        logits,
        reference_close=reference_close,
        chronos_high=chronos_high,
        chronos_low=chronos_low,
    )
    jax_decoded = decode_actions_jax(
        {key: jnp.asarray(value.numpy()) for key, value in logits.items()},
        reference_close=jnp.asarray(reference_close.numpy()),
        chronos_high=jnp.asarray(chronos_high.numpy()),
        chronos_low=jnp.asarray(chronos_low.numpy()),
        price_offset_pct=float(torch_model.price_offset_pct),
        min_price_gap_pct=float(torch_model.min_gap_pct),
        trade_amount_scale=float(torch_model.trade_amount_scale),
        use_midpoint_offsets=bool(torch_model.use_midpoint_offsets),
        max_hold_hours=float(torch_model.max_hold_hours),
    )
    for key in ("buy_price", "sell_price", "trade_amount", "buy_amount", "sell_amount"):
        np.testing.assert_allclose(
            np.asarray(torch_decoded[key]),
            np.asarray(jax_decoded[key]),
            rtol=1e-5,
            atol=1e-5,
        )
