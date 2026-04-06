from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

from binanceneural.model import build_policy, policy_config_from_payload
from src.torch_load_utils import torch_load_compat


CHECKPOINT_DIR = Path("unified_hourly_experiment/checkpoints/wd_0.06_s42")
CHECKPOINT_PATH = CHECKPOINT_DIR / "epoch_008.pt"


def _run_jax_policy_probe(case: str, payload: dict[str, object]) -> dict[str, object]:
    with tempfile.TemporaryDirectory() as tmpdir:
        input_path = Path(tmpdir) / "input.json"
        output_path = Path(tmpdir) / "output.json"
        input_path.write_text(json.dumps(payload))

        script = """
import json
import sys
from pathlib import Path

import numpy as np

case = sys.argv[1]
input_path = Path(sys.argv[2])
output_path = Path(sys.argv[3])
payload = json.loads(input_path.read_text())

if case == "classic_policy_logits":
    import jax.numpy as jnp

    from binanceneural.jax_policy import (
        JaxClassicPolicy,
        build_classic_policy_config,
        convert_torch_classic_state_dict,
    )
    from src.torch_load_utils import torch_load_compat

    training_meta_path = Path(payload["training_meta_path"])
    checkpoint_path = Path(payload["checkpoint_path"])
    checkpoint = torch_load_compat(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = checkpoint.get("state_dict", checkpoint)
    training_meta = json.loads(training_meta_path.read_text())
    jax_cfg = build_classic_policy_config(
        training_meta,
        input_dim=len(training_meta["feature_columns"]),
        state_dict=state_dict,
    )
    jax_model = JaxClassicPolicy(jax_cfg)
    jax_params = convert_torch_classic_state_dict(state_dict, config=jax_cfg)
    features = np.asarray(payload["features"], dtype=np.float32)
    outputs = jax_model.apply({"params": jax_params}, jnp.asarray(features), deterministic=True)
    result = {
        key: np.asarray(outputs[key]).tolist()
        for key in ("buy_price_logits", "sell_price_logits", "buy_amount_logits", "sell_amount_logits")
    }
elif case == "decode_actions":
    import jax.numpy as jnp

    from binanceneural.jax_policy import decode_actions_jax

    logits = {
        key: jnp.asarray(np.asarray(value, dtype=np.float32))
        for key, value in payload["logits"].items()
    }
    decoded = decode_actions_jax(
        logits,
        reference_close=jnp.asarray(np.asarray(payload["reference_close"], dtype=np.float32)),
        chronos_high=jnp.asarray(np.asarray(payload["chronos_high"], dtype=np.float32)),
        chronos_low=jnp.asarray(np.asarray(payload["chronos_low"], dtype=np.float32)),
        price_offset_pct=float(payload["price_offset_pct"]),
        min_price_gap_pct=float(payload["min_price_gap_pct"]),
        trade_amount_scale=float(payload["trade_amount_scale"]),
        use_midpoint_offsets=bool(payload["use_midpoint_offsets"]),
        max_hold_hours=float(payload["max_hold_hours"]),
    )
    result = {
        key: np.asarray(decoded[key]).tolist()
        for key in ("buy_price", "sell_price", "trade_amount", "buy_amount", "sell_amount")
    }
else:
    raise ValueError(f"Unknown probe case: {case}")

output_path.write_text(json.dumps(result))
"""

        env = os.environ.copy()
        env.setdefault("JAX_PLATFORMS", "cpu")
        proc = subprocess.run(
            [sys.executable, "-c", script, case, str(input_path), str(output_path)],
            capture_output=True,
            text=True,
            env=env,
        )
        if proc.returncode != 0:
            combined_output = ((proc.stdout or "") + "\n" + (proc.stderr or "")).strip()
            if proc.returncode < 0 or "Fatal Python error" in combined_output or "Aborted" in combined_output:
                pytest.skip(
                    "JAX policy test skipped due to native JAX/XLA compiler abort under this environment: "
                    f"{combined_output[-400:]}"
                )
            raise AssertionError(
                "JAX policy subprocess failed unexpectedly:\n"
                f"{combined_output or f'process exited with code {proc.returncode}'}"
            )
        return json.loads(output_path.read_text())


def test_jax_classic_policy_matches_torch_checkpoint_logits() -> None:
    payload = json.loads((CHECKPOINT_DIR / "training_meta.json").read_text())
    checkpoint = torch_load_compat(CHECKPOINT_PATH, map_location="cpu", weights_only=False)
    state_dict = checkpoint.get("state_dict", checkpoint)

    torch_cfg = policy_config_from_payload(payload, input_dim=len(payload["feature_columns"]), state_dict=state_dict)
    torch_model = build_policy(torch_cfg)
    torch_model.load_state_dict(state_dict, strict=False)
    torch_model.eval()

    rng = np.random.default_rng(42)
    features = rng.normal(size=(2, payload["sequence_length"], len(payload["feature_columns"]))).astype(np.float32)
    with torch.inference_mode():
        torch_outputs = torch_model(torch.from_numpy(features))
    jax_outputs = _run_jax_policy_probe(
        "classic_policy_logits",
        {
            "training_meta_path": str(CHECKPOINT_DIR / "training_meta.json"),
            "checkpoint_path": str(CHECKPOINT_PATH),
            "features": features.tolist(),
        },
    )

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
    jax_decoded = _run_jax_policy_probe(
        "decode_actions",
        {
            "logits": {key: value.numpy().tolist() for key, value in logits.items()},
            "reference_close": reference_close.numpy().tolist(),
            "chronos_high": chronos_high.numpy().tolist(),
            "chronos_low": chronos_low.numpy().tolist(),
            "price_offset_pct": float(torch_model.price_offset_pct),
            "min_price_gap_pct": float(torch_model.min_gap_pct),
            "trade_amount_scale": float(torch_model.trade_amount_scale),
            "use_midpoint_offsets": bool(torch_model.use_midpoint_offsets),
            "max_hold_hours": float(torch_model.max_hold_hours),
        },
    )
    for key in ("buy_price", "sell_price", "trade_amount", "buy_amount", "sell_amount"):
        np.testing.assert_allclose(
            np.asarray(torch_decoded[key]),
            np.asarray(jax_decoded[key]),
            rtol=1e-5,
            atol=1e-5,
        )
