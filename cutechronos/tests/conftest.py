"""Shared test helpers for cutechronos tests."""

from __future__ import annotations

import os
import sys

import pytest
import torch

from cutechronos.model import CuteChronos2Config, CuteChronos2Model


# ---------------------------------------------------------------------------
# Ensure chronos-forecasting source is importable (local checkout)
# ---------------------------------------------------------------------------
_CHRONOS_SRC = os.path.join(
    os.path.dirname(__file__), "..", "..", "chronos-forecasting", "src"
)
if os.path.isdir(_CHRONOS_SRC) and _CHRONOS_SRC not in sys.path:
    sys.path.insert(0, _CHRONOS_SRC)


# ---------------------------------------------------------------------------
# Standard Chronos2-base config used by build_model_pair / build_cute_only
# ---------------------------------------------------------------------------
_BASE_CONFIG_KWARGS = dict(
    d_model=768,
    d_kv=64,
    d_ff=3072,
    num_layers=12,
    num_heads=12,
    dropout_rate=0.0,
    layer_norm_epsilon=1e-6,
    dense_act_fn="relu",
    rope_theta=10000.0,
    vocab_size=2,
    reg_token_id=1,
    context_length=512,
    input_patch_size=16,
    input_patch_stride=16,
    output_patch_size=16,
    quantiles=[
        0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45,
        0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.99,
    ],
    use_reg_token=True,
    use_arcsinh=True,
    time_encoding_scale=512,
)


def build_cute_only() -> CuteChronos2Model:
    """Build a randomly-initialized CuteChronos2Model in eval mode.

    Uses the standard Chronos2-base architecture (12 layers, 768 d_model).
    No dependency on the upstream ``chronos`` package.
    """
    torch.manual_seed(42)
    config = CuteChronos2Config(**_BASE_CONFIG_KWARGS)
    model = CuteChronos2Model(config)
    model.eval()
    return model


def build_model_pair():
    """Build a (Chronos2Model, CuteChronos2Model) pair sharing identical weights.

    The original ``Chronos2Model`` is randomly initialized (not from a
    pretrained checkpoint) so that tests are reproducible without downloading
    weights.  ``CuteChronos2Model.from_original(original)`` copies every
    parameter exactly.

    Returns
    -------
    tuple[Chronos2Model, CuteChronos2Model]

    Raises
    ------
    pytest.skip
        If the upstream ``chronos`` package or ``transformers`` is not
        importable (e.g. in lightweight CI environments).
    """
    # Import the upstream Chronos2 model; skip the test if unavailable.
    try:
        from chronos.chronos2.config import Chronos2CoreConfig
        from chronos.chronos2.model import Chronos2Model
    except ImportError:
        pytest.skip(
            "chronos-forecasting package not available; "
            "skipping test that requires the original Chronos2Model"
        )

    torch.manual_seed(42)

    # Build a Chronos2CoreConfig that matches _BASE_CONFIG_KWARGS
    chronos_config = {
        "context_length": _BASE_CONFIG_KWARGS["context_length"],
        "input_patch_size": _BASE_CONFIG_KWARGS["input_patch_size"],
        "input_patch_stride": _BASE_CONFIG_KWARGS["input_patch_stride"],
        "output_patch_size": _BASE_CONFIG_KWARGS["output_patch_size"],
        "max_output_patches": 4,
        "quantiles": _BASE_CONFIG_KWARGS["quantiles"],
        "use_reg_token": _BASE_CONFIG_KWARGS["use_reg_token"],
        "use_arcsinh": _BASE_CONFIG_KWARGS["use_arcsinh"],
        "time_encoding_scale": _BASE_CONFIG_KWARGS["time_encoding_scale"],
    }
    core_config = Chronos2CoreConfig(
        d_model=_BASE_CONFIG_KWARGS["d_model"],
        d_kv=_BASE_CONFIG_KWARGS["d_kv"],
        d_ff=_BASE_CONFIG_KWARGS["d_ff"],
        num_layers=_BASE_CONFIG_KWARGS["num_layers"],
        num_heads=_BASE_CONFIG_KWARGS["num_heads"],
        dropout_rate=_BASE_CONFIG_KWARGS["dropout_rate"],
        layer_norm_epsilon=_BASE_CONFIG_KWARGS["layer_norm_epsilon"],
        feed_forward_proj=_BASE_CONFIG_KWARGS["dense_act_fn"],
        rope_theta=_BASE_CONFIG_KWARGS["rope_theta"],
        vocab_size=_BASE_CONFIG_KWARGS["vocab_size"],
        chronos_config=chronos_config,
        attn_implementation="eager",
    )

    original = Chronos2Model(core_config)
    original.eval()

    cute = CuteChronos2Model.from_original(original)
    return original, cute


def make_context(B: int, L: int, nan_frac: float = 0.1, seed: int = 42) -> torch.Tensor:
    """Create a random context tensor with some NaN values sprinkled in."""
    gen = torch.Generator().manual_seed(seed)
    ctx = torch.randn(B, L, generator=gen)
    mask = torch.rand(B, L, generator=gen) < nan_frac
    ctx[mask] = float("nan")
    return ctx


def compare_outputs(
    result: tuple,
    ref: tuple,
    atol: float = 1e-5,
    label: str = "",
):
    """Compare preprocessing output against reference, asserting shape and value match."""
    patched, attn, loc, scale = result
    ref_patched, ref_attn, ref_loc, ref_scale = ref

    prefix = f"[{label}] " if label else ""

    # Shapes
    assert patched.shape == ref_patched.shape, (
        f"{prefix}patched shape mismatch: {patched.shape} vs {ref_patched.shape}"
    )
    assert attn.shape == ref_attn.shape, (
        f"{prefix}attn_mask shape mismatch: {attn.shape} vs {ref_attn.shape}"
    )
    assert loc.shape == ref_loc.shape, (
        f"{prefix}loc shape mismatch: {loc.shape} vs {ref_loc.shape}"
    )
    assert scale.shape == ref_scale.shape, (
        f"{prefix}scale shape mismatch: {scale.shape} vs {ref_scale.shape}"
    )

    # Values
    max_err_patched = (patched.float() - ref_patched.float()).abs().max().item()
    max_err_attn = (attn.float() - ref_attn.float()).abs().max().item()
    max_err_loc = (loc.float() - ref_loc.float()).abs().max().item()
    max_err_scale = (scale.float() - ref_scale.float()).abs().max().item()

    assert max_err_loc < atol, f"{prefix}loc max error {max_err_loc} >= {atol}"
    assert max_err_scale < atol, f"{prefix}scale max error {max_err_scale} >= {atol}"
    assert max_err_attn < atol, f"{prefix}attn_mask max error {max_err_attn} >= {atol}"
    assert max_err_patched < atol, (
        f"{prefix}patched max error {max_err_patched} >= {atol}"
    )
