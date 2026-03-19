"""Tests for CuteChronos2Model vs original Chronos2Model.

Verifies that CuteChronos2Model produces outputs matching the original
within a tight tolerance (max abs error < 1e-4) after loading the same weights.

Uses randomly-initialized Chronos2 models for testing (the finetuned checkpoints
on disk have numerical instability in the unscaled attention with many inputs).
"""

import json
import os
import tempfile

import pytest
import torch

from cutechronos.model import CuteChronos2Model
from cutechronos.tests.conftest import build_model_pair

# All tests in this module require the original Chronos2Model (and therefore
# the upstream chronos-forecasting package).  build_model_pair() will call
# pytest.skip() if the package is not available, but the marker lets CI
# filter the entire module cheaply with ``-m "not model_required"``.
pytestmark = pytest.mark.model_required


# -------------------------------------------------------------------
# Test: matching outputs with from_original weight copy
# -------------------------------------------------------------------

@pytest.mark.parametrize(
    "batch_size,context_length",
    [(2, 512), (1, 256), (3, 128), (1, 64)],
    ids=["B2_L512", "B1_L256", "B3_L128", "B1_L64"],
)
def test_matches_original(batch_size: int, context_length: int):
    """Verify CuteChronos2Model matches original Chronos2Model output."""
    original, cute = build_model_pair()

    torch.manual_seed(0)
    context = torch.randn(batch_size, context_length) * 0.1 + 100

    with torch.no_grad():
        orig_out = original(context)
        cute_out = cute(context)

    orig_preds = orig_out.quantile_preds
    assert orig_preds.shape == cute_out.shape, (
        f"Shape mismatch: original {orig_preds.shape} vs cute {cute_out.shape}"
    )

    max_err = (orig_preds - cute_out).abs().max().item()
    assert max_err < 1e-4, (
        f"Max abs error {max_err:.2e} >= 1e-4 for B={batch_size}, L={context_length}"
    )


# -------------------------------------------------------------------
# Test: matching outputs with from_pretrained path loading
# -------------------------------------------------------------------

def test_matches_original_from_pretrained():
    """Verify from_pretrained produces same results as from_original."""
    original, _ = build_model_pair()

    with tempfile.TemporaryDirectory() as tmpdir:
        # Save config
        config_dict = {
            "architectures": ["Chronos2Model"],
            "d_model": 768, "d_kv": 64, "d_ff": 3072, "num_layers": 12, "num_heads": 12,
            "dropout_rate": 0.0, "layer_norm_epsilon": 1e-6, "dense_act_fn": "relu",
            "rope_theta": 10000.0, "vocab_size": 2, "reg_token_id": 1, "model_type": "t5",
            "chronos_config": {
                "context_length": 512, "input_patch_size": 16, "input_patch_stride": 16,
                "output_patch_size": 16, "max_output_patches": 4,
                "quantiles": [
                    0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45,
                    0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.99,
                ],
                "use_reg_token": True, "use_arcsinh": True, "time_encoding_scale": 512,
            },
        }
        with open(os.path.join(tmpdir, "config.json"), "w") as f:
            json.dump(config_dict, f)

        # Save weights as safetensors
        from safetensors.torch import save_file as safetensors_save_file
        safetensors_save_file(original.state_dict(), os.path.join(tmpdir, "model.safetensors"))

        cute = CuteChronos2Model.from_pretrained(tmpdir)

    torch.manual_seed(0)
    context = torch.randn(2, 512) * 0.1 + 100

    with torch.no_grad():
        orig_out = original(context)
        cute_out = cute(context)

    orig_preds = orig_out.quantile_preds
    assert orig_preds.shape == cute_out.shape
    max_err = (orig_preds - cute_out).abs().max().item()
    assert max_err < 1e-4, f"Max abs error {max_err:.2e} >= 1e-4"


# -------------------------------------------------------------------
# Test: output shape correctness
# -------------------------------------------------------------------

@pytest.mark.parametrize(
    "batch_size,context_length,num_output_patches",
    [(2, 512, 1), (1, 256, 2), (3, 128, 4)],
    ids=["B2_L512_P1", "B1_L256_P2", "B3_L128_P4"],
)
def test_output_shape(batch_size: int, context_length: int, num_output_patches: int):
    """Verify output shape is (B, Q, H) with correct dimensions."""
    _, cute = build_model_pair()

    torch.manual_seed(0)
    context = torch.randn(batch_size, context_length) * 0.1 + 100

    with torch.no_grad():
        cute_out = cute(context, num_output_patches=num_output_patches)

    expected_q = cute.config.num_quantiles  # 21
    expected_h = num_output_patches * cute.config.output_patch_size  # N * 16
    assert cute_out.shape == (batch_size, expected_q, expected_h), (
        f"Expected ({batch_size}, {expected_q}, {expected_h}), got {cute_out.shape}"
    )


# -------------------------------------------------------------------
# Test: multi-output-patch matching
# -------------------------------------------------------------------

@pytest.mark.parametrize("num_output_patches", [1, 2, 4], ids=["P1", "P2", "P4"])
def test_matches_original_multi_output_patches(num_output_patches: int):
    """Verify matching with multiple output patches."""
    original, cute = build_model_pair()

    torch.manual_seed(0)
    context = torch.randn(2, 512) * 0.1 + 100

    with torch.no_grad():
        orig_out = original(context, num_output_patches=num_output_patches)
        cute_out = cute(context, num_output_patches=num_output_patches)

    orig_preds = orig_out.quantile_preds
    assert orig_preds.shape == cute_out.shape
    max_err = (orig_preds - cute_out).abs().max().item()
    assert max_err < 1e-4, (
        f"Max abs error {max_err:.2e} >= 1e-4 with {num_output_patches} output patches"
    )


# -------------------------------------------------------------------
# Test: handling NaN inputs
# -------------------------------------------------------------------

def test_matches_with_nan_inputs():
    """Verify matching when context contains NaN values."""
    original, cute = build_model_pair()

    torch.manual_seed(0)
    context = torch.randn(2, 512) * 0.1 + 100
    context[0, :50] = float("nan")
    context[1, 200:300] = float("nan")

    with torch.no_grad():
        orig_out = original(context)
        cute_out = cute(context)

    orig_preds = orig_out.quantile_preds
    assert orig_preds.shape == cute_out.shape
    max_err = (orig_preds - cute_out).abs().max().item()
    assert max_err < 1e-4, f"Max abs error {max_err:.2e} >= 1e-4 with NaN inputs"


# -------------------------------------------------------------------
# Test: determinism
# -------------------------------------------------------------------

def test_deterministic():
    """Two forward passes with the same input should give identical results."""
    _, cute = build_model_pair()

    torch.manual_seed(0)
    context = torch.randn(2, 512) * 0.1 + 100

    with torch.no_grad():
        out1 = cute(context)
        out2 = cute(context)

    assert torch.equal(out1, out2), "Forward pass should be deterministic"


# -------------------------------------------------------------------
# Test: weight count matches
# -------------------------------------------------------------------

def test_parameter_count():
    """Verify CuteChronos2Model has the same number of parameters as original."""
    original, cute = build_model_pair()

    orig_params = sum(p.numel() for p in original.parameters())
    cute_params = sum(p.numel() for p in cute.parameters())

    assert orig_params == cute_params, (
        f"Parameter count mismatch: original {orig_params} vs cute {cute_params}"
    )


# -------------------------------------------------------------------
# Test: from_original copies all weights exactly
# -------------------------------------------------------------------

def test_weight_copy_exact():
    """Ensure from_original copies every weight parameter exactly."""
    original, cute = build_model_pair()

    orig_sd = original.state_dict()

    # Check shared embedding
    assert torch.equal(cute.shared.weight, original.shared.weight), "shared.weight mismatch"

    # Check input/output patch embeddings
    for prefix in ["input_patch_embedding", "output_patch_embedding"]:
        for suffix in ["hidden_layer.weight", "hidden_layer.bias",
                       "output_layer.weight", "output_layer.bias",
                       "residual_layer.weight", "residual_layer.bias"]:
            orig_key = f"{prefix}.{suffix}"
            assert orig_key in orig_sd, f"Missing key: {orig_key}"
            orig_w = orig_sd[orig_key]
            cute_w = getattr(cute, prefix)
            for part in suffix.split("."):
                cute_w = getattr(cute_w, part)
            assert torch.equal(cute_w, orig_w), f"Weight mismatch: {orig_key}"

    # Check encoder blocks
    for i in range(12):
        # Time attention weights
        for proj in ["q", "k", "v", "o"]:
            orig_key = f"encoder.block.{i}.layer.0.self_attention.{proj}.weight"
            cute_w = getattr(cute.blocks[i].time_attn, proj).weight
            assert torch.equal(cute_w, orig_sd[orig_key]), f"Weight mismatch: {orig_key}"

        # Time LN weight
        orig_key = f"encoder.block.{i}.layer.0.layer_norm.weight"
        assert torch.equal(cute.blocks[i].time_attn.layer_norm_weight, orig_sd[orig_key]), (
            f"Weight mismatch: {orig_key}"
        )

        # Group attention weights
        for proj in ["q", "k", "v", "o"]:
            orig_key = f"encoder.block.{i}.layer.1.self_attention.{proj}.weight"
            cute_w = getattr(cute.blocks[i].group_attn, proj).weight
            assert torch.equal(cute_w, orig_sd[orig_key]), f"Weight mismatch: {orig_key}"

        # Group LN weight
        orig_key = f"encoder.block.{i}.layer.1.layer_norm.weight"
        assert torch.equal(cute.blocks[i].group_attn.layer_norm_weight, orig_sd[orig_key]), (
            f"Weight mismatch: {orig_key}"
        )

        # FF weights
        orig_key_wi = f"encoder.block.{i}.layer.2.mlp.wi.weight"
        orig_key_wo = f"encoder.block.{i}.layer.2.mlp.wo.weight"
        orig_key_ln = f"encoder.block.{i}.layer.2.layer_norm.weight"
        assert torch.equal(cute.blocks[i].feed_forward.wi.weight, orig_sd[orig_key_wi])
        assert torch.equal(cute.blocks[i].feed_forward.wo.weight, orig_sd[orig_key_wo])
        assert torch.equal(cute.blocks[i].feed_forward.layer_norm_weight, orig_sd[orig_key_ln])

    # Final layer norm
    assert torch.equal(cute.final_layer_norm_weight, orig_sd["encoder.final_layer_norm.weight"])
