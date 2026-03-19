"""Tests for torch.compile integration with CuteChronos2Model.

Verifies that:
1. Compiled model outputs match eager model (< 1e-4 max abs error)
2. Warmup iterations work without errors
3. Compiled model produces deterministic results
4. predict() method works with torch.inference_mode
5. Benchmark function runs end-to-end
"""

import json
import os
import sys
import tempfile
import time

import pytest
import torch

# Ensure chronos2 source is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "chronos-forecasting", "src"))

from cutechronos.model import (
    CuteChronos2Config,
    CuteChronos2Model,
    _apply_torch_compile,
    benchmark_eager_vs_compiled,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_small_config() -> CuteChronos2Config:
    """Create a small config for fast testing (2 layers instead of 12)."""
    return CuteChronos2Config(
        d_model=128,
        d_kv=32,
        d_ff=256,
        num_layers=2,
        num_heads=4,
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
    )


def _build_eager_model() -> CuteChronos2Model:
    """Build a small eager model with random weights."""
    torch.manual_seed(42)
    config = _make_small_config()
    model = CuteChronos2Model(config)
    model.eval()
    return model


def _save_model_to_dir(model: CuteChronos2Model, tmpdir: str) -> None:
    """Save model weights and config to a temp directory for from_pretrained."""
    from safetensors.torch import save_file as safetensors_save_file

    config = model.config
    config_dict = {
        "d_model": config.d_model,
        "d_kv": config.d_kv,
        "d_ff": config.d_ff,
        "num_layers": config.num_layers,
        "num_heads": config.num_heads,
        "dropout_rate": config.dropout_rate,
        "layer_norm_epsilon": config.layer_norm_epsilon,
        "dense_act_fn": config.dense_act_fn,
        "rope_theta": config.rope_theta,
        "vocab_size": config.vocab_size,
        "reg_token_id": config.reg_token_id,
        "chronos_config": {
            "context_length": config.context_length,
            "input_patch_size": config.input_patch_size,
            "input_patch_stride": config.input_patch_stride,
            "output_patch_size": config.output_patch_size,
            "quantiles": config.quantiles,
            "use_reg_token": config.use_reg_token,
            "use_arcsinh": config.use_arcsinh,
            "time_encoding_scale": config.time_encoding_scale,
        },
    }
    with open(os.path.join(tmpdir, "config.json"), "w") as f:
        json.dump(config_dict, f)

    # Build the state dict in the format load_chronos2_weights expects
    state_dict = {}
    state_dict["shared.weight"] = model.shared.weight.data
    for name in ("input_patch_embedding", "output_patch_embedding"):
        block = getattr(model, name)
        for layer in ("hidden_layer", "output_layer", "residual_layer"):
            for param in ("weight", "bias"):
                key = f"{name}.{layer}.{param}"
                state_dict[key] = getattr(getattr(block, layer), param).data

    state_dict["encoder.final_layer_norm.weight"] = model.final_layer_norm_weight.data
    for i, block in enumerate(model.blocks):
        prefix = f"encoder.block.{i}"
        state_dict[f"{prefix}.layer.0.layer_norm.weight"] = block.time_attn.layer_norm_weight.data
        for proj in ("q", "k", "v", "o"):
            state_dict[f"{prefix}.layer.0.self_attention.{proj}.weight"] = getattr(block.time_attn, proj).weight.data
        state_dict[f"{prefix}.layer.1.layer_norm.weight"] = block.group_attn.layer_norm_weight.data
        for proj in ("q", "k", "v", "o"):
            state_dict[f"{prefix}.layer.1.self_attention.{proj}.weight"] = getattr(block.group_attn, proj).weight.data
        state_dict[f"{prefix}.layer.2.layer_norm.weight"] = block.feed_forward.layer_norm_weight.data
        state_dict[f"{prefix}.layer.2.mlp.wi.weight"] = block.feed_forward.wi.weight.data
        state_dict[f"{prefix}.layer.2.mlp.wo.weight"] = block.feed_forward.wo.weight.data

    safetensors_save_file(state_dict, os.path.join(tmpdir, "model.safetensors"))


# ---------------------------------------------------------------------------
# Tests: compiled model matches eager model
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "batch_size,context_length",
    [(2, 256), (1, 128), (3, 64)],
    ids=["B2_L256", "B1_L128", "B3_L64"],
)
def test_compiled_matches_eager(batch_size: int, context_length: int):
    """Compiled model outputs must match eager within 1e-4 tolerance."""
    eager_model = _build_eager_model()

    # Build compiled model from same weights
    compiled_model = _build_eager_model()
    compiled_model = _apply_torch_compile(compiled_model, compile_mode="reduce-overhead")

    torch.manual_seed(0)
    context = torch.randn(batch_size, context_length) * 0.1 + 100.0

    with torch.inference_mode():
        eager_out = eager_model(context)
        compiled_out = compiled_model(context)

    assert eager_out.shape == compiled_out.shape, (
        f"Shape mismatch: eager {eager_out.shape} vs compiled {compiled_out.shape}"
    )

    max_err = (eager_out - compiled_out).abs().max().item()
    assert max_err < 1e-4, (
        f"Max abs error {max_err:.2e} >= 1e-4 for B={batch_size}, L={context_length}"
    )


# ---------------------------------------------------------------------------
# Tests: warmup iterations work
# ---------------------------------------------------------------------------

def test_compiled_warmup():
    """Compiled model should handle multiple warmup iterations without error."""
    model = _build_eager_model()
    model = _apply_torch_compile(model, compile_mode="reduce-overhead")

    torch.manual_seed(0)
    context = torch.randn(2, 128) * 0.1 + 100.0

    # Run 5 warmup iterations — first triggers compilation, rest should reuse
    outputs = []
    with torch.inference_mode():
        for _ in range(5):
            out = model(context)
            outputs.append(out.clone())

    # All outputs should be identical after warmup
    for i in range(1, len(outputs)):
        assert torch.equal(outputs[0], outputs[i]), (
            f"Warmup iteration {i} produced different output"
        )


# ---------------------------------------------------------------------------
# Tests: deterministic results
# ---------------------------------------------------------------------------

def test_compiled_deterministic():
    """Two forward passes on the compiled model should give identical results."""
    model = _build_eager_model()
    model = _apply_torch_compile(model, compile_mode="reduce-overhead")

    torch.manual_seed(0)
    context = torch.randn(2, 256) * 0.1 + 100.0

    with torch.inference_mode():
        # Warmup
        _ = model(context)
        # Actual test
        out1 = model(context)
        out2 = model(context)

    assert torch.equal(out1, out2), "Compiled forward pass should be deterministic"


# ---------------------------------------------------------------------------
# Tests: predict method with inference_mode
# ---------------------------------------------------------------------------

def test_predict_method():
    """predict() should work and produce same output as forward()."""
    model = _build_eager_model()

    torch.manual_seed(0)
    context = torch.randn(2, 256) * 0.1 + 100.0

    with torch.inference_mode():
        forward_out = model.forward(context)

    predict_out = model.predict(context)

    assert torch.equal(forward_out, predict_out), (
        "predict() should produce identical output to forward()"
    )


def test_predict_method_compiled():
    """predict() should work on compiled model."""
    model = _build_eager_model()
    model = _apply_torch_compile(model, compile_mode="reduce-overhead")

    torch.manual_seed(0)
    context = torch.randn(2, 256) * 0.1 + 100.0

    # predict already wraps inference_mode
    out = model.predict(context)

    assert out.shape == (2, 21, 16), f"Unexpected shape: {out.shape}"
    assert torch.isfinite(out).all(), "Output contains non-finite values"


def test_predict_with_nan_inputs():
    """predict() should handle NaN inputs gracefully on compiled model."""
    model = _build_eager_model()
    model = _apply_torch_compile(model, compile_mode="reduce-overhead")

    torch.manual_seed(0)
    context = torch.randn(2, 256) * 0.1 + 100.0
    context[0, :30] = float("nan")
    context[1, 100:150] = float("nan")

    out = model.predict(context)
    assert out.shape == (2, 21, 16), f"Unexpected shape: {out.shape}"
    assert torch.isfinite(out).all(), "Output contains non-finite values with NaN inputs"


def test_predict_with_context_mask():
    """predict() should accept optional context_mask argument."""
    model = _build_eager_model()

    torch.manual_seed(0)
    context = torch.randn(2, 128) * 0.1 + 100.0
    mask = torch.ones(2, 128)
    mask[0, :20] = 0.0

    out = model.predict(context, context_mask=mask)
    assert out.shape == (2, 21, 16)
    assert torch.isfinite(out).all()


def test_predict_multi_output_patches():
    """predict() should support multiple output patches."""
    model = _build_eager_model()

    torch.manual_seed(0)
    context = torch.randn(1, 256) * 0.1 + 100.0

    out = model.predict(context, num_output_patches=3)
    expected_h = 3 * 16  # 3 patches * patch_size 16
    assert out.shape == (1, 21, expected_h), f"Expected (1, 21, {expected_h}), got {out.shape}"


# ---------------------------------------------------------------------------
# Tests: compiled matches eager with NaN inputs
# ---------------------------------------------------------------------------

def test_compiled_matches_eager_with_nans():
    """Compiled model handles NaN inputs same as eager."""
    eager_model = _build_eager_model()
    compiled_model = _build_eager_model()
    compiled_model = _apply_torch_compile(compiled_model, compile_mode="reduce-overhead")

    torch.manual_seed(0)
    context = torch.randn(2, 256) * 0.1 + 100.0
    context[0, :50] = float("nan")
    context[1, 100:200] = float("nan")

    with torch.inference_mode():
        eager_out = eager_model(context)
        compiled_out = compiled_model(context)

    max_err = (eager_out - compiled_out).abs().max().item()
    assert max_err < 1e-4, (
        f"Max abs error {max_err:.2e} >= 1e-4 with NaN inputs"
    )


# ---------------------------------------------------------------------------
# Tests: from_pretrained_compiled
# ---------------------------------------------------------------------------

def test_from_pretrained_compiled():
    """from_pretrained_compiled should load and compile successfully."""
    eager_model = _build_eager_model()

    with tempfile.TemporaryDirectory() as tmpdir:
        _save_model_to_dir(eager_model, tmpdir)

        compiled_model = CuteChronos2Model.from_pretrained_compiled(
            tmpdir, compile_mode="reduce-overhead"
        )

    torch.manual_seed(0)
    context = torch.randn(2, 128) * 0.1 + 100.0

    eager_out = eager_model.predict(context)
    compiled_out = compiled_model.predict(context)

    assert eager_out.shape == compiled_out.shape
    max_err = (eager_out - compiled_out).abs().max().item()
    assert max_err < 1e-4, (
        f"from_pretrained_compiled max abs error {max_err:.2e} >= 1e-4"
    )


def test_from_pretrained_compiled_default_mode():
    """from_pretrained_compiled should use reduce-overhead by default."""
    eager_model = _build_eager_model()

    with tempfile.TemporaryDirectory() as tmpdir:
        _save_model_to_dir(eager_model, tmpdir)

        # Just test it loads without error using default mode
        compiled_model = CuteChronos2Model.from_pretrained_compiled(tmpdir)

    out = compiled_model.predict(torch.randn(1, 64) * 0.1 + 100.0)
    assert out.shape == (1, 21, 16)


# ---------------------------------------------------------------------------
# Tests: _apply_torch_compile edge cases
# ---------------------------------------------------------------------------

def test_apply_compile_different_modes():
    """_apply_torch_compile should work with different compile modes."""
    for mode in ("default", "reduce-overhead"):
        model = _build_eager_model()
        compiled = _apply_torch_compile(model, compile_mode=mode)

        context = torch.randn(1, 64) * 0.1 + 100.0
        with torch.inference_mode():
            out = compiled(context)
        assert out.shape == (1, 21, 16), f"Failed with mode={mode}"


# ---------------------------------------------------------------------------
# Tests: benchmark function
# ---------------------------------------------------------------------------

def test_benchmark_runs(capsys):
    """benchmark_eager_vs_compiled should run end-to-end and print timing."""
    eager_model = _build_eager_model()

    with tempfile.TemporaryDirectory() as tmpdir:
        _save_model_to_dir(eager_model, tmpdir)

        result = benchmark_eager_vs_compiled(
            tmpdir,
            context_length=64,
            batch_size=1,
            warmup_iters=1,
            bench_iters=2,
            device="cpu",
            compile_mode="reduce-overhead",
        )

    assert "eager_ms" in result
    assert "compiled_ms" in result
    assert "speedup" in result
    assert result["eager_ms"] > 0
    assert result["compiled_ms"] > 0
    assert result["speedup"] > 0

    captured = capsys.readouterr()
    assert "eager=" in captured.out
    assert "compiled=" in captured.out
    assert "speedup=" in captured.out


# ---------------------------------------------------------------------------
# Tests: output shape invariance under compilation
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("num_output_patches", [1, 2, 4], ids=["P1", "P2", "P4"])
def test_compiled_output_shape(num_output_patches: int):
    """Compiled model should produce correct output shapes."""
    model = _build_eager_model()
    model = _apply_torch_compile(model, compile_mode="reduce-overhead")

    torch.manual_seed(0)
    context = torch.randn(2, 256) * 0.1 + 100.0

    out = model.predict(context, num_output_patches=num_output_patches)
    expected_h = num_output_patches * 16
    assert out.shape == (2, 21, expected_h), (
        f"Expected (2, 21, {expected_h}), got {out.shape}"
    )
