"""Tests for FusedFeedForward module.

Verifies numerical equivalence between the original Chronos2 FeedForward
and the optimized FusedFeedForward across:
- Multiple input shapes: (1, 34, 768), (4, 130, 768), (16, 514, 768)
- FP32 and BF16 precision
- Weight loading from original layer
"""

import pytest
import torch

from chronos.chronos2.config import Chronos2CoreConfig
from chronos.chronos2.layers import FeedForward

from cutechronos.modules.feedforward import FusedFeedForward

# Chronos2-base config: d_model=768, d_ff=3072, relu, 12 heads, 64 kv_dim
BASE_CONFIG = dict(
    d_model=768,
    d_kv=64,
    d_ff=3072,
    num_layers=12,
    num_heads=12,
    dropout_rate=0.0,  # Zero dropout for deterministic comparison
    layer_norm_epsilon=1e-6,
    feed_forward_proj="relu",
)


def make_config(**overrides) -> Chronos2CoreConfig:
    """Create a Chronos2CoreConfig with base settings and optional overrides."""
    kwargs = {**BASE_CONFIG, **overrides}
    return Chronos2CoreConfig(**kwargs)


def make_original_and_fused(device="cuda", dtype=torch.float32):
    """Create matched original FeedForward and FusedFeedForward on given device/dtype."""
    config = make_config()
    original = FeedForward(config).eval().to(device=device, dtype=dtype)

    fused = FusedFeedForward(
        d_model=config.d_model,
        d_ff=config.d_ff,
        eps=config.layer_norm_epsilon,
    ).eval().to(device=device, dtype=dtype)

    fused.load_from_original(original)
    return original, fused


# ---- FusedFeedForward correctness tests ----


INPUT_SHAPES = [
    (1, 34, 768),
    (4, 130, 768),
    (16, 514, 768),
]


class TestFusedFeedForwardFP32:
    """Test FusedFeedForward matches original in FP32."""

    @pytest.mark.parametrize("shape", INPUT_SHAPES, ids=lambda s: f"{s[0]}x{s[1]}x{s[2]}")
    def test_output_matches(self, shape):
        """FusedFeedForward output matches original FeedForward in FP32."""
        torch.manual_seed(42)
        original, fused = make_original_and_fused(device="cuda", dtype=torch.float32)

        x = torch.randn(*shape, device="cuda", dtype=torch.float32)

        with torch.no_grad():
            ref_out = original(x)
            fused_out = fused(x)

        max_err = (ref_out - fused_out).abs().max().item()
        assert max_err < 1e-5, f"FP32 shape {shape} max error: {max_err}"


class TestFusedFeedForwardBF16:
    """Test FusedFeedForward matches original in BF16."""

    @pytest.mark.parametrize("shape", INPUT_SHAPES, ids=lambda s: f"{s[0]}x{s[1]}x{s[2]}")
    def test_output_matches(self, shape):
        """FusedFeedForward output matches original FeedForward in BF16."""
        torch.manual_seed(42)
        original, fused = make_original_and_fused(device="cuda", dtype=torch.bfloat16)

        x = torch.randn(*shape, device="cuda", dtype=torch.bfloat16)

        with torch.no_grad():
            ref_out = original(x)
            fused_out = fused(x)

        max_err = (ref_out - fused_out).abs().max().item()
        assert max_err < 5e-3, f"BF16 shape {shape} max error: {max_err}"


# ---- Weight loading tests ----


class TestWeightLoading:
    """Test that weight loading from original layer works correctly."""

    def test_load_from_original_copies_all_weights(self):
        """All weights are correctly copied from original layer."""
        config = make_config()
        original = FeedForward(config).eval().cuda()
        fused = FusedFeedForward(
            d_model=config.d_model,
            d_ff=config.d_ff,
            eps=config.layer_norm_epsilon,
        ).eval().cuda()

        fused.load_from_original(original)

        assert torch.equal(fused.norm_weight.data, original.layer_norm.weight.data)
        assert torch.equal(fused.wi.weight.data, original.mlp.wi.weight.data)
        assert torch.equal(fused.wo.weight.data, original.mlp.wo.weight.data)

    def test_load_from_original_copies_eps(self):
        """Epsilon value is correctly copied from original layer."""
        config = make_config(layer_norm_epsilon=1e-5)
        original = FeedForward(config).eval().cuda()
        fused = FusedFeedForward(
            d_model=config.d_model,
            d_ff=config.d_ff,
        ).eval().cuda()

        fused.load_from_original(original)
        assert fused.eps == 1e-5

    def test_load_preserves_device_and_dtype(self):
        """Loaded weights match device and dtype of original."""
        original = FeedForward(make_config()).eval().cuda().bfloat16()
        fused = FusedFeedForward(d_model=768, d_ff=3072).eval().cuda().bfloat16()

        fused.load_from_original(original)

        assert fused.norm_weight.device == original.layer_norm.weight.device
        assert fused.wi.weight.dtype == original.mlp.wi.weight.dtype


# ---- Edge case tests ----


class TestEdgeCases:
    """Test edge cases and unusual inputs."""

    def test_single_token(self):
        """Works with single-token input (batch=1, seq=1)."""
        torch.manual_seed(42)
        original, fused = make_original_and_fused()
        x = torch.randn(1, 1, 768, device="cuda")

        with torch.no_grad():
            ref = original(x)
            out = fused(x)

        max_err = (ref - out).abs().max().item()
        assert max_err < 1e-5, f"Single token max error: {max_err}"

    def test_large_batch(self):
        """Works with large batch size."""
        torch.manual_seed(42)
        original, fused = make_original_and_fused()
        x = torch.randn(32, 10, 768, device="cuda")

        with torch.no_grad():
            ref = original(x)
            out = fused(x)

        max_err = (ref - out).abs().max().item()
        assert max_err < 1e-5, f"Large batch max error: {max_err}"

    def test_residual_connection(self):
        """Residual connection is correctly applied."""
        torch.manual_seed(42)
        original, fused = make_original_and_fused()

        # Use zeros so residual is easy to verify
        x = torch.zeros(1, 4, 768, device="cuda")
        with torch.no_grad():
            out = fused(x)

        # With zero input: norm output is 0, wi(0)=0, relu(0)=0, wo(0)=0, res=0+0=0
        assert torch.allclose(out, x, atol=1e-7), "Zero input should produce zero output"

    def test_deterministic_across_calls(self):
        """Same input produces same output across multiple calls."""
        torch.manual_seed(42)
        _, fused = make_original_and_fused()
        x = torch.randn(2, 34, 768, device="cuda")

        with torch.no_grad():
            out1 = fused(x.clone())
            out2 = fused(x.clone())

        assert torch.equal(out1, out2), "Outputs should be identical for same input"

    def test_non_standard_dims(self):
        """Works with non-standard d_model and d_ff."""
        config = make_config(d_model=256, d_ff=1024)
        original = FeedForward(config).eval().cuda()
        fused = FusedFeedForward(d_model=256, d_ff=1024, eps=1e-6).eval().cuda()
        fused.load_from_original(original)

        torch.manual_seed(42)
        x = torch.randn(2, 16, 256, device="cuda")

        with torch.no_grad():
            ref = original(x)
            out = fused(x)

        max_err = (ref - out).abs().max().item()
        assert max_err < 1e-5, f"Non-standard dims max error: {max_err}"
