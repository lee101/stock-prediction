"""Tests for FusedTimeSelfAttention vs original Chronos2 TimeSelfAttention.

Verifies that FusedTimeSelfAttention produces outputs matching the original
within a tight tolerance (max abs error < 1e-4) after loading weights via
load_from_original.
"""

import sys
import os
import pytest
import torch

# Ensure the chronos2 source is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "chronos-forecasting", "src"))

from chronos.chronos2.config import Chronos2CoreConfig
from chronos.chronos2.layers import TimeSelfAttention

from cutechronos.modules.time_attention import FusedTimeSelfAttention


def _make_config(d_model=768, num_heads=12, d_kv=64, attn_implementation="eager"):
    """Build a Chronos2CoreConfig for testing."""
    return Chronos2CoreConfig(
        d_model=d_model,
        d_kv=d_kv,
        num_heads=num_heads,
        dropout_rate=0.0,
        layer_norm_epsilon=1e-6,
        rope_theta=10000.0,
        attn_implementation=attn_implementation,
    )


def _build_pair(config):
    """Build an original and a fused layer sharing the same weights."""
    original = TimeSelfAttention(config).eval()
    fused = FusedTimeSelfAttention(
        d_model=config.d_model,
        num_heads=config.num_heads,
        d_kv=config.d_kv,
        layer_norm_eps=config.layer_norm_epsilon,
        rope_theta=config.rope_theta,
    ).eval()
    fused.load_from_original(original)
    return original, fused


def _make_inputs(batch_size, seq_len, d_model, num_heads, device="cpu"):
    """Generate random hidden_states, attention_mask, position_ids."""
    hidden_states = torch.randn(batch_size, seq_len, d_model, device=device)
    # Causal mask: 0 for attend, -inf for masked (upper triangle)
    mask = torch.zeros(batch_size, num_heads, seq_len, seq_len, device=device)
    causal = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1).bool()
    mask[:, :, causal] = float("-inf")
    position_ids = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
    return hidden_states, mask, position_ids


@pytest.mark.parametrize(
    "batch_size,seq_len",
    [(1, 34), (4, 130), (2, 1), (1, 512)],
    ids=["B1_S34", "B4_S130", "B2_S1", "B1_S512"],
)
def test_fused_matches_original_cpu(batch_size, seq_len):
    config = _make_config()
    original, fused = _build_pair(config)
    hidden_states, mask, position_ids = _make_inputs(
        batch_size, seq_len, config.d_model, config.num_heads
    )

    with torch.no_grad():
        orig_out = original(hidden_states, mask, position_ids)
        fused_out = fused(hidden_states, mask, position_ids)

    orig_hs = orig_out.hidden_states
    fused_hs = fused_out[0]

    max_err = (orig_hs - fused_hs).abs().max().item()
    assert max_err < 1e-4, f"Max abs error {max_err:.2e} >= 1e-4"


@pytest.mark.parametrize(
    "batch_size,seq_len",
    [(1, 34), (4, 130)],
    ids=["B1_S34", "B4_S130"],
)
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_fused_matches_original_cuda(batch_size, seq_len):
    device = "cuda"
    config = _make_config()
    original, fused = _build_pair(config)
    original = original.to(device)
    fused = fused.to(device)
    hidden_states, mask, position_ids = _make_inputs(
        batch_size, seq_len, config.d_model, config.num_heads, device=device
    )

    with torch.no_grad():
        orig_out = original(hidden_states, mask, position_ids)
        fused_out = fused(hidden_states, mask, position_ids)

    orig_hs = orig_out.hidden_states
    fused_hs = fused_out[0]

    max_err = (orig_hs - fused_hs).abs().max().item()
    assert max_err < 1e-4, f"Max abs error {max_err:.2e} >= 1e-4"


def test_output_shapes():
    """Verify output tensor shapes match expectations."""
    config = _make_config()
    _, fused = _build_pair(config)

    for B, S in [(1, 34), (4, 130)]:
        hs, mask, pos = _make_inputs(B, S, config.d_model, config.num_heads)
        with torch.no_grad():
            out, attn_w = fused(hs, mask, pos)
        assert out.shape == (B, S, config.d_model), f"Wrong shape: {out.shape}"
        assert attn_w is None, "attn_weights should be None at inference"


def test_load_from_original_copies_all_weights():
    """Ensure load_from_original copies every parameter exactly."""
    config = _make_config()
    original, fused = _build_pair(config)

    # Check layer norm
    assert torch.equal(fused.layer_norm_weight, original.layer_norm.weight)
    # Check projections
    assert torch.equal(fused.q.weight, original.self_attention.q.weight)
    assert torch.equal(fused.k.weight, original.self_attention.k.weight)
    assert torch.equal(fused.v.weight, original.self_attention.v.weight)
    assert torch.equal(fused.o.weight, original.self_attention.o.weight)
    # Check RoPE inv_freq
    assert torch.equal(fused.inv_freq, original.self_attention.rope_embed.inv_freq)


def test_residual_connection():
    """Output should differ from input due to attention, but the residual path
    should be present (output is NOT just the attention output)."""
    config = _make_config()
    _, fused = _build_pair(config)
    hs, mask, pos = _make_inputs(1, 10, config.d_model, config.num_heads)

    with torch.no_grad():
        out, _ = fused(hs, mask, pos)

    # Output should not be identical to input (attention changes it)
    assert not torch.allclose(out, hs, atol=1e-6), "Output should differ from input"

    # But difference should be bounded (residual keeps it close)
    diff = (out - hs).abs().max().item()
    assert diff < 100, f"Residual difference too large: {diff}"


def test_deterministic():
    """Two forward passes with the same input should give identical results."""
    config = _make_config()
    _, fused = _build_pair(config)
    hs, mask, pos = _make_inputs(2, 50, config.d_model, config.num_heads)

    with torch.no_grad():
        out1, _ = fused(hs, mask, pos)
        out2, _ = fused(hs, mask, pos)

    assert torch.equal(out1, out2), "Forward pass should be deterministic"
