"""Tests for FusedGroupSelfAttention vs original GroupSelfAttention.

Verifies that the fused implementation produces outputs matching the original
within 1e-4 max absolute error across various input shapes and mask patterns.
"""

import sys
import os

import torch
import pytest

# Add project root to path so chronos-forecasting is accessible
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(
    0,
    os.path.join(
        os.path.dirname(__file__), "..", "..", "chronos-forecasting", "src"
    ),
)

from chronos.chronos2.config import Chronos2CoreConfig
from chronos.chronos2.layers import GroupSelfAttention

from cutechronos.modules.group_attention import FusedGroupSelfAttention


def make_config(
    d_model: int = 768,
    d_kv: int = 64,
    num_heads: int = 12,
    dropout_rate: float = 0.0,
    layer_norm_epsilon: float = 1e-6,
    attn_implementation: str = "eager",
) -> Chronos2CoreConfig:
    """Create a Chronos2CoreConfig for testing."""
    return Chronos2CoreConfig(
        d_model=d_model,
        d_kv=d_kv,
        num_heads=num_heads,
        dropout_rate=dropout_rate,
        layer_norm_epsilon=layer_norm_epsilon,
        attn_implementation=attn_implementation,
    )


def build_group_mask(
    time_len: int, batch_size: int, num_heads: int, device: torch.device
) -> torch.Tensor:
    """Build an additive attention mask for group self-attention.

    The mask has shape (time_len, num_heads, batch_size, batch_size).
    All positions are valid (mask = 0), simulating a full-attention scenario
    where all series in the group attend to each other.
    """
    return torch.zeros(
        time_len, num_heads, batch_size, batch_size, device=device
    )


def build_partial_group_mask(
    time_len: int,
    batch_size: int,
    num_heads: int,
    device: torch.device,
    block_size: int = 2,
) -> torch.Tensor:
    """Build a block-diagonal group mask to test masking behavior.

    Groups series into blocks of `block_size`. Series in the same block
    can attend to each other; cross-block attention is masked out (-inf).
    """
    mask = torch.full(
        (time_len, num_heads, batch_size, batch_size),
        float("-inf"),
        device=device,
    )
    for i in range(0, batch_size, block_size):
        end = min(i + block_size, batch_size)
        mask[:, :, i:end, i:end] = 0.0
    return mask


@pytest.fixture(
    params=[
        {"batch": 4, "time": 34, "d_model": 768, "d_kv": 64, "n_heads": 12},
        {"batch": 8, "time": 130, "d_model": 768, "d_kv": 64, "n_heads": 12},
        {"batch": 2, "time": 16, "d_model": 512, "d_kv": 64, "n_heads": 8},
        {"batch": 1, "time": 5, "d_model": 768, "d_kv": 64, "n_heads": 12},
    ],
    ids=["4x34x768", "8x130x768", "2x16x512", "1x5x768"],
)
def shape_config(request):
    return request.param


@pytest.fixture
def device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _run_comparison(
    batch: int,
    time: int,
    d_model: int,
    d_kv: int,
    n_heads: int,
    device: torch.device,
    mask_fn,
    atol: float = 1e-4,
):
    """Run original vs fused and compare outputs."""
    config = make_config(
        d_model=d_model,
        d_kv=d_kv,
        num_heads=n_heads,
        dropout_rate=0.0,
        attn_implementation="eager",
    )

    # Create original layer
    original = GroupSelfAttention(config).to(device).eval()

    # Create fused layer and load weights
    fused = FusedGroupSelfAttention(
        d_model=d_model, n_heads=n_heads, d_kv=d_kv, eps=config.layer_norm_epsilon
    ).to(device).eval()
    fused.load_from_original(original)

    # Random input
    torch.manual_seed(42)
    x = torch.randn(batch, time, d_model, device=device)

    # Build mask
    mask = mask_fn(time, batch, n_heads, device)

    # Run both
    with torch.no_grad():
        orig_out = original(x, attention_mask=mask)
        fused_out = fused(x, attention_mask=mask)

    orig_hidden = orig_out.hidden_states
    max_err = (orig_hidden - fused_out).abs().max().item()

    assert max_err < atol, (
        f"Max abs error {max_err:.6e} exceeds tolerance {atol:.1e} "
        f"for shape ({batch}, {time}, {d_model})"
    )
    return max_err


class TestFusedGroupSelfAttention:
    """Test suite for FusedGroupSelfAttention."""

    def test_basic_shapes(self, shape_config, device):
        """Test various input shapes with full attention mask."""
        max_err = _run_comparison(
            batch=shape_config["batch"],
            time=shape_config["time"],
            d_model=shape_config["d_model"],
            d_kv=shape_config["d_kv"],
            n_heads=shape_config["n_heads"],
            device=device,
            mask_fn=build_group_mask,
        )
        print(
            f"  shape ({shape_config['batch']}, {shape_config['time']}, "
            f"{shape_config['d_model']}): max_err = {max_err:.2e}"
        )

    def test_partial_mask(self, device):
        """Test with block-diagonal partial masking."""
        max_err = _run_comparison(
            batch=6,
            time=20,
            d_model=768,
            d_kv=64,
            n_heads=12,
            device=device,
            mask_fn=lambda t, b, h, d: build_partial_group_mask(
                t, b, h, d, block_size=3
            ),
        )
        print(f"  partial mask (6, 20, 768): max_err = {max_err:.2e}")

    def test_single_series(self, device):
        """Test with batch_size=1 (single series, self-attention only)."""
        max_err = _run_comparison(
            batch=1,
            time=34,
            d_model=768,
            d_kv=64,
            n_heads=12,
            device=device,
            mask_fn=build_group_mask,
        )
        print(f"  single series (1, 34, 768): max_err = {max_err:.2e}")

    def test_weight_loading(self, device):
        """Verify that load_from_original copies all weights correctly."""
        config = make_config()
        original = GroupSelfAttention(config).to(device)
        fused = FusedGroupSelfAttention(
            d_model=768, n_heads=12, d_kv=64, eps=config.layer_norm_epsilon
        ).to(device)
        fused.load_from_original(original)

        assert torch.equal(fused.ln_weight, original.layer_norm.weight)
        assert torch.equal(fused.q_weight, original.self_attention.q.weight)
        assert torch.equal(fused.k_weight, original.self_attention.k.weight)
        assert torch.equal(fused.v_weight, original.self_attention.v.weight)
        assert torch.equal(fused.o_weight, original.self_attention.o.weight)

    def test_no_data_copy_on_transpose(self, device):
        """Verify that transpose uses stride manipulation, not data copy."""
        x = torch.randn(4, 34, 768, device=device)
        xt = x.transpose(0, 1)
        # After transpose, strides should be swapped, not contiguous
        assert xt.stride(0) == x.stride(1)
        assert xt.stride(1) == x.stride(0)
        # data_ptr should be the same (no copy)
        assert xt.data_ptr() == x.data_ptr()

    def test_sdpa_backend(self, device):
        """Test that the fused module works even when original uses SDPA backend."""
        config = make_config(attn_implementation="sdpa")
        original_sdpa = GroupSelfAttention(config).to(device).eval()

        # Build fused from SDPA original (weights are the same regardless of backend)
        fused = FusedGroupSelfAttention(
            d_model=768, n_heads=12, d_kv=64, eps=config.layer_norm_epsilon
        ).to(device).eval()
        fused.load_from_original(original_sdpa)

        # Also build an eager original with the same weights for comparison
        config_eager = make_config(attn_implementation="eager")
        original_eager = GroupSelfAttention(config_eager).to(device).eval()
        with torch.no_grad():
            original_eager.layer_norm.weight.copy_(original_sdpa.layer_norm.weight)
            original_eager.self_attention.q.weight.copy_(
                original_sdpa.self_attention.q.weight
            )
            original_eager.self_attention.k.weight.copy_(
                original_sdpa.self_attention.k.weight
            )
            original_eager.self_attention.v.weight.copy_(
                original_sdpa.self_attention.v.weight
            )
            original_eager.self_attention.o.weight.copy_(
                original_sdpa.self_attention.o.weight
            )

        torch.manual_seed(42)
        x = torch.randn(4, 34, 768, device=device)
        mask = build_group_mask(34, 4, 12, device)

        with torch.no_grad():
            eager_out = original_eager(x, attention_mask=mask).hidden_states
            fused_out = fused(x, attention_mask=mask)

        max_err = (eager_out - fused_out).abs().max().item()
        assert max_err < 1e-4, f"SDPA weight load error: {max_err:.6e}"

    def test_float16(self, device):
        """Test with float16 precision (relaxed tolerance)."""
        if device.type == "cpu":
            pytest.skip("float16 matmul not well supported on CPU")

        config = make_config(dropout_rate=0.0)
        original = GroupSelfAttention(config).to(device).half().eval()
        fused = FusedGroupSelfAttention(
            d_model=768, n_heads=12, d_kv=64, eps=config.layer_norm_epsilon
        ).to(device).half().eval()
        fused.load_from_original(original)

        torch.manual_seed(42)
        x = torch.randn(4, 34, 768, device=device, dtype=torch.float16)
        mask = build_group_mask(34, 4, 12, device)

        with torch.no_grad():
            orig_out = original(x, attention_mask=mask).hidden_states
            fused_out = fused(x, attention_mask=mask)

        max_err = (orig_out - fused_out).abs().max().item()
        # Relaxed tolerance for fp16
        assert max_err < 5e-3, f"fp16 max abs error {max_err:.6e}"

    def test_bfloat16(self, device):
        """Test with bfloat16 precision (relaxed tolerance)."""
        if device.type == "cpu":
            pytest.skip("bfloat16 matmul requires GPU")

        config = make_config(dropout_rate=0.0)
        original = GroupSelfAttention(config).to(device).to(torch.bfloat16).eval()
        fused = FusedGroupSelfAttention(
            d_model=768, n_heads=12, d_kv=64, eps=config.layer_norm_epsilon
        ).to(device).to(torch.bfloat16).eval()
        fused.load_from_original(original)

        torch.manual_seed(42)
        x = torch.randn(4, 34, 768, device=device, dtype=torch.bfloat16)
        mask = build_group_mask(34, 4, 12, device)

        with torch.no_grad():
            orig_out = original(x, attention_mask=mask).hidden_states
            fused_out = fused(x, attention_mask=mask)

        max_err = (orig_out - fused_out).abs().max().item()
        # bf16 has only 7 mantissa bits; Triton tiled attention uses FP32
        # online softmax which differs slightly from PyTorch eager bf16 softmax
        assert max_err < 2e-2, f"bf16 max abs error {max_err:.6e}"
