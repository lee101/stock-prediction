"""Tests for the fused output transform kernel and FusedOutputHead module.

Verifies that:
1. The Triton kernel for fused rearrange + sinh + unscale matches the
   reference PyTorch implementation.
2. The full FusedOutputHead produces identical results to the original
   ResidualBlock + rearrange + inverse instance norm pipeline.
"""

import pytest
import torch

from cutechronos.triton_kernels.fused_output import fused_output_transform
from cutechronos.modules.output import FusedOutputHead, _output_transform_fallback


# ------------------------------------------------------------------ #
# Reference implementations                                          #
# ------------------------------------------------------------------ #

def reference_output_transform(
    x: torch.Tensor,
    loc: torch.Tensor,
    scale: torch.Tensor,
    num_quantiles: int = 21,
    patch_size: int = 16,
    use_arcsinh: bool = True,
) -> torch.Tensor:
    """Reference rearrange + sinh + unscale using einops-style view/permute."""
    B, N, QP = x.shape
    Q = num_quantiles
    P = patch_size
    orig_dtype = x.dtype

    # rearrange "b n (q p) -> b q (n p)"
    x = x.view(B, N, Q, P).permute(0, 2, 1, 3).reshape(B, Q, N * P)

    # inverse instance norm in FP32, cast back to orig dtype
    x = x.float()
    if use_arcsinh:
        x = torch.sinh(x)
    x = x * scale.unsqueeze(1) + loc.unsqueeze(1)

    return x.to(orig_dtype)


class OriginalResidualBlock(torch.nn.Module):
    """Minimal replica of Chronos2 ResidualBlock for testing."""

    def __init__(self, in_dim: int, h_dim: int, out_dim: int):
        super().__init__()
        self.hidden_layer = torch.nn.Linear(in_dim, h_dim)
        self.output_layer = torch.nn.Linear(h_dim, out_dim)
        self.residual_layer = torch.nn.Linear(in_dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hid = torch.relu(self.hidden_layer(x))
        out = self.output_layer(hid)
        res = self.residual_layer(x)
        return out + res


# ------------------------------------------------------------------ #
# Test shapes                                                         #
# ------------------------------------------------------------------ #

# (B, P_out, Q*P) where Q=21, P=16
TRANSFORM_SHAPES = [
    (1, 2, 336),    # minimal: 1 batch, 2 output patches
    (4, 4, 336),    # moderate batch, 4 output patches
    (16, 8, 336),   # large batch, 8 output patches
]


# ------------------------------------------------------------------ #
# Tests: fused output transform kernel                                #
# ------------------------------------------------------------------ #

@pytest.mark.parametrize("B,N,QP", TRANSFORM_SHAPES)
def test_fused_transform_with_sinh(B: int, N: int, QP: int):
    """Verify fused kernel matches reference with sinh (use_arcsinh=True)."""
    device = "cuda"
    torch.manual_seed(42)

    x = torch.randn(B, N, QP, device=device, dtype=torch.float32)
    loc = torch.randn(B, 1, device=device, dtype=torch.float32)
    scale = torch.rand(B, 1, device=device, dtype=torch.float32) + 0.1

    ref = reference_output_transform(x, loc, scale, use_arcsinh=True)
    out = fused_output_transform(x, loc, scale, use_arcsinh=True)

    max_err = (ref - out).abs().max().item()
    assert max_err < 1e-5, (
        f"Max abs error {max_err:.6e} exceeds 1e-5 for shape ({B},{N},{QP}) with sinh"
    )


@pytest.mark.parametrize("B,N,QP", TRANSFORM_SHAPES)
def test_fused_transform_without_sinh(B: int, N: int, QP: int):
    """Verify fused kernel matches reference without sinh (use_arcsinh=False)."""
    device = "cuda"
    torch.manual_seed(123)

    x = torch.randn(B, N, QP, device=device, dtype=torch.float32)
    loc = torch.randn(B, 1, device=device, dtype=torch.float32)
    scale = torch.rand(B, 1, device=device, dtype=torch.float32) + 0.1

    ref = reference_output_transform(x, loc, scale, use_arcsinh=False)
    out = fused_output_transform(x, loc, scale, use_arcsinh=False)

    max_err = (ref - out).abs().max().item()
    assert max_err < 1e-5, (
        f"Max abs error {max_err:.6e} exceeds 1e-5 for shape ({B},{N},{QP}) without sinh"
    )


@pytest.mark.parametrize("B,N,QP", TRANSFORM_SHAPES)
def test_fused_transform_output_shape(B: int, N: int, QP: int):
    """Verify output shape is (B, Q, N*P)."""
    device = "cuda"
    Q, P = 21, 16
    torch.manual_seed(0)

    x = torch.randn(B, N, QP, device=device, dtype=torch.float32)
    loc = torch.zeros(B, 1, device=device, dtype=torch.float32)
    scale = torch.ones(B, 1, device=device, dtype=torch.float32)

    out = fused_output_transform(x, loc, scale)
    expected_shape = (B, Q, N * P)
    assert out.shape == expected_shape, f"Shape mismatch: {out.shape} vs {expected_shape}"


def test_fused_transform_bfloat16_input():
    """Verify kernel handles bfloat16 input correctly."""
    device = "cuda"
    B, N, QP = 4, 4, 336
    torch.manual_seed(77)

    x = torch.randn(B, N, QP, device=device, dtype=torch.bfloat16)
    loc = torch.randn(B, 1, device=device, dtype=torch.float32)
    scale = torch.rand(B, 1, device=device, dtype=torch.float32) + 0.1

    ref = reference_output_transform(x, loc, scale, use_arcsinh=True)
    out = fused_output_transform(x, loc, scale, use_arcsinh=True)

    # bf16 input has inherent quantization; sinh amplifies differences.
    # Use relative tolerance for numerically large outputs.
    max_err = (ref - out).abs().max().item()
    rel_err = ((ref - out).abs() / (ref.abs() + 1e-6)).max().item()
    assert rel_err < 5e-3 or max_err < 5e-2, (
        f"Max abs error {max_err:.6e}, max rel error {rel_err:.6e} "
        f"exceed tolerance for bfloat16 input"
    )


def test_fused_transform_deterministic():
    """Verify the kernel produces identical results on repeated calls."""
    device = "cuda"
    B, N, QP = 4, 4, 336
    torch.manual_seed(99)

    x = torch.randn(B, N, QP, device=device, dtype=torch.float32)
    loc = torch.randn(B, 1, device=device, dtype=torch.float32)
    scale = torch.rand(B, 1, device=device, dtype=torch.float32) + 0.1

    out1 = fused_output_transform(x, loc, scale)
    out2 = fused_output_transform(x, loc, scale)

    assert torch.equal(out1, out2), "Kernel is not deterministic"


def test_fallback_matches_reference():
    """Verify the PyTorch fallback matches the reference implementation."""
    device = "cuda"
    B, N, QP = 4, 4, 336
    torch.manual_seed(42)

    x = torch.randn(B, N, QP, device=device, dtype=torch.float32)
    loc = torch.randn(B, 1, device=device, dtype=torch.float32)
    scale = torch.rand(B, 1, device=device, dtype=torch.float32) + 0.1

    ref = reference_output_transform(x, loc, scale, use_arcsinh=True)
    fb = _output_transform_fallback(x, loc, scale, use_arcsinh=True)

    max_err = (ref - fb).abs().max().item()
    assert max_err < 1e-6, f"Fallback max abs error {max_err:.6e} exceeds 1e-6"


# ------------------------------------------------------------------ #
# Tests: FusedOutputHead end-to-end                                   #
# ------------------------------------------------------------------ #

@pytest.mark.parametrize("B,N", [(1, 2), (4, 4), (16, 8)])
def test_fused_output_head_matches_original(B: int, N: int):
    """Verify FusedOutputHead produces same results as original pipeline."""
    device = "cuda"
    in_dim = 768
    hidden_dim = 3072
    Q, P = 21, 16
    out_dim = Q * P
    torch.manual_seed(42)

    # Create original ResidualBlock with random weights
    original = OriginalResidualBlock(in_dim, hidden_dim, out_dim).to(device)

    # Create FusedOutputHead and load weights
    fused = FusedOutputHead(
        in_dim=in_dim,
        hidden_dim=hidden_dim,
        out_dim=out_dim,
        num_quantiles=Q,
        patch_size=P,
    ).to(device)
    fused.load_from_original(original)

    # Random inputs
    hidden_states = torch.randn(B, N, in_dim, device=device, dtype=torch.float32)
    loc = torch.randn(B, 1, device=device, dtype=torch.float32)
    scale = torch.rand(B, 1, device=device, dtype=torch.float32) + 0.1

    # Original pipeline
    with torch.no_grad():
        resblock_out = original(hidden_states)
        ref = reference_output_transform(resblock_out, loc, scale, Q, P, use_arcsinh=True)

    # Fused pipeline
    with torch.no_grad():
        fused_out = fused(hidden_states, loc, scale, use_arcsinh=True)

    max_err = (ref - fused_out).abs().max().item()
    assert max_err < 1e-5, (
        f"Max abs error {max_err:.6e} exceeds 1e-5 for FusedOutputHead "
        f"with B={B}, N={N}"
    )


@pytest.mark.parametrize("B,N", [(1, 2), (4, 4)])
def test_fused_output_head_no_arcsinh(B: int, N: int):
    """Verify FusedOutputHead works correctly without arcsinh."""
    device = "cuda"
    in_dim = 768
    hidden_dim = 3072
    Q, P = 21, 16
    out_dim = Q * P
    torch.manual_seed(7)

    original = OriginalResidualBlock(in_dim, hidden_dim, out_dim).to(device)
    fused = FusedOutputHead(
        in_dim=in_dim,
        hidden_dim=hidden_dim,
        out_dim=out_dim,
        num_quantiles=Q,
        patch_size=P,
    ).to(device)
    fused.load_from_original(original)

    hidden_states = torch.randn(B, N, in_dim, device=device, dtype=torch.float32)
    loc = torch.randn(B, 1, device=device, dtype=torch.float32)
    scale = torch.rand(B, 1, device=device, dtype=torch.float32) + 0.1

    with torch.no_grad():
        resblock_out = original(hidden_states)
        ref = reference_output_transform(resblock_out, loc, scale, Q, P, use_arcsinh=False)

    with torch.no_grad():
        fused_out = fused(hidden_states, loc, scale, use_arcsinh=False)

    max_err = (ref - fused_out).abs().max().item()
    assert max_err < 1e-5, (
        f"Max abs error {max_err:.6e} exceeds 1e-5 for no-arcsinh path"
    )


def test_fused_output_head_output_shape():
    """Verify FusedOutputHead returns correct shape."""
    device = "cuda"
    B, N = 4, 4
    in_dim = 768
    hidden_dim = 3072
    Q, P = 21, 16
    out_dim = Q * P

    fused = FusedOutputHead(
        in_dim=in_dim,
        hidden_dim=hidden_dim,
        out_dim=out_dim,
        num_quantiles=Q,
        patch_size=P,
    ).to(device)

    hidden_states = torch.randn(B, N, in_dim, device=device, dtype=torch.float32)
    loc = torch.zeros(B, 1, device=device, dtype=torch.float32)
    scale = torch.ones(B, 1, device=device, dtype=torch.float32)

    with torch.no_grad():
        out = fused(hidden_states, loc, scale)

    expected_shape = (B, Q, N * P)
    assert out.shape == expected_shape, f"Shape mismatch: {out.shape} vs {expected_shape}"


def test_load_from_original_copies_all_weights():
    """Verify load_from_original copies all parameters correctly."""
    device = "cuda"
    in_dim = 768
    hidden_dim = 3072
    Q, P = 21, 16
    out_dim = Q * P
    torch.manual_seed(42)

    original = OriginalResidualBlock(in_dim, hidden_dim, out_dim).to(device)
    fused = FusedOutputHead(
        in_dim=in_dim,
        hidden_dim=hidden_dim,
        out_dim=out_dim,
        num_quantiles=Q,
        patch_size=P,
    ).to(device)

    # Before loading, weights should differ
    fused.load_from_original(original)

    # After loading, all weights should match
    assert torch.equal(fused.hidden_layer.weight, original.hidden_layer.weight)
    assert torch.equal(fused.hidden_layer.bias, original.hidden_layer.bias)
    assert torch.equal(fused.output_layer.weight, original.output_layer.weight)
    assert torch.equal(fused.output_layer.bias, original.output_layer.bias)
    assert torch.equal(fused.residual_layer.weight, original.residual_layer.weight)
    assert torch.equal(fused.residual_layer.bias, original.residual_layer.bias)


def test_fused_transform_large_values():
    """Verify kernel handles large input values (sinh can overflow)."""
    device = "cuda"
    B, N, QP = 2, 2, 336
    torch.manual_seed(42)

    # Use values in a reasonable range (sinh overflows at ~89 for fp32)
    x = torch.randn(B, N, QP, device=device, dtype=torch.float32) * 5.0
    loc = torch.randn(B, 1, device=device, dtype=torch.float32)
    scale = torch.rand(B, 1, device=device, dtype=torch.float32) + 0.1

    ref = reference_output_transform(x, loc, scale, use_arcsinh=True)
    out = fused_output_transform(x, loc, scale, use_arcsinh=True)

    # sinh amplifies large inputs exponentially, so use relative tolerance.
    finite_mask = torch.isfinite(ref) & torch.isfinite(out)
    if finite_mask.any():
        abs_err = (ref[finite_mask] - out[finite_mask]).abs()
        rel_err = abs_err / (ref[finite_mask].abs() + 1e-6)
        max_rel_err = rel_err.max().item()
        assert max_rel_err < 1e-5, (
            f"Max relative error {max_rel_err:.6e} exceeds 1e-5 for large values"
        )
