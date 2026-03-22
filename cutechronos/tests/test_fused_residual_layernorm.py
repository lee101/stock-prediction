"""Tests for Triton fused residual-add + RMS LayerNorm kernel.

Validates numerical equivalence against the reference two-step approach:
    x = residual + delta
    normed = rms_layernorm(x, weight, eps)

Tests both the out-of-place and in-place variants across multiple shapes and dtypes.
"""

import pytest
import torch

from cutechronos.triton_kernels.fused_residual_layernorm import (
    fused_residual_layernorm,
    fused_residual_layernorm_inplace,
)


def reference_rms_layernorm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Reference implementation matching Chronos2LayerNorm.forward (T5-style)."""
    variance = x.to(torch.float32).pow(2).mean(-1, keepdim=True)
    normed = x * torch.rsqrt(variance + eps)
    if weight.dtype in (torch.float16, torch.bfloat16):
        normed = normed.to(weight.dtype)
    return weight * normed


def reference_fused(
    residual: torch.Tensor, delta: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reference two-step implementation."""
    x = residual + delta
    normed = reference_rms_layernorm(x, weight, eps)
    return x, normed


# Test shapes: (rows, D=768) — matching the task spec
SHAPES_2D = [
    (34, 768),
    (520, 768),
    (4160, 768),
]

# Additional 3D shapes for testing reshape handling
SHAPES_3D = [
    (1, 34, 768),
    (4, 130, 768),
    (16, 260, 768),
]


# ---------------------------------------------------------------------------
# Out-of-place variant tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("shape", SHAPES_2D, ids=[f"{'x'.join(map(str, s))}" for s in SHAPES_2D])
def test_fused_residual_layernorm_fp32_2d(shape):
    """FP32 out-of-place, 2D shapes."""
    torch.manual_seed(42)
    residual = torch.randn(shape, device="cuda", dtype=torch.float32)
    delta = torch.randn(shape, device="cuda", dtype=torch.float32)
    w = torch.randn(shape[-1], device="cuda", dtype=torch.float32)

    ref_sum, ref_normed = reference_fused(residual, delta, w)
    out_sum, out_normed = fused_residual_layernorm(residual, delta, w)

    max_err_sum = (out_sum - ref_sum).abs().max().item()
    max_err_normed = (out_normed - ref_normed).abs().max().item()
    assert max_err_sum < 1e-5, f"FP32 sum max abs error {max_err_sum} >= 1e-5"
    assert max_err_normed < 1e-5, f"FP32 normed max abs error {max_err_normed} >= 1e-5"


@pytest.mark.parametrize("shape", SHAPES_2D, ids=[f"{'x'.join(map(str, s))}" for s in SHAPES_2D])
def test_fused_residual_layernorm_bf16_2d(shape):
    """BF16 out-of-place, 2D shapes."""
    torch.manual_seed(42)
    residual = torch.randn(shape, device="cuda", dtype=torch.bfloat16)
    delta = torch.randn(shape, device="cuda", dtype=torch.bfloat16)
    w = torch.randn(shape[-1], device="cuda", dtype=torch.bfloat16)

    ref_sum, ref_normed = reference_fused(residual, delta, w)
    out_sum, out_normed = fused_residual_layernorm(residual, delta, w)

    # BF16 sum should be exact (just addition)
    assert torch.allclose(out_sum, ref_sum, atol=1e-3, rtol=1e-2), (
        f"BF16 sum mismatch: max abs error {(out_sum - ref_sum).abs().max().item()}"
    )
    # BF16 normed has reduction tree differences
    assert torch.allclose(out_normed, ref_normed, atol=5e-3, rtol=1e-2), (
        f"BF16 normed mismatch: max abs error {(out_normed - ref_normed).abs().max().item()}"
    )


@pytest.mark.parametrize("shape", SHAPES_3D, ids=[f"{'x'.join(map(str, s))}" for s in SHAPES_3D])
def test_fused_residual_layernorm_fp32_3d(shape):
    """FP32 out-of-place, 3D shapes (batch x seq x hidden)."""
    torch.manual_seed(42)
    residual = torch.randn(shape, device="cuda", dtype=torch.float32)
    delta = torch.randn(shape, device="cuda", dtype=torch.float32)
    w = torch.randn(shape[-1], device="cuda", dtype=torch.float32)

    ref_sum, ref_normed = reference_fused(residual, delta, w)
    out_sum, out_normed = fused_residual_layernorm(residual, delta, w)

    max_err_sum = (out_sum - ref_sum).abs().max().item()
    max_err_normed = (out_normed - ref_normed).abs().max().item()
    assert max_err_sum < 1e-5, f"FP32 3D sum max abs error {max_err_sum} >= 1e-5"
    assert max_err_normed < 1e-5, f"FP32 3D normed max abs error {max_err_normed} >= 1e-5"


@pytest.mark.parametrize("shape", SHAPES_3D, ids=[f"{'x'.join(map(str, s))}" for s in SHAPES_3D])
def test_fused_residual_layernorm_bf16_3d(shape):
    """BF16 out-of-place, 3D shapes."""
    torch.manual_seed(42)
    residual = torch.randn(shape, device="cuda", dtype=torch.bfloat16)
    delta = torch.randn(shape, device="cuda", dtype=torch.bfloat16)
    w = torch.randn(shape[-1], device="cuda", dtype=torch.bfloat16)

    ref_sum, ref_normed = reference_fused(residual, delta, w)
    out_sum, out_normed = fused_residual_layernorm(residual, delta, w)

    assert torch.allclose(out_sum, ref_sum, atol=1e-3, rtol=1e-2), (
        f"BF16 3D sum mismatch: max abs error {(out_sum - ref_sum).abs().max().item()}"
    )
    assert torch.allclose(out_normed, ref_normed, atol=5e-3, rtol=1e-2), (
        f"BF16 3D normed mismatch: max abs error {(out_normed - ref_normed).abs().max().item()}"
    )


# ---------------------------------------------------------------------------
# In-place variant tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("shape", SHAPES_2D, ids=[f"{'x'.join(map(str, s))}" for s in SHAPES_2D])
def test_fused_residual_layernorm_inplace_fp32(shape):
    """FP32 in-place variant."""
    torch.manual_seed(42)
    residual = torch.randn(shape, device="cuda", dtype=torch.float32)
    delta = torch.randn(shape, device="cuda", dtype=torch.float32)
    w = torch.randn(shape[-1], device="cuda", dtype=torch.float32)

    ref_sum, ref_normed = reference_fused(residual.clone(), delta, w)

    # In-place modifies residual
    residual_copy = residual.clone()
    out_normed = fused_residual_layernorm_inplace(residual_copy, delta, w)

    max_err_sum = (residual_copy - ref_sum).abs().max().item()
    max_err_normed = (out_normed - ref_normed).abs().max().item()
    assert max_err_sum < 1e-5, f"FP32 in-place sum max abs error {max_err_sum} >= 1e-5"
    assert max_err_normed < 1e-5, f"FP32 in-place normed max abs error {max_err_normed} >= 1e-5"


@pytest.mark.parametrize("shape", SHAPES_2D, ids=[f"{'x'.join(map(str, s))}" for s in SHAPES_2D])
def test_fused_residual_layernorm_inplace_bf16(shape):
    """BF16 in-place variant."""
    torch.manual_seed(42)
    residual = torch.randn(shape, device="cuda", dtype=torch.bfloat16)
    delta = torch.randn(shape, device="cuda", dtype=torch.bfloat16)
    w = torch.randn(shape[-1], device="cuda", dtype=torch.bfloat16)

    ref_sum, ref_normed = reference_fused(residual.clone(), delta, w)

    residual_copy = residual.clone()
    out_normed = fused_residual_layernorm_inplace(residual_copy, delta, w)

    assert torch.allclose(residual_copy, ref_sum, atol=1e-3, rtol=1e-2), (
        f"BF16 in-place sum mismatch: max abs error {(residual_copy - ref_sum).abs().max().item()}"
    )
    assert torch.allclose(out_normed, ref_normed, atol=5e-3, rtol=1e-2), (
        f"BF16 in-place normed mismatch: max abs error {(out_normed - ref_normed).abs().max().item()}"
    )


# ---------------------------------------------------------------------------
# Output dtype and shape preservation
# ---------------------------------------------------------------------------


def test_output_dtype_preserved_fp32():
    """Output dtypes must match input dtype for FP32."""
    residual = torch.randn(34, 768, device="cuda", dtype=torch.float32)
    delta = torch.randn(34, 768, device="cuda", dtype=torch.float32)
    w = torch.randn(768, device="cuda", dtype=torch.float32)

    out_sum, out_normed = fused_residual_layernorm(residual, delta, w)
    assert out_sum.dtype == torch.float32
    assert out_normed.dtype == torch.float32


def test_output_dtype_preserved_bf16():
    """Output dtypes must match input dtype for BF16."""
    residual = torch.randn(34, 768, device="cuda", dtype=torch.bfloat16)
    delta = torch.randn(34, 768, device="cuda", dtype=torch.bfloat16)
    w = torch.randn(768, device="cuda", dtype=torch.bfloat16)

    out_sum, out_normed = fused_residual_layernorm(residual, delta, w)
    assert out_sum.dtype == torch.bfloat16
    assert out_normed.dtype == torch.bfloat16


def test_output_shape_preserved():
    """Output shapes must match input shape exactly."""
    for shape in SHAPES_2D + SHAPES_3D:
        residual = torch.randn(shape, device="cuda")
        delta = torch.randn(shape, device="cuda")
        w = torch.randn(shape[-1], device="cuda")

        out_sum, out_normed = fused_residual_layernorm(residual, delta, w)
        assert out_sum.shape == residual.shape, f"Sum shape mismatch: {out_sum.shape} != {residual.shape}"
        assert out_normed.shape == residual.shape, f"Normed shape mismatch: {out_normed.shape} != {residual.shape}"


def test_inplace_output_shape_preserved():
    """In-place variant output shape must match input shape."""
    for shape in SHAPES_2D + SHAPES_3D:
        residual = torch.randn(shape, device="cuda")
        delta = torch.randn(shape, device="cuda")
        w = torch.randn(shape[-1], device="cuda")

        out_normed = fused_residual_layernorm_inplace(residual, delta, w)
        assert out_normed.shape == residual.shape, f"Normed shape mismatch: {out_normed.shape} != {residual.shape}"


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


def test_zero_delta():
    """When delta is zero, sum should equal residual and normed should match standalone rms_norm."""
    torch.manual_seed(42)
    residual = torch.randn(34, 768, device="cuda", dtype=torch.float32)
    delta = torch.zeros_like(residual)
    w = torch.randn(768, device="cuda", dtype=torch.float32)

    out_sum, out_normed = fused_residual_layernorm(residual, delta, w)

    # Sum should be just residual
    assert (out_sum - residual).abs().max().item() < 1e-6

    # Normed should match standalone rms_layernorm
    ref_normed = reference_rms_layernorm(residual, w)
    max_err = (out_normed - ref_normed).abs().max().item()
    assert max_err < 1e-5, f"Zero delta normed error {max_err} >= 1e-5"


def test_zero_residual():
    """When residual is zero, sum should equal delta."""
    torch.manual_seed(42)
    residual = torch.zeros(34, 768, device="cuda", dtype=torch.float32)
    delta = torch.randn(34, 768, device="cuda", dtype=torch.float32)
    w = torch.randn(768, device="cuda", dtype=torch.float32)

    out_sum, out_normed = fused_residual_layernorm(residual, delta, w)

    assert (out_sum - delta).abs().max().item() < 1e-6

    ref_normed = reference_rms_layernorm(delta, w)
    max_err = (out_normed - ref_normed).abs().max().item()
    assert max_err < 1e-5, f"Zero residual normed error {max_err} >= 1e-5"


def test_both_zero():
    """When both residual and delta are zero, outputs should be zero."""
    residual = torch.zeros(34, 768, device="cuda")
    delta = torch.zeros(34, 768, device="cuda")
    w = torch.ones(768, device="cuda")

    out_sum, out_normed = fused_residual_layernorm(residual, delta, w)

    assert out_sum.abs().max().item() == 0.0
    assert out_normed.abs().max().item() == 0.0


def test_consistency_between_variants():
    """Out-of-place and in-place variants must produce identical results."""
    torch.manual_seed(42)
    residual = torch.randn(520, 768, device="cuda", dtype=torch.float32)
    delta = torch.randn(520, 768, device="cuda", dtype=torch.float32)
    w = torch.randn(768, device="cuda", dtype=torch.float32)

    # Out-of-place
    out_sum, out_normed = fused_residual_layernorm(residual, delta, w)

    # In-place
    residual_copy = residual.clone()
    inplace_normed = fused_residual_layernorm_inplace(residual_copy, delta, w)

    assert (residual_copy - out_sum).abs().max().item() < 1e-6, "Sum differs between variants"
    assert (inplace_normed - out_normed).abs().max().item() < 1e-6, "Normed differs between variants"
