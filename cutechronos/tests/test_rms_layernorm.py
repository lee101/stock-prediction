"""Tests for Triton RMS LayerNorm kernel.

Validates numerical equivalence against the reference PyTorch implementation
(Chronos2LayerNorm) across multiple shapes and dtypes.
"""

import pytest
import torch

from cutechronos.triton_kernels.rms_layernorm import rms_layernorm, TritonRMSNorm


def reference_rms_layernorm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Reference implementation matching Chronos2LayerNorm.forward."""
    variance = x.to(torch.float32).pow(2).mean(-1, keepdim=True)
    normed = x * torch.rsqrt(variance + eps)
    if weight.dtype in (torch.float16, torch.bfloat16):
        normed = normed.to(weight.dtype)
    return weight * normed


SHAPES = [
    (1, 34, 768),
    (4, 130, 768),
    (16, 514, 768),
]


@pytest.mark.parametrize("shape", SHAPES, ids=[f"{'x'.join(map(str, s))}" for s in SHAPES])
def test_rms_layernorm_fp32(shape):
    torch.manual_seed(42)
    x = torch.randn(shape, device="cuda", dtype=torch.float32)
    w = torch.randn(shape[-1], device="cuda", dtype=torch.float32)

    ref = reference_rms_layernorm(x, w)
    out = rms_layernorm(x, w)

    max_err = (out - ref).abs().max().item()
    assert max_err < 1e-5, f"FP32 max abs error {max_err} >= 1e-5"


@pytest.mark.parametrize("shape", SHAPES, ids=[f"{'x'.join(map(str, s))}" for s in SHAPES])
def test_rms_layernorm_bf16(shape):
    torch.manual_seed(42)
    x = torch.randn(shape, device="cuda", dtype=torch.bfloat16)
    w = torch.randn(shape[-1], device="cuda", dtype=torch.bfloat16)

    ref = reference_rms_layernorm(x, w)
    out = rms_layernorm(x, w)

    # BF16 has ~8 bits of mantissa. Triton and PyTorch use different reduction
    # tree orders for sum, so intermediate FP32 values can differ at the LSB,
    # causing rare 1-ULP BF16 rounding differences. Use both absolute and
    # relative tolerance (torch.allclose semantics: |a-b| <= atol + rtol*|b|).
    assert torch.allclose(out, ref, atol=5e-3, rtol=1e-2), (
        f"BF16 mismatch: max abs error {(out - ref).abs().max().item()}"
    )


@pytest.mark.parametrize("shape", SHAPES, ids=[f"{'x'.join(map(str, s))}" for s in SHAPES])
def test_triton_rms_norm_module_fp32(shape):
    """Test the nn.Module wrapper with FP32 inputs."""
    torch.manual_seed(42)
    hidden_size = shape[-1]
    x = torch.randn(shape, device="cuda", dtype=torch.float32)

    module = TritonRMSNorm(hidden_size).cuda()
    ref = reference_rms_layernorm(x, module.weight.data)
    out = module(x)

    max_err = (out - ref).abs().max().item()
    assert max_err < 1e-5, f"Module FP32 max abs error {max_err} >= 1e-5"


@pytest.mark.parametrize("shape", SHAPES, ids=[f"{'x'.join(map(str, s))}" for s in SHAPES])
def test_triton_rms_norm_module_bf16(shape):
    """Test the nn.Module wrapper with BF16 inputs and weights."""
    torch.manual_seed(42)
    hidden_size = shape[-1]
    x = torch.randn(shape, device="cuda", dtype=torch.bfloat16)

    module = TritonRMSNorm(hidden_size).to(torch.bfloat16).cuda()
    ref = reference_rms_layernorm(x, module.weight.data)
    out = module(x)

    assert torch.allclose(out, ref, atol=5e-3, rtol=1e-2), (
        f"Module BF16 mismatch: max abs error {(out - ref).abs().max().item()}"
    )


def test_output_dtype_preserved_fp32():
    """Output dtype must match input dtype for FP32."""
    x = torch.randn(2, 10, 768, device="cuda", dtype=torch.float32)
    w = torch.randn(768, device="cuda", dtype=torch.float32)
    out = rms_layernorm(x, w)
    assert out.dtype == torch.float32


def test_output_dtype_preserved_bf16():
    """Output dtype must match input dtype for BF16."""
    x = torch.randn(2, 10, 768, device="cuda", dtype=torch.bfloat16)
    w = torch.randn(768, device="cuda", dtype=torch.bfloat16)
    out = rms_layernorm(x, w)
    assert out.dtype == torch.bfloat16


def test_output_shape_preserved():
    """Output shape must match input shape exactly."""
    for shape in SHAPES:
        x = torch.randn(shape, device="cuda")
        w = torch.randn(shape[-1], device="cuda")
        out = rms_layernorm(x, w)
        assert out.shape == x.shape, f"Shape mismatch: {out.shape} != {x.shape}"


def test_zero_input():
    """RMS norm of zero vector should be zero (0 * rsqrt(eps) * w = 0 ... actually = 0)."""
    x = torch.zeros(1, 5, 768, device="cuda")
    w = torch.ones(768, device="cuda")
    out = rms_layernorm(x, w)
    assert out.abs().max().item() == 0.0


def test_gradient_flow():
    """Verify gradients flow through the module."""
    module = TritonRMSNorm(768).cuda()
    x = torch.randn(2, 10, 768, device="cuda", requires_grad=True)
    out = module(x)
    loss = out.sum()
    loss.backward()
    assert x.grad is not None
    assert x.grad.shape == x.shape
    assert module.weight.grad is not None
