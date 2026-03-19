"""Tests for Triton fused RMS LayerNorm + Linear kernel.

Validates numerical equivalence against sequential
F.linear(rms_layernorm(x, w, eps), linear_w) across multiple shapes and dtypes.
"""

import pytest
import torch
import torch.nn.functional as F

from cutechronos.triton_kernels.fused_layernorm_linear import (
    fused_rms_norm_linear,
    fused_rms_norm_qkv,
)


def reference_rms_layernorm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Reference RMS LayerNorm matching T5/Chronos2 behavior."""
    variance = x.to(torch.float32).pow(2).mean(-1, keepdim=True)
    normed = x * torch.rsqrt(variance + eps)
    if weight.dtype in (torch.float16, torch.bfloat16):
        normed = normed.to(weight.dtype)
    return weight * normed


# Shapes from the task spec: (rows, input_dim) -> output_dim
# (34, 768) -> 768, (520, 768) -> 768, (4160, 768) -> 3072
FUSED_LINEAR_SHAPES = [
    ((34, 768), 768),
    ((520, 768), 768),
    ((4160, 768), 3072),
]

FUSED_LINEAR_IDS = [
    f"{rows}x{N}_to_{O}" for (rows, N), O in FUSED_LINEAR_SHAPES
]


# ---------------------------------------------------------------
# fused_rms_norm_linear tests
# ---------------------------------------------------------------

@pytest.mark.parametrize("shape_pair", FUSED_LINEAR_SHAPES, ids=FUSED_LINEAR_IDS)
def test_fused_rms_norm_linear_fp32(shape_pair):
    (rows, N), O = shape_pair
    torch.manual_seed(42)
    x = torch.randn(rows, N, device="cuda", dtype=torch.float32)
    norm_w = torch.randn(N, device="cuda", dtype=torch.float32)
    linear_w = torch.randn(O, N, device="cuda", dtype=torch.float32)

    normed = reference_rms_layernorm(x, norm_w)
    ref = F.linear(normed, linear_w)
    out = fused_rms_norm_linear(x, norm_w, linear_w)

    max_err = (out - ref).abs().max().item()
    assert max_err < 1e-4, f"FP32 max abs error {max_err} >= 1e-4"


@pytest.mark.parametrize("shape_pair", FUSED_LINEAR_SHAPES, ids=FUSED_LINEAR_IDS)
def test_fused_rms_norm_linear_bf16(shape_pair):
    """BF16 test: verify kernel is at least as accurate as PyTorch reference.

    BF16 matmul over 768 dims with unit-variance data produces outputs of
    magnitude ~O(sqrt(N)), where 1 ULP of BF16 at that scale exceeds 1e-4.
    Instead, we compare both against an FP32 gold standard and verify our
    kernel's error is no worse than the reference's.
    """
    (rows, N), O = shape_pair
    torch.manual_seed(42)
    x = torch.randn(rows, N, device="cuda", dtype=torch.bfloat16)
    norm_w = torch.randn(N, device="cuda", dtype=torch.bfloat16)
    linear_w = torch.randn(O, N, device="cuda", dtype=torch.bfloat16)

    # BF16 reference
    normed = reference_rms_layernorm(x, norm_w)
    ref = F.linear(normed, linear_w)
    out = fused_rms_norm_linear(x, norm_w, linear_w)

    # FP32 gold standard
    gold_normed = reference_rms_layernorm(x.float(), norm_w.float())
    gold = F.linear(gold_normed, linear_w.float())

    our_err = (out.float() - gold).abs().max().item()
    ref_err = (ref.float() - gold).abs().max().item()

    # Our kernel (FP32 accumulation) should be at least as accurate as cuBLAS BF16
    assert our_err <= ref_err * 1.1 + 1e-5, (
        f"BF16 kernel error {our_err} exceeds reference error {ref_err} by too much"
    )


# ---------------------------------------------------------------
# fused_rms_norm_qkv tests
# ---------------------------------------------------------------

# QKV shapes: (rows, 768) -> 768 (inner_dim = 12 heads * 64 d_kv)
QKV_SHAPES = [
    (34, 768, 768),
    (520, 768, 768),
    (4160, 768, 768),
]

QKV_IDS = [f"{rows}x{N}_to_{D}" for rows, N, D in QKV_SHAPES]


@pytest.mark.parametrize("shape", QKV_SHAPES, ids=QKV_IDS)
def test_fused_rms_norm_qkv_fp32(shape):
    rows, N, D = shape
    torch.manual_seed(42)
    x = torch.randn(rows, N, device="cuda", dtype=torch.float32)
    norm_w = torch.randn(N, device="cuda", dtype=torch.float32)
    wq = torch.randn(D, N, device="cuda", dtype=torch.float32)
    wk = torch.randn(D, N, device="cuda", dtype=torch.float32)
    wv = torch.randn(D, N, device="cuda", dtype=torch.float32)

    normed = reference_rms_layernorm(x, norm_w)
    ref_q = F.linear(normed, wq)
    ref_k = F.linear(normed, wk)
    ref_v = F.linear(normed, wv)

    out_q, out_k, out_v = fused_rms_norm_qkv(x, norm_w, wq, wk, wv)

    for name, out, ref in [("Q", out_q, ref_q), ("K", out_k, ref_k), ("V", out_v, ref_v)]:
        max_err = (out - ref).abs().max().item()
        assert max_err < 1e-4, f"FP32 {name} max abs error {max_err} >= 1e-4"


@pytest.mark.parametrize("shape", QKV_SHAPES, ids=QKV_IDS)
def test_fused_rms_norm_qkv_bf16(shape):
    """BF16 QKV test: verify kernel is at least as accurate as PyTorch reference."""
    rows, N, D = shape
    torch.manual_seed(42)
    x = torch.randn(rows, N, device="cuda", dtype=torch.bfloat16)
    norm_w = torch.randn(N, device="cuda", dtype=torch.bfloat16)
    wq = torch.randn(D, N, device="cuda", dtype=torch.bfloat16)
    wk = torch.randn(D, N, device="cuda", dtype=torch.bfloat16)
    wv = torch.randn(D, N, device="cuda", dtype=torch.bfloat16)

    # BF16 reference
    normed = reference_rms_layernorm(x, norm_w)
    ref_q = F.linear(normed, wq)
    ref_k = F.linear(normed, wk)
    ref_v = F.linear(normed, wv)

    out_q, out_k, out_v = fused_rms_norm_qkv(x, norm_w, wq, wk, wv)

    # FP32 gold standard
    gold_normed = reference_rms_layernorm(x.float(), norm_w.float())
    gold_q = F.linear(gold_normed, wq.float())
    gold_k = F.linear(gold_normed, wk.float())
    gold_v = F.linear(gold_normed, wv.float())

    for name, out, ref, gold in [
        ("Q", out_q, ref_q, gold_q),
        ("K", out_k, ref_k, gold_k),
        ("V", out_v, ref_v, gold_v),
    ]:
        our_err = (out.float() - gold).abs().max().item()
        ref_err = (ref.float() - gold).abs().max().item()
        assert our_err <= ref_err * 1.1 + 1e-5, (
            f"BF16 {name} kernel error {our_err} exceeds reference error {ref_err}"
        )


# ---------------------------------------------------------------
# Edge cases and shape preservation
# ---------------------------------------------------------------

def test_output_dtype_fp32():
    """Output dtype must match input dtype for FP32."""
    x = torch.randn(10, 768, device="cuda", dtype=torch.float32)
    norm_w = torch.randn(768, device="cuda", dtype=torch.float32)
    linear_w = torch.randn(768, 768, device="cuda", dtype=torch.float32)
    out = fused_rms_norm_linear(x, norm_w, linear_w)
    assert out.dtype == torch.float32


def test_output_dtype_bf16():
    """Output dtype must match input dtype for BF16."""
    x = torch.randn(10, 768, device="cuda", dtype=torch.bfloat16)
    norm_w = torch.randn(768, device="cuda", dtype=torch.bfloat16)
    linear_w = torch.randn(768, 768, device="cuda", dtype=torch.bfloat16)
    out = fused_rms_norm_linear(x, norm_w, linear_w)
    assert out.dtype == torch.bfloat16


def test_output_shape_2d():
    """2D input shape: (rows, N) -> (rows, O)."""
    x = torch.randn(34, 768, device="cuda")
    norm_w = torch.randn(768, device="cuda")
    linear_w = torch.randn(3072, 768, device="cuda")
    out = fused_rms_norm_linear(x, norm_w, linear_w)
    assert out.shape == (34, 3072)


def test_output_shape_3d():
    """3D input shape: (B, S, N) -> (B, S, O)."""
    x = torch.randn(2, 17, 768, device="cuda")
    norm_w = torch.randn(768, device="cuda")
    linear_w = torch.randn(3072, 768, device="cuda")
    out = fused_rms_norm_linear(x, norm_w, linear_w)
    assert out.shape == (2, 17, 3072)


def test_qkv_output_shapes():
    """QKV outputs must each have shape (..., D)."""
    x = torch.randn(2, 17, 768, device="cuda")
    norm_w = torch.randn(768, device="cuda")
    wq = torch.randn(768, 768, device="cuda")
    wk = torch.randn(768, 768, device="cuda")
    wv = torch.randn(768, 768, device="cuda")
    q, k, v = fused_rms_norm_qkv(x, norm_w, wq, wk, wv)
    assert q.shape == (2, 17, 768)
    assert k.shape == (2, 17, 768)
    assert v.shape == (2, 17, 768)


def test_qkv_output_dtype_bf16():
    """QKV output dtypes must match input dtype."""
    x = torch.randn(10, 768, device="cuda", dtype=torch.bfloat16)
    norm_w = torch.randn(768, device="cuda", dtype=torch.bfloat16)
    wq = torch.randn(768, 768, device="cuda", dtype=torch.bfloat16)
    wk = torch.randn(768, 768, device="cuda", dtype=torch.bfloat16)
    wv = torch.randn(768, 768, device="cuda", dtype=torch.bfloat16)
    q, k, v = fused_rms_norm_qkv(x, norm_w, wq, wk, wv)
    assert q.dtype == torch.bfloat16
    assert k.dtype == torch.bfloat16
    assert v.dtype == torch.bfloat16


def test_single_row():
    """Single-row input must work correctly."""
    torch.manual_seed(42)
    x = torch.randn(1, 768, device="cuda", dtype=torch.float32)
    norm_w = torch.randn(768, device="cuda", dtype=torch.float32)
    linear_w = torch.randn(768, 768, device="cuda", dtype=torch.float32)

    normed = reference_rms_layernorm(x, norm_w)
    ref = F.linear(normed, linear_w)
    out = fused_rms_norm_linear(x, norm_w, linear_w)

    max_err = (out - ref).abs().max().item()
    assert max_err < 1e-4, f"Single row max abs error {max_err} >= 1e-4"


def test_3d_numerical_equivalence():
    """3D input must produce same result as flattened 2D."""
    torch.manual_seed(42)
    x = torch.randn(2, 17, 768, device="cuda", dtype=torch.float32)
    norm_w = torch.randn(768, device="cuda", dtype=torch.float32)
    linear_w = torch.randn(3072, 768, device="cuda", dtype=torch.float32)

    normed = reference_rms_layernorm(x, norm_w)
    ref = F.linear(normed, linear_w)
    out = fused_rms_norm_linear(x, norm_w, linear_w)

    max_err = (out - ref).abs().max().item()
    assert max_err < 1e-4, f"3D max abs error {max_err} >= 1e-4"
