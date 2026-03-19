"""Tests for the Triton unscaled attention kernel.

Verifies that the custom Triton kernel produces results matching
the reference PyTorch implementation:

    output = softmax(Q @ K^T + mask) @ V

with scale=1.0 (no 1/sqrt(d_k) scaling).
"""

import pytest
import torch

from cutechronos.triton_kernels.attention import unscaled_attention


def reference_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Reference unscaled attention computed in FP32 for accuracy.

    The Triton kernel accumulates dot products in FP32 (even for BF16 inputs),
    so the reference must also use FP32 accumulation to be a fair comparison.
    This is actually more numerically accurate than Chronos2's eager path
    which does BF16 matmul.
    """
    # Upcast to FP32 for the dot product to match Triton's FP32 accumulation
    orig_dtype = q.dtype
    q32 = q.float()
    k32 = k.float()
    v32 = v.float()

    scores = torch.matmul(q32, k32.transpose(-2, -1))  # (B, H, S, S) in FP32
    if mask is not None:
        scores = scores + mask.float()
    attn_weights = torch.softmax(scores, dim=-1)  # already FP32
    out = torch.matmul(attn_weights, v32)
    return out.to(orig_dtype)


# ------------------------------------------------------------------ #
# Test parameters                                                      #
# ------------------------------------------------------------------ #

# Shapes matching typical Chronos2 usage
SHAPES = [
    (1, 12, 34, 64),   # ctx=512  -> 34 patches
    (4, 12, 130, 64),  # ctx=2048 -> 130 patches
    (2, 12, 514, 64),  # ctx=8192 -> 514 patches
]

DTYPES = [torch.float32, torch.bfloat16]


# ------------------------------------------------------------------ #
# Helpers                                                              #
# ------------------------------------------------------------------ #

def make_causal_mask(B: int, S: int, device: str, broadcast_batch: bool = True) -> torch.Tensor:
    """Create an additive causal mask: 0 for valid, -inf for masked."""
    mask = torch.full((S, S), float("-inf"), device=device)
    mask = torch.triu(mask, diagonal=1)  # upper triangle = -inf
    if broadcast_batch:
        return mask.unsqueeze(0).unsqueeze(0)  # (1, 1, S, S)
    return mask.unsqueeze(0).unsqueeze(0).expand(B, 1, S, S).contiguous()


# ------------------------------------------------------------------ #
# Tests: no mask                                                       #
# ------------------------------------------------------------------ #

@pytest.mark.parametrize("B,H,S,D", SHAPES)
@pytest.mark.parametrize("dtype", DTYPES, ids=["fp32", "bf16"])
def test_no_mask(B: int, H: int, S: int, D: int, dtype: torch.dtype):
    """Verify kernel matches reference without any attention mask."""
    device = "cuda"
    torch.manual_seed(42)

    q = torch.randn(B, H, S, D, device=device, dtype=dtype)
    k = torch.randn(B, H, S, D, device=device, dtype=dtype)
    v = torch.randn(B, H, S, D, device=device, dtype=dtype)

    ref = reference_attention(q, k, v, mask=None)
    out = unscaled_attention(q, k, v, mask=None)

    if dtype == torch.float32:
        atol = 1e-4
    else:
        atol = 2e-2

    max_err = (ref - out).abs().max().item()
    assert max_err < atol, (
        f"Max abs error {max_err:.6e} exceeds tolerance {atol} "
        f"for shape ({B},{H},{S},{D}) dtype={dtype}"
    )


# ------------------------------------------------------------------ #
# Tests: with causal mask                                              #
# ------------------------------------------------------------------ #

@pytest.mark.parametrize("B,H,S,D", SHAPES)
@pytest.mark.parametrize("dtype", DTYPES, ids=["fp32", "bf16"])
def test_causal_mask(B: int, H: int, S: int, D: int, dtype: torch.dtype):
    """Verify kernel matches reference with a causal (lower-triangular) mask."""
    device = "cuda"
    torch.manual_seed(123)

    q = torch.randn(B, H, S, D, device=device, dtype=dtype)
    k = torch.randn(B, H, S, D, device=device, dtype=dtype)
    v = torch.randn(B, H, S, D, device=device, dtype=dtype)
    mask = make_causal_mask(B, S, device)  # (1, 1, S, S) broadcast

    ref = reference_attention(q, k, v, mask=mask)
    out = unscaled_attention(q, k, v, mask=mask)

    if dtype == torch.float32:
        atol = 1e-4
    else:
        atol = 2e-2

    max_err = (ref - out).abs().max().item()
    assert max_err < atol, (
        f"Max abs error {max_err:.6e} exceeds tolerance {atol} "
        f"for shape ({B},{H},{S},{D}) dtype={dtype} with causal mask"
    )


# ------------------------------------------------------------------ #
# Tests: per-batch mask (no broadcast on batch dim)                    #
# ------------------------------------------------------------------ #

@pytest.mark.parametrize("B,H,S,D", [(2, 12, 34, 64), (4, 12, 130, 64)])
def test_per_batch_mask(B: int, H: int, S: int, D: int):
    """Verify kernel handles masks with explicit batch dimension (no broadcast)."""
    device = "cuda"
    dtype = torch.float32
    torch.manual_seed(77)

    q = torch.randn(B, H, S, D, device=device, dtype=dtype)
    k = torch.randn(B, H, S, D, device=device, dtype=dtype)
    v = torch.randn(B, H, S, D, device=device, dtype=dtype)
    mask = make_causal_mask(B, S, device, broadcast_batch=False)  # (B, 1, S, S)

    ref = reference_attention(q, k, v, mask=mask)
    out = unscaled_attention(q, k, v, mask=mask)

    max_err = (ref - out).abs().max().item()
    assert max_err < 1e-4, f"Max abs error {max_err:.6e} exceeds 1e-4 for per-batch mask"


# ------------------------------------------------------------------ #
# Tests: determinism                                                   #
# ------------------------------------------------------------------ #

def test_deterministic():
    """Verify the kernel produces identical results on repeated calls."""
    device = "cuda"
    B, H, S, D = 2, 12, 34, 64
    torch.manual_seed(99)

    q = torch.randn(B, H, S, D, device=device, dtype=torch.float32)
    k = torch.randn(B, H, S, D, device=device, dtype=torch.float32)
    v = torch.randn(B, H, S, D, device=device, dtype=torch.float32)

    out1 = unscaled_attention(q, k, v)
    out2 = unscaled_attention(q, k, v)

    assert torch.equal(out1, out2), "Kernel is not deterministic"


# ------------------------------------------------------------------ #
# Tests: output shape and dtype                                        #
# ------------------------------------------------------------------ #

@pytest.mark.parametrize("dtype", DTYPES, ids=["fp32", "bf16"])
def test_output_shape_and_dtype(dtype: torch.dtype):
    """Verify output shape and dtype match the input."""
    device = "cuda"
    B, H, S, D = 1, 12, 34, 64
    torch.manual_seed(0)

    q = torch.randn(B, H, S, D, device=device, dtype=dtype)
    k = torch.randn(B, H, S, D, device=device, dtype=dtype)
    v = torch.randn(B, H, S, D, device=device, dtype=dtype)

    out = unscaled_attention(q, k, v)
    assert out.shape == (B, H, S, D), f"Shape mismatch: {out.shape}"
    assert out.dtype == dtype, f"Dtype mismatch: {out.dtype}"


# ------------------------------------------------------------------ #
# Tests: small example for sanity check                                #
# ------------------------------------------------------------------ #

def test_small_identity():
    """With identity-like Q, K and uniform V, output should approximate V."""
    device = "cuda"
    B, H, S, D = 1, 1, 4, 64

    # Q = K = zeros => all scores equal => softmax uniform => output = mean(V)
    q = torch.zeros(B, H, S, D, device=device, dtype=torch.float32)
    k = torch.zeros(B, H, S, D, device=device, dtype=torch.float32)
    v = torch.randn(B, H, S, D, device=device, dtype=torch.float32)

    out = unscaled_attention(q, k, v)
    # Each output row should be the mean of all V rows (uniform attention)
    expected = v.mean(dim=2, keepdim=True).expand_as(v)
    max_err = (out - expected).abs().max().item()
    assert max_err < 1e-5, f"Uniform attention error {max_err:.6e}"
