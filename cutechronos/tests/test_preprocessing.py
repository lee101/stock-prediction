"""Tests for cutechronos fused preprocessing kernel.

Validates CUDA and fallback outputs against a pure-PyTorch reference
that exactly reproduces the Chronos2 _prepare_patched_context pipeline.
"""

import sys
from pathlib import Path

import pytest
import torch

# Ensure the repo root is on sys.path so cutechronos is importable
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from cutechronos.kernels import fused_preprocess, _fallback_preprocess


# ---------------------------------------------------------------------------
# Reference implementation -- mirrors Chronos2 InstanceNorm + Patch exactly
# ---------------------------------------------------------------------------

def _reference_preprocess(
    context: torch.Tensor,
    patch_size: int,
    context_length: int,
    use_arcsinh: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Reference implementation matching chronos2 model.py _prepare_patched_context."""
    x = context.clone().float()
    B, L = x.shape

    # Truncate
    if L > context_length:
        x = x[:, -context_length:]
        L = context_length

    # Build context_mask from NaN (same as model.py line 399)
    context_mask = torch.isnan(x).logical_not().float()

    # InstanceNorm (chronos_bolt.py lines 81-98)
    # nanmean
    loc = torch.nan_to_num(torch.nanmean(x, dim=-1, keepdim=True), nan=0.0)
    # nanstd: sqrt(nanmean((x-loc)^2))
    scale = torch.nan_to_num(
        (x - loc).square().nanmean(dim=-1, keepdim=True).sqrt(), nan=1.0
    )
    scale = torch.where(scale == 0, torch.tensor(1e-5, dtype=scale.dtype), scale)

    scaled_x = (x - loc) / scale
    if use_arcsinh:
        scaled_x = torch.arcsinh(scaled_x)

    # InstanceNorm returns scaled_x cast back to orig dtype (float here)
    # Fill NaN positions with 0 is done after patching in model.py

    # Patching (chronos_bolt.py Patch.forward)
    # Left-pad to multiple of patch_size with NaN, then unfold
    length = scaled_x.shape[-1]
    if length % patch_size != 0:
        pad_size = patch_size - (length % patch_size)
        padding = torch.full((B, pad_size), float("nan"), dtype=scaled_x.dtype, device=scaled_x.device)
        scaled_x = torch.cat([padding, scaled_x], dim=-1)
        # Also pad context_mask with NaN so patching works consistently
        mask_padding = torch.full((B, pad_size), float("nan"), dtype=context_mask.dtype, device=context_mask.device)
        context_mask = torch.cat([mask_padding, context_mask], dim=-1)

    patched_context = scaled_x.unfold(-1, patch_size, patch_size)
    patched_mask = torch.nan_to_num(
        context_mask.unfold(-1, patch_size, patch_size), nan=0.0
    )
    patched_context = torch.where(patched_mask > 0.0, patched_context, torch.tensor(0.0))

    # attention_mask
    attn_mask = (patched_mask.sum(dim=-1) > 0).float()
    num_patches = attn_mask.shape[-1]

    # Time encoding (model.py lines 426-438)
    final_ctx_len = num_patches * patch_size
    time_enc = torch.arange(-final_ctx_len, 0, device=x.device, dtype=torch.float32)
    time_enc = (
        time_enc.reshape(1, num_patches, patch_size)
        .expand(B, -1, -1)
        .div(context_length)
    )

    # Concat
    patched_out = torch.cat([time_enc, patched_context, patched_mask], dim=-1)

    return patched_out, attn_mask, loc, scale


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_context(B: int, L: int, nan_frac: float = 0.1, seed: int = 42) -> torch.Tensor:
    """Create a random context tensor with some NaN values sprinkled in."""
    gen = torch.Generator().manual_seed(seed)
    ctx = torch.randn(B, L, generator=gen)
    # Sprinkle NaN
    mask = torch.rand(B, L, generator=gen) < nan_frac
    ctx[mask] = float("nan")
    return ctx


def _compare_outputs(
    result: tuple, ref: tuple, atol: float = 1e-5, label: str = ""
):
    patched, attn, loc, scale = result
    ref_patched, ref_attn, ref_loc, ref_scale = ref

    prefix = f"[{label}] " if label else ""

    # Shapes
    assert patched.shape == ref_patched.shape, (
        f"{prefix}patched shape mismatch: {patched.shape} vs {ref_patched.shape}"
    )
    assert attn.shape == ref_attn.shape, (
        f"{prefix}attn_mask shape mismatch: {attn.shape} vs {ref_attn.shape}"
    )
    assert loc.shape == ref_loc.shape, (
        f"{prefix}loc shape mismatch: {loc.shape} vs {ref_loc.shape}"
    )
    assert scale.shape == ref_scale.shape, (
        f"{prefix}scale shape mismatch: {scale.shape} vs {ref_scale.shape}"
    )

    # Values
    max_err_patched = (patched.float() - ref_patched.float()).abs().max().item()
    max_err_attn = (attn.float() - ref_attn.float()).abs().max().item()
    max_err_loc = (loc.float() - ref_loc.float()).abs().max().item()
    max_err_scale = (scale.float() - ref_scale.float()).abs().max().item()

    assert max_err_loc < atol, f"{prefix}loc max error {max_err_loc} >= {atol}"
    assert max_err_scale < atol, f"{prefix}scale max error {max_err_scale} >= {atol}"
    assert max_err_attn < atol, f"{prefix}attn_mask max error {max_err_attn} >= {atol}"
    assert max_err_patched < atol, (
        f"{prefix}patched max error {max_err_patched} >= {atol}"
    )


# ---------------------------------------------------------------------------
# Tests — PyTorch fallback
# ---------------------------------------------------------------------------

class TestFallbackPreprocess:
    """Test the pure-PyTorch fallback against the reference."""

    @pytest.mark.parametrize("B,L", [(1, 512), (4, 512), (16, 2048)])
    def test_shapes_and_values(self, B: int, L: int):
        ctx = _make_context(B, L)
        ref = _reference_preprocess(ctx, patch_size=16, context_length=512, use_arcsinh=False)
        result = _fallback_preprocess(ctx, patch_size=16, context_length=512, use_arcsinh=False)
        _compare_outputs(result, ref, atol=1e-5, label=f"fallback B={B} L={L}")

    def test_arcsinh(self):
        ctx = _make_context(4, 512)
        ref = _reference_preprocess(ctx, patch_size=16, context_length=512, use_arcsinh=True)
        result = _fallback_preprocess(ctx, patch_size=16, context_length=512, use_arcsinh=True)
        _compare_outputs(result, ref, atol=1e-5, label="fallback arcsinh")

    def test_no_nans(self):
        torch.manual_seed(123)
        ctx = torch.randn(2, 512)
        ref = _reference_preprocess(ctx, patch_size=16, context_length=512, use_arcsinh=False)
        result = _fallback_preprocess(ctx, patch_size=16, context_length=512, use_arcsinh=False)
        _compare_outputs(result, ref, atol=1e-5, label="fallback no-nan")

    def test_all_nans(self):
        ctx = torch.full((2, 512), float("nan"))
        ref = _reference_preprocess(ctx, patch_size=16, context_length=512, use_arcsinh=False)
        result = _fallback_preprocess(ctx, patch_size=16, context_length=512, use_arcsinh=False)
        _compare_outputs(result, ref, atol=1e-5, label="fallback all-nan")

    def test_non_multiple_length(self):
        """Context length not a multiple of patch_size triggers padding."""
        ctx = _make_context(3, 500)  # 500 % 16 = 4 != 0
        ref = _reference_preprocess(ctx, patch_size=16, context_length=512, use_arcsinh=False)
        result = _fallback_preprocess(ctx, patch_size=16, context_length=512, use_arcsinh=False)
        _compare_outputs(result, ref, atol=1e-5, label="fallback non-multiple")

    def test_context_truncation(self):
        """Context longer than context_length gets truncated."""
        ctx = _make_context(2, 1024)
        ref = _reference_preprocess(ctx, patch_size=16, context_length=512, use_arcsinh=False)
        result = _fallback_preprocess(ctx, patch_size=16, context_length=512, use_arcsinh=False)
        _compare_outputs(result, ref, atol=1e-5, label="fallback truncation")

    def test_output_shapes_typical(self):
        """Verify output shapes for the typical Chronos2 config."""
        ctx = _make_context(8, 512)
        patched, attn, loc, scale = _fallback_preprocess(
            ctx, patch_size=16, context_length=512, use_arcsinh=False
        )
        assert patched.shape == (8, 32, 48), f"Expected (8,32,48), got {patched.shape}"
        assert attn.shape == (8, 32), f"Expected (8,32), got {attn.shape}"
        assert loc.shape == (8, 1), f"Expected (8,1), got {loc.shape}"
        assert scale.shape == (8, 1), f"Expected (8,1), got {scale.shape}"


# ---------------------------------------------------------------------------
# Tests — CUDA kernel
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
class TestCUDAPreprocess:
    """Test the CUDA fused kernel against the reference."""

    @pytest.mark.parametrize("B,L", [(1, 512), (4, 512), (16, 2048)])
    def test_shapes_and_values(self, B: int, L: int):
        ctx = _make_context(B, L).cuda()
        ref = _reference_preprocess(ctx.cpu(), patch_size=16, context_length=512, use_arcsinh=False)
        result = fused_preprocess(
            ctx, patch_size=16, context_length=512, use_arcsinh=False
        )
        # Move CUDA results to CPU for comparison
        result_cpu = tuple(t.cpu() for t in result)
        _compare_outputs(result_cpu, ref, atol=1e-5, label=f"cuda B={B} L={L}")

    def test_arcsinh(self):
        ctx = _make_context(4, 512).cuda()
        ref = _reference_preprocess(ctx.cpu(), patch_size=16, context_length=512, use_arcsinh=True)
        result = fused_preprocess(ctx, patch_size=16, context_length=512, use_arcsinh=True)
        result_cpu = tuple(t.cpu() for t in result)
        _compare_outputs(result_cpu, ref, atol=1e-5, label="cuda arcsinh")

    def test_no_nans(self):
        torch.manual_seed(123)
        ctx = torch.randn(2, 512).cuda()
        ref = _reference_preprocess(ctx.cpu(), patch_size=16, context_length=512, use_arcsinh=False)
        result = fused_preprocess(ctx, patch_size=16, context_length=512, use_arcsinh=False)
        result_cpu = tuple(t.cpu() for t in result)
        _compare_outputs(result_cpu, ref, atol=1e-5, label="cuda no-nan")

    def test_all_nans(self):
        ctx = torch.full((2, 512), float("nan")).cuda()
        ref = _reference_preprocess(ctx.cpu(), patch_size=16, context_length=512, use_arcsinh=False)
        result = fused_preprocess(ctx, patch_size=16, context_length=512, use_arcsinh=False)
        result_cpu = tuple(t.cpu() for t in result)
        _compare_outputs(result_cpu, ref, atol=1e-5, label="cuda all-nan")

    def test_non_multiple_length(self):
        ctx = _make_context(3, 500).cuda()
        ref = _reference_preprocess(ctx.cpu(), patch_size=16, context_length=512, use_arcsinh=False)
        result = fused_preprocess(ctx, patch_size=16, context_length=512, use_arcsinh=False)
        result_cpu = tuple(t.cpu() for t in result)
        _compare_outputs(result_cpu, ref, atol=1e-5, label="cuda non-multiple")

    def test_context_truncation(self):
        ctx = _make_context(2, 1024).cuda()
        ref = _reference_preprocess(ctx.cpu(), patch_size=16, context_length=512, use_arcsinh=False)
        result = fused_preprocess(ctx, patch_size=16, context_length=512, use_arcsinh=False)
        result_cpu = tuple(t.cpu() for t in result)
        _compare_outputs(result_cpu, ref, atol=1e-5, label="cuda truncation")

    def test_output_shapes_typical(self):
        ctx = _make_context(8, 512).cuda()
        patched, attn, loc, scale = fused_preprocess(
            ctx, patch_size=16, context_length=512, use_arcsinh=False
        )
        assert patched.shape == (8, 32, 48)
        assert attn.shape == (8, 32)
        assert loc.shape == (8, 1)
        assert scale.shape == (8, 1)

    def test_half_precision_input(self):
        """Test with float16 input -- kernel should still produce float32 outputs."""
        ctx = _make_context(4, 512).half().cuda()
        ref = _reference_preprocess(ctx.cpu().float(), patch_size=16, context_length=512, use_arcsinh=False)
        result = fused_preprocess(ctx, patch_size=16, context_length=512, use_arcsinh=False)
        result_cpu = tuple(t.cpu() for t in result)
        # Relaxed tolerance for half precision
        _compare_outputs(result_cpu, ref, atol=5e-3, label="cuda half")

    def test_bfloat16_input(self):
        """Test with bfloat16 input."""
        ctx = _make_context(4, 512).bfloat16().cuda()
        ref = _reference_preprocess(ctx.cpu().float(), patch_size=16, context_length=512, use_arcsinh=False)
        result = fused_preprocess(ctx, patch_size=16, context_length=512, use_arcsinh=False)
        result_cpu = tuple(t.cpu() for t in result)
        _compare_outputs(result_cpu, ref, atol=5e-3, label="cuda bf16")

    def test_large_values(self):
        """Test with large values to ensure numerical stability."""
        torch.manual_seed(99)
        ctx = torch.randn(2, 512) * 1e6
        ctx_cuda = ctx.cuda()
        ref = _reference_preprocess(ctx, patch_size=16, context_length=512, use_arcsinh=False)
        result = fused_preprocess(ctx_cuda, patch_size=16, context_length=512, use_arcsinh=False)
        result_cpu = tuple(t.cpu() for t in result)
        # Relaxed tolerance for large values: float32 reduction ordering
        # causes ~0.1 absolute error when values are ~1e6 (relative ~1e-7).
        _compare_outputs(result_cpu, ref, atol=0.1, label="cuda large-values")

    def test_constant_series(self):
        """Constant series has std=0, should use eps."""
        ctx = torch.full((2, 512), 42.0).cuda()
        ref = _reference_preprocess(ctx.cpu(), patch_size=16, context_length=512, use_arcsinh=False)
        result = fused_preprocess(ctx, patch_size=16, context_length=512, use_arcsinh=False)
        result_cpu = tuple(t.cpu() for t in result)
        _compare_outputs(result_cpu, ref, atol=1e-5, label="cuda constant")


# ---------------------------------------------------------------------------
# Tests — fused_preprocess API (uses CUDA if available, else fallback)
# ---------------------------------------------------------------------------

class TestFusedPreprocessAPI:
    """Test the top-level fused_preprocess function."""

    def test_cpu_fallback(self):
        """Ensure fused_preprocess works on CPU tensors (uses fallback)."""
        ctx = _make_context(2, 512)
        ref = _reference_preprocess(ctx, patch_size=16, context_length=512, use_arcsinh=False)
        result = fused_preprocess(ctx, patch_size=16, context_length=512, use_arcsinh=False)
        _compare_outputs(result, ref, atol=1e-5, label="api cpu")

    def test_returns_four_tensors(self):
        ctx = _make_context(2, 512)
        result = fused_preprocess(ctx, patch_size=16, context_length=512, use_arcsinh=False)
        assert len(result) == 4, f"Expected 4 outputs, got {len(result)}"
        for i, t in enumerate(result):
            assert isinstance(t, torch.Tensor), f"Output {i} is not a tensor"
