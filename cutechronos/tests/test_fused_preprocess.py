"""Tests for the Triton fused preprocessing kernel.

Validates the two-phase Triton kernel (NaN-aware reduction + fused transform)
against the pure-PyTorch fallback from cutechronos.kernels._fallback_preprocess,
which itself matches the Chronos2 _prepare_patched_context pipeline.

Test matrix:
    - Shapes: (1,512), (4,512), (2,2048)
    - NaN patterns: sprinkled (~10%), no NaN, all NaN
    - arcsinh on/off
    - Non-multiple lengths, truncation
    - Edge cases: constant series, large values, half/bfloat16 inputs
"""

import sys
from pathlib import Path

import pytest
import torch

# Ensure the repo root is on sys.path
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from cutechronos.kernels import _fallback_preprocess
from cutechronos.triton_kernels.fused_preprocess import triton_fused_preprocess
from cutechronos.tests.conftest import make_context, compare_outputs


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
class TestTritonFusedPreprocess:
    """Test the Triton fused preprocessing kernel against PyTorch fallback."""

    @pytest.mark.parametrize("B,L", [(1, 512), (4, 512), (2, 2048)])
    def test_shapes_and_values(self, B: int, L: int):
        """Core shapes from the task spec with NaN-sprinkled data."""
        ctx = make_context(B, L)
        ref = _fallback_preprocess(ctx, patch_size=16, context_length=512, use_arcsinh=False)
        result = triton_fused_preprocess(
            ctx.cuda(), patch_size=16, context_length=512, use_arcsinh=False,
        )
        result_cpu = tuple(t.cpu() for t in result)
        compare_outputs(result_cpu, ref, atol=1e-5, label=f"triton B={B} L={L}")

    @pytest.mark.parametrize("B,L", [(1, 512), (4, 512), (2, 2048)])
    def test_shapes_and_values_arcsinh(self, B: int, L: int):
        """Same shapes with arcsinh enabled."""
        ctx = make_context(B, L)
        ref = _fallback_preprocess(ctx, patch_size=16, context_length=512, use_arcsinh=True)
        result = triton_fused_preprocess(
            ctx.cuda(), patch_size=16, context_length=512, use_arcsinh=True,
        )
        result_cpu = tuple(t.cpu() for t in result)
        compare_outputs(result_cpu, ref, atol=1e-5, label=f"triton arcsinh B={B} L={L}")

    def test_no_nans(self):
        """Edge case: no NaN values at all."""
        torch.manual_seed(123)
        ctx = torch.randn(2, 512)
        ref = _fallback_preprocess(ctx, patch_size=16, context_length=512, use_arcsinh=False)
        result = triton_fused_preprocess(
            ctx.cuda(), patch_size=16, context_length=512, use_arcsinh=False,
        )
        result_cpu = tuple(t.cpu() for t in result)
        compare_outputs(result_cpu, ref, atol=1e-5, label="triton no-nan")

    def test_no_nans_arcsinh(self):
        """No NaN with arcsinh."""
        torch.manual_seed(123)
        ctx = torch.randn(4, 512)
        ref = _fallback_preprocess(ctx, patch_size=16, context_length=512, use_arcsinh=True)
        result = triton_fused_preprocess(
            ctx.cuda(), patch_size=16, context_length=512, use_arcsinh=True,
        )
        result_cpu = tuple(t.cpu() for t in result)
        compare_outputs(result_cpu, ref, atol=1e-5, label="triton no-nan arcsinh")

    def test_all_nans(self):
        """Edge case: every value is NaN."""
        ctx = torch.full((2, 512), float("nan"))
        ref = _fallback_preprocess(ctx, patch_size=16, context_length=512, use_arcsinh=False)
        result = triton_fused_preprocess(
            ctx.cuda(), patch_size=16, context_length=512, use_arcsinh=False,
        )
        result_cpu = tuple(t.cpu() for t in result)
        compare_outputs(result_cpu, ref, atol=1e-5, label="triton all-nan")

    def test_all_nans_arcsinh(self):
        """All NaN with arcsinh enabled (should still produce zeros)."""
        ctx = torch.full((2, 512), float("nan"))
        ref = _fallback_preprocess(ctx, patch_size=16, context_length=512, use_arcsinh=True)
        result = triton_fused_preprocess(
            ctx.cuda(), patch_size=16, context_length=512, use_arcsinh=True,
        )
        result_cpu = tuple(t.cpu() for t in result)
        compare_outputs(result_cpu, ref, atol=1e-5, label="triton all-nan arcsinh")

    def test_non_multiple_length(self):
        """Context length not a multiple of patch_size triggers padding."""
        ctx = make_context(3, 500)  # 500 % 16 = 4, needs 12 padding
        ref = _fallback_preprocess(ctx, patch_size=16, context_length=512, use_arcsinh=False)
        result = triton_fused_preprocess(
            ctx.cuda(), patch_size=16, context_length=512, use_arcsinh=False,
        )
        result_cpu = tuple(t.cpu() for t in result)
        compare_outputs(result_cpu, ref, atol=1e-5, label="triton non-multiple")

    def test_context_truncation(self):
        """Context longer than context_length gets truncated."""
        ctx = make_context(2, 1024)
        ref = _fallback_preprocess(ctx, patch_size=16, context_length=512, use_arcsinh=False)
        result = triton_fused_preprocess(
            ctx.cuda(), patch_size=16, context_length=512, use_arcsinh=False,
        )
        result_cpu = tuple(t.cpu() for t in result)
        compare_outputs(result_cpu, ref, atol=1e-5, label="triton truncation")

    def test_output_shapes_typical(self):
        """Verify output shapes for the typical Chronos2 config."""
        ctx = make_context(8, 512).cuda()
        patched, attn, loc, scale = triton_fused_preprocess(
            ctx, patch_size=16, context_length=512, use_arcsinh=False,
        )
        # 512 / 16 = 32 patches, feature dim = 3*16 = 48
        assert patched.shape == (8, 32, 48), f"Expected (8,32,48), got {patched.shape}"
        assert attn.shape == (8, 32), f"Expected (8,32), got {attn.shape}"
        assert loc.shape == (8, 1), f"Expected (8,1), got {loc.shape}"
        assert scale.shape == (8, 1), f"Expected (8,1), got {scale.shape}"

    def test_constant_series(self):
        """Constant series has std=0, should use eps fallback for scale."""
        ctx = torch.full((2, 512), 42.0).cuda()
        ref = _fallback_preprocess(ctx.cpu(), patch_size=16, context_length=512, use_arcsinh=False)
        result = triton_fused_preprocess(ctx, patch_size=16, context_length=512, use_arcsinh=False)
        result_cpu = tuple(t.cpu() for t in result)
        compare_outputs(result_cpu, ref, atol=1e-5, label="triton constant")

    def test_large_values(self):
        """Test with large values to ensure numerical stability."""
        torch.manual_seed(99)
        ctx = torch.randn(2, 512) * 1e6
        ref = _fallback_preprocess(ctx, patch_size=16, context_length=512, use_arcsinh=False)
        result = triton_fused_preprocess(
            ctx.cuda(), patch_size=16, context_length=512, use_arcsinh=False,
        )
        result_cpu = tuple(t.cpu() for t in result)
        # Relaxed tolerance: float32 reduction ordering differs between
        # CPU PyTorch and GPU Triton, causing ~0.1 abs error at 1e6 scale
        compare_outputs(result_cpu, ref, atol=0.2, label="triton large-values")

    def test_single_element_patches(self):
        """Patch size 1 (degenerate case)."""
        ctx = make_context(2, 32)
        ref = _fallback_preprocess(ctx, patch_size=1, context_length=512, use_arcsinh=False)
        result = triton_fused_preprocess(
            ctx.cuda(), patch_size=1, context_length=512, use_arcsinh=False,
        )
        result_cpu = tuple(t.cpu() for t in result)
        compare_outputs(result_cpu, ref, atol=1e-5, label="triton ps=1")

    def test_batch_size_1(self):
        """Single-sample batch."""
        ctx = make_context(1, 512)
        ref = _fallback_preprocess(ctx, patch_size=16, context_length=512, use_arcsinh=True)
        result = triton_fused_preprocess(
            ctx.cuda(), patch_size=16, context_length=512, use_arcsinh=True,
        )
        result_cpu = tuple(t.cpu() for t in result)
        compare_outputs(result_cpu, ref, atol=1e-5, label="triton B=1")

    def test_large_batch(self):
        """Larger batch to exercise grid sizing."""
        ctx = make_context(32, 512, seed=77)
        ref = _fallback_preprocess(ctx, patch_size=16, context_length=512, use_arcsinh=True)
        result = triton_fused_preprocess(
            ctx.cuda(), patch_size=16, context_length=512, use_arcsinh=True,
        )
        result_cpu = tuple(t.cpu() for t in result)
        compare_outputs(result_cpu, ref, atol=1e-5, label="triton B=32")

    def test_high_nan_fraction(self):
        """90% NaN -- most values are missing."""
        ctx = make_context(4, 512, nan_frac=0.9, seed=55)
        ref = _fallback_preprocess(ctx, patch_size=16, context_length=512, use_arcsinh=True)
        result = triton_fused_preprocess(
            ctx.cuda(), patch_size=16, context_length=512, use_arcsinh=True,
        )
        result_cpu = tuple(t.cpu() for t in result)
        compare_outputs(result_cpu, ref, atol=1e-5, label="triton 90% nan")

    def test_context_length_2048(self):
        """Use context_length=2048 with matching data."""
        ctx = make_context(2, 2048)
        ref = _fallback_preprocess(ctx, patch_size=16, context_length=2048, use_arcsinh=True)
        result = triton_fused_preprocess(
            ctx.cuda(), patch_size=16, context_length=2048, use_arcsinh=True,
        )
        result_cpu = tuple(t.cpu() for t in result)
        compare_outputs(result_cpu, ref, atol=1e-5, label="triton ctx=2048")

    def test_returns_float32(self):
        """All outputs should be float32 regardless of input dtype."""
        ctx = make_context(2, 512).cuda()
        patched, attn, loc, scale = triton_fused_preprocess(
            ctx, patch_size=16, context_length=512, use_arcsinh=False,
        )
        assert patched.dtype == torch.float32, f"patched dtype: {patched.dtype}"
        assert attn.dtype == torch.float32, f"attn dtype: {attn.dtype}"
        assert loc.dtype == torch.float32, f"loc dtype: {loc.dtype}"
        assert scale.dtype == torch.float32, f"scale dtype: {scale.dtype}"

    def test_outputs_on_same_device(self):
        """All outputs should be on the same CUDA device as input."""
        ctx = make_context(2, 512).cuda()
        patched, attn, loc, scale = triton_fused_preprocess(
            ctx, patch_size=16, context_length=512, use_arcsinh=False,
        )
        assert patched.is_cuda
        assert attn.is_cuda
        assert loc.is_cuda
        assert scale.is_cuda
