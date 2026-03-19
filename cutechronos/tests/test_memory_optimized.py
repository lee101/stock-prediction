"""Tests for CuteChronos2Model memory optimizations.

Verifies that:
1. Optimized model produces outputs matching the unoptimized original (<1e-4 error)
2. In-place add_ and reshape (vs contiguous().view()) preserve correctness
3. Position ID caching works correctly across different seq lengths
4. profile_allocations method reports meaningful stats
5. torch.inference_mode() usage is compatible with all code paths
"""

import pytest
import torch

from cutechronos.model import CuteChronos2Model
from cutechronos.tests.conftest import build_model_pair, build_cute_only


# -------------------------------------------------------------------
# Test: optimized output matches original within tight tolerance
# -------------------------------------------------------------------

@pytest.mark.model_required
@pytest.mark.parametrize(
    "batch_size,context_length",
    [(2, 512), (1, 256), (3, 128), (1, 64)],
    ids=["B2_L512", "B1_L256", "B3_L128", "B1_L64"],
)
def test_optimized_matches_original(batch_size: int, context_length: int):
    """Verify optimized CuteChronos2Model matches original within 1e-4."""
    original, cute = build_model_pair()

    torch.manual_seed(0)
    context = torch.randn(batch_size, context_length) * 0.1 + 100

    with torch.no_grad():
        orig_out = original(context)
        cute_out = cute(context)

    orig_preds = orig_out.quantile_preds
    assert orig_preds.shape == cute_out.shape, (
        f"Shape mismatch: original {orig_preds.shape} vs cute {cute_out.shape}"
    )

    max_err = (orig_preds - cute_out).abs().max().item()
    assert max_err < 1e-4, (
        f"Max abs error {max_err:.2e} >= 1e-4 for B={batch_size}, L={context_length}"
    )


# -------------------------------------------------------------------
# Test: optimized output matches with multiple output patches
# -------------------------------------------------------------------

@pytest.mark.model_required
@pytest.mark.parametrize("num_output_patches", [1, 2, 4], ids=["P1", "P2", "P4"])
def test_optimized_multi_output_patches(num_output_patches: int):
    """Verify matching with multiple output patches after optimization."""
    original, cute = build_model_pair()

    torch.manual_seed(0)
    context = torch.randn(2, 512) * 0.1 + 100

    with torch.no_grad():
        orig_out = original(context, num_output_patches=num_output_patches)
        cute_out = cute(context, num_output_patches=num_output_patches)

    orig_preds = orig_out.quantile_preds
    assert orig_preds.shape == cute_out.shape
    max_err = (orig_preds - cute_out).abs().max().item()
    assert max_err < 1e-4, (
        f"Max abs error {max_err:.2e} >= 1e-4 with {num_output_patches} output patches"
    )


# -------------------------------------------------------------------
# Test: position_ids caching correctness
# -------------------------------------------------------------------

def test_position_ids_caching():
    """Verify position_ids are cached and reused correctly."""
    cute = build_cute_only()

    torch.manual_seed(0)
    context1 = torch.randn(2, 512) * 0.1 + 100

    with torch.no_grad():
        out1 = cute(context1)

    cached_len_1 = cute._cached_seq_length
    cached_ids_1 = cute._cached_position_ids.clone()
    assert cached_len_1 > 0, "Position IDs should be cached after first forward"

    # Second call with same context_length should reuse cache
    with torch.no_grad():
        out2 = cute(context1)

    assert cute._cached_seq_length == cached_len_1, "Cache length should not change"
    assert torch.equal(cute._cached_position_ids, cached_ids_1), "Cached IDs should be identical"
    assert torch.equal(out1, out2), "Outputs should be identical with cached position IDs"


def test_position_ids_cache_invalidation():
    """Verify position_ids cache is invalidated when seq_length changes."""
    cute = build_cute_only()

    torch.manual_seed(0)
    context_512 = torch.randn(2, 512) * 0.1 + 100
    context_256 = torch.randn(1, 256) * 0.1 + 100

    with torch.no_grad():
        cute(context_512)
        len_after_512 = cute._cached_seq_length

        cute(context_256)
        len_after_256 = cute._cached_seq_length

    assert len_after_512 != len_after_256, (
        "Cache should be invalidated when seq_length changes"
    )


# -------------------------------------------------------------------
# Test: reshape vs contiguous().view() equivalence
# -------------------------------------------------------------------

def test_reshape_equivalence():
    """Verify that reshape produces identical results to contiguous().view()."""
    B, S, H, D = 2, 34, 12, 64
    x = torch.randn(B, H, S, D)

    original = x.transpose(1, 2).contiguous().view(B, S, H * D)
    optimized = x.transpose(1, 2).reshape(B, S, H * D)

    assert torch.equal(original, optimized), "reshape should produce identical results to contiguous().view()"


def test_permute_reshape_equivalence():
    """Verify permute+reshape produces identical results to permute+contiguous+view."""
    B, N, Q, P = 2, 3, 21, 16
    x = torch.randn(B, N, Q, P)

    original = x.permute(0, 2, 1, 3).contiguous().view(B, Q, N * P)
    optimized = x.permute(0, 2, 1, 3).reshape(B, Q, N * P)

    assert torch.equal(original, optimized), "permute+reshape should match permute+contiguous+view"


# -------------------------------------------------------------------
# Test: in-place residual add equivalence
# -------------------------------------------------------------------

def test_inplace_add_equivalence():
    """Verify in-place add_ produces identical results to out-of-place add."""
    torch.manual_seed(0)
    x = torch.randn(2, 34, 768)
    y = torch.randn(2, 34, 768)

    result_oop = x + y

    x_clone = x.clone()
    x_clone.add_(y)

    assert torch.equal(result_oop, x_clone), "add_ should produce identical results to +"


# -------------------------------------------------------------------
# Test: inference_mode compatibility
# -------------------------------------------------------------------

def test_inference_mode_forward():
    """Verify model works correctly under torch.inference_mode()."""
    cute = build_cute_only()

    torch.manual_seed(0)
    context = torch.randn(2, 512) * 0.1 + 100

    with torch.no_grad():
        out_nograd = cute(context)

    with torch.inference_mode():
        out_inference = cute(context)

    max_err = (out_nograd - out_inference).abs().max().item()
    assert max_err < 1e-6, (
        f"inference_mode output differs from no_grad by {max_err:.2e}"
    )


def test_inference_mode_with_nan_inputs():
    """Verify inference_mode handles NaN inputs correctly."""
    cute = build_cute_only()

    torch.manual_seed(0)
    context = torch.randn(2, 512) * 0.1 + 100
    context[0, :50] = float("nan")
    context[1, 200:300] = float("nan")

    with torch.inference_mode():
        out = cute(context)

    assert not torch.isnan(out).any(), "Output should not contain NaN"
    assert out.shape == (2, 21, 16), f"Unexpected shape: {out.shape}"


# -------------------------------------------------------------------
# Test: profile_allocations method
# -------------------------------------------------------------------

def test_profile_allocations_cpu():
    """Verify profile_allocations works on CPU (returns zero stats)."""
    cute = build_cute_only().cpu()

    torch.manual_seed(0)
    context = torch.randn(2, 128) * 0.1 + 100

    with cute.profile_allocations() as stats:
        with torch.no_grad():
            out = cute(context)

    assert "allocation_count" in stats, "Should have allocation_count"
    assert "peak_memory_mb" in stats, "Should have peak_memory_mb"
    assert "allocated_memory_mb" in stats, "Should have allocated_memory_mb"
    assert "memory_delta_mb" in stats, "Should have memory_delta_mb"

    assert stats["allocation_count"] == 0
    assert stats["peak_memory_mb"] == 0.0

    assert out.shape[0] == 2
    assert not torch.isnan(out).any()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_profile_allocations_cuda():
    """Verify profile_allocations reports non-zero stats on CUDA."""
    cute = build_cute_only().cuda()

    torch.manual_seed(0)
    context = torch.randn(2, 128, device="cuda") * 0.1 + 100

    with cute.profile_allocations() as stats:
        with torch.no_grad():
            out = cute(context)

    assert stats["peak_memory_mb"] > 0, f"Peak memory should be > 0, got {stats['peak_memory_mb']}"
    assert stats["allocation_count"] >= 0, f"Allocation count should be >= 0, got {stats['allocation_count']}"

    assert out.shape == (2, 21, 16)
    assert not torch.isnan(out).any()
    print(f"\n  CUDA profile stats: alloc_count={stats['allocation_count']}, "
          f"peak={stats['peak_memory_mb']:.1f}MB, delta={stats['memory_delta_mb']:.1f}MB")


# -------------------------------------------------------------------
# Test: determinism preserved after optimizations
# -------------------------------------------------------------------

def test_determinism_after_optimization():
    """Two forward passes should give identical results after optimization."""
    cute = build_cute_only()

    torch.manual_seed(0)
    context = torch.randn(2, 512) * 0.1 + 100

    with torch.no_grad():
        out1 = cute(context.clone())
        out2 = cute(context.clone())

    assert torch.equal(out1, out2), "Forward pass should be deterministic after optimization"


def test_determinism_across_batch_sizes():
    """Same series should produce same predictions regardless of batch composition."""
    cute = build_cute_only()

    torch.manual_seed(0)
    single_context = torch.randn(1, 256) * 0.1 + 100

    with torch.no_grad():
        out_single = cute(single_context.clone())

    batch_context = torch.cat([single_context, torch.randn(1, 256) * 0.1 + 50], dim=0)
    with torch.no_grad():
        out_batch = cute(batch_context.clone())

    max_err = (out_single[0] - out_batch[0]).abs().max().item()
    assert max_err < 1e-4, (
        f"Single vs batch[0] max error {max_err:.2e} >= 1e-4"
    )


# -------------------------------------------------------------------
# Test: position_ids correctness
# -------------------------------------------------------------------

def test_position_ids_values():
    """Verify cached position IDs have correct values."""
    cute = build_cute_only()

    torch.manual_seed(0)
    context = torch.randn(2, 512) * 0.1 + 100

    with torch.no_grad():
        cute(context)

    pos_ids = cute._cached_position_ids
    seq_len = cute._cached_seq_length

    assert pos_ids.shape == (1, seq_len), f"Expected shape (1, {seq_len}), got {pos_ids.shape}"
    expected = torch.arange(seq_len, dtype=torch.long, device=pos_ids.device).unsqueeze(0)
    assert torch.equal(pos_ids, expected), "Position IDs should be sequential from 0"


# -------------------------------------------------------------------
# Test: memory optimization regression check (before vs after)
# -------------------------------------------------------------------

def test_no_unnecessary_contiguous_calls():
    """Verify the model source code has no remaining .contiguous().view() patterns."""
    import inspect
    source = inspect.getsource(CuteChronos2Model)
    assert ".contiguous().view(" not in source, (
        "Found .contiguous().view() in CuteChronos2Model - should use .reshape() instead"
    )

    from cutechronos.model import TimeSelfAttentionFallback, GroupSelfAttentionFallback
    for cls in [TimeSelfAttentionFallback, GroupSelfAttentionFallback]:
        src = inspect.getsource(cls)
        assert ".contiguous().view(" not in src, (
            f"Found .contiguous().view() in {cls.__name__} - should use .reshape() instead"
        )


def test_residual_uses_inplace():
    """Verify that residual connections use in-place add_."""
    import inspect
    from cutechronos.model import (
        TimeSelfAttentionFallback,
        GroupSelfAttentionFallback,
        FeedForwardFallback,
    )
    for cls in [TimeSelfAttentionFallback, GroupSelfAttentionFallback, FeedForwardFallback]:
        src = inspect.getsource(cls.forward)
        assert ".add_(" in src, (
            f"{cls.__name__}.forward should use in-place .add_() for residual connections"
        )
