"""Tests for alternative attention backends (SDPA, FlexAttention, eager).

Verifies that all backends match the reference unscaled attention:
    softmax(Q @ K^T + mask) @ V   (no 1/sqrt(d_k) scaling)

within max abs error < 1e-4.
"""

import pytest
import torch

from cutechronos.modules.flex_attention import (
    sdpa_unscaled_attention,
    flex_unscaled_attention,
    eager_unscaled_attention,
    get_attention_backend,
    get_best_attention_backend,
    list_backends,
    benchmark_backends,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _reference_unscaled_attention(q, k, v, mask=None):
    """Reference implementation: no scaling, float32 softmax."""
    scores = torch.matmul(q, k.transpose(-2, -1))
    if mask is not None:
        scores = scores + mask
    weights = torch.softmax(scores.float(), dim=-1).to(q.dtype)
    return torch.matmul(weights, v)


def _make_causal_mask(batch_size, num_heads, seq_len, device="cpu"):
    """Create a causal additive mask: (B, num_heads, S, S)."""
    mask = torch.zeros(batch_size, num_heads, seq_len, seq_len, device=device)
    causal = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1).bool()
    mask[:, :, causal] = float("-inf")
    return mask


def _make_padding_mask(batch_size, seq_len, pad_count, device="cpu"):
    """Create a padding-style additive mask: (B, 1, 1, S).

    Masks the last ``pad_count`` tokens for all batches.
    """
    mask = torch.zeros(batch_size, 1, 1, seq_len, device=device)
    if pad_count > 0:
        mask[:, :, :, -pad_count:] = float("-inf")
    return mask


# ---------------------------------------------------------------------------
# Test shapes from the task spec
# ---------------------------------------------------------------------------

_SHAPES = [
    (1, 12, 34, 64),
    (4, 12, 130, 64),
    (2, 12, 514, 64),
]

_SHAPE_IDS = [f"B{s[0]}_H{s[1]}_S{s[2]}_D{s[3]}" for s in _SHAPES]


# ---------------------------------------------------------------------------
# CPU tests (always run)
# ---------------------------------------------------------------------------

class TestSDPAUnscaledAttentionCPU:
    """Test SDPA backend on CPU."""

    @pytest.mark.parametrize("shape", _SHAPES, ids=_SHAPE_IDS)
    def test_no_mask(self, shape):
        B, H, S, D = shape
        torch.manual_seed(42)
        q = torch.randn(B, H, S, D)
        k = torch.randn(B, H, S, D)
        v = torch.randn(B, H, S, D)

        out = sdpa_unscaled_attention(q, k, v)
        ref = _reference_unscaled_attention(q, k, v)

        max_err = (out - ref).abs().max().item()
        assert max_err < 1e-4, f"SDPA no-mask max err {max_err:.2e} >= 1e-4"

    @pytest.mark.parametrize("shape", _SHAPES, ids=_SHAPE_IDS)
    def test_causal_mask(self, shape):
        B, H, S, D = shape
        torch.manual_seed(42)
        q = torch.randn(B, H, S, D)
        k = torch.randn(B, H, S, D)
        v = torch.randn(B, H, S, D)
        mask = _make_causal_mask(B, H, S)

        out = sdpa_unscaled_attention(q, k, v, mask)
        ref = _reference_unscaled_attention(q, k, v, mask)

        max_err = (out - ref).abs().max().item()
        assert max_err < 1e-4, f"SDPA causal max err {max_err:.2e} >= 1e-4"

    @pytest.mark.parametrize("shape", _SHAPES, ids=_SHAPE_IDS)
    def test_padding_mask(self, shape):
        B, H, S, D = shape
        torch.manual_seed(42)
        q = torch.randn(B, H, S, D)
        k = torch.randn(B, H, S, D)
        v = torch.randn(B, H, S, D)
        pad_count = max(1, S // 10)
        mask = _make_padding_mask(B, S, pad_count)

        out = sdpa_unscaled_attention(q, k, v, mask)
        ref = _reference_unscaled_attention(q, k, v, mask)

        max_err = (out - ref).abs().max().item()
        assert max_err < 1e-4, f"SDPA padding max err {max_err:.2e} >= 1e-4"


class TestFlexUnscaledAttentionCPU:
    """Test FlexAttention backend on CPU (falls back to SDPA for masked case)."""

    @pytest.mark.parametrize("shape", _SHAPES, ids=_SHAPE_IDS)
    def test_no_mask(self, shape):
        B, H, S, D = shape
        torch.manual_seed(42)
        q = torch.randn(B, H, S, D)
        k = torch.randn(B, H, S, D)
        v = torch.randn(B, H, S, D)

        out = flex_unscaled_attention(q, k, v)
        ref = _reference_unscaled_attention(q, k, v)

        max_err = (out - ref).abs().max().item()
        assert max_err < 1e-4, f"Flex no-mask max err {max_err:.2e} >= 1e-4"

    @pytest.mark.parametrize("shape", _SHAPES, ids=_SHAPE_IDS)
    def test_with_mask(self, shape):
        B, H, S, D = shape
        torch.manual_seed(42)
        q = torch.randn(B, H, S, D)
        k = torch.randn(B, H, S, D)
        v = torch.randn(B, H, S, D)
        mask = _make_causal_mask(B, H, S)

        out = flex_unscaled_attention(q, k, v, mask)
        ref = _reference_unscaled_attention(q, k, v, mask)

        max_err = (out - ref).abs().max().item()
        assert max_err < 1e-4, f"Flex mask max err {max_err:.2e} >= 1e-4"


class TestEagerUnscaledAttentionCPU:
    """Test eager (reference) backend on CPU."""

    @pytest.mark.parametrize("shape", _SHAPES, ids=_SHAPE_IDS)
    def test_no_mask(self, shape):
        B, H, S, D = shape
        torch.manual_seed(42)
        q = torch.randn(B, H, S, D)
        k = torch.randn(B, H, S, D)
        v = torch.randn(B, H, S, D)

        out = eager_unscaled_attention(q, k, v)
        ref = _reference_unscaled_attention(q, k, v)

        max_err = (out - ref).abs().max().item()
        assert max_err < 1e-4, f"Eager no-mask max err {max_err:.2e} >= 1e-4"

    @pytest.mark.parametrize("shape", _SHAPES, ids=_SHAPE_IDS)
    def test_with_mask(self, shape):
        B, H, S, D = shape
        torch.manual_seed(42)
        q = torch.randn(B, H, S, D)
        k = torch.randn(B, H, S, D)
        v = torch.randn(B, H, S, D)
        mask = _make_causal_mask(B, H, S)

        out = eager_unscaled_attention(q, k, v, mask)
        ref = _reference_unscaled_attention(q, k, v, mask)

        max_err = (out - ref).abs().max().item()
        assert max_err < 1e-4, f"Eager mask max err {max_err:.2e} >= 1e-4"


# ---------------------------------------------------------------------------
# CUDA tests
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
class TestSDPAUnscaledAttentionCUDA:
    """Test SDPA backend on CUDA."""

    @pytest.mark.parametrize("shape", _SHAPES, ids=_SHAPE_IDS)
    def test_no_mask(self, shape):
        B, H, S, D = shape
        torch.manual_seed(42)
        q = torch.randn(B, H, S, D, device="cuda")
        k = torch.randn(B, H, S, D, device="cuda")
        v = torch.randn(B, H, S, D, device="cuda")

        out = sdpa_unscaled_attention(q, k, v)
        ref = _reference_unscaled_attention(q, k, v)

        max_err = (out - ref).abs().max().item()
        assert max_err < 1e-4, f"SDPA CUDA no-mask max err {max_err:.2e} >= 1e-4"

    @pytest.mark.parametrize("shape", _SHAPES, ids=_SHAPE_IDS)
    def test_causal_mask(self, shape):
        B, H, S, D = shape
        torch.manual_seed(42)
        q = torch.randn(B, H, S, D, device="cuda")
        k = torch.randn(B, H, S, D, device="cuda")
        v = torch.randn(B, H, S, D, device="cuda")
        mask = _make_causal_mask(B, H, S, device="cuda")

        out = sdpa_unscaled_attention(q, k, v, mask)
        ref = _reference_unscaled_attention(q, k, v, mask)

        max_err = (out - ref).abs().max().item()
        assert max_err < 1e-4, f"SDPA CUDA causal max err {max_err:.2e} >= 1e-4"

    @pytest.mark.parametrize("shape", _SHAPES, ids=_SHAPE_IDS)
    def test_padding_mask(self, shape):
        B, H, S, D = shape
        torch.manual_seed(42)
        q = torch.randn(B, H, S, D, device="cuda")
        k = torch.randn(B, H, S, D, device="cuda")
        v = torch.randn(B, H, S, D, device="cuda")
        pad_count = max(1, S // 10)
        mask = _make_padding_mask(B, S, pad_count, device="cuda")

        out = sdpa_unscaled_attention(q, k, v, mask)
        ref = _reference_unscaled_attention(q, k, v, mask)

        max_err = (out - ref).abs().max().item()
        assert max_err < 1e-4, f"SDPA CUDA padding max err {max_err:.2e} >= 1e-4"

    def test_broadcast_mask_shapes(self):
        """SDPA handles various broadcast mask shapes correctly."""
        B, H, S, D = 2, 12, 34, 64
        torch.manual_seed(42)
        q = torch.randn(B, H, S, D, device="cuda")
        k = torch.randn(B, H, S, D, device="cuda")
        v = torch.randn(B, H, S, D, device="cuda")

        # (B, 1, 1, S) - typical Chronos2 time attention mask
        mask_1 = torch.zeros(B, 1, 1, S, device="cuda")
        mask_1[:, :, :, -5:] = float("-inf")
        out1 = sdpa_unscaled_attention(q, k, v, mask_1)
        ref1 = _reference_unscaled_attention(q, k, v, mask_1)
        assert (out1 - ref1).abs().max().item() < 1e-4

        # (B, H, S, S) - fully expanded mask
        mask_2 = mask_1.expand(B, H, S, S).contiguous()
        out2 = sdpa_unscaled_attention(q, k, v, mask_2)
        ref2 = _reference_unscaled_attention(q, k, v, mask_2)
        assert (out2 - ref2).abs().max().item() < 1e-4


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
class TestFlexUnscaledAttentionCUDA:
    """Test FlexAttention backend on CUDA."""

    @pytest.mark.parametrize("shape", _SHAPES, ids=_SHAPE_IDS)
    def test_no_mask(self, shape):
        B, H, S, D = shape
        torch.manual_seed(42)
        q = torch.randn(B, H, S, D, device="cuda")
        k = torch.randn(B, H, S, D, device="cuda")
        v = torch.randn(B, H, S, D, device="cuda")

        out = flex_unscaled_attention(q, k, v)
        ref = _reference_unscaled_attention(q, k, v)

        max_err = (out - ref).abs().max().item()
        assert max_err < 1e-4, f"Flex CUDA no-mask max err {max_err:.2e} >= 1e-4"

    @pytest.mark.parametrize("shape", _SHAPES, ids=_SHAPE_IDS)
    def test_with_mask_falls_back_to_sdpa(self, shape):
        """When mask is provided, flex falls back to SDPA."""
        B, H, S, D = shape
        torch.manual_seed(42)
        q = torch.randn(B, H, S, D, device="cuda")
        k = torch.randn(B, H, S, D, device="cuda")
        v = torch.randn(B, H, S, D, device="cuda")
        mask = _make_causal_mask(B, H, S, device="cuda")

        out_flex = flex_unscaled_attention(q, k, v, mask)
        out_sdpa = sdpa_unscaled_attention(q, k, v, mask)

        # Should produce identical results since flex falls back to SDPA
        max_err = (out_flex - out_sdpa).abs().max().item()
        assert max_err == 0.0, f"Flex should match SDPA when masked, got err {max_err:.2e}"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
class TestEagerUnscaledAttentionCUDA:
    """Test eager backend on CUDA."""

    @pytest.mark.parametrize("shape", _SHAPES, ids=_SHAPE_IDS)
    def test_matches_reference(self, shape):
        B, H, S, D = shape
        torch.manual_seed(42)
        q = torch.randn(B, H, S, D, device="cuda")
        k = torch.randn(B, H, S, D, device="cuda")
        v = torch.randn(B, H, S, D, device="cuda")
        mask = _make_causal_mask(B, H, S, device="cuda")

        out = eager_unscaled_attention(q, k, v, mask)
        ref = _reference_unscaled_attention(q, k, v, mask)

        max_err = (out - ref).abs().max().item()
        assert max_err < 1e-4, f"Eager CUDA max err {max_err:.2e} >= 1e-4"


# ---------------------------------------------------------------------------
# Cross-backend consistency tests
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
class TestCrossBackendConsistency:
    """All backends should produce the same output for the same inputs."""

    @pytest.mark.parametrize("shape", _SHAPES, ids=_SHAPE_IDS)
    def test_all_backends_match_no_mask(self, shape):
        B, H, S, D = shape
        torch.manual_seed(42)
        q = torch.randn(B, H, S, D, device="cuda")
        k = torch.randn(B, H, S, D, device="cuda")
        v = torch.randn(B, H, S, D, device="cuda")

        ref = _reference_unscaled_attention(q, k, v)
        out_sdpa = sdpa_unscaled_attention(q, k, v)
        out_flex = flex_unscaled_attention(q, k, v)
        out_eager = eager_unscaled_attention(q, k, v)

        for name, out in [("sdpa", out_sdpa), ("flex", out_flex), ("eager", out_eager)]:
            max_err = (out - ref).abs().max().item()
            assert max_err < 1e-4, f"{name} vs ref (no mask) max err {max_err:.2e} >= 1e-4"

    @pytest.mark.parametrize("shape", _SHAPES, ids=_SHAPE_IDS)
    def test_all_backends_match_with_mask(self, shape):
        B, H, S, D = shape
        torch.manual_seed(42)
        q = torch.randn(B, H, S, D, device="cuda")
        k = torch.randn(B, H, S, D, device="cuda")
        v = torch.randn(B, H, S, D, device="cuda")
        mask = _make_padding_mask(B, S, max(1, S // 10), device="cuda")

        ref = _reference_unscaled_attention(q, k, v, mask)
        out_sdpa = sdpa_unscaled_attention(q, k, v, mask)
        out_flex = flex_unscaled_attention(q, k, v, mask)
        out_eager = eager_unscaled_attention(q, k, v, mask)

        for name, out in [("sdpa", out_sdpa), ("flex", out_flex), ("eager", out_eager)]:
            max_err = (out - ref).abs().max().item()
            assert max_err < 1e-4, f"{name} vs ref (mask) max err {max_err:.2e} >= 1e-4"


# ---------------------------------------------------------------------------
# API tests
# ---------------------------------------------------------------------------

class TestBackendAPI:
    """Test the backend registry and selection API."""

    def test_list_backends(self):
        backends = list_backends()
        assert "sdpa" in backends
        assert "flex" in backends
        assert "eager" in backends

    def test_get_attention_backend(self):
        fn = get_attention_backend("sdpa")
        assert fn is sdpa_unscaled_attention

        fn = get_attention_backend("flex")
        assert fn is flex_unscaled_attention

        fn = get_attention_backend("eager")
        assert fn is eager_unscaled_attention

    def test_get_unknown_backend_raises(self):
        with pytest.raises(KeyError, match="Unknown attention backend"):
            get_attention_backend("nonexistent")

    def test_output_shapes(self):
        """All backends produce correct output shapes."""
        for shape in _SHAPES:
            B, H, S, D = shape
            q = torch.randn(B, H, S, D)
            k = torch.randn(B, H, S, D)
            v = torch.randn(B, H, S, D)

            for name in list_backends():
                if name == "triton":
                    continue  # Triton requires CUDA tensors
                fn = get_attention_backend(name)
                out = fn(q, k, v)
                assert out.shape == (B, H, S, D), (
                    f"{name} output shape {out.shape} != expected {(B, H, S, D)}"
                )

    def test_deterministic(self):
        """Two calls with same input produce identical output."""
        B, H, S, D = 2, 12, 34, 64
        torch.manual_seed(42)
        q = torch.randn(B, H, S, D)
        k = torch.randn(B, H, S, D)
        v = torch.randn(B, H, S, D)

        for name in list_backends():
            if name == "triton":
                continue  # Triton requires CUDA tensors
            fn = get_attention_backend(name)
            out1 = fn(q, k, v)
            out2 = fn(q, k, v)
            assert torch.equal(out1, out2), f"{name} is not deterministic"


# ---------------------------------------------------------------------------
# Benchmark (prints timing, always passes)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
class TestBenchmark:
    """Benchmark tests that print timing comparisons."""

    def test_benchmark_with_mask(self, capsys):
        """Benchmark all backends with mask and print results."""
        timings = benchmark_backends(
            batch_size=4,
            num_heads=12,
            seq_len=130,
            d_kv=64,
            with_mask=True,
            warmup=10,
            repeats=50,
        )

        print("\n--- Benchmark: B=4, H=12, S=130, D=64, with_mask=True ---")
        for name, ms in sorted(timings.items(), key=lambda x: x[1]):
            print(f"  {name:>8s}: {ms:.3f} ms")

        # Sanity: all timings should be positive
        for name, ms in timings.items():
            assert ms > 0, f"{name} timing should be positive"

    def test_benchmark_no_mask(self, capsys):
        """Benchmark all backends without mask and print results."""
        timings = benchmark_backends(
            batch_size=4,
            num_heads=12,
            seq_len=130,
            d_kv=64,
            with_mask=False,
            warmup=10,
            repeats=50,
        )

        print("\n--- Benchmark: B=4, H=12, S=130, D=64, with_mask=False ---")
        for name, ms in sorted(timings.items(), key=lambda x: x[1]):
            print(f"  {name:>8s}: {ms:.3f} ms")

        for name, ms in timings.items():
            assert ms > 0, f"{name} timing should be positive"

    def test_benchmark_large_seq(self, capsys):
        """Benchmark with larger sequence length (closer to real usage)."""
        timings = benchmark_backends(
            batch_size=2,
            num_heads=12,
            seq_len=514,
            d_kv=64,
            with_mask=True,
            warmup=10,
            repeats=50,
        )

        print("\n--- Benchmark: B=2, H=12, S=514, D=64, with_mask=True ---")
        for name, ms in sorted(timings.items(), key=lambda x: x[1]):
            print(f"  {name:>8s}: {ms:.3f} ms")

        for name, ms in timings.items():
            assert ms > 0, f"{name} timing should be positive"

    def test_get_best_backend(self, capsys):
        """Test that get_best_attention_backend returns a valid backend."""
        name, fn = get_best_attention_backend(with_mask=True, device="cuda")
        print(f"\n--- Best backend (masked): {name} ---")
        assert name in list_backends()
        assert callable(fn)

        name_nomask, fn_nomask = get_best_attention_backend(with_mask=False, device="cuda")
        print(f"--- Best backend (no mask): {name_nomask} ---")
        assert name_nomask in list_backends()
        assert callable(fn_nomask)


# ---------------------------------------------------------------------------
# Integration: wire into FusedTimeSelfAttention
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
class TestIntegrationWithTimeSelfAttention:
    """Test that SDPA backend works as a drop-in for FusedTimeSelfAttention."""

    def test_sdpa_matches_fallback_in_attention_step(self):
        """SDPA produces same result as the eager fallback for the attention
        step within FusedTimeSelfAttention."""
        B, H, S, D = 2, 12, 34, 64
        torch.manual_seed(42)

        q = torch.randn(B, H, S, D, device="cuda")
        k = torch.randn(B, H, S, D, device="cuda")
        v = torch.randn(B, H, S, D, device="cuda")

        # Simulate Chronos2 time attention mask: (B, 1, 1, S)
        mask = torch.zeros(B, 1, 1, S, device="cuda")
        mask[1, 0, 0, -5:] = float("-inf")

        from cutechronos.modules._fallbacks import unscaled_attention as fallback_attn

        out_fallback = fallback_attn(q, k, v, mask)
        out_sdpa = sdpa_unscaled_attention(q, k, v, mask)

        max_err = (out_sdpa - out_fallback).abs().max().item()
        assert max_err < 1e-4, (
            f"SDPA vs fallback max err {max_err:.2e} >= 1e-4 "
            f"(should be drop-in compatible)"
        )
