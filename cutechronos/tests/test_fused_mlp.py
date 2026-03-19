"""Tests for Triton fused MLP kernels.

Tests fused_linear_relu and fused_mlp_relu against PyTorch reference
implementations across multiple shapes and dtypes, with timing comparison.

Shapes tested: (34, 768), (520, 768), (4160, 768) -- matching Chronos2
typical sequence lengths (34 = small context, 520 = ~8k context / 16 patch,
4160 = large batched inference).

Chronos2 MLP dimensions: d_model=768, d_ff=3072, activation=ReLU.
"""

import time

import pytest
import torch
import torch.nn.functional as F

from cutechronos.triton_kernels.fused_mlp import fused_linear_relu, fused_mlp_relu


# ---------------------------------------------------------------------------
# Constants matching Chronos2-base config
# ---------------------------------------------------------------------------

D_MODEL = 768
D_FF = 3072

# Test shapes: (M, K) where K = d_model = 768
TEST_SHAPES = [
    (34, D_MODEL),
    (520, D_MODEL),
    (4160, D_MODEL),
]


# ---------------------------------------------------------------------------
# Reference implementations
# ---------------------------------------------------------------------------

def ref_linear_relu(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """Reference: F.relu(F.linear(x, weight))"""
    return F.relu(F.linear(x, weight))


def ref_mlp_relu(
    x: torch.Tensor, wi: torch.Tensor, wo: torch.Tensor
) -> torch.Tensor:
    """Reference: F.linear(F.relu(F.linear(x, wi)), wo)"""
    return F.linear(F.relu(F.linear(x, wi)), wo)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_weights(d_in: int, d_hidden: int, d_out: int, dtype, device="cuda"):
    """Create wi and wo weight matrices."""
    wi = torch.randn(d_hidden, d_in, device=device, dtype=dtype) * 0.02
    wo = torch.randn(d_out, d_hidden, device=device, dtype=dtype) * 0.02
    return wi, wo


def benchmark_fn(fn, warmup=10, iters=50):
    """Benchmark a function, returning median time in microseconds."""
    # Warmup
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    times = []
    for _ in range(iters):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1e6)  # microseconds

    times.sort()
    return times[len(times) // 2]  # median


# ===========================================================================
# Tests for fused_linear_relu
# ===========================================================================

class TestFusedLinearReluFP32:
    """Test fused_linear_relu matches reference in FP32."""

    @pytest.mark.parametrize(
        "shape", TEST_SHAPES, ids=lambda s: f"M={s[0]}_K={s[1]}"
    )
    def test_correctness(self, shape):
        M, K = shape
        torch.manual_seed(42)
        x = torch.randn(M, K, device="cuda", dtype=torch.float32)
        wi, _ = make_weights(K, D_FF, D_MODEL, torch.float32)

        ref = ref_linear_relu(x, wi)
        out = fused_linear_relu(x, wi)

        max_err = (ref - out).abs().max().item()
        assert max_err < 1e-5, f"FP32 fused_linear_relu max error: {max_err}"


class TestFusedLinearReluBF16:
    """Test fused_linear_relu matches reference in BF16."""

    @pytest.mark.parametrize(
        "shape", TEST_SHAPES, ids=lambda s: f"M={s[0]}_K={s[1]}"
    )
    def test_correctness(self, shape):
        M, K = shape
        torch.manual_seed(42)
        x = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
        wi, _ = make_weights(K, D_FF, D_MODEL, torch.bfloat16)

        ref = ref_linear_relu(x, wi)
        out = fused_linear_relu(x, wi)

        # BF16 has lower precision: accumulation in FP32 but inputs are BF16
        max_err = (ref - out).abs().max().item()
        assert max_err < 1e-2, f"BF16 fused_linear_relu max error: {max_err}"


# ===========================================================================
# Tests for fused_mlp_relu
# ===========================================================================

class TestFusedMlpReluFP32:
    """Test fused_mlp_relu matches reference in FP32."""

    @pytest.mark.parametrize(
        "shape", TEST_SHAPES, ids=lambda s: f"M={s[0]}_K={s[1]}"
    )
    def test_correctness(self, shape):
        M, K = shape
        torch.manual_seed(42)
        x = torch.randn(M, K, device="cuda", dtype=torch.float32)
        wi, wo = make_weights(K, D_FF, D_MODEL, torch.float32)

        ref = ref_mlp_relu(x, wi, wo)
        out = fused_mlp_relu(x, wi, wo)

        max_err = (ref - out).abs().max().item()
        assert max_err < 1e-5, f"FP32 fused_mlp_relu max error: {max_err}"


class TestFusedMlpReluBF16:
    """Test fused_mlp_relu matches reference in BF16."""

    @pytest.mark.parametrize(
        "shape", TEST_SHAPES, ids=lambda s: f"M={s[0]}_K={s[1]}"
    )
    def test_correctness(self, shape):
        M, K = shape
        torch.manual_seed(42)
        x = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
        wi, wo = make_weights(K, D_FF, D_MODEL, torch.bfloat16)

        ref = ref_mlp_relu(x, wi, wo)
        out = fused_mlp_relu(x, wi, wo)

        # BF16: two matmuls compound error, use slightly larger tolerance
        max_err = (ref - out).abs().max().item()
        assert max_err < 5e-2, f"BF16 fused_mlp_relu max error: {max_err}"


# ===========================================================================
# 3D input tests (batch, seq, d_model)
# ===========================================================================

class TestBatchedInput:
    """Test that 3D inputs (batch, seq, d_model) work correctly."""

    def test_fused_linear_relu_3d(self):
        torch.manual_seed(42)
        x = torch.randn(4, 34, D_MODEL, device="cuda", dtype=torch.float32)
        wi, _ = make_weights(D_MODEL, D_FF, D_MODEL, torch.float32)

        ref = ref_linear_relu(x.reshape(-1, D_MODEL), wi).reshape(4, 34, D_FF)
        out = fused_linear_relu(x, wi)

        max_err = (ref - out).abs().max().item()
        assert max_err < 1e-5, f"3D fused_linear_relu max error: {max_err}"
        assert out.shape == (4, 34, D_FF)

    def test_fused_mlp_relu_3d(self):
        torch.manual_seed(42)
        x = torch.randn(4, 34, D_MODEL, device="cuda", dtype=torch.float32)
        wi, wo = make_weights(D_MODEL, D_FF, D_MODEL, torch.float32)

        ref = ref_mlp_relu(x.reshape(-1, D_MODEL), wi, wo).reshape(4, 34, D_MODEL)
        out = fused_mlp_relu(x, wi, wo)

        max_err = (ref - out).abs().max().item()
        assert max_err < 1e-5, f"3D fused_mlp_relu max error: {max_err}"
        assert out.shape == (4, 34, D_MODEL)


# ===========================================================================
# Edge cases
# ===========================================================================

class TestEdgeCases:
    """Edge case tests."""

    def test_single_row(self):
        """Single-row input."""
        torch.manual_seed(42)
        x = torch.randn(1, D_MODEL, device="cuda", dtype=torch.float32)
        wi, wo = make_weights(D_MODEL, D_FF, D_MODEL, torch.float32)

        ref = ref_mlp_relu(x, wi, wo)
        out = fused_mlp_relu(x, wi, wo)

        max_err = (ref - out).abs().max().item()
        assert max_err < 1e-5, f"Single row max error: {max_err}"

    def test_relu_zeros_negative(self):
        """ReLU should zero out negative values."""
        torch.manual_seed(42)
        # Use large negative input so pre-activation is mostly negative
        x = torch.full((4, D_MODEL), -10.0, device="cuda", dtype=torch.float32)
        wi, _ = make_weights(D_MODEL, D_FF, D_MODEL, torch.float32)

        ref = ref_linear_relu(x, wi)
        out = fused_linear_relu(x, wi)

        # Slightly relaxed tolerance for extreme constant inputs where
        # sum-of-768-terms at magnitude 0.2 amplifies FP32 rounding
        max_err = (ref - out).abs().max().item()
        assert max_err < 5e-5, f"Negative input max error: {max_err}"
        # Verify ReLU is working: most values should be zero
        ref_zero_frac = (ref == 0).float().mean().item()
        out_zero_frac = (out == 0).float().mean().item()
        assert abs(ref_zero_frac - out_zero_frac) < 0.01

    def test_deterministic(self):
        """Same input gives same output across calls."""
        torch.manual_seed(42)
        x = torch.randn(34, D_MODEL, device="cuda", dtype=torch.float32)
        wi, wo = make_weights(D_MODEL, D_FF, D_MODEL, torch.float32)

        out1 = fused_mlp_relu(x.clone(), wi, wo)
        out2 = fused_mlp_relu(x.clone(), wi, wo)
        assert torch.equal(out1, out2), "Outputs differ across calls"


# ===========================================================================
# Benchmark (prints timing, always passes)
# ===========================================================================

class TestBenchmark:
    """Benchmark comparison: Triton fused vs PyTorch reference.

    These tests always pass but print timing information.
    """

    @pytest.mark.parametrize(
        "shape", TEST_SHAPES, ids=lambda s: f"M={s[0]}"
    )
    @pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16], ids=["fp32", "bf16"])
    def test_benchmark_linear_relu(self, shape, dtype, capsys):
        """Benchmark fused_linear_relu vs F.relu(F.linear(...))."""
        M, K = shape
        torch.manual_seed(42)
        x = torch.randn(M, K, device="cuda", dtype=dtype)
        wi, _ = make_weights(K, D_FF, D_MODEL, dtype)

        t_ref = benchmark_fn(lambda: ref_linear_relu(x, wi))
        t_fused = benchmark_fn(lambda: fused_linear_relu(x, wi))

        speedup = t_ref / t_fused if t_fused > 0 else float("inf")
        with capsys.disabled():
            print(
                f"\n  linear_relu M={M} {dtype}: "
                f"ref={t_ref:.0f}us, triton={t_fused:.0f}us, "
                f"speedup={speedup:.2f}x"
            )

    @pytest.mark.parametrize(
        "shape", TEST_SHAPES, ids=lambda s: f"M={s[0]}"
    )
    @pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16], ids=["fp32", "bf16"])
    def test_benchmark_full_mlp(self, shape, dtype, capsys):
        """Benchmark fused_mlp_relu vs F.linear(F.relu(F.linear(...)))."""
        M, K = shape
        torch.manual_seed(42)
        x = torch.randn(M, K, device="cuda", dtype=dtype)
        wi, wo = make_weights(K, D_FF, D_MODEL, dtype)

        t_ref = benchmark_fn(lambda: ref_mlp_relu(x, wi, wo))
        t_fused = benchmark_fn(lambda: fused_mlp_relu(x, wi, wo))

        speedup = t_ref / t_fused if t_fused > 0 else float("inf")
        with capsys.disabled():
            print(
                f"\n  full_mlp M={M} {dtype}: "
                f"ref={t_ref:.0f}us, triton={t_fused:.0f}us, "
                f"speedup={speedup:.2f}x"
            )
