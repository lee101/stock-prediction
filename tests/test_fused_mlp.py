"""Tests for the fused_mlp_relu Triton kernel.

Verifies:
- Output matches BF16 PyTorch reference (atol=5e-2 for Triton path, 1e-5 for FP32 fallback)
- Works for batch sizes 1, 64, 128, 256
- Fallback path (HAS_TRITON=False) produces correct results
- Various in/hidden/out dimension combinations
"""

import pytest
import torch
import torch.nn.functional as F


# ─── Helpers ──────────────────────────────────────────────────────────────────

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _is_cuda_resource_pressure_error(exc: BaseException) -> bool:
    return "out of memory" in str(exc).lower()


def _skip_for_cuda_resource_pressure(exc: BaseException) -> None:
    if _is_cuda_resource_pressure_error(exc):
        pytest.skip(f"Fused MLP Triton test skipped under shared-GPU resource pressure: {exc}")


def _allocate_on_device_or_skip(factory, *args, **kwargs):
    try:
        return factory(*args, **kwargs)
    except Exception as exc:
        _skip_for_cuda_resource_pressure(exc)
        raise


def _to_device_or_skip(tensor, device, dtype=None):
    try:
        if dtype is None:
            return tensor.to(device)
        return tensor.to(device, dtype)
    except Exception as exc:
        _skip_for_cuda_resource_pressure(exc)
        raise


def _make_weights(in_dim, hidden, out_dim, seed=42):
    """Create deterministic weight tensors for reproducibility."""
    gen = torch.Generator()
    gen.manual_seed(seed)
    W1 = torch.randn(hidden, in_dim,  generator=gen, dtype=torch.float32)
    b1 = torch.randn(hidden,          generator=gen, dtype=torch.float32) * 0.1
    W2 = torch.randn(out_dim, hidden, generator=gen, dtype=torch.float32)
    b2 = torch.randn(out_dim,         generator=gen, dtype=torch.float32) * 0.1
    return W1, b1, W2, b2


def _sequential_ref(x, W1, b1, W2, b2):
    """Pure PyTorch reference in FP32: relu(x @ W1.T + b1) @ W2.T + b2."""
    h = F.linear(x.float(), W1.float(), b1.float())
    h = F.relu(h)
    return F.linear(h, W2.float(), b2.float())


def _sequential_ref_bf16(x, W1, b1, W2, b2, device):
    """BF16 PyTorch reference (matches Triton kernel's internal precision)."""
    x_ = _to_device_or_skip(x, device, torch.bfloat16)
    W1_ = _to_device_or_skip(W1, device, torch.bfloat16)
    b1_ = _to_device_or_skip(b1, device, torch.bfloat16)
    W2_ = _to_device_or_skip(W2, device, torch.bfloat16)
    b2_ = _to_device_or_skip(b2, device, torch.bfloat16)
    h = F.linear(x_, W1_, b1_)
    h = F.relu(h)
    return F.linear(h, W2_, b2_)


# ─── Import under test ────────────────────────────────────────────────────────

import pufferlib_market.kernels.fused_mlp as _fused_mlp_module
from pufferlib_market.kernels.fused_mlp import fused_mlp_relu, HAS_TRITON


# ─── Tests: fallback path (always runs) ───────────────────────────────────────

class TestFallbackPath:
    """Tests that use the pure-PyTorch fallback regardless of Triton availability."""

    def _run_fallback(self, x, W1, b1, W2, b2):
        """Call fallback directly, bypassing HAS_TRITON check."""
        return _fused_mlp_module._fused_mlp_relu_fallback(x, W1, b1, W2, b2)

    @pytest.mark.parametrize("batch", [1, 64, 128, 256])
    def test_fallback_matches_reference(self, batch):
        """Fallback must match F.linear + F.relu reference in FP32 to 1e-5."""
        in_dim, hidden, out_dim = 64, 128, 64
        W1, b1, W2, b2 = _make_weights(in_dim, hidden, out_dim)
        x = torch.randn(batch, in_dim)

        ref = _sequential_ref(x, W1, b1, W2, b2)
        got = self._run_fallback(x, W1, b1, W2, b2)

        assert got.shape == (batch, out_dim), f"Shape mismatch: {got.shape}"
        torch.testing.assert_close(got.float(), ref.float(), atol=1e-5, rtol=1e-5)

    def test_fallback_relu_zeros_negatives(self):
        """Verify ReLU zeroes negative pre-activations."""
        # Build weights s.t. all pre-activations of layer1 are negative
        in_dim, hidden, out_dim = 8, 16, 8
        W1 = -torch.ones(hidden, in_dim)   # all negative weights
        b1 = -torch.ones(hidden)           # negative bias -> all negative
        W2 = torch.eye(out_dim, hidden)    # identity-like
        b2 = torch.zeros(out_dim)
        x  = torch.ones(1, in_dim)

        got = self._run_fallback(x, W1, b1, W2, b2)
        # ReLU zeros all negatives; output should be exactly b2 = 0
        torch.testing.assert_close(got, torch.zeros(1, out_dim), atol=1e-6, rtol=0)

    def test_fallback_3d_input(self):
        """Fallback handles (..., D_IN) shaped inputs."""
        W1, b1, W2, b2 = _make_weights(32, 64, 32)
        x = torch.randn(4, 8, 32)  # 3D input
        got = _fused_mlp_relu_fallback_wrapper(x, W1, b1, W2, b2)
        assert got.shape == (4, 8, 32)


def _fused_mlp_relu_fallback_wrapper(x, W1, b1, W2, b2):
    """Wrap fallback through the public API (forces fallback via monkey-patch)."""
    orig = _fused_mlp_module.HAS_TRITON
    _fused_mlp_module.HAS_TRITON = False
    try:
        return fused_mlp_relu(x, W1, b1, W2, b2)
    finally:
        _fused_mlp_module.HAS_TRITON = orig


class TestForcedFallback:
    """Test the public fused_mlp_relu API with HAS_TRITON forced to False."""

    @pytest.mark.parametrize("batch", [1, 64, 128, 256])
    def test_forced_fallback_matches_reference(self, batch):
        in_dim, hidden, out_dim = 64, 128, 64
        W1, b1, W2, b2 = _make_weights(in_dim, hidden, out_dim)
        x = torch.randn(batch, in_dim)

        ref = _sequential_ref(x, W1, b1, W2, b2)
        got = _fused_mlp_relu_fallback_wrapper(x, W1, b1, W2, b2)

        assert got.shape == (batch, out_dim)
        torch.testing.assert_close(got.float(), ref.float(), atol=1e-5, rtol=1e-5)

    def test_forced_fallback_output_dtype_matches_input(self):
        W1, b1, W2, b2 = _make_weights(32, 64, 32)
        x = torch.randn(16, 32, dtype=torch.float32)
        got = _fused_mlp_relu_fallback_wrapper(x, W1, b1, W2, b2)
        assert got.dtype == torch.float32

    def test_forced_fallback_bf16(self):
        W1, b1, W2, b2 = _make_weights(32, 64, 32)
        x   = torch.randn(16, 32, dtype=torch.bfloat16)
        W1f = W1.bfloat16(); b1f = b1.bfloat16()
        W2f = W2.bfloat16(); b2f = b2.bfloat16()
        got = _fused_mlp_relu_fallback_wrapper(x, W1f, b1f, W2f, b2f)
        assert got.dtype == torch.bfloat16


# ─── Tests: Triton path (skipped if no CUDA/Triton) ──────────────────────────

@pytest.mark.skipif(not (HAS_TRITON and torch.cuda.is_available()), reason="Triton + CUDA required")
class TestTritonPath:
    """Tests for the Triton kernel on CUDA."""

    @pytest.mark.parametrize("batch", [1, 64, 128, 256])
    def test_triton_matches_reference_bf16(self, batch):
        """Triton BF16 output must match BF16 PyTorch reference exactly."""
        in_dim, hidden, out_dim = 64, 256, 64
        W1, b1, W2, b2 = _make_weights(in_dim, hidden, out_dim)

        x = _allocate_on_device_or_skip(torch.randn, batch, in_dim, device=DEVICE, dtype=torch.bfloat16)
        W1 = _to_device_or_skip(W1, DEVICE, torch.bfloat16)
        b1 = _to_device_or_skip(b1, DEVICE, torch.bfloat16)
        W2 = _to_device_or_skip(W2, DEVICE, torch.bfloat16)
        b2 = _to_device_or_skip(b2, DEVICE, torch.bfloat16)

        # Use BF16 reference — Triton kernel accumulates in FP32 for precision,
        # so may differ slightly from BF16 F.linear (which uses BF16 tensor cores).
        ref = _sequential_ref_bf16(x, W1, b1, W2, b2, DEVICE)
        got = fused_mlp_relu(x, W1, b1, W2, b2)

        assert got.shape == (batch, out_dim)
        assert got.dtype == torch.bfloat16
        # Kernel accumulates in FP32 then casts to BF16 — slight diff from BF16 F.linear ok
        torch.testing.assert_close(got.float(), ref.float(), atol=5e-2, rtol=1e-2)

    @pytest.mark.parametrize("in_dim,hidden,out_dim", [
        (221, 1024, 1024),  # crypto12 default
        (64,  256,  64),
        (128, 512,  128),
        (32,  64,   16),
    ])
    def test_triton_various_dims(self, in_dim, hidden, out_dim):
        """Kernel works for various dimension combinations.

        Compares Triton BF16 output against FP32 ground truth.
        BF16 matmul error scales with hidden size (each accumulator step
        rounds to BF16). At hidden=1024 output values can reach ~1600,
        so absolute tolerance of 8.0 (0.5% of range) is appropriate.
        """
        batch = 128
        W1, b1, W2, b2 = _make_weights(in_dim, hidden, out_dim, seed=7)

        gen = torch.Generator(); gen.manual_seed(99)
        x = _to_device_or_skip(
            torch.randn(batch, in_dim, generator=gen, dtype=torch.bfloat16),
            DEVICE,
        )
        W1 = _to_device_or_skip(W1, DEVICE, torch.bfloat16)
        b1 = _to_device_or_skip(b1, DEVICE, torch.bfloat16)
        W2 = _to_device_or_skip(W2, DEVICE, torch.bfloat16)
        b2 = _to_device_or_skip(b2, DEVICE, torch.bfloat16)

        # FP32 reference — Triton accumulates in FP32 internally so it should
        # match FP32 ground truth as closely as any BF16-input matmul.
        ref = _sequential_ref(x.cpu(), W1.cpu(), b1.cpu(), W2.cpu(), b2.cpu())
        got = fused_mlp_relu(x, W1, b1, W2, b2)

        assert got.shape == (batch, out_dim)
        # BF16 rounding error at hidden=1024: empirically ~5 abs vs FP32 (output ~1600).
        # atol=8 gives 0.5% headroom above observed max.
        torch.testing.assert_close(got.float().cpu(), ref, atol=8.0, rtol=0.0)

    def test_triton_output_dtype_is_bf16(self):
        in_dim, hidden, out_dim = 64, 128, 64
        W1, b1, W2, b2 = _make_weights(in_dim, hidden, out_dim)
        x = _allocate_on_device_or_skip(torch.randn, 32, in_dim, device=DEVICE, dtype=torch.bfloat16)
        W1 = _to_device_or_skip(W1, DEVICE, torch.bfloat16)
        b1 = _to_device_or_skip(b1, DEVICE, torch.bfloat16)
        W2 = _to_device_or_skip(W2, DEVICE, torch.bfloat16)
        b2 = _to_device_or_skip(b2, DEVICE, torch.bfloat16)

        got = fused_mlp_relu(x, W1, b1, W2, b2)
        assert got.dtype == torch.bfloat16, f"Expected BF16, got {got.dtype}"
        assert got.device.type == "cuda"

    def test_triton_relu_zeros_negatives(self):
        """ReLU must zero out negative hidden pre-activations."""
        hidden = 64
        # W1 all -1 and b1 all -1 so hidden = relu(-sum - 1) = 0
        W1 = -_allocate_on_device_or_skip(torch.ones, hidden, 8, device=DEVICE, dtype=torch.bfloat16)
        b1 = -_allocate_on_device_or_skip(torch.ones, hidden, device=DEVICE, dtype=torch.bfloat16)
        W2 = _allocate_on_device_or_skip(torch.eye, 8, hidden, device=DEVICE, dtype=torch.bfloat16)
        b2 = _allocate_on_device_or_skip(torch.zeros, 8, device=DEVICE, dtype=torch.bfloat16)
        x = _allocate_on_device_or_skip(torch.ones, 4, 8, device=DEVICE, dtype=torch.bfloat16)

        got = fused_mlp_relu(x, W1, b1, W2, b2)
        torch.testing.assert_close(
            got.float(),
            _allocate_on_device_or_skip(torch.zeros, 4, 8, device=DEVICE),
            atol=1e-4,
            rtol=0,
        )

    def test_triton_non_power_of_2_batch(self):
        """Kernel handles batch sizes that are not powers of 2."""
        in_dim, hidden, out_dim = 64, 128, 64
        W1, b1, W2, b2 = _make_weights(in_dim, hidden, out_dim)
        x = _allocate_on_device_or_skip(torch.randn, 73, in_dim, device=DEVICE, dtype=torch.bfloat16)
        W1 = _to_device_or_skip(W1, DEVICE, torch.bfloat16)
        b1 = _to_device_or_skip(b1, DEVICE, torch.bfloat16)
        W2 = _to_device_or_skip(W2, DEVICE, torch.bfloat16)
        b2 = _to_device_or_skip(b2, DEVICE, torch.bfloat16)

        ref = _sequential_ref_bf16(x, W1, b1, W2, b2, DEVICE)
        got = fused_mlp_relu(x, W1, b1, W2, b2)
        assert got.shape == (73, out_dim)
        torch.testing.assert_close(got.float(), ref.float(), atol=5e-2, rtol=1e-2)

    def test_triton_batch1(self):
        """Kernel works for batch_size=1 (important for inference)."""
        in_dim, hidden, out_dim = 64, 128, 64
        W1, b1, W2, b2 = _make_weights(in_dim, hidden, out_dim)
        x = _allocate_on_device_or_skip(torch.randn, 1, in_dim, device=DEVICE, dtype=torch.bfloat16)
        W1 = _to_device_or_skip(W1, DEVICE, torch.bfloat16)
        b1 = _to_device_or_skip(b1, DEVICE, torch.bfloat16)
        W2 = _to_device_or_skip(W2, DEVICE, torch.bfloat16)
        b2 = _to_device_or_skip(b2, DEVICE, torch.bfloat16)

        ref = _sequential_ref_bf16(x, W1, b1, W2, b2, DEVICE)
        got = fused_mlp_relu(x, W1, b1, W2, b2)
        assert got.shape == (1, out_dim)
        torch.testing.assert_close(got.float(), ref.float(), atol=5e-2, rtol=1e-2)
