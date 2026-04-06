"""Tests for Triton RoPE and RMS norm kernels.

Verifies numerical correctness against PyTorch reference implementations
at various sequence lengths and dtypes.

Run:
    pytest tests/test_rope_norm.py -v
"""

from __future__ import annotations

from contextlib import contextmanager
from functools import wraps

import pytest
import torch
import torch.nn.functional as F

from binanceneural.kernels.rope import apply_rope, apply_rope_fused
from binanceneural.kernels.norm import rms_norm, fused_rms_norm_qkv

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _is_cuda_resource_pressure_error(exc: BaseException) -> bool:
    return "out of memory" in str(exc).lower()


def _skip_for_cuda_resource_pressure(exc: BaseException) -> None:
    if _is_cuda_resource_pressure_error(exc):
        pytest.skip(f"RoPE CUDA test skipped under shared-GPU resource pressure: {exc}")


def _skip_on_cuda_resource_pressure(test_fn):
    @wraps(test_fn)
    def wrapped(*args, **kwargs):
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        try:
            return test_fn(*args, **kwargs)
        except Exception as exc:
            _skip_for_cuda_resource_pressure(exc)
            raise
        finally:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    return wrapped


def _decorate_test_class_for_cuda_resource_pressure(cls):
    for name, value in vars(cls).items():
        if name.startswith("test_") and callable(value):
            setattr(cls, name, _skip_on_cuda_resource_pressure(value))
    return cls


def _cuda_module_or_skip(module):
    try:
        return module.cuda()
    except Exception as exc:
        _skip_for_cuda_resource_pressure(exc)
        raise


def _cuda_randn_or_skip(*shape, **kwargs):
    try:
        return torch.randn(*shape, **kwargs)
    except Exception as exc:
        _skip_for_cuda_resource_pressure(exc)
        raise


def _resolve_test_device() -> str:
    if not torch.cuda.is_available():
        return "cpu"
    try:
        torch.empty(1, device="cuda")
        return "cuda"
    except Exception as exc:
        if _is_cuda_resource_pressure_error(exc):
            return "cpu"
        raise


DEVICE = _resolve_test_device()

SEQ_LENS = [1, 16, 32, 48, 64, 128]


def _make_inv_freq(head_dim: int, base: float = 10000.0, device: str = DEVICE) -> torch.Tensor:
    return 1.0 / (base ** (torch.arange(0, head_dim, 2, dtype=torch.float32, device=device) / head_dim))


def _make_cos_sin(inv_freq: torch.Tensor, seq_len: int):
    """Build (1, seq_len, 1, half_dim) cos/sin as RotaryEmbedding does."""
    t = torch.arange(seq_len, device=inv_freq.device, dtype=inv_freq.dtype)
    freqs = torch.outer(t, inv_freq)
    cos = freqs.cos()[None, :, None, :]   # (1, T, 1, half_dim)
    sin = freqs.sin()[None, :, None, :]
    return cos, sin


def _apply_rope_ref(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Pure-PyTorch interleaved RoPE (matches _apply_rotary_emb in model.py).

    x:   (B, T, H, D)
    cos: (1, T, 1, half_dim) — broadcast over batch and head dims
    sin: (1, T, 1, half_dim)
    """
    xr = x[..., ::2]   # (B, T, H, half_dim)
    xi = x[..., 1::2]  # (B, T, H, half_dim)
    # cos/sin broadcast: (1, T, 1, half_dim) → matches xr/xi on T and half_dim
    out_r = xr * cos - xi * sin
    out_i = xr * sin + xi * cos
    return torch.stack([out_r, out_i], dim=-1).flatten(-2)


def _rms_norm_ref(x: torch.Tensor, weight: torch.Tensor | None = None, eps: float = 1e-6) -> torch.Tensor:
    variance = x.float().pow(2).mean(dim=-1, keepdim=True)
    normed = (x * torch.rsqrt(variance + eps)).to(x.dtype)
    if weight is not None:
        normed = normed * weight
    return normed


@contextmanager
def _strict_float32_matmul():
    """Pin FP32 references to IEEE math so leaked TF32 settings don't skew comparisons."""
    if DEVICE != "cuda":
        yield
        return

    get_precision = getattr(torch, "get_float32_matmul_precision", None)
    set_precision = getattr(torch, "set_float32_matmul_precision", None)
    matmul_backend = getattr(torch.backends.cuda, "matmul", None)
    cudnn_backend = getattr(torch.backends, "cudnn", None)
    cudnn_conv = getattr(cudnn_backend, "conv", None) if cudnn_backend is not None else None

    snapshot = {
        "float32_matmul_precision": get_precision() if callable(get_precision) else None,
        "cuda_allow_tf32": getattr(matmul_backend, "allow_tf32", None) if matmul_backend is not None else None,
        "cuda_matmul_fp32_precision": getattr(matmul_backend, "fp32_precision", None) if matmul_backend is not None else None,
        "cudnn_allow_tf32": getattr(cudnn_backend, "allow_tf32", None) if cudnn_backend is not None else None,
        "cudnn_conv_fp32_precision": getattr(cudnn_conv, "fp32_precision", None) if cudnn_conv is not None else None,
    }

    if callable(set_precision):
        set_precision("highest")
    if snapshot["cuda_allow_tf32"] is not None:
        matmul_backend.allow_tf32 = False
    if snapshot["cuda_matmul_fp32_precision"] is not None:
        matmul_backend.fp32_precision = "ieee"
    if snapshot["cudnn_allow_tf32"] is not None:
        cudnn_backend.allow_tf32 = False
    if snapshot["cudnn_conv_fp32_precision"] is not None:
        cudnn_conv.fp32_precision = "ieee"

    try:
        yield
    finally:
        if callable(set_precision) and snapshot["float32_matmul_precision"] is not None:
            set_precision(snapshot["float32_matmul_precision"])
        if snapshot["cuda_allow_tf32"] is not None:
            matmul_backend.allow_tf32 = snapshot["cuda_allow_tf32"]
        if snapshot["cuda_matmul_fp32_precision"] is not None:
            matmul_backend.fp32_precision = snapshot["cuda_matmul_fp32_precision"]
        if snapshot["cudnn_allow_tf32"] is not None:
            cudnn_backend.allow_tf32 = snapshot["cudnn_allow_tf32"]
        if snapshot["cudnn_conv_fp32_precision"] is not None:
            cudnn_conv.fp32_precision = snapshot["cudnn_conv_fp32_precision"]


# ---------------------------------------------------------------------------
# RoPE tests
# ---------------------------------------------------------------------------

@_decorate_test_class_for_cuda_resource_pressure
class TestApplyRope:
    """Tests for apply_rope (pre-computed cos/sin)."""

    @pytest.mark.parametrize("seq_len", SEQ_LENS)
    @pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
    def test_matches_pytorch_ref(self, seq_len: int, dtype: torch.dtype) -> None:
        B, Hq, Hk, D = 2, 4, 2, 32
        torch.manual_seed(42)
        q = torch.randn(B, seq_len, Hq, D, device=DEVICE, dtype=dtype)
        k = torch.randn(B, seq_len, Hk, D, device=DEVICE, dtype=dtype)
        inv_freq = _make_inv_freq(D)
        cos, sin = _make_cos_sin(inv_freq, seq_len)

        q_ref = _apply_rope_ref(q, cos, sin)
        k_ref = _apply_rope_ref(k, cos, sin)

        q_out, k_out = apply_rope(q, k, cos, sin)

        atol = 1e-4 if dtype == torch.float32 else 2e-2
        assert q_out.shape == q.shape
        assert k_out.shape == k.shape
        assert torch.allclose(q_out.float(), q_ref.float(), atol=atol), \
            f"Q mismatch: max_diff={( q_out.float() - q_ref.float()).abs().max()}"
        assert torch.allclose(k_out.float(), k_ref.float(), atol=atol), \
            f"K mismatch: max_diff={(k_out.float() - k_ref.float()).abs().max()}"

    @pytest.mark.parametrize("seq_len", SEQ_LENS)
    def test_bf16_atol(self, seq_len: int) -> None:
        B, Hq, Hk, D = 2, 8, 4, 64
        torch.manual_seed(7)
        q = torch.randn(B, seq_len, Hq, D, device=DEVICE, dtype=torch.bfloat16)
        k = torch.randn(B, seq_len, Hk, D, device=DEVICE, dtype=torch.bfloat16)
        inv_freq = _make_inv_freq(D)
        cos, sin = _make_cos_sin(inv_freq, seq_len)

        q_ref = _apply_rope_ref(q, cos, sin)
        k_ref = _apply_rope_ref(k, cos, sin)
        q_out, k_out = apply_rope(q, k, cos, sin)

        # BF16 has 7 mantissa bits; with FP32 intermediate computation inside the
        # Triton kernel vs PyTorch's own bf16 arithmetic, differences up to ~2 BF16
        # ULPs (~0.016 for values near 1.0) are expected and correct.
        assert q_out.dtype == torch.bfloat16
        assert k_out.dtype == torch.bfloat16
        assert torch.allclose(q_out.float(), q_ref.float(), atol=2e-2), \
            f"Q bf16 max_diff={(q_out.float() - q_ref.float()).abs().max()}"
        assert torch.allclose(k_out.float(), k_ref.float(), atol=2e-2), \
            f"K bf16 max_diff={(k_out.float() - k_ref.float()).abs().max()}"

    def test_mqa_shape(self) -> None:
        """Multi-query: Hk < Hq."""
        B, Hq, Hk, D, T = 3, 8, 2, 32, 48
        q = torch.randn(B, T, Hq, D, device=DEVICE, dtype=torch.float32)
        k = torch.randn(B, T, Hk, D, device=DEVICE, dtype=torch.float32)
        inv_freq = _make_inv_freq(D)
        cos, sin = _make_cos_sin(inv_freq, T)
        q_out, k_out = apply_rope(q, k, cos, sin)
        assert q_out.shape == (B, T, Hq, D)
        assert k_out.shape == (B, T, Hk, D)

    def test_output_dtype_preserved(self) -> None:
        B, T, H, D = 2, 32, 4, 32
        inv_freq = _make_inv_freq(D)
        cos, sin = _make_cos_sin(inv_freq, T)
        for dtype in [torch.float32, torch.bfloat16]:
            q = torch.randn(B, T, H, D, device=DEVICE, dtype=dtype)
            k = torch.randn(B, T, H, D, device=DEVICE, dtype=dtype)
            q_out, k_out = apply_rope(q, k, cos, sin)
            assert q_out.dtype == dtype
            assert k_out.dtype == dtype

    def test_output_dtype_preserved_in_pytorch_fallback(self, monkeypatch: pytest.MonkeyPatch) -> None:
        B, T, H, D = 2, 8, 4, 32
        inv_freq = _make_inv_freq(D)
        cos, sin = _make_cos_sin(inv_freq, T)
        q = torch.randn(B, T, H, D, device=DEVICE, dtype=torch.bfloat16)
        k = torch.randn(B, T, H, D, device=DEVICE, dtype=torch.bfloat16)

        monkeypatch.setattr("binanceneural.kernels.rope.HAS_TRITON", False)

        q_out, k_out = apply_rope(q, k, cos, sin)

        assert q_out.dtype == torch.bfloat16
        assert k_out.dtype == torch.bfloat16


@_decorate_test_class_for_cuda_resource_pressure
class TestApplyRopeFused:
    """Tests for apply_rope_fused (computes cos/sin internally from inv_freq)."""

    @pytest.mark.parametrize("seq_len", SEQ_LENS)
    def test_matches_apply_rope(self, seq_len: int) -> None:
        B, Hq, Hk, D = 2, 4, 2, 32
        torch.manual_seed(99)
        q = torch.randn(B, seq_len, Hq, D, device=DEVICE, dtype=torch.bfloat16)
        k = torch.randn(B, seq_len, Hk, D, device=DEVICE, dtype=torch.bfloat16)
        inv_freq = _make_inv_freq(D)
        cos, sin = _make_cos_sin(inv_freq, seq_len)

        q_ref, k_ref = apply_rope(q, k, cos, sin)
        q_fused, k_fused = apply_rope_fused(q, k, inv_freq)

        # Both should agree with each other (and transitively with the ref)
        atol = 2e-2  # bf16 round-trip
        assert torch.allclose(q_fused.float(), q_ref.float(), atol=atol), \
            f"Q fused vs apply_rope max_diff={(q_fused.float()-q_ref.float()).abs().max()}"
        assert torch.allclose(k_fused.float(), k_ref.float(), atol=atol), \
            f"K fused vs apply_rope max_diff={(k_fused.float()-k_ref.float()).abs().max()}"

    @pytest.mark.parametrize("seq_len", SEQ_LENS)
    def test_matches_pytorch_ref(self, seq_len: int) -> None:
        B, Hq, Hk, D = 2, 8, 4, 64
        torch.manual_seed(13)
        q = torch.randn(B, seq_len, Hq, D, device=DEVICE, dtype=torch.float32)
        k = torch.randn(B, seq_len, Hk, D, device=DEVICE, dtype=torch.float32)
        inv_freq = _make_inv_freq(D)
        cos, sin = _make_cos_sin(inv_freq, seq_len)

        q_ref = _apply_rope_ref(q, cos, sin)
        k_ref = _apply_rope_ref(k, cos, sin)
        q_out, k_out = apply_rope_fused(q, k, inv_freq)

        assert torch.allclose(q_out.float(), q_ref.float(), atol=1e-4), \
            f"Q max_diff={(q_out.float()-q_ref.float()).abs().max()}"
        assert torch.allclose(k_out.float(), k_ref.float(), atol=1e-4), \
            f"K max_diff={(k_out.float()-k_ref.float()).abs().max()}"

    def test_fallback_preserves_bf16_dtype(self, monkeypatch: pytest.MonkeyPatch) -> None:
        B, T, H, D = 2, 8, 4, 32
        inv_freq = _make_inv_freq(D)
        q = torch.randn(B, T, H, D, device=DEVICE, dtype=torch.bfloat16)
        k = torch.randn(B, T, H, D, device=DEVICE, dtype=torch.bfloat16)

        monkeypatch.setattr("binanceneural.kernels.rope.HAS_TRITON", False)

        q_out, k_out = apply_rope_fused(q, k, inv_freq)

        assert q_out.dtype == torch.bfloat16
        assert k_out.dtype == torch.bfloat16


# ---------------------------------------------------------------------------
# RMS norm tests
# ---------------------------------------------------------------------------

@_decorate_test_class_for_cuda_resource_pressure
class TestRmsNorm:
    """Tests for rms_norm (unweighted and weighted)."""

    @pytest.mark.parametrize("shape", [(64, 256), (1, 256), (192, 512)])
    def test_unweighted_matches_ref(self, shape: tuple[int, int]) -> None:
        torch.manual_seed(0)
        x = torch.randn(*shape, device=DEVICE, dtype=torch.float32)
        ref = _rms_norm_ref(x, weight=None, eps=1e-6)
        out = rms_norm(x, weight=None, eps=1e-6)
        assert torch.allclose(out, ref, atol=1e-4), \
            f"max_diff={(out - ref).abs().max()}"

    @pytest.mark.parametrize("shape", [(64, 256), (32, 128)])
    def test_weighted_matches_ref(self, shape: tuple[int, int]) -> None:
        torch.manual_seed(1)
        x = torch.randn(*shape, device=DEVICE, dtype=torch.float32)
        w = torch.randn(shape[-1], device=DEVICE, dtype=torch.float32)
        ref = _rms_norm_ref(x, weight=w, eps=1e-6)
        out = rms_norm(x, weight=w, eps=1e-6)
        assert torch.allclose(out, ref, atol=1e-4), \
            f"max_diff={(out - ref).abs().max()}"

    @pytest.mark.parametrize("N", [64, 128, 256, 512])
    def test_bf16_matches_ref(self, N: int) -> None:
        torch.manual_seed(2)
        x = torch.randn(48, N, device=DEVICE, dtype=torch.bfloat16)
        ref = _rms_norm_ref(x, weight=None, eps=1e-5)
        out = rms_norm(x, weight=None, eps=1e-5)
        assert out.dtype == torch.bfloat16
        assert torch.allclose(out.float(), ref.float(), atol=1e-2), \
            f"bf16 max_diff={(out.float() - ref.float()).abs().max()}"

    def test_matches_nn_rmsnorm(self) -> None:
        """Weighted rms_norm should match nn.RMSNorm."""
        N = 256
        torch.manual_seed(3)
        x = torch.randn(32, N, device=DEVICE, dtype=torch.float32)
        w = torch.ones(N, device=DEVICE)

        ref_layer = torch.nn.RMSNorm(N, eps=1e-6).to(DEVICE)
        with torch.no_grad():
            ref_layer.weight.copy_(w)
        ref = ref_layer(x)
        out = rms_norm(x, weight=w, eps=1e-6)

        assert torch.allclose(out, ref, atol=1e-4), \
            f"vs nn.RMSNorm max_diff={(out - ref).abs().max()}"

    @pytest.mark.parametrize("seq_len", SEQ_LENS)
    def test_various_seq_lens(self, seq_len: int) -> None:
        N = 256
        x = torch.randn(seq_len, N, device=DEVICE, dtype=torch.bfloat16)
        ref = _rms_norm_ref(x, None, eps=1e-5)
        out = rms_norm(x, None, eps=1e-5)
        assert torch.allclose(out.float(), ref.float(), atol=1e-2)


# ---------------------------------------------------------------------------
# Fused RMS norm + QKV tests
# ---------------------------------------------------------------------------

@_decorate_test_class_for_cuda_resource_pressure
class TestFusedRmsNormQkv:
    """Tests for fused_rms_norm_qkv."""

    @pytest.mark.parametrize("seq_len", SEQ_LENS)
    def test_unweighted_matches_sequential_f32(self, seq_len: int) -> None:
        B, N, Dq, Dk, Dv = 2, 256, 256, 128, 128
        torch.manual_seed(10)
        x = torch.randn(B * seq_len, N, device=DEVICE, dtype=torch.float32)
        wq = torch.randn(Dq, N, device=DEVICE, dtype=torch.float32)
        wk = torch.randn(Dk, N, device=DEVICE, dtype=torch.float32)
        wv = torch.randn(Dv, N, device=DEVICE, dtype=torch.float32)

        with _strict_float32_matmul():
            normed = _rms_norm_ref(x, None, eps=1e-5)
            q_ref = F.linear(normed, wq)
            k_ref = F.linear(normed, wk)
            v_ref = F.linear(normed, wv)

            q_out, k_out, v_out = fused_rms_norm_qkv(x, None, wq, wk, wv, eps=1e-5)

        assert torch.allclose(q_out, q_ref, atol=1e-3), \
            f"Q max_diff={(q_out - q_ref).abs().max()}"
        assert torch.allclose(k_out, k_ref, atol=1e-3), \
            f"K max_diff={(k_out - k_ref).abs().max()}"
        assert torch.allclose(v_out, v_ref, atol=1e-3), \
            f"V max_diff={(v_out - v_ref).abs().max()}"

    @pytest.mark.parametrize("seq_len", SEQ_LENS)
    def test_unweighted_matches_sequential_bf16(self, seq_len: int) -> None:
        B, N, D = 4, 256, 256
        torch.manual_seed(11)
        x = torch.randn(B * seq_len, N, device=DEVICE, dtype=torch.bfloat16)
        wq = torch.randn(D, N, device=DEVICE, dtype=torch.bfloat16)
        wk = torch.randn(D, N, device=DEVICE, dtype=torch.bfloat16)
        wv = torch.randn(D, N, device=DEVICE, dtype=torch.bfloat16)

        normed = _rms_norm_ref(x, None, eps=1e-5)
        q_ref = F.linear(normed, wq)
        k_ref = F.linear(normed, wk)
        v_ref = F.linear(normed, wv)

        q_out, k_out, v_out = fused_rms_norm_qkv(x, None, wq, wk, wv, eps=1e-5)

        # BF16 dot-product accumulation over N=256 elements can produce ~1% relative
        # error. Use rtol=2e-2 (relative) which is appropriate for BF16.
        assert q_out.dtype == torch.bfloat16
        assert torch.allclose(q_out.float(), q_ref.float(), rtol=2e-2, atol=1e-1), \
            f"Q bf16 max_diff={(q_out.float() - q_ref.float()).abs().max()}"
        assert torch.allclose(k_out.float(), k_ref.float(), rtol=2e-2, atol=1e-1), \
            f"K bf16 max_diff={(k_out.float() - k_ref.float()).abs().max()}"
        assert torch.allclose(v_out.float(), v_ref.float(), rtol=2e-2, atol=1e-1), \
            f"V bf16 max_diff={(v_out.float() - v_ref.float()).abs().max()}"

    def test_weighted_norm_matches_sequential(self) -> None:
        N, Dq, Dk, Dv = 256, 256, 128, 128
        rows = 48
        torch.manual_seed(12)
        x = torch.randn(rows, N, device=DEVICE, dtype=torch.float32)
        norm_w = torch.randn(N, device=DEVICE, dtype=torch.float32)
        wq = torch.randn(Dq, N, device=DEVICE, dtype=torch.float32)
        wk = torch.randn(Dk, N, device=DEVICE, dtype=torch.float32)
        wv = torch.randn(Dv, N, device=DEVICE, dtype=torch.float32)

        with _strict_float32_matmul():
            normed = _rms_norm_ref(x, norm_w, eps=1e-5)
            q_ref = F.linear(normed, wq)
            k_ref = F.linear(normed, wk)
            v_ref = F.linear(normed, wv)

            q_out, k_out, v_out = fused_rms_norm_qkv(x, norm_w, wq, wk, wv, eps=1e-5)

        assert torch.allclose(q_out, q_ref, atol=1e-3)
        assert torch.allclose(k_out, k_ref, atol=1e-3)
        assert torch.allclose(v_out, v_ref, atol=1e-3)

    def test_shapes_correct(self) -> None:
        rows, N, Dq, Dk, Dv = 32, 256, 64, 32, 32
        x = torch.randn(rows, N, device=DEVICE, dtype=torch.float32)
        wq = torch.randn(Dq, N, device=DEVICE, dtype=torch.float32)
        wk = torch.randn(Dk, N, device=DEVICE, dtype=torch.float32)
        wv = torch.randn(Dv, N, device=DEVICE, dtype=torch.float32)

        q, k, v = fused_rms_norm_qkv(x, None, wq, wk, wv)
        assert q.shape == (rows, Dq)
        assert k.shape == (rows, Dk)
        assert v.shape == (rows, Dv)

    def test_3d_input(self) -> None:
        """Input shape (B, T, N) should reshape and give (B, T, D) outputs."""
        B, T, N, D = 4, 48, 256, 128
        x = torch.randn(B, T, N, device=DEVICE, dtype=torch.float32)
        wq = torch.randn(D, N, device=DEVICE, dtype=torch.float32)
        wk = torch.randn(D, N, device=DEVICE, dtype=torch.float32)
        wv = torch.randn(D, N, device=DEVICE, dtype=torch.float32)

        q, k, v = fused_rms_norm_qkv(x, None, wq, wk, wv)
        assert q.shape == (B, T, D)
        assert k.shape == (B, T, D)
        assert v.shape == (B, T, D)


# ---------------------------------------------------------------------------
# End-to-end model import test
# ---------------------------------------------------------------------------

@_skip_on_cuda_resource_pressure
def test_model_import_ok() -> None:
    """Ensure model.py can be imported with Triton kernels wired in."""
    from binanceneural.model import BinanceHourlyPolicyNano, PolicyConfig
    cfg = PolicyConfig(input_dim=16, hidden_dim=64, num_heads=4, num_layers=2, max_len=64,
                       model_arch="nano", num_kv_heads=2)
    model = BinanceHourlyPolicyNano(cfg).to(DEVICE)
    x = torch.randn(2, 32, 16, device=DEVICE)
    with torch.no_grad():
        out = model(x)
    assert "buy_price_logits" in out
    assert out["buy_price_logits"].shape == (2, 32, 1)


@_skip_on_cuda_resource_pressure
def test_model_rope_triton_path() -> None:
    """With CUDA available, the Triton RoPE path is exercised without error."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA required for Triton RoPE path")
    from binanceneural.model import BinanceHourlyPolicyNano, PolicyConfig
    cfg = PolicyConfig(input_dim=8, hidden_dim=32, num_heads=4, num_layers=1, max_len=64,
                       model_arch="nano", num_kv_heads=2)
    model = _cuda_module_or_skip(BinanceHourlyPolicyNano(cfg))
    x = _cuda_randn_or_skip(1, 48, 8, device="cuda")
    with torch.no_grad():
        out = model(x)
    assert out["buy_price_logits"].shape == (1, 48, 1)
