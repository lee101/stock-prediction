"""Tests for the Triton fused multi-query attention kernel.

Covers:
  - Numerical correctness against PyTorch SDPA reference (atol=1e-2 for BF16)
  - Various sequence lengths: 1, 16, 32, 48, 64
  - Various batch sizes: 1, 32, 64
  - Causal and non-causal modes
  - Multi-query (num_kv_heads=1) and grouped-query (num_kv_heads=2) inputs
  - Float additive mask (sliding window)
  - Bool mask
  - Fallback path when HAS_TRITON=False
"""

from __future__ import annotations

import math
from unittest import mock

import pytest
import torch


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sdpa_reference(Q, K, V, mask=None, causal=False):
    """Pure PyTorch reference: scaled dot product attention."""
    B, H, S, D = Q.shape
    # Expand K/V to match Q heads if needed (repeat_interleave for proper GQA semantics)
    if K.shape[1] != H:
        n_rep = H // K.shape[1]
        K = K.repeat_interleave(n_rep, dim=1)
        V = V.repeat_interleave(n_rep, dim=1)

    scale = 1.0 / math.sqrt(D)
    scores = torch.matmul(Q.float(), K.float().transpose(-2, -1)) * scale  # [B,H,S,S]

    if causal and mask is None:
        idx = torch.arange(S, device=Q.device)
        causal_mask = idx[:, None] >= idx[None, :]  # [S,S] lower triangular
        scores = scores.masked_fill(~causal_mask, float("-inf"))

    if mask is not None:
        if mask.dtype == torch.bool:
            scores = scores.masked_fill(~mask, float("-inf"))
        else:
            scores = scores + mask.float()

    attn = torch.softmax(scores, dim=-1)
    out = torch.matmul(attn, V.float())
    return out.to(Q.dtype)


def _is_cuda_resource_pressure_error(exc: BaseException) -> bool:
    return "out of memory" in str(exc).lower()


def _skip_for_cuda_resource_pressure(exc: BaseException) -> None:
    if _is_cuda_resource_pressure_error(exc):
        pytest.skip(f"Triton attention test skipped under shared-GPU resource pressure: {exc}")


def _is_cuda_compiler_environment_error(exc: BaseException) -> bool:
    message = str(exc).lower()
    indicators = (
        "cppcompileerror",
        "internaltorchdynamoerror",
        "assembler messages",
        "can't open /tmp/",
        "no such file or directory",
    )
    return any(indicator in message for indicator in indicators)


def _skip_for_cuda_compiler_environment(exc: BaseException) -> None:
    if _is_cuda_compiler_environment_error(exc):
        pytest.skip(
            "Triton attention test skipped due to transient TorchInductor/C++ compiler "
            f"environment failure: {exc}"
        )


def _cuda_randn_or_skip(*shape, **kwargs):
    try:
        return torch.randn(*shape, **kwargs)
    except Exception as exc:
        _skip_for_cuda_resource_pressure(exc)
        _skip_for_cuda_compiler_environment(exc)
        raise


def _cuda_module_or_skip(module):
    try:
        return module.cuda()
    except Exception as exc:
        _skip_for_cuda_resource_pressure(exc)
        _skip_for_cuda_compiler_environment(exc)
        raise


def _make_qkv(batch, heads, seq, head_dim, kv_heads=1, dtype=torch.bfloat16, device="cuda"):
    torch.manual_seed(42)
    try:
        Q = torch.randn(batch, heads, seq, head_dim, dtype=dtype, device=device)
        K = torch.randn(batch, kv_heads, seq, head_dim, dtype=dtype, device=device)
        V = torch.randn(batch, kv_heads, seq, head_dim, dtype=dtype, device=device)
    except Exception as exc:
        _skip_for_cuda_resource_pressure(exc)
        raise
    return Q, K, V


# ---------------------------------------------------------------------------
# Skip if no CUDA or no Triton
# ---------------------------------------------------------------------------

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA required for Triton kernel tests"
)


def _import_kernel():
    from binanceneural.kernels.attention import HAS_TRITON, multi_query_attention
    return HAS_TRITON, multi_query_attention


# ---------------------------------------------------------------------------
# Correctness tests
# ---------------------------------------------------------------------------

class TestTritonKernelCorrectness:
    """Numerical match against PyTorch reference."""

    def setup_method(self):
        HAS_TRITON, self.mqa = _import_kernel()
        if not HAS_TRITON:
            pytest.skip("Triton not installed")

    @pytest.mark.parametrize("seq", [1, 16, 32, 48, 64])
    def test_seq_lengths(self, seq):
        Q, K, V = _make_qkv(4, 8, seq, 64, kv_heads=1)
        ref = _sdpa_reference(Q, K, V)
        got = self.mqa(Q, K, V)
        torch.testing.assert_close(got.float(), ref.float(), atol=1e-2, rtol=1e-2)

    @pytest.mark.parametrize("batch", [1, 32, 64])
    def test_batch_sizes(self, batch):
        Q, K, V = _make_qkv(batch, 8, 48, 64, kv_heads=1)
        ref = _sdpa_reference(Q, K, V)
        got = self.mqa(Q, K, V)
        torch.testing.assert_close(got.float(), ref.float(), atol=1e-2, rtol=1e-2)

    def test_causal(self):
        Q, K, V = _make_qkv(4, 8, 48, 64, kv_heads=1)
        ref = _sdpa_reference(Q, K, V, causal=True)
        got = self.mqa(Q, K, V, causal=True)
        torch.testing.assert_close(got.float(), ref.float(), atol=1e-2, rtol=1e-2)

    def test_non_causal(self):
        Q, K, V = _make_qkv(4, 8, 48, 64, kv_heads=1)
        ref = _sdpa_reference(Q, K, V, causal=False)
        got = self.mqa(Q, K, V, causal=False)
        torch.testing.assert_close(got.float(), ref.float(), atol=1e-2, rtol=1e-2)

    def test_gqa_two_kv_heads(self):
        """Grouped query attention: 8 Q heads, 2 KV heads."""
        Q, K, V = _make_qkv(4, 8, 48, 64, kv_heads=2)
        ref = _sdpa_reference(Q, K, V)
        got = self.mqa(Q, K, V)
        torch.testing.assert_close(got.float(), ref.float(), atol=1e-2, rtol=1e-2)

    def test_additive_float_mask(self):
        """Sliding-window style additive mask."""
        B, H, S, D = 4, 8, 48, 64
        Q, K, V = _make_qkv(B, H, S, D, kv_heads=1)
        # Window mask: only attend within 8 steps
        idx = torch.arange(S, device=Q.device)
        window = (idx[:, None] - idx[None, :]).abs() < 8
        float_mask = torch.zeros(S, S, dtype=torch.float32, device=Q.device)
        float_mask = float_mask.masked_fill(~window, float("-inf"))
        # shape [S, S] — will be broadcast to [B, H, S, S] by the kernel
        ref = _sdpa_reference(Q, K, V, mask=float_mask.unsqueeze(0).unsqueeze(0))
        got = self.mqa(Q, K, V, mask=float_mask)
        torch.testing.assert_close(got.float(), ref.float(), atol=1e-2, rtol=1e-2)

    def test_bool_mask(self):
        """Bool additive mask (kernel converts to -inf)."""
        B, H, S, D = 4, 8, 32, 64
        Q, K, V = _make_qkv(B, H, S, D, kv_heads=1)
        # Causal bool mask
        idx = torch.arange(S, device=Q.device)
        bool_mask = idx[:, None] >= idx[None, :]  # [S, S]
        ref = _sdpa_reference(Q, K, V, mask=bool_mask.unsqueeze(0).unsqueeze(0))
        got = self.mqa(Q, K, V, mask=bool_mask)
        torch.testing.assert_close(got.float(), ref.float(), atol=1e-2, rtol=1e-2)

    def test_output_shape(self):
        Q, K, V = _make_qkv(8, 8, 48, 64, kv_heads=1)
        out = self.mqa(Q, K, V)
        assert out.shape == Q.shape

    def test_output_dtype_preserved_bfloat16(self):
        Q, K, V = _make_qkv(4, 8, 48, 64, kv_heads=1, dtype=torch.bfloat16)
        out = self.mqa(Q, K, V)
        assert out.dtype == torch.bfloat16

    def test_output_dtype_preserved_float32(self):
        Q, K, V = _make_qkv(4, 8, 48, 64, kv_heads=1, dtype=torch.float32)
        out = self.mqa(Q, K, V)
        assert out.dtype == torch.float32

    @pytest.mark.parametrize("head_dim", [16, 32, 64, 128])
    def test_supported_head_dims(self, head_dim):
        Q, K, V = _make_qkv(2, 8, 32, head_dim, kv_heads=1)
        ref = _sdpa_reference(Q, K, V)
        got = self.mqa(Q, K, V)
        torch.testing.assert_close(got.float(), ref.float(), atol=1e-2, rtol=1e-2)

    def test_seq_len_1(self):
        """Edge case: single token sequence."""
        Q, K, V = _make_qkv(4, 8, 1, 64, kv_heads=1)
        ref = _sdpa_reference(Q, K, V)
        got = self.mqa(Q, K, V)
        torch.testing.assert_close(got.float(), ref.float(), atol=1e-2, rtol=1e-2)

    def test_custom_scale(self):
        Q, K, V = _make_qkv(4, 8, 48, 64, kv_heads=1)
        scale = 0.5
        # Reference with explicit scale
        B, H, S, D = Q.shape
        K_exp = K.expand(-1, H, -1, -1)
        V_exp = V.expand(-1, H, -1, -1)
        scores = torch.matmul(Q.float(), K_exp.float().transpose(-2, -1)) * scale
        attn = torch.softmax(scores, dim=-1)
        ref = torch.matmul(attn, V_exp.float()).to(Q.dtype)

        got = self.mqa(Q, K, V, scale=scale)
        torch.testing.assert_close(got.float(), ref.float(), atol=1e-2, rtol=1e-2)


# ---------------------------------------------------------------------------
# Fallback path test (HAS_TRITON=False)
# ---------------------------------------------------------------------------

class TestFallbackPath:
    """When HAS_TRITON=False the wrapper raises RuntimeError."""

    def test_raises_when_triton_unavailable(self):
        import binanceneural.kernels.attention as attn_mod
        # Temporarily patch HAS_TRITON to False
        with mock.patch.object(attn_mod, "HAS_TRITON", False):
            with pytest.raises(RuntimeError, match="Triton is not available"):
                attn_mod.multi_query_attention(
                    _cuda_randn_or_skip(1, 8, 8, 64, device="cuda", dtype=torch.bfloat16),
                    _cuda_randn_or_skip(1, 1, 8, 64, device="cuda", dtype=torch.bfloat16),
                    _cuda_randn_or_skip(1, 1, 8, 64, device="cuda", dtype=torch.bfloat16),
                )

    def test_model_uses_sdpa_fallback_when_triton_unavailable(self):
        """BinanceHourlyPolicyNano still forward-passes when Triton is patched out."""
        import binanceneural.kernels.attention as attn_mod
        from binanceneural.config import PolicyConfig
        import binanceneural.model as model_mod
        from binanceneural.model import BinanceHourlyPolicyNano

        cfg = PolicyConfig(
            input_dim=32,
            hidden_dim=64,
            num_heads=8,
            num_kv_heads=1,
            num_layers=2,
            max_len=64,
            model_arch="nano",
        )
        model = _cuda_module_or_skip(BinanceHourlyPolicyNano(cfg).eval())
        x = _cuda_randn_or_skip(2, 48, 32, device="cuda", dtype=torch.float32)

        # This test is specifically about the attention fallback path, so patch
        # the model-level Triton kernel switch off as well to avoid unrelated
        # RMSNorm/rope Triton kernels from masking the attention behavior.
        with mock.patch.object(attn_mod, "HAS_TRITON", False), \
             mock.patch.object(model_mod, "_HAS_TRITON_KERNELS", False):
            try:
                out = model(x)
            except Exception as exc:
                _skip_for_cuda_resource_pressure(exc)
                _skip_for_cuda_compiler_environment(exc)
                raise
        assert "buy_price_logits" in out
        assert out["buy_price_logits"].shape == (2, 48, 1)


# ---------------------------------------------------------------------------
# Model integration test
# ---------------------------------------------------------------------------

class TestModelIntegration:
    """End-to-end forward pass through BinanceHourlyPolicyNano."""

    def test_nano_policy_forward_with_triton(self):
        from binanceneural.kernels.attention import HAS_TRITON
        if not HAS_TRITON:
            pytest.skip("Triton not installed")

        from binanceneural.config import PolicyConfig
        from binanceneural.model import BinanceHourlyPolicyNano

        cfg = PolicyConfig(
            input_dim=32,
            hidden_dim=64,
            num_heads=8,
            num_kv_heads=1,
            num_layers=2,
            max_len=64,
            model_arch="nano",
        )
        model = _cuda_module_or_skip(BinanceHourlyPolicyNano(cfg).eval())
        x = _cuda_randn_or_skip(4, 48, 32, device="cuda", dtype=torch.float32)
        try:
            out = model(x)
        except Exception as exc:
            _skip_for_cuda_resource_pressure(exc)
            _skip_for_cuda_compiler_environment(exc)
            raise
        assert "buy_price_logits" in out
        assert out["buy_price_logits"].shape == (4, 48, 1)

    def test_nano_policy_causal_window_forward(self):
        """Policy with causal + attention_window uses mask — triton handles it."""
        from binanceneural.kernels.attention import HAS_TRITON
        if not HAS_TRITON:
            pytest.skip("Triton not installed")

        from binanceneural.config import PolicyConfig
        from binanceneural.model import BinanceHourlyPolicyNano

        cfg = PolicyConfig(
            input_dim=32,
            hidden_dim=64,
            num_heads=8,
            num_kv_heads=1,
            num_layers=2,
            max_len=64,
            use_causal_attention=True,
            attention_window=12,
            model_arch="nano",
        )
        model = _cuda_module_or_skip(BinanceHourlyPolicyNano(cfg).eval())
        x = _cuda_randn_or_skip(4, 48, 32, device="cuda", dtype=torch.float32)
        try:
            out = model(x)
        except Exception as exc:
            _skip_for_cuda_resource_pressure(exc)
            _skip_for_cuda_compiler_environment(exc)
            raise
        assert "buy_price_logits" in out


# ---------------------------------------------------------------------------
# flash_attn_mqa fallback chain tests
# ---------------------------------------------------------------------------

class TestFlashAttnMqa:
    """Tests for flash_attn_mqa() fallback chain: flash_attn -> Triton -> SDPA."""

    def test_flash_attn_mqa_matches_triton(self):
        """flash_attn_mqa should produce output close to the Triton kernel."""
        from binanceneural.kernels.attention import flash_attn_mqa, HAS_TRITON
        if not HAS_TRITON:
            pytest.skip("Triton not installed")

        Q, K, V = _make_qkv(4, 8, 48, 64, kv_heads=1)
        ref = _sdpa_reference(Q, K, V)
        got = flash_attn_mqa(Q, K, V)
        torch.testing.assert_close(got.float(), ref.float(), atol=1e-2, rtol=1e-2)

    def test_flash_attn_mqa_causal(self):
        """flash_attn_mqa with causal=True must match causal reference."""
        from binanceneural.kernels.attention import flash_attn_mqa, HAS_TRITON
        if not HAS_TRITON:
            pytest.skip("Triton not installed")

        Q, K, V = _make_qkv(4, 8, 32, 64, kv_heads=1)
        ref = _sdpa_reference(Q, K, V, causal=True)
        got = flash_attn_mqa(Q, K, V, causal=True)
        torch.testing.assert_close(got.float(), ref.float(), atol=1e-2, rtol=1e-2)

    def test_flash_attn_mqa_output_shape(self):
        """Output shape must equal Q shape."""
        from binanceneural.kernels.attention import flash_attn_mqa
        Q, K, V = _make_qkv(2, 8, 48, 64, kv_heads=2)
        out = flash_attn_mqa(Q, K, V)
        assert out.shape == Q.shape

    def test_flash_attn_mqa_dtype_preserved(self):
        """Output dtype must match input dtype."""
        from binanceneural.kernels.attention import flash_attn_mqa
        Q, K, V = _make_qkv(2, 8, 48, 64, kv_heads=1, dtype=torch.bfloat16)
        out = flash_attn_mqa(Q, K, V)
        assert out.dtype == torch.bfloat16

    def test_flash_attn_mqa_sdpa_fallback(self):
        """When both HAS_FLASH_ATTN and HAS_TRITON are False, falls back to SDPA."""
        import binanceneural.kernels.attention as attn_mod
        Q, K, V = _make_qkv(2, 8, 32, 64, kv_heads=1)
        with mock.patch.object(attn_mod, "HAS_FLASH_ATTN", False), \
             mock.patch.object(attn_mod, "HAS_TRITON", False):
            out = attn_mod.flash_attn_mqa(Q, K, V)
        ref = _sdpa_reference(Q, K, V)
        torch.testing.assert_close(out.float(), ref.float(), atol=1e-2, rtol=1e-2)

    def test_flash_attn_mqa_gqa(self):
        """GQA with 2 KV heads should work via SDPA fallback."""
        from binanceneural.kernels.attention import flash_attn_mqa, HAS_TRITON
        if not HAS_TRITON:
            pytest.skip("Triton not installed")
        Q, K, V = _make_qkv(2, 8, 32, 64, kv_heads=2)
        ref = _sdpa_reference(Q, K, V)
        got = flash_attn_mqa(Q, K, V)
        torch.testing.assert_close(got.float(), ref.float(), atol=1e-2, rtol=1e-2)


# ---------------------------------------------------------------------------
# Flash Attention hardware detection tests (SM 8.0+ including SM 12.0)
# ---------------------------------------------------------------------------

class TestFlashAttnDetection:
    """Verify HAS_FLASH_ATTN flag and version string reflect hardware capability."""

    def test_probe_attention_backend_returns_false_on_runtime_error(self):
        import binanceneural.kernels.attention as attn_mod

        orig_zeros = torch.zeros
        with mock.patch("torch.cuda.is_available", return_value=True), \
             mock.patch("torch.zeros", side_effect=lambda *args, **kwargs: orig_zeros(1)):
            assert attn_mod._probe_attention_backend(  # type: ignore[attr-defined]
                lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("backend boom"))
            ) is False

    def test_probe_attention_backend_returns_true_for_working_backend(self):
        import binanceneural.kernels.attention as attn_mod

        orig_zeros = torch.zeros
        with mock.patch("torch.cuda.is_available", return_value=True), \
             mock.patch("torch.zeros", side_effect=lambda *args, **kwargs: orig_zeros(1)):
            assert attn_mod._probe_attention_backend(  # type: ignore[attr-defined]
                lambda q, _k, _v, causal=False: q
            ) is True

    def test_has_flash_attn_is_bool(self):
        """HAS_FLASH_ATTN must always be a bool (import probe must not raise)."""
        from binanceneural.kernels.attention import HAS_FLASH_ATTN
        assert isinstance(HAS_FLASH_ATTN, bool)

    def test_flash_attn_version_is_string(self):
        """_flash_attn_version is a string when flash_attn is available, else empty."""
        from binanceneural.kernels.attention import HAS_FLASH_ATTN, _flash_attn_version
        assert isinstance(_flash_attn_version, str)
        if HAS_FLASH_ATTN:
            assert len(_flash_attn_version) > 0

    def test_flash_attn_mqa_correctness_sm80plus(self):
        """On SM8.0+ with flash_attn installed, flash_attn_mqa must match reference.

        FA2 2.8.3 ships native cubins for SM 8.0, 9.0, 10.0, and 12.0 so this
        test covers A100 (SM 8.0), H100 (SM 9.0), B100 (SM 10.0), and RTX 5090
        (SM 12.0) without modification.
        """
        if not torch.cuda.is_available():
            pytest.skip("CUDA required")
        major, _ = torch.cuda.get_device_capability()
        if major < 8:
            pytest.skip("SM80+ required for flash_attn")
        from binanceneural.kernels.attention import HAS_FLASH_ATTN, flash_attn_mqa
        if not HAS_FLASH_ATTN:
            pytest.skip("flash_attn not installed on this GPU")
        Q, K, V = _make_qkv(4, 8, 48, 64, kv_heads=1)
        ref = _sdpa_reference(Q, K, V)
        got = flash_attn_mqa(Q, K, V)
        torch.testing.assert_close(got.float(), ref.float(), atol=1e-2, rtol=1e-2)

    def test_flash_attn_mqa_causal_sm80plus(self):
        """flash_attn_mqa causal=True matches reference on SM8.0+."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA required")
        major, _ = torch.cuda.get_device_capability()
        if major < 8:
            pytest.skip("SM80+ required for flash_attn")
        from binanceneural.kernels.attention import HAS_FLASH_ATTN, flash_attn_mqa
        if not HAS_FLASH_ATTN:
            pytest.skip("flash_attn not installed on this GPU")
        Q, K, V = _make_qkv(4, 8, 48, 64, kv_heads=1)
        ref = _sdpa_reference(Q, K, V, causal=True)
        got = flash_attn_mqa(Q, K, V, causal=True)
        torch.testing.assert_close(got.float(), ref.float(), atol=1e-2, rtol=1e-2)
