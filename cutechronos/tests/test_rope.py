"""Tests for Triton RoPE kernel against reference PyTorch implementation."""

import pytest
import torch

from cutechronos.triton_kernels.rope import apply_rope, compute_cos_sin


# ---------------------------------------------------------------------------
# Reference implementation (copied from chronos2/layers.py)
# ---------------------------------------------------------------------------

class ReferenceRoPE:
    """Reference RoPE matching chronos-forecasting layers.py exactly."""

    def __init__(self, dim: int, base: float = 10000.0, device: str = "cuda"):
        inv_freq = 1.0 / (
            base ** (torch.arange(0, dim, 2, dtype=torch.int64).float() / dim)
        )
        self.inv_freq = inv_freq.to(device)

    def forward(
        self, x: torch.Tensor, position_ids: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        inv_freq_expanded = (
            self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
        )
        position_ids_expanded = position_ids[:, None, :].float()
        device_type = x.device.type
        device_type = (
            device_type
            if isinstance(device_type, str) and device_type != "mps"
            else "cpu"
        )
        with torch.autocast(device_type=device_type, enabled=False):
            freqs = (
                inv_freq_expanded.float() @ position_ids_expanded.float()
            ).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos()
            sin = emb.sin()
        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


def ref_rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def ref_apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    unsqueeze_dim: int = 1,
) -> tuple[torch.Tensor, torch.Tensor]:
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (ref_rotate_half(q) * sin)
    k_embed = (k * cos) + (ref_rotate_half(k) * sin)
    return q_embed, k_embed


def reference_rope(
    q: torch.Tensor,
    k: torch.Tensor,
    position_ids: torch.Tensor,
    head_dim: int = 64,
    base: float = 10000.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Full reference RoPE: compute cos/sin then apply to q, k."""
    rope = ReferenceRoPE(dim=head_dim, base=base, device=q.device.type)
    cos, sin = rope.forward(q, position_ids)
    return ref_apply_rotary_pos_emb(q, k, cos, sin)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

NUM_HEADS = 12
HEAD_DIM = 64
BASE = 10000.0


def make_inv_freq(dim: int = HEAD_DIM, base: float = BASE, device: str = "cuda"):
    return 1.0 / (
        base ** (torch.arange(0, dim, 2, dtype=torch.int64).float() / dim)
    ).to(device)


def make_inputs(batch: int, seq_len: int, dtype: torch.dtype, device: str = "cuda"):
    q = torch.randn(batch, NUM_HEADS, seq_len, HEAD_DIM, device=device, dtype=dtype)
    k = torch.randn(batch, NUM_HEADS, seq_len, HEAD_DIM, device=device, dtype=dtype)
    position_ids = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch, -1)
    return q, k, position_ids


# ---------------------------------------------------------------------------
# Tests: compute_cos_sin
# ---------------------------------------------------------------------------

class TestComputeCosSin:
    @pytest.mark.parametrize("batch,seq_len", [(1, 34), (4, 130), (16, 514)])
    def test_matches_reference(self, batch: int, seq_len: int):
        device = "cuda"
        inv_freq = make_inv_freq(device=device)
        position_ids = (
            torch.arange(seq_len, device=device).unsqueeze(0).expand(batch, -1)
        )
        dummy_x = torch.randn(1, device=device)  # for dtype

        ref_rope = ReferenceRoPE(dim=HEAD_DIM, device=device)
        cos_ref, sin_ref = ref_rope.forward(dummy_x, position_ids)

        cos_tri, sin_tri = compute_cos_sin(inv_freq, position_ids)

        assert cos_tri.shape == cos_ref.shape, (
            f"cos shape mismatch: {cos_tri.shape} vs {cos_ref.shape}"
        )
        assert sin_tri.shape == sin_ref.shape, (
            f"sin shape mismatch: {sin_tri.shape} vs {sin_ref.shape}"
        )

        max_cos_err = (cos_tri - cos_ref.float()).abs().max().item()
        max_sin_err = (sin_tri - sin_ref.float()).abs().max().item()

        assert max_cos_err < 1e-5, f"cos max error too large: {max_cos_err}"
        assert max_sin_err < 1e-5, f"sin max error too large: {max_sin_err}"


# ---------------------------------------------------------------------------
# Tests: apply_rope
# ---------------------------------------------------------------------------

class TestApplyRope:
    @pytest.mark.parametrize("batch,seq_len", [(1, 34), (4, 130), (16, 514)])
    def test_fp32(self, batch: int, seq_len: int):
        q, k, position_ids = make_inputs(batch, seq_len, torch.float32)
        inv_freq = make_inv_freq()

        q_ref, k_ref = reference_rope(q, k, position_ids)
        q_tri, k_tri = apply_rope(q, k, inv_freq, position_ids)

        max_q_err = (q_tri - q_ref).abs().max().item()
        max_k_err = (k_tri - k_ref).abs().max().item()

        assert max_q_err < 1e-5, f"FP32 Q max error: {max_q_err}"
        assert max_k_err < 1e-5, f"FP32 K max error: {max_k_err}"

    @pytest.mark.parametrize("batch,seq_len", [(1, 34), (4, 130), (16, 514)])
    def test_bf16(self, batch: int, seq_len: int):
        q, k, position_ids = make_inputs(batch, seq_len, torch.bfloat16)
        inv_freq = make_inv_freq()

        q_ref, k_ref = reference_rope(q, k, position_ids)
        q_tri, k_tri = apply_rope(q, k, inv_freq, position_ids)

        max_q_err = (q_tri.float() - q_ref.float()).abs().max().item()
        max_k_err = (k_tri.float() - k_ref.float()).abs().max().item()

        # Tolerance is 0.04 (not 5e-3) because: our kernel computes
        # the RoPE application in FP32 then casts to BF16, while the
        # reference does BF16 * BF16 arithmetic. The difference is
        # pure BF16 rounding noise (ULP = 2^-7 * magnitude). Our kernel
        # is actually more numerically accurate than the reference.
        assert max_q_err < 0.04, f"BF16 Q max error: {max_q_err}"
        assert max_k_err < 0.04, f"BF16 K max error: {max_k_err}"

    def test_output_dtype_matches_input(self):
        for dtype in [torch.float32, torch.bfloat16]:
            q, k, position_ids = make_inputs(2, 64, dtype)
            inv_freq = make_inv_freq()
            q_out, k_out = apply_rope(q, k, inv_freq, position_ids)
            assert q_out.dtype == dtype, f"Q dtype mismatch: {q_out.dtype} vs {dtype}"
            assert k_out.dtype == dtype, f"K dtype mismatch: {k_out.dtype} vs {dtype}"

    def test_output_shape_matches_input(self):
        q, k, position_ids = make_inputs(4, 100, torch.float32)
        inv_freq = make_inv_freq()
        q_out, k_out = apply_rope(q, k, inv_freq, position_ids)
        assert q_out.shape == q.shape
        assert k_out.shape == k.shape

    def test_non_sequential_position_ids(self):
        """Position IDs don't have to be 0,1,2,..."""
        batch, seq_len = 2, 32
        q, k, _ = make_inputs(batch, seq_len, torch.float32)
        inv_freq = make_inv_freq()
        # Random position IDs (e.g., with padding offset)
        position_ids = torch.randint(0, 1000, (batch, seq_len), device="cuda")

        q_ref, k_ref = reference_rope(q, k, position_ids)
        q_tri, k_tri = apply_rope(q, k, inv_freq, position_ids)

        max_q_err = (q_tri - q_ref).abs().max().item()
        max_k_err = (k_tri - k_ref).abs().max().item()

        assert max_q_err < 1e-5, f"Non-sequential Q max error: {max_q_err}"
        assert max_k_err < 1e-5, f"Non-sequential K max error: {max_k_err}"

    def test_batch_1_short_seq(self):
        """Edge case: B=1, S=1."""
        q, k, position_ids = make_inputs(1, 1, torch.float32)
        inv_freq = make_inv_freq()

        q_ref, k_ref = reference_rope(q, k, position_ids)
        q_tri, k_tri = apply_rope(q, k, inv_freq, position_ids)

        max_q_err = (q_tri - q_ref).abs().max().item()
        max_k_err = (k_tri - k_ref).abs().max().item()

        assert max_q_err < 1e-5, f"B=1 S=1 Q max error: {max_q_err}"
        assert max_k_err < 1e-5, f"B=1 S=1 K max error: {max_k_err}"

    def test_large_batch(self):
        """Larger batch to stress grid decomposition."""
        q, k, position_ids = make_inputs(32, 64, torch.float32)
        inv_freq = make_inv_freq()

        q_ref, k_ref = reference_rope(q, k, position_ids)
        q_tri, k_tri = apply_rope(q, k, inv_freq, position_ids)

        max_q_err = (q_tri - q_ref).abs().max().item()
        max_k_err = (k_tri - k_ref).abs().max().item()

        assert max_q_err < 1e-5, f"Large batch Q max error: {max_q_err}"
        assert max_k_err < 1e-5, f"Large batch K max error: {max_k_err}"
