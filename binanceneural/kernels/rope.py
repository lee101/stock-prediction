"""Triton fused RoPE kernel for binanceneural.

Fuses the full RoPE computation: cos/sin application to Q and K.

The model uses interleaved convention: even indices are "real", odd are "imaginary".
  out[..., 2i]   = x[..., 2i]   * cos[i] - x[..., 2i+1] * sin[i]
  out[..., 2i+1] = x[..., 2i]   * sin[i] + x[..., 2i+1] * cos[i]

This matches _apply_rotary_emb() in binanceneural/model.py which uses
x[..., ::2] and x[..., 1::2] (not the split-half convention).

cos/sin from RotaryEmbedding.forward() have shape (1, seq_len, 1, half_dim).
apply_rope() accepts those shapes directly.

apply_rope_fused() takes raw Q, K and computes cos/sin from inv_freq internally.

Design: Q and K are processed by separate kernel launches (one per tensor).
Grid is (B * H * num_seq_blocks,) — no head loop inside the kernel.
"""

from __future__ import annotations

import torch

try:
    import triton
    import triton.language as tl
    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False


if HAS_TRITON:
    @triton.jit
    def _rope_apply_kernel(
        X_ptr,
        Out_ptr,
        cos_ptr,          # (T, half_dim) FP32
        sin_ptr,          # (T, half_dim) FP32
        stride_xbh,       # stride for the (B*H) flat dimension
        stride_xs,        # sequence stride
        stride_xd,        # head-dim stride
        stride_obh,
        stride_os,
        stride_od,
        stride_cos_s,
        stride_cos_d,
        stride_sin_s,
        stride_sin_d,
        seq_len,
        half_dim: tl.constexpr,
        BLOCK_S: tl.constexpr,
        IS_BF16: tl.constexpr,
    ):
        """Apply interleaved RoPE to one (Q or K) tensor viewed as (B*H, T, D).

        Grid: (B * H * ceil(T / BLOCK_S),)
        """
        pid = tl.program_id(0)
        num_seq_blocks = tl.cdiv(seq_len, BLOCK_S)

        pid_s = pid % num_seq_blocks
        pid_bh = pid // num_seq_blocks

        s_offs = pid_s * BLOCK_S + tl.arange(0, BLOCK_S)
        s_mask = s_offs < seq_len
        d_offs = tl.arange(0, half_dim)

        cos_ptrs = cos_ptr + s_offs[:, None] * stride_cos_s + d_offs[None, :] * stride_cos_d
        sin_ptrs = sin_ptr + s_offs[:, None] * stride_sin_s + d_offs[None, :] * stride_sin_d
        cos_val = tl.load(cos_ptrs, mask=s_mask[:, None], other=0.0).to(tl.float32)
        sin_val = tl.load(sin_ptrs, mask=s_mask[:, None], other=0.0).to(tl.float32)

        real_offs = d_offs * 2
        imag_offs = d_offs * 2 + 1

        x_base = X_ptr + pid_bh * stride_xbh
        o_base = Out_ptr + pid_bh * stride_obh

        real_ptrs = x_base + s_offs[:, None] * stride_xs + real_offs[None, :] * stride_xd
        imag_ptrs = x_base + s_offs[:, None] * stride_xs + imag_offs[None, :] * stride_xd

        x_r = tl.load(real_ptrs, mask=s_mask[:, None], other=0.0).to(tl.float32)
        x_i = tl.load(imag_ptrs, mask=s_mask[:, None], other=0.0).to(tl.float32)

        out_r = x_r * cos_val - x_i * sin_val
        out_i = x_r * sin_val + x_i * cos_val

        out_real_ptrs = o_base + s_offs[:, None] * stride_os + real_offs[None, :] * stride_od
        out_imag_ptrs = o_base + s_offs[:, None] * stride_os + imag_offs[None, :] * stride_od

        if IS_BF16:
            tl.store(out_real_ptrs, out_r.to(tl.bfloat16), mask=s_mask[:, None])
            tl.store(out_imag_ptrs, out_i.to(tl.bfloat16), mask=s_mask[:, None])
        else:
            tl.store(out_real_ptrs, out_r, mask=s_mask[:, None])
            tl.store(out_imag_ptrs, out_i, mask=s_mask[:, None])

    @triton.jit
    def _rope_fused_kernel(
        X_ptr,
        Out_ptr,
        inv_freq_ptr,     # (half_dim,) FP32
        stride_xbh,
        stride_xs,
        stride_xd,
        stride_obh,
        stride_os,
        stride_od,
        seq_len,
        half_dim: tl.constexpr,
        BLOCK_S: tl.constexpr,
        IS_BF16: tl.constexpr,
    ):
        """Fully fused: compute cos/sin from inv_freq, then apply to one tensor.

        Grid: (B * H * ceil(T / BLOCK_S),)
        """
        pid = tl.program_id(0)
        num_seq_blocks = tl.cdiv(seq_len, BLOCK_S)

        pid_s = pid % num_seq_blocks
        pid_bh = pid // num_seq_blocks

        s_offs = pid_s * BLOCK_S + tl.arange(0, BLOCK_S)
        s_mask = s_offs < seq_len
        d_offs = tl.arange(0, half_dim)

        pos = s_offs.to(tl.float32)
        inv_freq = tl.load(inv_freq_ptr + d_offs).to(tl.float32)
        freqs = pos[:, None] * inv_freq[None, :]
        cos_val = tl.cos(freqs)
        sin_val = tl.sin(freqs)

        real_offs = d_offs * 2
        imag_offs = d_offs * 2 + 1

        x_base = X_ptr + pid_bh * stride_xbh
        o_base = Out_ptr + pid_bh * stride_obh

        real_ptrs = x_base + s_offs[:, None] * stride_xs + real_offs[None, :] * stride_xd
        imag_ptrs = x_base + s_offs[:, None] * stride_xs + imag_offs[None, :] * stride_xd

        x_r = tl.load(real_ptrs, mask=s_mask[:, None], other=0.0).to(tl.float32)
        x_i = tl.load(imag_ptrs, mask=s_mask[:, None], other=0.0).to(tl.float32)

        out_r = x_r * cos_val - x_i * sin_val
        out_i = x_r * sin_val + x_i * cos_val

        out_real_ptrs = o_base + s_offs[:, None] * stride_os + real_offs[None, :] * stride_od
        out_imag_ptrs = o_base + s_offs[:, None] * stride_os + imag_offs[None, :] * stride_od

        if IS_BF16:
            tl.store(out_real_ptrs, out_r.to(tl.bfloat16), mask=s_mask[:, None])
            tl.store(out_imag_ptrs, out_i.to(tl.bfloat16), mask=s_mask[:, None])
        else:
            tl.store(out_real_ptrs, out_r, mask=s_mask[:, None])
            tl.store(out_imag_ptrs, out_i, mask=s_mask[:, None])


def _launch_apply(x, x_out, cos_2d, sin_2d, BLOCK_S, is_bf16, half_dim_pow2):
    """Launch _rope_apply_kernel for (B, T, H, D) tensor.

    Permutes to (B, H, T, D) so the (B*H) flat dim is contiguous,
    then permutes the output back to (B, T, H, D).
    """
    B, T, H, D = x.shape
    # (B, T, H, D) -> (B, H, T, D): contiguous so (B*H) rows are flat
    x_perm = x.permute(0, 2, 1, 3).contiguous()   # strides: (H*T*D, T*D, D, 1)
    o_perm = torch.empty_like(x_perm)

    # stride_xbh = stride of one (b, h) element = x_perm.stride(1) = T*D
    grid = (B * H * triton.cdiv(T, BLOCK_S),)
    _rope_apply_kernel[grid](
        x_perm, o_perm,
        cos_2d, sin_2d,
        x_perm.stride(1),       # stride_xbh
        x_perm.stride(2),       # stride_xs
        x_perm.stride(3),       # stride_xd
        o_perm.stride(1),       # stride_obh
        o_perm.stride(2),       # stride_os
        o_perm.stride(3),       # stride_od
        cos_2d.stride(0), cos_2d.stride(1),
        sin_2d.stride(0), sin_2d.stride(1),
        T,
        half_dim=half_dim_pow2,
        BLOCK_S=BLOCK_S,
        IS_BF16=is_bf16,
    )
    x_out.copy_(o_perm.permute(0, 2, 1, 3))


def _launch_fused(x, x_out, inv_freq_c, BLOCK_S, is_bf16, half_dim_pow2):
    """Launch _rope_fused_kernel for (B, T, H, D) tensor."""
    B, T, H, D = x.shape
    x_perm = x.permute(0, 2, 1, 3).contiguous()   # (B, H, T, D)
    o_perm = torch.empty_like(x_perm)

    grid = (B * H * triton.cdiv(T, BLOCK_S),)
    _rope_fused_kernel[grid](
        x_perm, o_perm,
        inv_freq_c,
        x_perm.stride(1),
        x_perm.stride(2),
        x_perm.stride(3),
        o_perm.stride(1),
        o_perm.stride(2),
        o_perm.stride(3),
        T,
        half_dim=half_dim_pow2,
        BLOCK_S=BLOCK_S,
        IS_BF16=is_bf16,
    )
    x_out.copy_(o_perm.permute(0, 2, 1, 3))


# ---------------------------------------------------------------------------
# PyTorch fallback
# ---------------------------------------------------------------------------

def _rope_pytorch(q, k, cos, sin):
    """Interleaved RoPE fallback. cos/sin: (1, T, 1, half_dim)."""
    def _rot(x):
        xr = x[..., ::2]
        xi = x[..., 1::2]
        return torch.stack([xr * cos - xi * sin, xr * sin + xi * cos], dim=-1).flatten(-2)
    return _rot(q), _rot(k)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def apply_rope(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply pre-computed cos/sin RoPE to Q and K (interleaved convention).

    cos/sin: (1, seq_len, 1, half_dim) as returned by RotaryEmbedding.forward()
    Q: (batch, seq, num_q_heads, head_dim)
    K: (batch, seq, num_k_heads, head_dim)

    Returns rotated Q, K with same shapes/dtype.
    """
    if not HAS_TRITON or not q.is_cuda:
        return _rope_pytorch(q, k, cos, sin)

    B, T, Hq, D = q.shape
    half_dim = D // 2

    if D % 2 != 0:
        raise ValueError(f"head_dim must be even, got {D}")
    half_dim_pow2 = triton.next_power_of_2(half_dim)
    if half_dim_pow2 > 1024:
        return _rope_pytorch(q, k, cos, sin)

    # (1, T, 1, half_dim) -> (T, half_dim) FP32
    cos_2d = cos.reshape(T, half_dim).contiguous().float()
    sin_2d = sin.reshape(T, half_dim).contiguous().float()

    q_cont = q.contiguous()
    k_cont = k.contiguous()
    q_out = torch.empty_like(q_cont)
    k_out = torch.empty_like(k_cont)

    BLOCK_S = triton.next_power_of_2(min(T, 64))
    is_bf16 = q.dtype == torch.bfloat16

    _launch_apply(q_cont, q_out, cos_2d, sin_2d, BLOCK_S, is_bf16, half_dim_pow2)
    _launch_apply(k_cont, k_out, cos_2d, sin_2d, BLOCK_S, is_bf16, half_dim_pow2)

    return q_out, k_out


def apply_rope_fused(
    q: torch.Tensor,
    k: torch.Tensor,
    inv_freq: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fully fused RoPE: compute cos/sin from inv_freq and apply to Q, K.

    Q: (batch, seq, num_q_heads, head_dim)
    K: (batch, seq, num_k_heads, head_dim)
    inv_freq: (head_dim // 2,)

    Returns rotated Q, K with same shapes/dtype.
    """
    B, T, Hq, D = q.shape
    half_dim = D // 2

    def _fallback():
        t = torch.arange(T, device=q.device, dtype=torch.float32)
        freqs = torch.outer(t, inv_freq.float())
        cos = freqs.cos()[None, :, None, :]
        sin = freqs.sin()[None, :, None, :]
        return _rope_pytorch(q, k, cos, sin)

    if not HAS_TRITON or not q.is_cuda:
        return _fallback()
    if D % 2 != 0:
        raise ValueError(f"head_dim must be even, got {D}")
    half_dim_pow2 = triton.next_power_of_2(half_dim)
    if half_dim_pow2 > 1024:
        return _fallback()

    q_cont = q.contiguous()
    k_cont = k.contiguous()
    q_out = torch.empty_like(q_cont)
    k_out = torch.empty_like(k_cont)

    BLOCK_S = triton.next_power_of_2(min(T, 64))
    is_bf16 = q.dtype == torch.bfloat16
    inv_freq_c = inv_freq.contiguous().float()

    _launch_fused(q_cont, q_out, inv_freq_c, BLOCK_S, is_bf16, half_dim_pow2)
    _launch_fused(k_cont, k_out, inv_freq_c, BLOCK_S, is_bf16, half_dim_pow2)

    return q_out, k_out


__all__ = ["HAS_TRITON", "apply_rope", "apply_rope_fused"]
