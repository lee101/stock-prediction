"""Triton fused RoPE (Rotary Position Embedding) kernel for Chronos2.

Fuses inv_freq + cos/sin generation + application to Q and K tensors
in a single kernel launch per (Q, K) pair.

Reference: chronos-forecasting/src/chronos/chronos2/layers.py (class RoPE)
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _apply_rope_to_tensor(
    x_ptr,
    out_ptr,
    cos_val,
    sin_val,
    s_offs,
    s_mask,
    d_offs,
    stride_xs,
    stride_xd,
    stride_os,
    stride_od,
    half_dim: tl.constexpr,
    IS_BF16: tl.constexpr,
):
    """Apply RoPE to a single (Q or K) tensor tile.

    RoPE math (using rotate_half convention):
      result_lo[d] = x_lo[d] * cos[d] - x_hi[d] * sin[d]
      result_hi[d] = x_hi[d] * cos[d] + x_lo[d] * sin[d]
    """
    ptrs_lo = x_ptr + s_offs[:, None] * stride_xs + d_offs[None, :] * stride_xd
    ptrs_hi = x_ptr + s_offs[:, None] * stride_xs + (d_offs[None, :] + half_dim) * stride_xd

    x_lo = tl.load(ptrs_lo, mask=s_mask[:, None], other=0.0).to(tl.float32)
    x_hi = tl.load(ptrs_hi, mask=s_mask[:, None], other=0.0).to(tl.float32)

    out_lo = x_lo * cos_val - x_hi * sin_val
    out_hi = x_hi * cos_val + x_lo * sin_val

    o_ptrs_lo = out_ptr + s_offs[:, None] * stride_os + d_offs[None, :] * stride_od
    o_ptrs_hi = out_ptr + s_offs[:, None] * stride_os + (d_offs[None, :] + half_dim) * stride_od

    if IS_BF16:
        tl.store(o_ptrs_lo, out_lo.to(tl.bfloat16), mask=s_mask[:, None])
        tl.store(o_ptrs_hi, out_hi.to(tl.bfloat16), mask=s_mask[:, None])
    else:
        tl.store(o_ptrs_lo, out_lo, mask=s_mask[:, None])
        tl.store(o_ptrs_hi, out_hi, mask=s_mask[:, None])


@triton.jit
def _rope_fwd_kernel(
    Q_ptr,
    K_ptr,
    Q_out_ptr,
    K_out_ptr,
    inv_freq_ptr,
    position_ids_ptr,
    stride_qb, stride_qh, stride_qs, stride_qd,
    stride_kb, stride_kh, stride_ks, stride_kd,
    stride_qob, stride_qoh, stride_qos, stride_qod,
    stride_kob, stride_koh, stride_kos, stride_kod,
    stride_pb, stride_ps,
    num_heads,
    seq_len,
    half_dim: tl.constexpr,
    BLOCK_S: tl.constexpr,
    IS_BF16: tl.constexpr,
):
    """Fused RoPE kernel.

    Each program handles one (batch, head, seq_block) tile.
    Loads inv_freq and position_ids, computes cos/sin in FP32,
    then applies RoPE to both Q and K.
    """
    pid_bs = tl.program_id(0)
    num_seq_blocks = tl.cdiv(seq_len, BLOCK_S)

    pid_b = pid_bs // (num_heads * num_seq_blocks)
    remainder = pid_bs % (num_heads * num_seq_blocks)
    pid_h = remainder // num_seq_blocks
    pid_s = remainder % num_seq_blocks

    s_offs = pid_s * BLOCK_S + tl.arange(0, BLOCK_S)
    s_mask = s_offs < seq_len

    # Load position_ids for this batch and seq block (FP32)
    pos_ptrs = position_ids_ptr + pid_b * stride_pb + s_offs * stride_ps
    pos_ids = tl.load(pos_ptrs, mask=s_mask, other=0.0).to(tl.float32)

    # Load inv_freq (FP32)
    d_offs = tl.arange(0, half_dim)
    inv_freq = tl.load(inv_freq_ptr + d_offs).to(tl.float32)

    # Compute freqs and cos/sin in FP32
    freqs = pos_ids[:, None] * inv_freq[None, :]
    cos_val = tl.cos(freqs)
    sin_val = tl.sin(freqs)

    # Apply RoPE to Q
    q_base = Q_ptr + pid_b * stride_qb + pid_h * stride_qh
    qo_base = Q_out_ptr + pid_b * stride_qob + pid_h * stride_qoh
    _apply_rope_to_tensor(
        q_base, qo_base, cos_val, sin_val,
        s_offs, s_mask, d_offs,
        stride_qs, stride_qd, stride_qos, stride_qod,
        half_dim, IS_BF16,
    )

    # Apply RoPE to K
    k_base = K_ptr + pid_b * stride_kb + pid_h * stride_kh
    ko_base = K_out_ptr + pid_b * stride_kob + pid_h * stride_koh
    _apply_rope_to_tensor(
        k_base, ko_base, cos_val, sin_val,
        s_offs, s_mask, d_offs,
        stride_ks, stride_kd, stride_kos, stride_kod,
        half_dim, IS_BF16,
    )


def apply_rope(
    q: torch.Tensor,
    k: torch.Tensor,
    inv_freq: torch.Tensor,
    position_ids: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply RoPE to Q and K tensors using a fused Triton kernel.

    Args:
        q: Query tensor, shape (batch, num_heads, seq_len, head_dim)
        k: Key tensor, shape (batch, num_heads, seq_len, head_dim)
        inv_freq: Inverse frequency buffer, shape (head_dim // 2,)
        position_ids: Position IDs, shape (batch, seq_len)

    Returns:
        Tuple of (q_embed, k_embed) with RoPE applied, same shapes/dtype as inputs.
    """
    batch_size, num_heads, seq_len, head_dim = q.shape
    half_dim = head_dim // 2
    assert head_dim % 2 == 0, f"head_dim must be even, got {head_dim}"
    assert inv_freq.shape == (half_dim,), (
        f"inv_freq shape mismatch: {inv_freq.shape} vs ({half_dim},)"
    )
    assert position_ids.shape == (batch_size, seq_len), (
        f"position_ids shape mismatch: {position_ids.shape} vs ({batch_size}, {seq_len})"
    )

    q = q.contiguous()
    k = k.contiguous()
    inv_freq = inv_freq.contiguous().float()
    position_ids = position_ids.contiguous().float()

    q_out = torch.empty_like(q)
    k_out = torch.empty_like(k)

    BLOCK_S = triton.next_power_of_2(min(seq_len, 128))
    num_seq_blocks = triton.cdiv(seq_len, BLOCK_S)
    grid = (batch_size * num_heads * num_seq_blocks,)

    is_bf16 = q.dtype == torch.bfloat16

    _rope_fwd_kernel[grid](
        q, k, q_out, k_out,
        inv_freq, position_ids,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        q_out.stride(0), q_out.stride(1), q_out.stride(2), q_out.stride(3),
        k_out.stride(0), k_out.stride(1), k_out.stride(2), k_out.stride(3),
        position_ids.stride(0), position_ids.stride(1),
        num_heads, seq_len, half_dim,
        BLOCK_S=BLOCK_S,
        IS_BF16=is_bf16,
    )

    return q_out, k_out


@triton.jit
def _compute_cos_sin_kernel(
    cos_ptr,
    sin_ptr,
    inv_freq_ptr,
    position_ids_ptr,
    stride_cb, stride_cs, stride_cd,
    stride_sb, stride_ss, stride_sd,
    stride_pb, stride_ps,
    seq_len,
    half_dim: tl.constexpr,
    BLOCK_S: tl.constexpr,
):
    """Compute cos/sin embeddings from inv_freq and position_ids."""
    pid = tl.program_id(0)
    num_seq_blocks = tl.cdiv(seq_len, BLOCK_S)
    pid_b = pid // num_seq_blocks
    pid_s = pid % num_seq_blocks

    s_offs = pid_s * BLOCK_S + tl.arange(0, BLOCK_S)
    s_mask = s_offs < seq_len

    pos_ptrs = position_ids_ptr + pid_b * stride_pb + s_offs * stride_ps
    pos_ids = tl.load(pos_ptrs, mask=s_mask, other=0.0).to(tl.float32)

    d_offs = tl.arange(0, half_dim)
    inv_freq = tl.load(inv_freq_ptr + d_offs).to(tl.float32)

    freqs = pos_ids[:, None] * inv_freq[None, :]
    cos_val = tl.cos(freqs)
    sin_val = tl.sin(freqs)

    # Store duplicated (matching cat(freqs, freqs) in reference)
    cos_base = cos_ptr + pid_b * stride_cb
    sin_base = sin_ptr + pid_b * stride_sb

    cos_ptrs_lo = cos_base + s_offs[:, None] * stride_cs + d_offs[None, :] * stride_cd
    cos_ptrs_hi = cos_base + s_offs[:, None] * stride_cs + (d_offs[None, :] + half_dim) * stride_cd
    sin_ptrs_lo = sin_base + s_offs[:, None] * stride_ss + d_offs[None, :] * stride_sd
    sin_ptrs_hi = sin_base + s_offs[:, None] * stride_ss + (d_offs[None, :] + half_dim) * stride_sd

    tl.store(cos_ptrs_lo, cos_val, mask=s_mask[:, None])
    tl.store(cos_ptrs_hi, cos_val, mask=s_mask[:, None])
    tl.store(sin_ptrs_lo, sin_val, mask=s_mask[:, None])
    tl.store(sin_ptrs_hi, sin_val, mask=s_mask[:, None])


def compute_cos_sin(
    inv_freq: torch.Tensor,
    position_ids: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute cos/sin embeddings matching the reference RoPE.forward().

    Args:
        inv_freq: Inverse frequency buffer, shape (half_dim,)
        position_ids: Position IDs, shape (batch, seq_len)

    Returns:
        Tuple of (cos, sin) each shape (batch, seq_len, head_dim) in FP32.
        head_dim = 2 * half_dim, with values duplicated across halves.
    """
    batch_size, seq_len = position_ids.shape
    half_dim = inv_freq.shape[0]
    head_dim = 2 * half_dim

    inv_freq = inv_freq.contiguous().float()
    position_ids = position_ids.contiguous().float()

    cos_out = torch.empty(
        batch_size, seq_len, head_dim, device=inv_freq.device, dtype=torch.float32
    )
    sin_out = torch.empty(
        batch_size, seq_len, head_dim, device=inv_freq.device, dtype=torch.float32
    )

    BLOCK_S = triton.next_power_of_2(min(seq_len, 128))
    num_seq_blocks = triton.cdiv(seq_len, BLOCK_S)
    grid = (batch_size * num_seq_blocks,)

    _compute_cos_sin_kernel[grid](
        cos_out, sin_out, inv_freq, position_ids,
        cos_out.stride(0), cos_out.stride(1), cos_out.stride(2),
        sin_out.stride(0), sin_out.stride(1), sin_out.stride(2),
        position_ids.stride(0), position_ids.stride(1),
        seq_len, half_dim,
        BLOCK_S=BLOCK_S,
    )

    return cos_out, sin_out
