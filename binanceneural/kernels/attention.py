"""Triton fused multi-query attention kernel for binanceneural.

Implements online-softmax (FlashAttention-style tiling) to avoid materializing
the full S x S attention matrix.  Handles:
  - Multi-query attention: Q has num_heads heads, K/V have 1 head (broadcast)
  - Standard scaled dot product (scale = 1/sqrt(head_dim))
  - Optional causal masking built into the kernel
  - Optional additive float mask (sliding-window, dilated, etc.)
  - FP32 accumulation with BF16/FP16/FP32 output

Input shapes:
  Q : [batch, num_heads,    seq, head_dim]
  K : [batch, num_kv_heads, seq, head_dim]   num_kv_heads == 1 for MQA
  V : [batch, num_kv_heads, seq, head_dim]

For GQA/MQA the kernel maps Q head i to KV head i // (num_heads // num_kv_heads),
matching PyTorch repeat_interleave semantics.  No pre-expansion of K/V required.
"""

from __future__ import annotations

import math

import torch

try:
    import triton
    import triton.language as tl

    HAS_TRITON = True
except Exception:
    HAS_TRITON = False


# ---------------------------------------------------------------------------
# Triton kernel
# ---------------------------------------------------------------------------

if HAS_TRITON:

    @triton.jit
    def _mqa_attention_fwd(
        Q_ptr,
        K_ptr,
        V_ptr,
        Mask_ptr,
        Out_ptr,
        # Q strides: [batch, heads, seq, head_dim]
        stride_qb,
        stride_qh,
        stride_qs,
        stride_qd,
        # K strides: [batch, kv_heads, seq, head_dim]
        stride_kb,
        stride_kh,
        stride_ks,
        stride_kd,
        # V strides
        stride_vb,
        stride_vh,
        stride_vs,
        stride_vd,
        # Mask strides (additive float mask, shape broadcastable to [B,H,S,S])
        stride_mb,
        stride_mh,
        stride_ms,
        stride_mn,
        # Out strides
        stride_ob,
        stride_oh,
        stride_os,
        stride_od,
        SEQ_LEN: tl.constexpr,
        D: tl.constexpr,
        SCALE: tl.constexpr,          # 1/sqrt(head_dim) as float
        NUM_KV_HEADS: tl.constexpr,   # 1 for pure MQA, else GQA ratio
        CAUSAL: tl.constexpr,
        HAS_MASK: tl.constexpr,
        MASK_BATCH_STRIDE_ZERO: tl.constexpr,
        MASK_HEAD_STRIDE_ZERO: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        """One program handles (batch, head, BLOCK_M query rows)."""
        pid_m = tl.program_id(0)   # query block index
        pid_bh = tl.program_id(1)  # flattened (batch * num_heads)

        # num_programs(2) == num_heads (Q heads), used to extract batch/head
        num_heads = tl.num_programs(2)
        batch_idx = pid_bh // num_heads
        head_idx = pid_bh % num_heads

        # K/V head: GQA mapping — consecutive groups of (H/KV_HEADS) Q heads share a KV head.
        # Matches PyTorch repeat_interleave semantics: Q head i -> KV head i // (H // KV_HEADS).
        kv_head_idx = head_idx // (num_heads // NUM_KV_HEADS)

        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)  # [BLOCK_M]
        offs_d = tl.arange(0, D)                           # [D]

        # Load Q block [BLOCK_M, D] -> upcasted to FP32
        mask_m = offs_m < SEQ_LEN
        q_ptrs = (
            Q_ptr
            + batch_idx * stride_qb
            + head_idx * stride_qh
            + offs_m[:, None] * stride_qs
            + offs_d[None, :] * stride_qd
        )
        q = tl.load(q_ptrs, mask=mask_m[:, None], other=0.0).to(tl.float32)

        # Apply scale factor to Q (fused: equivalent to scaling scores)
        q = q * SCALE

        # Online-softmax running state
        m_i = tl.full([BLOCK_M], value=float("-inf"), dtype=tl.float32)
        l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
        acc = tl.zeros([BLOCK_M, D], dtype=tl.float32)

        num_blocks_n = tl.cdiv(SEQ_LEN, BLOCK_N)

        for block_n in range(num_blocks_n):
            offs_n = block_n * BLOCK_N + tl.arange(0, BLOCK_N)  # [BLOCK_N]
            mask_n = offs_n < SEQ_LEN

            # Load K block [BLOCK_N, D]
            k_ptrs = (
                K_ptr
                + batch_idx * stride_kb
                + kv_head_idx * stride_kh
                + offs_n[:, None] * stride_ks
                + offs_d[None, :] * stride_kd
            )
            k = tl.load(k_ptrs, mask=mask_n[:, None], other=0.0).to(tl.float32)

            # scores = Q @ K^T  [BLOCK_M, BLOCK_N]  (Q is already scaled)
            scores = tl.dot(q, tl.trans(k), input_precision="ieee")

            # Causal mask: query at position i may only attend to keys j <= i
            if CAUSAL:
                causal_ok = offs_m[:, None] >= offs_n[None, :]
                scores = tl.where(causal_ok, scores, float("-inf"))

            # Additive attention mask (e.g. sliding window, dilated)
            if HAS_MASK:
                if MASK_BATCH_STRIDE_ZERO:
                    mb_off = 0
                else:
                    mb_off = batch_idx * stride_mb
                if MASK_HEAD_STRIDE_ZERO:
                    mh_off = 0
                else:
                    mh_off = head_idx * stride_mh

                m_ptrs = (
                    Mask_ptr
                    + mb_off
                    + mh_off
                    + offs_m[:, None] * stride_ms
                    + offs_n[None, :] * stride_mn
                )
                attn_mask = tl.load(
                    m_ptrs,
                    mask=mask_m[:, None] & mask_n[None, :],
                    other=float("-inf"),
                )
                scores = scores + attn_mask.to(tl.float32)

            # Mask out-of-bounds positions
            scores = tl.where(
                mask_m[:, None] & mask_n[None, :],
                scores,
                float("-inf"),
            )

            # Online softmax update
            m_ij = tl.max(scores, axis=1)              # [BLOCK_M]
            m_new = tl.maximum(m_i, m_ij)
            # Guard against -inf - (-inf) = NaN when all scores are masked.
            # If m_new == -inf the block is entirely masked; correction is 0.
            safe_diff = tl.where(m_new == float("-inf"), 0.0, m_i - m_new)
            alpha = tl.exp(safe_diff)                  # [BLOCK_M] correction
            # Guard scores - m_new when m_new == -inf (all-masked block).
            safe_scores = tl.where(m_new[:, None] == float("-inf"), float("-inf"), scores - m_new[:, None])
            p = tl.exp(safe_scores)                     # [BLOCK_M, BLOCK_N]
            l_i = l_i * alpha + tl.sum(p, axis=1)
            acc = acc * alpha[:, None]

            # Load V block [BLOCK_N, D]
            v_ptrs = (
                V_ptr
                + batch_idx * stride_vb
                + kv_head_idx * stride_vh
                + offs_n[:, None] * stride_vs
                + offs_d[None, :] * stride_vd
            )
            v = tl.load(v_ptrs, mask=mask_n[:, None], other=0.0).to(tl.float32)

            acc += tl.dot(p, v, input_precision="ieee")
            m_i = m_new

        # Normalize; guard against division by zero for fully-masked rows.
        acc = tl.where(l_i[:, None] == 0.0, 0.0, acc / l_i[:, None])

        # Store output [BLOCK_M, D]
        out_ptrs = (
            Out_ptr
            + batch_idx * stride_ob
            + head_idx * stride_oh
            + offs_m[:, None] * stride_os
            + offs_d[None, :] * stride_od
        )
        tl.store(out_ptrs, acc.to(Out_ptr.dtype.element_ty), mask=mask_m[:, None])


# ---------------------------------------------------------------------------
# Python wrapper
# ---------------------------------------------------------------------------

def multi_query_attention(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    *,
    causal: bool = False,
    scale: float | None = None,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Fused multi-query attention via Triton online-softmax kernel.

    Parameters
    ----------
    Q : [batch, num_heads, seq, head_dim]
    K : [batch, num_kv_heads, seq, head_dim]  (num_kv_heads divides num_heads)
    V : [batch, num_kv_heads, seq, head_dim]
    causal : apply causal (lower-triangular) mask
    scale : query scale factor; defaults to 1/sqrt(head_dim)
    mask : additive float attention mask, shape broadcastable to [B, H, S, S].
           Masked positions should contain -inf (or a large negative value).
           If bool tensor, -inf is applied where mask is False.

    Returns
    -------
    [batch, num_heads, seq, head_dim] same dtype as Q
    """
    if not HAS_TRITON:
        raise RuntimeError("Triton is not available; cannot call multi_query_attention kernel.")

    B, H, S, D = Q.shape
    _, KVH, _, _ = K.shape

    if H % KVH != 0:
        raise ValueError(f"num_heads ({H}) must be divisible by num_kv_heads ({KVH}).")
    if D not in (16, 32, 64, 128):
        raise ValueError(f"head_dim D={D} not supported (must be 16, 32, 64, or 128).")

    if scale is None:
        scale = 1.0 / math.sqrt(D)

    # Ensure contiguous inputs
    Q = Q.contiguous()
    K = K.contiguous()
    V = V.contiguous()

    out = torch.empty_like(Q)

    # Block sizes tuned for short sequences typical in this model (S=48)
    if S <= 64:
        BLOCK_M = 32
        BLOCK_N = 32
    elif S <= 256:
        BLOCK_M = 64
        BLOCK_N = 64
    else:
        BLOCK_M = 64
        BLOCK_N = 32

    # Handle mask: convert bool to float additive mask
    HAS_MASK = mask is not None
    if HAS_MASK:
        if mask.dtype == torch.bool:
            mask = torch.full(mask.shape, float("-inf"), dtype=torch.float32, device=mask.device).masked_fill_(mask, 0.0)
        mask = mask.contiguous()
        # Ensure mask is at least 4D: [B, H, S, S]
        while mask.dim() < 4:
            mask = mask.unsqueeze(0)
        mask_batch_stride_zero = mask.shape[0] == 1
        mask_head_stride_zero = mask.shape[1] == 1
        stride_mb = mask.stride(0)
        stride_mh = mask.stride(1)
        stride_ms = mask.stride(2)
        stride_mn = mask.stride(3)
    else:
        mask_batch_stride_zero = True
        mask_head_stride_zero = True
        stride_mb = stride_mh = stride_ms = stride_mn = 0
        mask = Q  # dummy pointer; HAS_MASK=False prevents any loads

    num_blocks_m = triton.cdiv(S, BLOCK_M)
    # grid: (query_blocks, batch*heads, num_heads)
    # The 3rd dim carries num_heads so the kernel can recover head_idx via modulo.
    grid = (num_blocks_m, B * H, H)

    _mqa_attention_fwd[grid](
        Q, K, V, mask, out,
        # Q strides
        Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
        # K strides
        K.stride(0), K.stride(1), K.stride(2), K.stride(3),
        # V strides
        V.stride(0), V.stride(1), V.stride(2), V.stride(3),
        # Mask strides
        stride_mb, stride_mh, stride_ms, stride_mn,
        # Out strides
        out.stride(0), out.stride(1), out.stride(2), out.stride(3),
        # Constants
        SEQ_LEN=S,
        D=D,
        SCALE=scale,
        NUM_KV_HEADS=KVH,
        CAUSAL=causal,
        HAS_MASK=HAS_MASK,
        MASK_BATCH_STRIDE_ZERO=mask_batch_stride_zero,
        MASK_HEAD_STRIDE_ZERO=mask_head_stride_zero,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
    )

    return out
