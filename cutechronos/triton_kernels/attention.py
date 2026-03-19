"""Triton kernel for unscaled multi-head attention (scale=1.0).

Chronos2 uses attention WITHOUT the standard 1/sqrt(d_k) scaling factor.
Standard FlashAttention implementations always apply this scaling internally,
so we need a custom kernel that computes:

    output = softmax(Q @ K^T + mask) @ V

with scale=1.0 (no division by sqrt(d_k)).

The kernel uses the online softmax algorithm (FlashAttention-style tiling)
to avoid materializing the full S x S attention matrix, keeping memory O(S)
instead of O(S^2).

Softmax is always computed in FP32 for numerical stability regardless of
input dtype.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _unscaled_attention_fwd(
    Q_ptr,
    K_ptr,
    V_ptr,
    Mask_ptr,
    Out_ptr,
    stride_qb,
    stride_qh,
    stride_qs,
    stride_qd,
    stride_kb,
    stride_kh,
    stride_ks,
    stride_kd,
    stride_vb,
    stride_vh,
    stride_vs,
    stride_vd,
    stride_mb,
    stride_mh,
    stride_ms,
    stride_md,
    stride_ob,
    stride_oh,
    stride_os,
    stride_od,
    SEQ_LEN: tl.constexpr,
    D: tl.constexpr,
    HAS_MASK: tl.constexpr,
    MASK_BATCH_STRIDE_ZERO: tl.constexpr,
    MASK_HEAD_STRIDE_ZERO: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """Forward pass of unscaled tiled attention.

    Each program instance handles one (batch, head, block_of_query_rows) tile.
    It iterates over all K/V blocks, maintaining a running online softmax.
    """
    pid_m = tl.program_id(0)  # which query block
    pid_bh = tl.program_id(1)  # flattened (batch, head)

    # Compute batch and head indices from the flattened pid
    # num_programs(2) holds num_heads (H), used for modular arithmetic
    batch_idx = pid_bh // tl.num_programs(2)
    head_idx = pid_bh % tl.num_programs(2)

    # Offsets for this query block
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)  # [BLOCK_M]
    offs_d = tl.arange(0, D)  # [D]

    # Pointers to Q for this (batch, head, query_block)
    q_ptrs = (
        Q_ptr
        + batch_idx * stride_qb
        + head_idx * stride_qh
        + offs_m[:, None] * stride_qs
        + offs_d[None, :] * stride_qd
    )

    # Load Q block [BLOCK_M, D] - mask out-of-bounds rows
    # Upcast to FP32 once here to avoid repeated casts inside the K/V loop.
    mask_m = offs_m < SEQ_LEN
    q = tl.load(q_ptrs, mask=mask_m[:, None], other=0.0).to(tl.float32)

    # Running accumulators for online softmax
    # m_i: running row-wise max of scores (for numerical stability)
    # l_i: running row-wise sum of exp(scores - m_i)
    # acc: running weighted sum of V
    m_i = tl.full([BLOCK_M], value=float("-inf"), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, D], dtype=tl.float32)

    # Iterate over K/V blocks
    num_blocks_n = tl.cdiv(SEQ_LEN, BLOCK_N)
    for block_n in range(num_blocks_n):
        offs_n = block_n * BLOCK_N + tl.arange(0, BLOCK_N)  # [BLOCK_N]
        mask_n = offs_n < SEQ_LEN

        # Load K block [BLOCK_N, D]
        k_ptrs = (
            K_ptr
            + batch_idx * stride_kb
            + head_idx * stride_kh
            + offs_n[:, None] * stride_ks
            + offs_d[None, :] * stride_kd
        )
        k = tl.load(k_ptrs, mask=mask_n[:, None], other=0.0)

        # Compute raw attention scores: Q @ K^T -> [BLOCK_M, BLOCK_N]
        # NO scaling by 1/sqrt(d_k) -- this is the key difference from standard attention.
        # Without scaling, raw scores can reach ~30-35 for D=64 with unit-variance
        # inputs, so we always compute in FP32 with IEEE precision to avoid
        # TF32/BF16 tensor core rounding errors that compound through softmax.
        scores = tl.dot(
            q, tl.trans(k.to(tl.float32)), input_precision="ieee"
        )

        # Apply additive attention mask if present
        if HAS_MASK:
            # Determine mask batch/head offsets, handling broadcasting
            if MASK_BATCH_STRIDE_ZERO:
                mask_b_off = 0
            else:
                mask_b_off = batch_idx * stride_mb

            if MASK_HEAD_STRIDE_ZERO:
                mask_h_off = 0
            else:
                mask_h_off = head_idx * stride_mh

            mask_ptrs = (
                Mask_ptr
                + mask_b_off
                + mask_h_off
                + offs_m[:, None] * stride_ms
                + offs_n[None, :] * stride_md
            )
            attn_mask = tl.load(
                mask_ptrs,
                mask=mask_m[:, None] & mask_n[None, :],
                other=float("-inf"),
            )
            scores = scores + attn_mask.to(tl.float32)

        # Mask out-of-bounds positions with -inf
        scores = tl.where(
            mask_m[:, None] & mask_n[None, :],
            scores,
            float("-inf"),
        )

        # Online softmax update
        # 1. Compute block-wise row max
        m_ij = tl.max(scores, axis=1)  # [BLOCK_M]

        # 2. New running max
        m_new = tl.maximum(m_i, m_ij)  # [BLOCK_M]

        # 3. Correction factor for previously accumulated values
        alpha = tl.exp(m_i - m_new)  # [BLOCK_M]

        # 4. Exponentiate current scores with new max
        p = tl.exp(scores - m_new[:, None])  # [BLOCK_M, BLOCK_N]

        # 5. Update running sum
        l_i = l_i * alpha + tl.sum(p, axis=1)  # [BLOCK_M]

        # 6. Rescale accumulated output and add new contribution
        acc = acc * alpha[:, None]

        # Load V block [BLOCK_N, D]
        v_ptrs = (
            V_ptr
            + batch_idx * stride_vb
            + head_idx * stride_vh
            + offs_n[:, None] * stride_vs
            + offs_d[None, :] * stride_vd
        )
        v = tl.load(v_ptrs, mask=mask_n[:, None], other=0.0)

        # Accumulate p @ V (always in FP32 for numerical stability)
        # p is FP32 from the softmax computation. Upcast V to FP32 and
        # use IEEE precision to avoid TF32 rounding on Blackwell GPUs.
        acc += tl.dot(p, v.to(tl.float32), input_precision="ieee")

        # 7. Update running max
        m_i = m_new

    # Normalize by the softmax denominator
    acc = acc / l_i[:, None]

    # Store output [BLOCK_M, D]
    out_ptrs = (
        Out_ptr
        + batch_idx * stride_ob
        + head_idx * stride_oh
        + offs_m[:, None] * stride_os
        + offs_d[None, :] * stride_od
    )
    tl.store(out_ptrs, acc.to(Out_ptr.dtype.element_ty), mask=mask_m[:, None])


def unscaled_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compute unscaled multi-head attention using a custom Triton kernel.

    Computes: softmax(Q @ K^T + mask) @ V  (no 1/sqrt(d_k) scaling)

    Parameters
    ----------
    q : torch.Tensor
        Query tensor of shape (B, H, S, D).
    k : torch.Tensor
        Key tensor of shape (B, H, S, D).
    v : torch.Tensor
        Value tensor of shape (B, H, S, D).
    mask : torch.Tensor or None
        Additive attention mask, broadcastable to (B, H, S, S).
        Typical shapes: (B, 1, S, S), (1, 1, S, S), or None.
        Masked positions should contain -inf (or large negative).

    Returns
    -------
    torch.Tensor
        Output tensor of shape (B, H, S, D).
    """
    B, H, S, D = q.shape
    assert k.shape == (B, H, S, D), f"K shape mismatch: {k.shape} vs expected {(B, H, S, D)}"
    assert v.shape == (B, H, S, D), f"V shape mismatch: {v.shape} vs expected {(B, H, S, D)}"

    # Ensure inputs are contiguous
    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()

    out = torch.empty_like(q)

    # Choose block sizes based on sequence length and dtype.
    # Shared memory usage scales with BLOCK_M * D + BLOCK_N * D (for Q, K, V tiles).
    # RTX 5090 (SM 120) has 101376 bytes shared memory per SM.
    # FP32 uses 4 bytes/element, BF16 uses 2 bytes/element.
    if S <= 64:
        BLOCK_M = 32
        BLOCK_N = 32
    elif S <= 256:
        BLOCK_M = 64
        BLOCK_N = 64
    else:
        BLOCK_M = 64
        BLOCK_N = 32

    # D must be a power of 2 for tl.trans; pad if needed
    # For Chronos2, D=64 which is already a power of 2
    assert D in (16, 32, 64, 128), f"Head dimension D={D} not supported (must be 16, 32, 64, or 128)"

    # Determine mask broadcasting
    HAS_MASK = mask is not None
    if HAS_MASK:
        mask = mask.contiguous()
        mask_batch_stride_zero = mask.shape[0] == 1
        mask_head_stride_zero = mask.shape[1] == 1
        stride_mb = mask.stride(0)
        stride_mh = mask.stride(1)
        stride_ms = mask.stride(2)
        stride_md = mask.stride(3)
    else:
        mask_batch_stride_zero = True
        mask_head_stride_zero = True
        stride_mb = 0
        stride_mh = 0
        stride_ms = 0
        stride_md = 0
        # Create a dummy pointer (won't be dereferenced)
        mask = q  # just need a valid pointer; HAS_MASK=False prevents loads

    num_blocks_m = triton.cdiv(S, BLOCK_M)
    grid = (num_blocks_m, B * H, H)  # 3rd dim carries num_heads for modular arithmetic

    _unscaled_attention_fwd[grid](
        q,
        k,
        v,
        mask,
        out,
        # Q strides
        q.stride(0),
        q.stride(1),
        q.stride(2),
        q.stride(3),
        # K strides
        k.stride(0),
        k.stride(1),
        k.stride(2),
        k.stride(3),
        # V strides
        v.stride(0),
        v.stride(1),
        v.stride(2),
        v.stride(3),
        # Mask strides
        stride_mb,
        stride_mh,
        stride_ms,
        stride_md,
        # Out strides
        out.stride(0),
        out.stride(1),
        out.stride(2),
        out.stride(3),
        # Constants
        SEQ_LEN=S,
        D=D,
        HAS_MASK=HAS_MASK,
        MASK_BATCH_STRIDE_ZERO=mask_batch_stride_zero,
        MASK_HEAD_STRIDE_ZERO=mask_head_stride_zero,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
    )

    return out
