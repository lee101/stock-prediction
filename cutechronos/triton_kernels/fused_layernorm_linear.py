"""Triton fused RMS LayerNorm + Linear kernel.

Fuses RMS LayerNorm followed by a linear projection into a single kernel,
avoiding materializing the normalized intermediate tensor in global memory.

RMS LayerNorm: out = weight * (x * rsqrt(mean(x^2) + eps))  (T5-style, FP32 variance)
Linear: y = x_normed @ W^T  (no bias, matching F.linear convention)

Also provides fused_rms_norm_qkv for producing Q, K, V from one normalized input.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _fused_rms_norm_linear_kernel(
    X_ptr,
    NW_ptr,       # norm weight (N,)
    LW_ptr,       # linear weight (O, N) stored row-major
    Y_ptr,
    stride_x_row,
    stride_y_row,
    stride_lw_row,
    N,             # input hidden dim
    O,             # output dim
    eps,
    BLOCK_N: tl.constexpr,   # >= N, power of 2
    BLOCK_O: tl.constexpr,   # output tile size
):
    """Fused RMS LayerNorm + Linear kernel.

    Grid: (num_rows, cdiv(O, BLOCK_O))
    Each program handles one row of x and one tile of output columns.
    """
    row_idx = tl.program_id(0)
    col_block = tl.program_id(1)

    # Pointers
    x_row_ptr = X_ptr + row_idx * stride_x_row
    y_row_ptr = Y_ptr + row_idx * stride_y_row

    # Load input row
    n_offsets = tl.arange(0, BLOCK_N)
    n_mask = n_offsets < N
    x = tl.load(x_row_ptr + n_offsets, mask=n_mask, other=0.0)

    # Compute RMS norm in FP32
    x_fp32 = x.to(tl.float32)
    sq_sum = tl.sum(x_fp32 * x_fp32, axis=0)
    mean_sq = sq_sum / N
    rrms = tl.rsqrt(mean_sq + eps)

    # Load norm weight and apply normalization
    nw = tl.load(NW_ptr + n_offsets, mask=n_mask, other=0.0)
    # Match reference: cast normalized to input dtype before weight multiply
    x_normed = (x_fp32 * rrms).to(x.dtype) * nw  # shape (BLOCK_N,)

    # Now compute linear projection for this output tile
    o_start = col_block * BLOCK_O
    o_offsets = o_start + tl.arange(0, BLOCK_O)
    o_mask = o_offsets < O

    # For each output column o, compute dot(x_normed, LW[o, :])
    # Load a tile of linear weight: shape (BLOCK_O, BLOCK_N)
    # LW[o, n] is at LW_ptr + o * stride_lw_row + n
    lw_ptrs = LW_ptr + o_offsets[:, None] * stride_lw_row + n_offsets[None, :]
    lw_mask = o_mask[:, None] & n_mask[None, :]
    lw = tl.load(lw_ptrs, mask=lw_mask, other=0.0)

    # Dot product: sum over N dimension
    # x_normed is (BLOCK_N,), lw is (BLOCK_O, BLOCK_N)
    # result is (BLOCK_O,)
    x_normed_fp32 = x_normed.to(tl.float32)
    result = tl.sum(lw.to(tl.float32) * x_normed_fp32[None, :], axis=1)

    # Store output
    tl.store(y_row_ptr + o_offsets, result.to(x.dtype), mask=o_mask)


def fused_rms_norm_linear(
    x: torch.Tensor,
    norm_weight: torch.Tensor,
    linear_weight: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Fused RMS LayerNorm + Linear projection.

    Equivalent to F.linear(rms_layernorm(x, norm_weight, eps), linear_weight)
    but without materializing the intermediate normalized tensor.

    Args:
        x: Input tensor of shape (..., N).
        norm_weight: RMS LayerNorm scale of shape (N,).
        linear_weight: Linear weight of shape (O, N), same convention as F.linear.
        eps: Epsilon for numerical stability.

    Returns:
        Output tensor of shape (..., O).
    """
    orig_shape = x.shape
    N = orig_shape[-1]
    O = linear_weight.shape[0]
    assert linear_weight.shape[1] == N, (
        f"linear_weight shape {linear_weight.shape} incompatible with input dim {N}"
    )

    x_2d = x.reshape(-1, N).contiguous()
    norm_weight = norm_weight.contiguous()
    linear_weight = linear_weight.contiguous()
    num_rows = x_2d.shape[0]

    y = torch.empty(num_rows, O, device=x.device, dtype=x.dtype)

    BLOCK_N = triton.next_power_of_2(N)
    # Choose output tile size: balance occupancy vs register pressure
    BLOCK_O = min(128, triton.next_power_of_2(O))

    grid = (num_rows, triton.cdiv(O, BLOCK_O))

    _fused_rms_norm_linear_kernel[grid](
        x_2d,
        norm_weight,
        linear_weight,
        y,
        x_2d.stride(0),
        y.stride(0),
        linear_weight.stride(0),
        N,
        O,
        eps,
        BLOCK_N=BLOCK_N,
        BLOCK_O=BLOCK_O,
    )

    out_shape = orig_shape[:-1] + (O,)
    return y.reshape(out_shape)


@triton.jit
def _fused_rms_norm_qkv_kernel(
    X_ptr,
    NW_ptr,        # norm weight (N,)
    WQ_ptr,        # Q weight (D, N)
    WK_ptr,        # K weight (D, N)
    WV_ptr,        # V weight (D, N)
    YQ_ptr,
    YK_ptr,
    YV_ptr,
    stride_x_row,
    stride_yq_row,
    stride_yk_row,
    stride_yv_row,
    stride_wq_row,
    stride_wk_row,
    stride_wv_row,
    N,              # input hidden dim
    D,              # output dim per head (inner_dim = num_heads * d_kv)
    eps,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """Fused RMS LayerNorm + QKV projection kernel.

    Grid: (num_rows, cdiv(D, BLOCK_D))
    Each program handles one row and one tile of the output dimension.
    Normalizes once, then projects into Q, K, V.
    """
    row_idx = tl.program_id(0)
    d_block = tl.program_id(1)

    x_row_ptr = X_ptr + row_idx * stride_x_row

    # Load input row
    n_offsets = tl.arange(0, BLOCK_N)
    n_mask = n_offsets < N
    x = tl.load(x_row_ptr + n_offsets, mask=n_mask, other=0.0)

    # RMS norm in FP32
    x_fp32 = x.to(tl.float32)
    sq_sum = tl.sum(x_fp32 * x_fp32, axis=0)
    mean_sq = sq_sum / N
    rrms = tl.rsqrt(mean_sq + eps)

    nw = tl.load(NW_ptr + n_offsets, mask=n_mask, other=0.0)
    x_normed = (x_fp32 * rrms).to(x.dtype) * nw
    x_normed_fp32 = x_normed.to(tl.float32)

    # Output tile -- shared offsets and mask for all three projections
    d_start = d_block * BLOCK_D
    d_offsets = d_start + tl.arange(0, BLOCK_D)
    d_mask = d_offsets < D
    w_mask = d_mask[:, None] & n_mask[None, :]

    # Q projection
    wq_ptrs = WQ_ptr + d_offsets[:, None] * stride_wq_row + n_offsets[None, :]
    wq = tl.load(wq_ptrs, mask=w_mask, other=0.0)
    q_result = tl.sum(wq.to(tl.float32) * x_normed_fp32[None, :], axis=1)
    tl.store(YQ_ptr + row_idx * stride_yq_row + d_offsets, q_result.to(x.dtype), mask=d_mask)

    # K projection
    wk_ptrs = WK_ptr + d_offsets[:, None] * stride_wk_row + n_offsets[None, :]
    wk = tl.load(wk_ptrs, mask=w_mask, other=0.0)
    k_result = tl.sum(wk.to(tl.float32) * x_normed_fp32[None, :], axis=1)
    tl.store(YK_ptr + row_idx * stride_yk_row + d_offsets, k_result.to(x.dtype), mask=d_mask)

    # V projection
    wv_ptrs = WV_ptr + d_offsets[:, None] * stride_wv_row + n_offsets[None, :]
    wv = tl.load(wv_ptrs, mask=w_mask, other=0.0)
    v_result = tl.sum(wv.to(tl.float32) * x_normed_fp32[None, :], axis=1)
    tl.store(YV_ptr + row_idx * stride_yv_row + d_offsets, v_result.to(x.dtype), mask=d_mask)


def fused_rms_norm_qkv(
    x: torch.Tensor,
    norm_weight: torch.Tensor,
    wq: torch.Tensor,
    wk: torch.Tensor,
    wv: torch.Tensor,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fused RMS LayerNorm + Q/K/V projection.

    Normalizes x once and projects into Q, K, V simultaneously, avoiding
    three redundant reads of the normalized intermediate.

    Equivalent to:
        normed = rms_layernorm(x, norm_weight, eps)
        q = F.linear(normed, wq)
        k = F.linear(normed, wk)
        v = F.linear(normed, wv)

    Args:
        x: Input tensor of shape (..., N).
        norm_weight: RMS LayerNorm scale of shape (N,).
        wq: Q projection weight of shape (D, N).
        wk: K projection weight of shape (D, N).
        wv: V projection weight of shape (D, N).
        eps: Epsilon for numerical stability.

    Returns:
        Tuple (Q, K, V), each of shape (..., D).
    """
    orig_shape = x.shape
    N = orig_shape[-1]
    D = wq.shape[0]
    assert wq.shape == (D, N), f"wq shape {wq.shape} != ({D}, {N})"
    assert wk.shape == (D, N), f"wk shape {wk.shape} != ({D}, {N})"
    assert wv.shape == (D, N), f"wv shape {wv.shape} != ({D}, {N})"

    x_2d = x.reshape(-1, N).contiguous()
    norm_weight = norm_weight.contiguous()
    wq = wq.contiguous()
    wk = wk.contiguous()
    wv = wv.contiguous()
    num_rows = x_2d.shape[0]

    yq = torch.empty(num_rows, D, device=x.device, dtype=x.dtype)
    yk = torch.empty(num_rows, D, device=x.device, dtype=x.dtype)
    yv = torch.empty(num_rows, D, device=x.device, dtype=x.dtype)

    BLOCK_N = triton.next_power_of_2(N)
    BLOCK_D = min(128, triton.next_power_of_2(D))

    grid = (num_rows, triton.cdiv(D, BLOCK_D))

    _fused_rms_norm_qkv_kernel[grid](
        x_2d,
        norm_weight,
        wq,
        wk,
        wv,
        yq,
        yk,
        yv,
        x_2d.stride(0),
        yq.stride(0),
        yk.stride(0),
        yv.stride(0),
        wq.stride(0),
        wk.stride(0),
        wv.stride(0),
        N,
        D,
        eps,
        BLOCK_N=BLOCK_N,
        BLOCK_D=BLOCK_D,
    )

    out_shape = orig_shape[:-1] + (D,)
    return yq.reshape(out_shape), yk.reshape(out_shape), yv.reshape(out_shape)
