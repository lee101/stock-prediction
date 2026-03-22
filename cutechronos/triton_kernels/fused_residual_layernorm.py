"""Triton kernel fusing residual-add + RMS LayerNorm.

In each encoder sub-layer the pattern is:

    hidden_states = hidden_states + delta      # residual add
    normed = rms_layernorm(hidden_states, w, eps)   # reads hidden_states again

This kernel fuses both operations into a single pass:
1. Compute x = residual + delta in registers
2. Compute norm(x) without storing x to global memory first
3. Write both x (for next residual) and norm(x) (for next projection)

This eliminates one full read-write of the hidden state per sub-layer.

RMS LayerNorm (T5-style): weight * (x * rsqrt(mean(x^2) + eps))
Variance computed in FP32 for numerical stability.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _fused_residual_rms_norm_kernel(
    Residual_ptr,
    Delta_ptr,
    Sum_ptr,
    Normed_ptr,
    W_ptr,
    stride_residual_row,
    stride_delta_row,
    stride_sum_row,
    stride_normed_row,
    N,
    eps,
    INPLACE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Fused residual-add + RMS LayerNorm forward kernel. One program per row.

    Each program:
    1. Loads a row of residual and delta (BLOCK_SIZE elements, masked for N)
    2. Computes x = residual + delta in registers
    3. Computes sum-of-squares of x in FP32
    4. Computes rsqrt(mean_sq + eps)
    5. Multiplies by weight, stores both x and normed in original dtype

    When INPLACE=True, the sum is written back to the residual buffer
    and Sum_ptr / stride_sum_row are unused.
    """
    row_idx = tl.program_id(0)

    residual_row_ptr = Residual_ptr + row_idx * stride_residual_row
    delta_row_ptr = Delta_ptr + row_idx * stride_delta_row
    normed_row_ptr = Normed_ptr + row_idx * stride_normed_row

    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < N

    # Load residual and delta
    residual = tl.load(residual_row_ptr + col_offsets, mask=mask, other=0.0)
    delta = tl.load(delta_row_ptr + col_offsets, mask=mask, other=0.0)
    w = tl.load(W_ptr + col_offsets, mask=mask, other=0.0)

    # Fused residual add — stays in registers
    x = residual + delta

    # Store the sum: in-place writes back to residual, otherwise to Sum_ptr
    if INPLACE:
        tl.store(residual_row_ptr + col_offsets, x, mask=mask)
    else:
        sum_row_ptr = Sum_ptr + row_idx * stride_sum_row
        tl.store(sum_row_ptr + col_offsets, x, mask=mask)

    # Compute RMS norm in FP32 for numerical stability
    x_fp32 = x.to(tl.float32)
    sq_sum = tl.sum(x_fp32 * x_fp32, axis=0)
    mean_sq = sq_sum / N
    rrms = tl.rsqrt(mean_sq + eps)

    # Normalize and scale: cast to output dtype before weight multiply
    # to match Chronos2LayerNorm behavior (bf16 * bf16 for weight multiply)
    normed = (x_fp32 * rrms).to(x.dtype) * w
    tl.store(normed_row_ptr + col_offsets, normed, mask=mask)


def fused_residual_layernorm(
    residual: torch.Tensor,
    delta: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fused residual-add + RMS LayerNorm.

    Computes:
        x = residual + delta
        normed = rms_layernorm(x, weight, eps)

    in a single Triton kernel pass.

    Args:
        residual: Tensor of shape (..., N) — the residual stream.
        delta: Tensor of shape (..., N) — the sub-layer output to add.
        weight: Scale parameter of shape (N,).
        eps: Small constant for numerical stability.

    Returns:
        Tuple of (sum, normed) where:
        - sum has the same shape/dtype as residual (residual + delta)
        - normed has the same shape/dtype as residual (RMS-normed sum)
    """
    orig_shape = residual.shape
    N = orig_shape[-1]

    residual_2d = residual.reshape(-1, N).contiguous()
    delta_2d = delta.reshape(-1, N).contiguous()
    weight = weight.contiguous()
    num_rows = residual_2d.shape[0]

    sum_out = torch.empty_like(residual_2d)
    normed_out = torch.empty_like(residual_2d)

    BLOCK_SIZE = triton.next_power_of_2(N)

    _fused_residual_rms_norm_kernel[(num_rows,)](
        residual_2d,
        delta_2d,
        sum_out,
        normed_out,
        weight,
        residual_2d.stride(0),
        delta_2d.stride(0),
        sum_out.stride(0),
        normed_out.stride(0),
        N,
        eps,
        INPLACE=False,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    return sum_out.reshape(orig_shape), normed_out.reshape(orig_shape)


def fused_residual_layernorm_inplace(
    residual: torch.Tensor,
    delta: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Fused residual-add (in-place) + RMS LayerNorm.

    Computes:
        residual += delta                         (written back to residual)
        normed = rms_layernorm(residual, weight, eps)

    in a single Triton kernel pass.

    Args:
        residual: Tensor of shape (..., N) — modified IN-PLACE with residual + delta.
            Must be contiguous.
        delta: Tensor of shape (..., N) — the sub-layer output to add.
        weight: Scale parameter of shape (N,).
        eps: Small constant for numerical stability.

    Returns:
        normed: Tensor of same shape/dtype as residual (RMS-normed updated residual).
        Note: residual is also modified in-place to contain residual + delta.
    """
    orig_shape = residual.shape
    N = orig_shape[-1]

    # For in-place, residual must be contiguous so we can write back
    assert residual.is_contiguous(), "residual must be contiguous for in-place variant"

    residual_2d = residual.reshape(-1, N)
    delta_2d = delta.reshape(-1, N).contiguous()
    weight = weight.contiguous()
    num_rows = residual_2d.shape[0]

    normed_out = torch.empty_like(residual_2d)

    BLOCK_SIZE = triton.next_power_of_2(N)

    # Sum_ptr and stride_sum_row are unused when INPLACE=True but Triton
    # requires positional args to match the kernel signature.
    _fused_residual_rms_norm_kernel[(num_rows,)](
        residual_2d,
        delta_2d,
        residual_2d,  # unused placeholder (INPLACE writes to Residual_ptr)
        normed_out,
        weight,
        residual_2d.stride(0),
        delta_2d.stride(0),
        residual_2d.stride(0),  # unused placeholder
        normed_out.stride(0),
        N,
        eps,
        INPLACE=True,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    return normed_out.reshape(orig_shape)
