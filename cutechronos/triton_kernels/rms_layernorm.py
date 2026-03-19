"""Triton RMS LayerNorm kernel for Chronos2 T5-style normalization.

T5 RMS LayerNorm: out = weight * (x * rsqrt(mean(x^2) + eps))
No mean subtraction, no bias. Variance computed in FP32 for numerical stability.
"""

import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def _rms_norm_fwd_kernel(
    X_ptr,
    W_ptr,
    Y_ptr,
    stride_x_row,
    stride_y_row,
    N,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    """RMS LayerNorm forward kernel. One program per row.

    Each program:
    1. Loads a row of X (BLOCK_SIZE elements, masked for N)
    2. Computes sum-of-squares in FP32
    3. Computes rsqrt(mean_sq + eps)
    4. Multiplies by weight, stores in original dtype
    """
    row_idx = tl.program_id(0)

    x_row_ptr = X_ptr + row_idx * stride_x_row
    y_row_ptr = Y_ptr + row_idx * stride_y_row

    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < N

    # Load row and weight
    x = tl.load(x_row_ptr + col_offsets, mask=mask, other=0.0)
    w = tl.load(W_ptr + col_offsets, mask=mask, other=0.0)

    # Compute variance in FP32 for numerical stability
    x_fp32 = x.to(tl.float32)
    sq_sum = tl.sum(x_fp32 * x_fp32, axis=0)
    mean_sq = sq_sum / N
    rrms = tl.rsqrt(mean_sq + eps)

    # Normalize and scale: cast to output dtype before weight multiply
    # to match Chronos2LayerNorm behavior (bf16 * bf16 for weight multiply)
    out = (x_fp32 * rrms).to(x.dtype) * w
    tl.store(y_row_ptr + col_offsets, out, mask=mask)


def _rms_norm_fwd(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """Launch the Triton RMS LayerNorm forward kernel."""
    orig_shape = x.shape
    N = orig_shape[-1]

    x_2d = x.reshape(-1, N).contiguous()
    weight = weight.contiguous()
    num_rows = x_2d.shape[0]

    y = torch.empty_like(x_2d)

    BLOCK_SIZE = triton.next_power_of_2(N)

    _rms_norm_fwd_kernel[(num_rows,)](
        x_2d,
        weight,
        y,
        x_2d.stride(0),
        y.stride(0),
        N,
        eps,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    return y.reshape(orig_shape)


class _RMSNormAutograd(torch.autograd.Function):
    """Autograd wrapper: Triton forward, PyTorch backward."""

    @staticmethod
    def forward(ctx, x, weight, eps):
        ctx.save_for_backward(x, weight)
        ctx.eps = eps
        return _rms_norm_fwd(x, weight, eps)

    @staticmethod
    def backward(ctx, grad_output):
        x, weight = ctx.saved_tensors
        eps = ctx.eps

        # Recompute normalization factor (cheaper than saving from forward)
        x_fp32 = x.float()
        rrms = torch.rsqrt(x_fp32.pow(2).mean(-1, keepdim=True) + eps)
        normed = x_fp32 * rrms

        # grad w.r.t. weight: sum over all rows of (grad_output * normed)
        grad_weight = (grad_output.float() * normed).reshape(-1, x.shape[-1]).sum(0)

        # grad w.r.t. x (standard RMSNorm backward)
        d_normed = grad_output.float() * weight.float()
        grad_x = rrms * (d_normed - normed * (normed * d_normed).mean(-1, keepdim=True))

        return grad_x.to(x.dtype), grad_weight.to(weight.dtype), None


def rms_layernorm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Apply T5-style RMS LayerNorm using a Triton kernel.

    Args:
        x: Input tensor of shape (..., N) where N is the hidden dimension.
        weight: Scale parameter of shape (N,).
        eps: Small constant for numerical stability.

    Returns:
        Normalized tensor of same shape and dtype as x.
    """
    return _RMSNormAutograd.apply(x, weight, eps)


class TritonRMSNorm(nn.Module):
    """Drop-in replacement for Chronos2LayerNorm using Triton kernel.

    T5-style RMS LayerNorm: no mean subtraction, no bias.
    Variance computed in FP32 for numerical stability.
    """

    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return rms_layernorm(hidden_states, self.weight, self.variance_epsilon)
