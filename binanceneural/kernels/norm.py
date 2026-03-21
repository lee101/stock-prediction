"""Triton RMS norm and fused RMS norm + projection kernels for binanceneural.

Three public functions:

  rms_norm(x, weight=None, eps=1e-6)
      T5-style RMS norm. weight is optional (None = no learnable scale,
      matching the unweighted _rms_norm() used throughout model.py).

  fused_rms_norm_linear(x, norm_weight, W, bias=None, eps=1e-6)
      Fused RMS norm + single linear projection. norm_weight may be None.

  fused_rms_norm_qkv(x, norm_weight, W_q, W_k, W_v, bias_q=None,
                     bias_k=None, bias_v=None, eps=1e-6)
      One RMS norm pass feeding three linear projections (Q, K, V).
      norm_weight may be None (unscaled norm, as used in model.py blocks).

All kernels use FP32 accumulation for variance and dot products; outputs
are cast back to the input dtype (BF16 or FP32).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

try:
    import triton
    import triton.language as tl

    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False


# ---------------------------------------------------------------------------
# Triton kernels
# ---------------------------------------------------------------------------

if HAS_TRITON:
    @triton.jit
    def _rms_norm_kernel(
        X_ptr,
        W_ptr,          # may be None-equivalent via HAS_WEIGHT constexpr
        Y_ptr,
        stride_x,
        stride_y,
        N,
        eps,
        HAS_WEIGHT: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        """RMS norm kernel, one program per row."""
        row = tl.program_id(0)
        offs = tl.arange(0, BLOCK_N)
        mask = offs < N

        x = tl.load(X_ptr + row * stride_x + offs, mask=mask, other=0.0)
        x_fp32 = x.to(tl.float32)
        mean_sq = tl.sum(x_fp32 * x_fp32, axis=0) / N
        rrms = tl.rsqrt(mean_sq + eps)
        out = (x_fp32 * rrms).to(x.dtype)

        if HAS_WEIGHT:
            w = tl.load(W_ptr + offs, mask=mask, other=1.0)
            out = out * w

        tl.store(Y_ptr + row * stride_y + offs, out, mask=mask)

    @triton.jit
    def _fused_rms_norm_qkv_kernel(
        X_ptr,
        NW_ptr,          # norm weight; loaded only if HAS_NORM_WEIGHT
        WQ_ptr,
        WK_ptr,
        WV_ptr,
        BQ_ptr,          # bias; loaded only if HAS_BIAS
        BK_ptr,
        BV_ptr,
        YQ_ptr,
        YK_ptr,
        YV_ptr,
        stride_x,
        stride_yq,
        stride_yk,
        stride_yv,
        stride_wq,
        stride_wk,
        stride_wv,
        N,
        Dq,              # output dim for Q
        Dk,              # output dim for K
        Dv,              # output dim for V
        eps,
        HAS_NORM_WEIGHT: tl.constexpr,
        HAS_BIAS: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        """Fused RMS norm + QKV projection.

        Grid: (num_rows, max(cdiv(Dq,BLOCK_D), cdiv(Dk,BLOCK_D), cdiv(Dv,BLOCK_D)))
        """
        row = tl.program_id(0)
        d_block = tl.program_id(1)

        # Load and normalize input row
        n_offs = tl.arange(0, BLOCK_N)
        n_mask = n_offs < N
        x = tl.load(X_ptr + row * stride_x + n_offs, mask=n_mask, other=0.0)
        x_fp32 = x.to(tl.float32)
        mean_sq = tl.sum(x_fp32 * x_fp32, axis=0) / N
        rrms = tl.rsqrt(mean_sq + eps)
        x_normed = (x_fp32 * rrms).to(x.dtype)

        if HAS_NORM_WEIGHT:
            nw = tl.load(NW_ptr + n_offs, mask=n_mask, other=1.0)
            x_normed = x_normed * nw

        x_normed_fp32 = x_normed.to(tl.float32)

        d_start = d_block * BLOCK_D
        d_offs = d_start + tl.arange(0, BLOCK_D)

        # --- Q projection ---
        q_mask = d_offs < Dq
        w_mask_q = q_mask[:, None] & n_mask[None, :]
        wq = tl.load(WQ_ptr + d_offs[:, None] * stride_wq + n_offs[None, :], mask=w_mask_q, other=0.0)
        q_res = tl.sum(wq.to(tl.float32) * x_normed_fp32[None, :], axis=1)
        if HAS_BIAS:
            bq = tl.load(BQ_ptr + d_offs, mask=q_mask, other=0.0).to(tl.float32)
            q_res = q_res + bq
        tl.store(YQ_ptr + row * stride_yq + d_offs, q_res.to(x.dtype), mask=q_mask)

        # --- K projection ---
        k_mask = d_offs < Dk
        w_mask_k = k_mask[:, None] & n_mask[None, :]
        wk = tl.load(WK_ptr + d_offs[:, None] * stride_wk + n_offs[None, :], mask=w_mask_k, other=0.0)
        k_res = tl.sum(wk.to(tl.float32) * x_normed_fp32[None, :], axis=1)
        if HAS_BIAS:
            bk = tl.load(BK_ptr + d_offs, mask=k_mask, other=0.0).to(tl.float32)
            k_res = k_res + bk
        tl.store(YK_ptr + row * stride_yk + d_offs, k_res.to(x.dtype), mask=k_mask)

        # --- V projection ---
        v_mask = d_offs < Dv
        w_mask_v = v_mask[:, None] & n_mask[None, :]
        wv = tl.load(WV_ptr + d_offs[:, None] * stride_wv + n_offs[None, :], mask=w_mask_v, other=0.0)
        v_res = tl.sum(wv.to(tl.float32) * x_normed_fp32[None, :], axis=1)
        if HAS_BIAS:
            bv = tl.load(BV_ptr + d_offs, mask=v_mask, other=0.0).to(tl.float32)
            v_res = v_res + bv
        tl.store(YV_ptr + row * stride_yv + d_offs, v_res.to(x.dtype), mask=v_mask)


# ---------------------------------------------------------------------------
# PyTorch reference fallbacks
# ---------------------------------------------------------------------------

def _rms_norm_ref(x: torch.Tensor, weight: torch.Tensor | None, eps: float) -> torch.Tensor:
    if hasattr(F, "rms_norm"):
        return F.rms_norm(x, (x.size(-1),), weight=weight, eps=eps)
    variance = x.float().pow(2).mean(dim=-1, keepdim=True)
    normed = (x * torch.rsqrt(variance + eps)).to(x.dtype)
    if weight is not None:
        normed = normed * weight
    return normed


# ---------------------------------------------------------------------------
# Public functions
# ---------------------------------------------------------------------------

def rms_norm(
    x: torch.Tensor,
    weight: torch.Tensor | None = None,
    eps: float = 1e-6,
) -> torch.Tensor:
    """T5-style RMS norm with optional learnable weight.

    Args:
        x: Input tensor (..., N).
        weight: Optional scale parameter (N,). None = unweighted (as in model.py).
        eps: Epsilon for numerical stability.

    Returns:
        Normalized tensor of same shape and dtype as x.
    """
    if not HAS_TRITON or not x.is_cuda:
        return _rms_norm_ref(x, weight, eps)

    orig_shape = x.shape
    N = orig_shape[-1]

    x_2d = x.reshape(-1, N).contiguous()
    num_rows = x_2d.shape[0]
    y = torch.empty_like(x_2d)

    BLOCK_N = triton.next_power_of_2(N)
    has_weight = weight is not None
    w_ptr = weight.contiguous() if has_weight else x_2d  # placeholder pointer

    _rms_norm_kernel[(num_rows,)](
        x_2d,
        w_ptr,
        y,
        x_2d.stride(0),
        y.stride(0),
        N,
        eps,
        HAS_WEIGHT=has_weight,
        BLOCK_N=BLOCK_N,
    )
    return y.reshape(orig_shape)


def fused_rms_norm_linear(
    x: torch.Tensor,
    norm_weight: torch.Tensor | None,
    W: torch.Tensor,
    bias: torch.Tensor | None = None,
    eps: float = 1e-6,
) -> torch.Tensor:
    """RMS norm followed by a single linear projection.

    Equivalent to F.linear(rms_norm(x, norm_weight, eps), W, bias).
    For the single-projection case the intermediate is materialized; the
    high-value fused path is fused_rms_norm_qkv (three projections, one norm).

    Args:
        x: Input (..., N).
        norm_weight: Optional RMS norm scale (N,). None = unweighted.
        W: Linear weight (O, N).
        bias: Optional bias (O,).
        eps: Epsilon.

    Returns:
        Output (..., O) in input dtype.
    """
    normed = rms_norm(x, norm_weight, eps)
    return F.linear(normed, W, bias)


def fused_rms_norm_qkv(
    x: torch.Tensor,
    norm_weight: torch.Tensor | None,
    W_q: torch.Tensor,
    W_k: torch.Tensor,
    W_v: torch.Tensor,
    bias_q: torch.Tensor | None = None,
    bias_k: torch.Tensor | None = None,
    bias_v: torch.Tensor | None = None,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fused RMS norm + Q/K/V projections in one kernel pass.

    Normalizes x once, then simultaneously computes Q = x_n @ W_q^T,
    K = x_n @ W_k^T, V = x_n @ W_v^T, avoiding three redundant reads
    of the normalized intermediate tensor.

    Args:
        x: Input (..., N).
        norm_weight: Optional scale (N,). None = unweighted (matching model.py).
        W_q: Q weight (Dq, N).
        W_k: K weight (Dk, N).
        W_v: V weight (Dv, N).
        bias_q/k/v: Optional biases.
        eps: Epsilon.

    Returns:
        (Q, K, V) each of shape (..., D).
    """
    orig_shape = x.shape
    N = orig_shape[-1]
    Dq = W_q.shape[0]
    Dk = W_k.shape[0]
    Dv = W_v.shape[0]

    if not HAS_TRITON or not x.is_cuda:
        normed = _rms_norm_ref(x, norm_weight, eps)
        return (
            F.linear(normed, W_q, bias_q),
            F.linear(normed, W_k, bias_k),
            F.linear(normed, W_v, bias_v),
        )

    x_2d = x.reshape(-1, N).contiguous()
    num_rows = x_2d.shape[0]

    nw = norm_weight.contiguous() if norm_weight is not None else x_2d  # placeholder
    wq = W_q.contiguous()
    wk = W_k.contiguous()
    wv = W_v.contiguous()

    yq = torch.empty(num_rows, Dq, device=x.device, dtype=x.dtype)
    yk = torch.empty(num_rows, Dk, device=x.device, dtype=x.dtype)
    yv = torch.empty(num_rows, Dv, device=x.device, dtype=x.dtype)

    has_bias = bias_q is not None
    bq = bias_q.contiguous() if has_bias else yq  # placeholder
    bk = bias_k.contiguous() if has_bias else yk
    bv = bias_v.contiguous() if has_bias else yv

    BLOCK_N = triton.next_power_of_2(N)
    D_max = max(Dq, Dk, Dv)
    BLOCK_D = min(128, triton.next_power_of_2(D_max))

    grid = (num_rows, triton.cdiv(D_max, BLOCK_D))

    _fused_rms_norm_qkv_kernel[grid](
        x_2d,
        nw,
        wq, wk, wv,
        bq, bk, bv,
        yq, yk, yv,
        x_2d.stride(0),
        yq.stride(0),
        yk.stride(0),
        yv.stride(0),
        wq.stride(0),
        wk.stride(0),
        wv.stride(0),
        N,
        Dq, Dk, Dv,
        eps,
        HAS_NORM_WEIGHT=(norm_weight is not None),
        HAS_BIAS=has_bias,
        BLOCK_N=BLOCK_N,
        BLOCK_D=BLOCK_D,
    )

    out_q = yq.reshape(orig_shape[:-1] + (Dq,))
    out_k = yk.reshape(orig_shape[:-1] + (Dk,))
    out_v = yv.reshape(orig_shape[:-1] + (Dv,))
    return out_q, out_k, out_v


__all__ = [
    "HAS_TRITON",
    "rms_norm",
    "fused_rms_norm_linear",
    "fused_rms_norm_qkv",
]
