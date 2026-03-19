"""Triton kernel fusing rearrange + sinh + unscale for Chronos2 output head.

After the output ResidualBlock produces a tensor of shape (B, P_out, Q*P),
the original code performs three steps:

    1. rearrange "b n (q p) -> b q (n p)"  (pure index permutation)
    2. optionally apply sinh (inverse of arcsinh instance normalisation)
    3. unscale: x * scale + loc

This kernel fuses all three into a single pass, avoiding extra memory reads
and writes for each intermediate result.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _fused_output_transform_kernel(
    X_ptr,
    Out_ptr,
    Loc_ptr,
    Scale_ptr,
    # strides for X: (B, N, Q*P)
    stride_xb,
    stride_xn,
    stride_xqp,
    # strides for Out: (B, Q, N*P)
    stride_ob,
    stride_oq,
    stride_onp,
    # dimensions
    N,  # num_output_patches
    Q: tl.constexpr,  # num_quantiles
    P: tl.constexpr,  # patch_size
    USE_SINH: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Fused rearrange + sinh + unscale kernel.

    Each program handles one batch element.  It iterates over all (Q * N * P)
    output elements, computing the source index from the rearrange mapping:

        out[b, q, n*P + p] = transform(x[b, n, q*P + p])

    where transform(v) = v * scale[b] + loc[b]   (optionally with sinh first).
    """
    pid_b = tl.program_id(0)

    # Load loc and scale for this batch element
    loc = tl.load(Loc_ptr + pid_b).to(tl.float32)
    scale = tl.load(Scale_ptr + pid_b).to(tl.float32)

    total_elems = Q * N * P

    for offset in range(0, total_elems, BLOCK_SIZE):
        idx = offset + tl.arange(0, BLOCK_SIZE)
        mask = idx < total_elems

        # Decompose linear index into (q, n, p) for the output tensor
        # Output is (B, Q, N*P) laid out as q * (N*P) + n * P + p
        q = idx // (N * P)
        remainder = idx % (N * P)
        n = remainder // P
        p = remainder % P

        # Source index in X: x[b, n, q*P + p]
        x_offset = pid_b * stride_xb + n * stride_xn + (q * P + p) * stride_xqp
        val = tl.load(X_ptr + x_offset, mask=mask, other=0.0).to(tl.float32)

        # Apply sinh if using arcsinh normalisation
        if USE_SINH:
            val = tl.extra.cuda.libdevice.sinh(val)

        # Unscale
        val = val * scale + loc

        # Destination index in Out: out[b, q, n*P + p]
        o_offset = pid_b * stride_ob + q * stride_oq + (n * P + p) * stride_onp
        tl.store(Out_ptr + o_offset, val, mask=mask)


def fused_output_transform(
    x: torch.Tensor,
    loc: torch.Tensor,
    scale: torch.Tensor,
    num_quantiles: int = 21,
    patch_size: int = 16,
    use_arcsinh: bool = True,
) -> torch.Tensor:
    """Fused rearrange + sinh + unscale for Chronos2 output transform.

    Replaces the sequence:
        x = rearrange(x, "b n (q p) -> b q (n p)", q=Q, p=P)
        if use_arcsinh: x = sinh(x)
        x = x * scale.unsqueeze(1) + loc.unsqueeze(1)

    Parameters
    ----------
    x : torch.Tensor
        Output of the ResidualBlock, shape (B, N, Q*P) where
        N = num_output_patches, Q = num_quantiles, P = patch_size.
    loc : torch.Tensor
        Location parameter, shape (B, 1).
    scale : torch.Tensor
        Scale parameter, shape (B, 1).
    num_quantiles : int
        Number of quantile levels (default 21).
    patch_size : int
        Patch size (default 16).
    use_arcsinh : bool
        Whether to apply sinh (inverse of arcsinh normalisation).

    Returns
    -------
    torch.Tensor
        Quantile predictions of shape (B, Q, N*P).
    """
    B, N, QP = x.shape
    Q = num_quantiles
    P = patch_size
    assert QP == Q * P, (
        f"Last dim of x ({QP}) must equal num_quantiles * patch_size ({Q * P})"
    )

    # Flatten loc/scale to (B,) for simpler indexing
    loc_flat = loc.reshape(B).contiguous().float()
    scale_flat = scale.reshape(B).contiguous().float()

    x = x.contiguous()
    out = torch.empty(B, Q, N * P, device=x.device, dtype=x.dtype)

    BLOCK_SIZE = min(1024, triton.next_power_of_2(Q * N * P))

    grid = (B,)
    _fused_output_transform_kernel[grid](
        x,
        out,
        loc_flat,
        scale_flat,
        x.stride(0),
        x.stride(1),
        x.stride(2),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        N,
        Q=Q,
        P=P,
        USE_SINH=use_arcsinh,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    return out
