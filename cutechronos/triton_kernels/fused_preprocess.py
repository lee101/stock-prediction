"""Triton fused preprocessing kernel for Chronos2 time series.

Replaces the multi-step Python preprocessing pipeline:
    NaN mask -> nanmean/nanstd -> normalize -> arcsinh -> unfold -> time_enc -> concat

with two fused Triton kernels:
    Phase 1: Per-series NaN-aware reduction to compute loc (mean) and scale (std).
    Phase 2: Per-element transform: normalize, arcsinh, patch, and write
             [time_enc, normalized_patch, mask_patch] into the output tensor.

All computation is done in FP32 for numerical stability, matching the reference
InstanceNorm + Patch pipeline from cutechronos/model.py.
"""

import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Phase 1: NaN-aware reduction (one program per batch row)
# ---------------------------------------------------------------------------

@triton.jit
def _nan_reduce_kernel(
    X_ptr,
    Loc_ptr,
    Scale_ptr,
    L,
    EPS,
    BLOCK_SIZE: tl.constexpr,
):
    """Compute NaN-aware mean and std for a single row.

    For row b:
        loc[b]   = nanmean(x[b, :])   (0.0 if all NaN)
        scale[b] = sqrt(nanmean((x[b,:] - loc)^2))  (1.0 if all NaN, eps if zero)
    """
    row = tl.program_id(0)

    x_base = X_ptr + row * L
    cols = tl.arange(0, BLOCK_SIZE)
    in_bounds = cols < L

    # Load row -- use NaN for out-of-bounds so they are excluded from valid count
    x = tl.load(x_base + cols, mask=in_bounds, other=float("nan"))

    # Detect NaN: NaN != NaN (also excludes out-of-bounds positions)
    is_valid = (x == x)  # True where not NaN and in-bounds
    valid_f = is_valid.to(tl.float32)

    # Replace NaN with 0 for summation
    x_clean = tl.where(is_valid, x.to(tl.float32), 0.0)

    count = tl.sum(valid_f, axis=0)
    sum_x = tl.sum(x_clean, axis=0)

    # Compute mean
    has_valid = count > 0.0
    count_safe = tl.where(has_valid, count, 1.0)
    loc = tl.where(has_valid, sum_x / count_safe, 0.0)

    # Compute variance: nanmean((x - loc)^2)
    diff = tl.where(is_valid, x_clean - loc, 0.0)
    sum_sq = tl.sum(diff * diff, axis=0)
    variance = sum_sq / count_safe
    scale = tl.sqrt(variance)
    scale = tl.where(has_valid, scale, 1.0)
    scale = tl.where(scale == 0.0, EPS, scale)

    tl.store(Loc_ptr + row, loc)
    tl.store(Scale_ptr + row, scale)


# ---------------------------------------------------------------------------
# Phase 2: Fused normalize + arcsinh + patch + time_enc + concat
# ---------------------------------------------------------------------------

@triton.jit
def _fused_transform_kernel(
    X_ptr,
    Out_ptr,
    Loc_ptr,
    Scale_ptr,
    AttnMask_ptr,
    # Dimensions
    L,               # original (possibly truncated) length
    PAD_LEN,         # number of padding elements prepended (0 or >0)
    L_PADDED,        # L + PAD_LEN (= num_patches * PATCH_SIZE)
    NUM_PATCHES,
    TIME_ENC_SCALE,  # context_length (float), used as divisor for time encoding
    # Strides for output: (B, num_patches, 3*PATCH_SIZE)
    stride_ob,
    stride_op,
    stride_of,
    # Constants
    PATCH_SIZE: tl.constexpr,
    USE_ARCSINH: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Fused transform kernel: one program per (batch, patch) pair.

    For output patch p of batch b, writes:
        out[b, p, 0:PS]       = time_enc values
        out[b, p, PS:2*PS]    = normalized (and optionally arcsinh'd) values
        out[b, p, 2*PS:3*PS]  = mask values (0 or 1)

    Where position in the padded series is: global_pos = p * PATCH_SIZE + local_pos
    Padding positions (global_pos < PAD_LEN) get normalized=0, mask=0.
    NaN positions get normalized=0, mask=0.
    Valid positions get normalized=(x-loc)/scale [optionally arcsinh'd], mask=1.
    """
    pid = tl.program_id(0)
    batch_idx = pid // NUM_PATCHES
    patch_idx = pid % NUM_PATCHES

    loc = tl.load(Loc_ptr + batch_idx).to(tl.float32)
    scale = tl.load(Scale_ptr + batch_idx).to(tl.float32)

    local_pos = tl.arange(0, BLOCK_SIZE)
    pos_mask = local_pos < PATCH_SIZE

    # Global position in the padded series
    global_pos = patch_idx * PATCH_SIZE + local_pos

    # Determine if this position is in the original data (not padding)
    # Padding is prepended: positions [0, PAD_LEN) are padding
    is_data = (global_pos >= PAD_LEN) & pos_mask

    # Load original data at (global_pos - PAD_LEN) offset from batch start
    data_idx = global_pos - PAD_LEN
    x_offset = batch_idx * L + data_idx
    x_val = tl.load(X_ptr + x_offset, mask=is_data, other=0.0).to(tl.float32)

    # Detect NaN in loaded values
    is_valid = (x_val == x_val) & is_data  # not NaN and not padding

    # Normalize
    normalized = tl.where(is_valid, (x_val - loc) / scale, 0.0)

    # Optionally apply arcsinh
    if USE_ARCSINH:
        normalized = tl.where(
            is_valid,
            tl.extra.cuda.libdevice.asinh(normalized),
            0.0,
        )

    # Mask: 1.0 where valid, 0.0 otherwise
    mask_val = tl.where(is_valid, 1.0, 0.0)

    # Time encoding: position relative to end, divided by context_length
    # time_enc[i] = (-(L_PADDED - global_pos)) / time_enc_scale
    #             = (global_pos - L_PADDED) / time_enc_scale
    time_val = (global_pos - L_PADDED).to(tl.float32) / TIME_ENC_SCALE

    # Write to output: out[batch, patch, feature]
    # Layout: [time_enc (PS), normalized (PS), mask (PS)]
    out_base = batch_idx * stride_ob + patch_idx * stride_op

    # Time encoding slice [0 : PATCH_SIZE]
    tl.store(
        Out_ptr + out_base + local_pos * stride_of,
        time_val,
        mask=pos_mask,
    )

    # Normalized data slice [PATCH_SIZE : 2*PATCH_SIZE]
    tl.store(
        Out_ptr + out_base + (PATCH_SIZE + local_pos) * stride_of,
        normalized,
        mask=pos_mask,
    )

    # Mask slice [2*PATCH_SIZE : 3*PATCH_SIZE]
    tl.store(
        Out_ptr + out_base + (2 * PATCH_SIZE + local_pos) * stride_of,
        mask_val,
        mask=pos_mask,
    )

    # Compute attention mask for this patch: 1 if any valid position, 0 otherwise
    any_valid = tl.sum(mask_val, axis=0)
    attn_val = tl.where(any_valid > 0.0, 1.0, 0.0)
    tl.store(AttnMask_ptr + batch_idx * NUM_PATCHES + patch_idx, attn_val)


# ---------------------------------------------------------------------------
# Python wrapper
# ---------------------------------------------------------------------------

def triton_fused_preprocess(
    context: torch.Tensor,
    patch_size: int = 16,
    context_length: int = 512,
    use_arcsinh: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fused Chronos2 preprocessing using Triton kernels.

    Two-phase approach:
        Phase 1: NaN-aware reduction for per-series mean (loc) and std (scale).
        Phase 2: Fused normalize + arcsinh + patch + time_encoding into output.

    Parameters
    ----------
    context : torch.Tensor
        Raw input of shape (B, L). May contain NaN values. Must be on CUDA.
    patch_size : int
        Size of each patch (default 16).
    context_length : int
        Maximum context length / time encoding scale (default 512).
    use_arcsinh : bool
        Apply arcsinh after normalization (default True).

    Returns
    -------
    patched : torch.Tensor
        Shape (B, P, 3*patch_size) with [time_enc, normalized, mask].
    attention_mask : torch.Tensor
        Shape (B, P), 1.0 if any non-NaN in patch, 0.0 otherwise.
    loc : torch.Tensor
        Shape (B, 1), per-series NaN-aware mean.
    scale : torch.Tensor
        Shape (B, 1), per-series NaN-aware std.
    """
    assert context.is_cuda, "triton_fused_preprocess requires CUDA tensors"

    ctx = context.float()
    B, L = ctx.shape

    # Truncate to context_length
    if L > context_length:
        ctx = ctx[:, -context_length:]
        L = context_length

    # Must be contiguous AFTER truncation (slicing can leave non-contiguous)
    ctx = ctx.contiguous()

    # Compute padding
    if L % patch_size != 0:
        pad_len = patch_size - (L % patch_size)
    else:
        pad_len = 0

    L_padded = L + pad_len
    num_patches = L_padded // patch_size

    # Phase 1: Reduction for loc and scale
    loc = torch.empty(B, dtype=torch.float32, device=ctx.device)
    scale = torch.empty(B, dtype=torch.float32, device=ctx.device)

    BLOCK_SIZE_REDUCE = triton.next_power_of_2(L)

    _nan_reduce_kernel[(B,)](
        ctx,
        loc,
        scale,
        L,
        1e-5,  # EPS
        BLOCK_SIZE=BLOCK_SIZE_REDUCE,
    )

    # Phase 2: Fused transform
    out = torch.empty(
        B, num_patches, 3 * patch_size,
        dtype=torch.float32, device=ctx.device,
    )
    attn_mask = torch.empty(
        B, num_patches,
        dtype=torch.float32, device=ctx.device,
    )

    BLOCK_SIZE_PATCH = triton.next_power_of_2(patch_size)

    grid = (B * num_patches,)
    _fused_transform_kernel[grid](
        ctx,
        out,
        loc,
        scale,
        attn_mask,
        L,
        pad_len,
        L_padded,
        num_patches,
        float(context_length),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        PATCH_SIZE=patch_size,
        USE_ARCSINH=use_arcsinh,
        BLOCK_SIZE=BLOCK_SIZE_PATCH,
    )

    # Reshape loc and scale to (B, 1) for API consistency
    loc = loc.unsqueeze(1)
    scale = scale.unsqueeze(1)

    return out, attn_mask, loc, scale
