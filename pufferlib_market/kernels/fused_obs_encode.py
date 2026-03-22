"""Triton kernel for fused observation normalization + first linear layer + ReLU.

Replaces the three-step sequence:
    obs_norm = (obs - mean) / (std + eps)        # large intermediate allocation
    h        = F.linear(obs_norm, weight, bias)  # second large allocation
    h        = F.relu(h)                         # in-place, but still a pass

with a single Triton kernel that tiles over (B, H) and accumulates in registers,
so only the (B, H) output tensor is allocated.

Usage:
    from pufferlib_market.kernels.fused_obs_encode import fused_obs_norm_linear_relu, HAS_TRITON

    # obs: (B, OBS) float32
    # mean, std: (OBS,) float32  -- running statistics from RunningObsNorm
    # weight: (H, OBS) bfloat16 or float32
    # bias:   (H,)    float32
    out = fused_obs_norm_linear_relu(obs, mean, std, weight, bias)
    # out: (B, H) bfloat16 (or float32 on fallback)

When Triton is unavailable, falls back to pure PyTorch (three-step path).
"""

import torch
import torch.nn.functional as F

try:
    import triton
    import triton.language as tl
    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False


# ---------------------------------------------------------------------------
# CC-aware autotune config selection.
#
# The kernel tiles over (batch B, hidden H) and accumulates along OBS.
# BLOCK_B × BLOCK_H is the output tile held in registers.
# BLOCK_OBS is the K-reduction tile width; must be a power of 2.
#
# Memory per tile (FP32 accumulators):
#   acc:      BLOCK_B * BLOCK_H * 4 bytes
#   obs tile: BLOCK_B * BLOCK_OBS * 4 bytes (FP32 after normalize)
#   w tile:   BLOCK_H * BLOCK_OBS * 2 bytes (BF16)
#   mean/std: BLOCK_OBS * 4 bytes each
#
# Example at BLOCK_B=64, BLOCK_H=128, BLOCK_OBS=64:
#   acc=32KB, obs=16KB, w=16KB, stats=0.5KB => ~65KB per SM stage
# ---------------------------------------------------------------------------


def _get_autotune_configs():
    """Return CC-appropriate autotune configs for fused obs-encode kernel.

    Falls back to a single conservative config when CUDA is not available.
    """
    if not torch.cuda.is_available():
        return [
            triton.Config(
                {"BLOCK_B": 16, "BLOCK_H": 32, "BLOCK_OBS": 32},
                num_stages=2,
                num_warps=4,
            )
        ]
    major, _minor = torch.cuda.get_device_capability()
    if major >= 9:
        # H100 (CC 9.0), RTX 5090 (CC 12.0+):
        # H100: Triton warp specialization (num_warps=8 with BLOCK_H=128) matches
        # H100 tensor core tile sizes (16x16 MMA tiles, warp group = 4 warps).
        # For very large hidden dims (>=2048), cuBLAS may still win; bench with
        # bench_obs_encode.py first.
        # enable_warp_specialization=True is Triton 3.x+ only; check
        # triton.__version__ >= "3.0" before enabling.
        return [
            triton.Config({"BLOCK_B": 64,  "BLOCK_H": 128, "BLOCK_OBS": 128}, num_stages=6, num_warps=8),
            triton.Config({"BLOCK_B": 32,  "BLOCK_H": 128, "BLOCK_OBS": 128}, num_stages=5, num_warps=8),
            triton.Config({"BLOCK_B": 64,  "BLOCK_H": 64,  "BLOCK_OBS": 128}, num_stages=5, num_warps=8),
            triton.Config({"BLOCK_B": 32,  "BLOCK_H": 64,  "BLOCK_OBS": 128}, num_stages=4, num_warps=8),
            triton.Config({"BLOCK_B": 64,  "BLOCK_H": 128, "BLOCK_OBS": 64},  num_stages=4, num_warps=8),
            triton.Config({"BLOCK_B": 32,  "BLOCK_H": 128, "BLOCK_OBS": 64},  num_stages=4, num_warps=8),
            triton.Config({"BLOCK_B": 64,  "BLOCK_H": 64,  "BLOCK_OBS": 64},  num_stages=4, num_warps=4),
            triton.Config({"BLOCK_B": 32,  "BLOCK_H": 64,  "BLOCK_OBS": 64},  num_stages=3, num_warps=4),
        ]
    elif major == 8:
        # A100 (CC 8.0), RTX 3090/4090 (CC 8.6/8.9): 192 KB SRAM/SM.
        # Slightly smaller tiles than CC 9.x to stay within SRAM budget across
        # pipelined stages.
        return [
            triton.Config({"BLOCK_B": 32,  "BLOCK_H": 64,  "BLOCK_OBS": 64},  num_stages=4, num_warps=4),
            triton.Config({"BLOCK_B": 16,  "BLOCK_H": 64,  "BLOCK_OBS": 64},  num_stages=4, num_warps=4),
            triton.Config({"BLOCK_B": 32,  "BLOCK_H": 128, "BLOCK_OBS": 64},  num_stages=3, num_warps=8),
            triton.Config({"BLOCK_B": 64,  "BLOCK_H": 64,  "BLOCK_OBS": 64},  num_stages=3, num_warps=4),
            triton.Config({"BLOCK_B": 32,  "BLOCK_H": 64,  "BLOCK_OBS": 128}, num_stages=3, num_warps=4),
            triton.Config({"BLOCK_B": 16,  "BLOCK_H": 32,  "BLOCK_OBS": 64},  num_stages=4, num_warps=4),
        ]
    else:
        # CC 7.x (V100) and older: 96 KB SRAM/SM — conservative tiles.
        return [
            triton.Config({"BLOCK_B": 16,  "BLOCK_H": 32,  "BLOCK_OBS": 32},  num_stages=4, num_warps=4),
            triton.Config({"BLOCK_B": 32,  "BLOCK_H": 32,  "BLOCK_OBS": 32},  num_stages=4, num_warps=4),
            triton.Config({"BLOCK_B": 16,  "BLOCK_H": 64,  "BLOCK_OBS": 32},  num_stages=3, num_warps=4),
            triton.Config({"BLOCK_B": 32,  "BLOCK_H": 64,  "BLOCK_OBS": 32},  num_stages=3, num_warps=4),
        ]


# ---------------------------------------------------------------------------
# Triton kernel: fused (obs - mean) / (std + eps)  +  linear  +  relu
#
# Strategy:
#   Each program handles a tile of shape (BLOCK_B, BLOCK_H) in the output.
#   For each tile, iterate over the OBS dimension in BLOCK_OBS chunks:
#     1. Load obs slice: obs[b_tile, obs_start:obs_end]  — (BLOCK_B, BLOCK_OBS)
#     2. Load mean/std slices: (BLOCK_OBS,)
#     3. Normalize in FP32: obs_norm = (obs - mean) / (std + eps)
#     4. Load weight slice: W[h_tile, obs_start:obs_end] — (BLOCK_H, BLOCK_OBS)
#     5. acc += obs_norm @ W.T  via tl.dot
#   After loop: add bias, apply ReLU, store as BF16.
#
# Accumulation in FP32 for numerical stability.
# Output stored as BF16 (same convention as fused_mlp.py).
# ---------------------------------------------------------------------------

if HAS_TRITON:
    @triton.autotune(
        configs=_get_autotune_configs(),
        key=["B", "OBS", "H"],
    )
    @triton.jit
    def _fused_obs_norm_linear_relu_kernel(
        obs_ptr,     # (B, OBS) — float32 raw observations
        mean_ptr,    # (OBS,)   — float32 running mean
        std_ptr,     # (OBS,)   — float32 running std
        weight_ptr,  # (H, OBS) — bfloat16 weight matrix
        bias_ptr,    # (H,)     — float32 bias
        out_ptr,     # (B, H)   — bfloat16 output
        B,           # batch size (runtime int)
        OBS,         # obs_size  (runtime int)
        H,           # hidden_size (runtime int)
        stride_ob,   # stride for obs row (= OBS)
        stride_wh,   # stride for weight row (= OBS)
        eps: tl.constexpr,
        BLOCK_B: tl.constexpr,
        BLOCK_H: tl.constexpr,
        BLOCK_OBS: tl.constexpr,
    ):
        """Fused (obs-mean)/std + linear(W,b) + ReLU kernel.

        Each program handles a (BLOCK_B, BLOCK_H) output tile.
        Accumulates in FP32, stores as BF16.
        """
        pid = tl.program_id(0)
        num_pid_h = tl.cdiv(H, BLOCK_H)
        pid_b = pid // num_pid_h
        pid_h = pid % num_pid_h

        offs_b = pid_b * BLOCK_B + tl.arange(0, BLOCK_B)
        offs_h = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)

        # Accumulator for the matmul result in FP32
        acc = tl.zeros((BLOCK_B, BLOCK_H), dtype=tl.float32)

        # Iterate over the OBS (K) dimension
        offs_obs = tl.arange(0, BLOCK_OBS)
        for obs_start in range(0, OBS, BLOCK_OBS):
            obs_remaining = OBS - obs_start
            obs_mask = offs_obs < obs_remaining

            # 1. Load obs tile: obs[offs_b, obs_start + offs_obs]
            obs_ptrs = obs_ptr + offs_b[:, None] * stride_ob + (obs_start + offs_obs[None, :])
            obs_tile = tl.load(
                obs_ptrs,
                mask=(offs_b[:, None] < B) & obs_mask[None, :],
                other=0.0,
            ).to(tl.float32)

            # 2. Load mean and std slices: (BLOCK_OBS,)
            mean_vals = tl.load(mean_ptr + obs_start + offs_obs, mask=obs_mask, other=0.0).to(tl.float32)
            std_vals  = tl.load(std_ptr  + obs_start + offs_obs, mask=obs_mask, other=1.0).to(tl.float32)

            # 3. Normalize in FP32: (obs - mean) / (std + eps)
            obs_norm = (obs_tile - mean_vals[None, :]) / (std_vals[None, :] + eps)

            # 4. Load weight tile: W[offs_h, obs_start + offs_obs] — (BLOCK_H, BLOCK_OBS)
            w_ptrs = weight_ptr + offs_h[:, None] * stride_wh + (obs_start + offs_obs[None, :])
            w_tile = tl.load(
                w_ptrs,
                mask=(offs_h[:, None] < H) & obs_mask[None, :],
                other=0.0,
            ).to(tl.float32)

            # 5. acc += obs_norm @ W.T  =>  (BLOCK_B, BLOCK_OBS) @ (BLOCK_OBS, BLOCK_H)
            acc = tl.dot(obs_norm, tl.trans(w_tile), acc, input_precision="ieee")

        # Add bias: (H,) broadcast over B
        b_vals = tl.load(bias_ptr + offs_h, mask=offs_h < H, other=0.0).to(tl.float32)
        acc = acc + b_vals[None, :]

        # Fused ReLU
        acc = tl.maximum(acc, 0.0)

        # Store as BF16
        out_ptrs = out_ptr + offs_b[:, None] * H + offs_h[None, :]
        out_mask = (offs_b[:, None] < B) & (offs_h[None, :] < H)
        tl.store(out_ptrs, acc.to(tl.bfloat16), mask=out_mask)


# ---------------------------------------------------------------------------
# Python wrappers
# ---------------------------------------------------------------------------

def _fused_obs_norm_linear_relu_triton(
    obs: torch.Tensor,
    mean: torch.Tensor,
    std: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    """Run the Triton fused kernel. All inputs must be on CUDA and 2D."""
    B, OBS = obs.shape
    H = weight.shape[0]

    out = torch.empty(B, H, device=obs.device, dtype=torch.bfloat16)

    grid = lambda META: (
        triton.cdiv(B, META["BLOCK_B"]) * triton.cdiv(H, META["BLOCK_H"]),
    )

    _fused_obs_norm_linear_relu_kernel[grid](
        obs.contiguous(),
        mean.contiguous(),
        std.contiguous(),
        weight.contiguous(),
        bias.contiguous(),
        out,
        B, OBS, H,
        obs.stride(0),
        weight.stride(0),
        eps=eps,
    )
    return out


def _fused_obs_norm_linear_relu_fallback(
    obs: torch.Tensor,
    mean: torch.Tensor,
    std: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    """Pure PyTorch fallback: normalize then linear+relu."""
    obs_norm = (obs.float() - mean.float()) / (std.float() + eps)
    h = F.linear(obs_norm.to(weight.dtype), weight, bias)
    return F.relu(h)


def fused_obs_norm_linear_relu(
    obs: torch.Tensor,
    mean: torch.Tensor,
    std: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    eps: float = 1e-5,
) -> torch.Tensor:
    """Fused (obs - mean) / (std + eps) + linear + ReLU.

    Replaces the three-allocation sequence:
        obs_norm = (obs - mean) / (std + eps)   # (B, OBS) — eliminated
        h        = F.linear(obs_norm, W, b)      # (B, H)   — eliminated
        h        = F.relu(h)                     # in-place pass — eliminated
    with a single Triton kernel writing only the (B, H) output.

    When Triton is unavailable or inputs are not on CUDA, falls back to the
    equivalent three-step PyTorch computation.

    Args:
        obs:    (B, OBS) float32 raw observations.
        mean:   (OBS,) float32 running mean from RunningObsNorm.
        std:    (OBS,) float32 running std from RunningObsNorm.
        weight: (H, OBS) bfloat16 or float32 weight of the first linear layer.
        bias:   (H,) float32 bias of the first linear layer.
        eps:    Small constant added to std for numerical stability.

    Returns:
        (B, H) tensor:
          - bfloat16 when using the Triton kernel
          - same dtype as the fallback F.linear output otherwise
    """
    orig_shape = obs.shape
    obs_2d = obs.reshape(-1, obs.shape[-1])

    if HAS_TRITON and obs.is_cuda:
        # Cast inputs to dtypes expected by the kernel
        def _fp32(t):
            return t if (t.dtype == torch.float32 and t.is_contiguous()) else t.to(torch.float32).contiguous()
        def _bf16(t):
            return t if (t.dtype == torch.bfloat16 and t.is_contiguous()) else t.to(torch.bfloat16).contiguous()

        out = _fused_obs_norm_linear_relu_triton(
            _fp32(obs_2d),
            _fp32(mean),
            _fp32(std),
            _bf16(weight),
            _fp32(bias),
            eps,
        )
    else:
        out = _fused_obs_norm_linear_relu_fallback(obs_2d, mean, std, weight, bias, eps)

    out_shape = orig_shape[:-1] + (out.shape[-1],)
    return out.reshape(out_shape)
