"""Triton kernels for fused MLP operations in Chronos2 FeedForward layers.

Chronos2 FeedForward MLP: relu(x @ wi.T) @ wo.T  with 768->3072->768.
The 3072-wide intermediate is a major allocation target.

This module provides:
1. fused_linear_relu(x, weight) -- fuses linear + ReLU in one Triton kernel,
   avoiding a separate ReLU pass over the 3072-wide intermediate.
2. fused_mlp_relu(x, wi_weight, wo_weight) -- attempts full tiled fusion
   that avoids materializing the 3072-wide intermediate entirely, by tiling
   over the hidden dimension and accumulating partial sums.

Benchmark note: For production sizes (768->3072->768), cuBLAS GEMM via
F.linear is extremely fast on modern GPUs (RTX 5090, H100) and typically
outperforms both Triton kernels at these sizes due to highly tuned tensor
core usage. The fused_linear_relu saves a separate ReLU kernel launch but
cuBLAS matmul dominates the total time. The fused_mlp_relu is slower due
to redundant x loads per hidden chunk. Both are provided for correctness
reference and for potential wins on different hardware or larger batch sizes.
"""

import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Kernel 1: Fused Linear + ReLU
# Computes: out = relu(x @ weight.T)
# x: (M, K), weight: (N, K), out: (M, N)
# ---------------------------------------------------------------------------

@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 32}, num_stages=4, num_warps=4),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64, "BLOCK_K": 32}, num_stages=4, num_warps=4),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 32}, num_stages=4, num_warps=4),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 32}, num_stages=4, num_warps=8),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 64}, num_stages=3, num_warps=4),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64, "BLOCK_K": 64}, num_stages=3, num_warps=4),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 64}, num_stages=3, num_warps=4),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 64}, num_stages=3, num_warps=8),
    ],
    key=["M", "N", "K"],
)
@triton.jit
def _fused_linear_relu_kernel(
    X_ptr,
    W_ptr,
    Out_ptr,
    M,
    N,
    K,
    stride_xm,
    stride_xk,
    stride_wn,
    stride_wk,
    stride_om,
    stride_on,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """Fused linear + ReLU: out[m,n] = max(0, sum_k x[m,k] * w[n,k])

    Standard matmul tiling with ReLU applied to the accumulator before store.
    This avoids a separate elementwise ReLU kernel launch and the associated
    memory round-trip for the 3072-wide intermediate.
    """
    pid = tl.program_id(0)
    num_pid_n = tl.cdiv(N, BLOCK_N)

    # 2D grid decomposition
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    # Pointers for x[offs_m, offs_k] and w[offs_n, offs_k]
    x_ptrs = X_ptr + offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk
    w_ptrs = W_ptr + offs_n[:, None] * stride_wn + offs_k[None, :] * stride_wk

    # Accumulate in FP32 for numerical stability
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k_start in range(0, K, BLOCK_K):
        k_remaining = K - k_start
        k_mask = offs_k[None, :] < k_remaining

        x_tile = tl.load(x_ptrs, mask=(offs_m[:, None] < M) & k_mask, other=0.0)
        w_tile = tl.load(w_ptrs, mask=(offs_n[:, None] < N) & k_mask, other=0.0)

        # x @ w.T: (BLOCK_M, BLOCK_K) @ (BLOCK_K, BLOCK_N)
        # input_precision="ieee" avoids TF32 rounding on FP32 inputs
        acc = tl.dot(x_tile, tl.trans(w_tile), acc, input_precision="ieee")

        x_ptrs += BLOCK_K * stride_xk
        w_ptrs += BLOCK_K * stride_wk

    # Fused ReLU
    acc = tl.maximum(acc, 0.0)

    # Store result
    out_ptrs = Out_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on
    out_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(out_ptrs, acc.to(Out_ptr.dtype.element_ty), mask=out_mask)


def fused_linear_relu(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """Compute relu(x @ weight.T) with a single fused Triton kernel.

    Avoids materializing the pre-ReLU intermediate and eliminates the
    separate ReLU elementwise kernel launch.

    Args:
        x: Input tensor, shape (M, K) or (*, K) -- will be reshaped to 2D.
        weight: Weight matrix, shape (N, K).

    Returns:
        Output tensor, shape (M, N) or (*, N) matching input batch dims.
    """
    orig_shape = x.shape
    x_2d = x.reshape(-1, x.shape[-1]).contiguous()
    weight = weight.contiguous()

    M, K = x_2d.shape
    N = weight.shape[0]
    assert weight.shape[1] == K, f"Weight K dim {weight.shape[1]} != input K dim {K}"

    out = torch.empty(M, N, device=x.device, dtype=x.dtype)

    grid = lambda META: (
        triton.cdiv(M, META["BLOCK_M"]) * triton.cdiv(N, META["BLOCK_N"]),
    )

    _fused_linear_relu_kernel[grid](
        x_2d, weight, out,
        M, N, K,
        x_2d.stride(0), x_2d.stride(1),
        weight.stride(0), weight.stride(1),
        out.stride(0), out.stride(1),
    )

    out_shape = orig_shape[:-1] + (N,)
    return out.reshape(out_shape)


# ---------------------------------------------------------------------------
# Kernel 2: Fully fused MLP (Linear + ReLU + Linear)
# Computes: out = relu(x @ wi.T) @ wo.T
# Tiles over the hidden dimension to avoid materializing the full 3072 intermediate.
#
# Strategy: For each output tile (BLOCK_M, BLOCK_D_OUT), iterate over hidden
# dimension in chunks of BLOCK_H. For each chunk:
#   1. Compute partial_hidden = x_tile @ wi_chunk.T  (BLOCK_M, BLOCK_H)
#   2. Apply ReLU
#   3. Accumulate into output: acc += relu_hidden @ wo_chunk.T  (BLOCK_M, BLOCK_D_OUT)
#
# This avoids ever allocating the full (M, 3072) intermediate.
# ---------------------------------------------------------------------------

@triton.jit
def _fused_mlp_relu_kernel(
    X_ptr,
    Wi_ptr,
    Wo_ptr,
    Out_ptr,
    M,
    D_IN,
    D_HIDDEN,
    D_OUT,
    stride_xm,
    stride_xk,
    stride_wih,
    stride_wik,
    stride_wod,
    stride_woh,
    stride_om,
    stride_od,
    BLOCK_M: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """Fully fused MLP: out = relu(x @ wi.T) @ wo.T

    Tiles over hidden dimension in BLOCK_H chunks to avoid materializing
    the full (M, D_HIDDEN) intermediate. For each hidden chunk:
    1. Compute sub_hidden = x @ wi[h_start:h_end, :].T via inner K loop
    2. Apply ReLU
    3. Accumulate: acc += sub_hidden @ wo[:, h_start:h_end].T

    Note: This trades compute (redundant x loads per hidden chunk) for memory.
    For large D_HIDDEN (3072) this can be beneficial when memory-bound.
    """
    pid = tl.program_id(0)
    num_pid_d = tl.cdiv(D_OUT, BLOCK_D)
    pid_m = pid // num_pid_d
    pid_d = pid % num_pid_d

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)

    # Final output accumulator: (BLOCK_M, BLOCK_D)
    out_acc = tl.zeros((BLOCK_M, BLOCK_D), dtype=tl.float32)

    # Iterate over hidden dimension in BLOCK_H chunks
    for h_start in range(0, D_HIDDEN, BLOCK_H):
        offs_h = h_start + tl.arange(0, BLOCK_H)

        # Step 1: Compute partial hidden = x @ wi[offs_h, :].T
        # Result shape: (BLOCK_M, BLOCK_H)
        hidden_acc = tl.zeros((BLOCK_M, BLOCK_H), dtype=tl.float32)

        offs_k = tl.arange(0, BLOCK_K)
        for k_start in range(0, D_IN, BLOCK_K):
            k_remaining = D_IN - k_start
            k_mask = offs_k < k_remaining

            # Load x[offs_m, k_start:k_start+BLOCK_K]
            x_ptrs = X_ptr + offs_m[:, None] * stride_xm + (k_start + offs_k[None, :]) * stride_xk
            x_tile = tl.load(x_ptrs, mask=(offs_m[:, None] < M) & k_mask[None, :], other=0.0)

            # Load wi[offs_h, k_start:k_start+BLOCK_K]
            wi_ptrs = Wi_ptr + offs_h[:, None] * stride_wih + (k_start + offs_k[None, :]) * stride_wik
            wi_tile = tl.load(wi_ptrs, mask=(offs_h[:, None] < D_HIDDEN) & k_mask[None, :], other=0.0)

            # hidden_acc += x_tile @ wi_tile.T
            hidden_acc = tl.dot(x_tile, tl.trans(wi_tile), hidden_acc, input_precision="ieee")

        # Step 2: ReLU on the hidden chunk
        hidden_relu = tl.maximum(hidden_acc, 0.0)

        # Step 3: Accumulate output += hidden_relu @ wo[offs_d, offs_h].T
        # wo is (D_OUT, D_HIDDEN), so wo[offs_d, offs_h] is the slice we need
        # out += hidden_relu @ wo[offs_d, offs_h].T
        #      = (BLOCK_M, BLOCK_H) @ (BLOCK_H, BLOCK_D)
        # Load wo[offs_d, offs_h] as (BLOCK_D, BLOCK_H), then transpose
        wo_ptrs = Wo_ptr + offs_d[:, None] * stride_wod + offs_h[None, :] * stride_woh
        wo_tile = tl.load(wo_ptrs, mask=(offs_d[:, None] < D_OUT) & (offs_h[None, :] < D_HIDDEN), other=0.0)

        # hidden_relu: (BLOCK_M, BLOCK_H), wo_tile: (BLOCK_D, BLOCK_H)
        # We want (BLOCK_M, BLOCK_D) = hidden_relu @ wo_tile.T
        out_acc = tl.dot(hidden_relu.to(wo_tile.dtype), tl.trans(wo_tile), out_acc, input_precision="ieee")

    # Store output
    out_ptrs = Out_ptr + offs_m[:, None] * stride_om + offs_d[None, :] * stride_od
    out_mask = (offs_m[:, None] < M) & (offs_d[None, :] < D_OUT)
    tl.store(out_ptrs, out_acc.to(Out_ptr.dtype.element_ty), mask=out_mask)


def fused_mlp_relu(
    x: torch.Tensor,
    wi_weight: torch.Tensor,
    wo_weight: torch.Tensor,
) -> torch.Tensor:
    """Compute relu(x @ wi.T) @ wo.T with a single fused Triton kernel.

    Avoids materializing the full (M, D_HIDDEN) intermediate by tiling over
    the hidden dimension internally.

    Note: For production sizes (768->3072->768), cuBLAS via F.linear is
    typically faster because it uses tensor cores and highly tuned tiling.
    This kernel is provided as a demonstration of the fusion technique.
    Use fused_linear_relu + F.linear for the best practical performance.

    Args:
        x: Input tensor, shape (M, D_IN) or (*, D_IN).
        wi_weight: First layer weight, shape (D_HIDDEN, D_IN).
        wo_weight: Second layer weight, shape (D_OUT, D_HIDDEN).

    Returns:
        Output tensor, shape (M, D_OUT) or (*, D_OUT).
    """
    orig_shape = x.shape
    x_2d = x.reshape(-1, x.shape[-1]).contiguous()
    wi_weight = wi_weight.contiguous()
    wo_weight = wo_weight.contiguous()

    M, D_IN = x_2d.shape
    D_HIDDEN, D_IN_w = wi_weight.shape
    D_OUT, D_HIDDEN_w = wo_weight.shape
    assert D_IN == D_IN_w, f"wi K dim {D_IN_w} != input K dim {D_IN}"
    assert D_HIDDEN == D_HIDDEN_w, f"wo hidden dim {D_HIDDEN_w} != wi hidden dim {D_HIDDEN}"

    out = torch.empty(M, D_OUT, device=x.device, dtype=x.dtype)

    # Block sizes chosen for Chronos2 dims (768->3072->768)
    # BLOCK_H must divide D_HIDDEN evenly for correctness at boundaries
    # (the kernel masks handle non-divisible cases, but powers of 2 are best)
    BLOCK_M = 64
    BLOCK_H = 128  # Hidden dim tile: 3072 / 128 = 24 iterations
    BLOCK_K = 64   # K reduction tile for the inner matmul
    BLOCK_D = 64   # Output dim tile

    grid = (
        triton.cdiv(M, BLOCK_M) * triton.cdiv(D_OUT, BLOCK_D),
    )

    _fused_mlp_relu_kernel[grid](
        x_2d, wi_weight, wo_weight, out,
        M, D_IN, D_HIDDEN, D_OUT,
        x_2d.stride(0), x_2d.stride(1),
        wi_weight.stride(0), wi_weight.stride(1),
        wo_weight.stride(0), wo_weight.stride(1),
        out.stride(0), out.stride(1),
        BLOCK_M=BLOCK_M,
        BLOCK_H=BLOCK_H,
        BLOCK_K=BLOCK_K,
        BLOCK_D=BLOCK_D,
        num_warps=4,
        num_stages=2,
    )

    out_shape = orig_shape[:-1] + (D_OUT,)
    return out.reshape(out_shape)
