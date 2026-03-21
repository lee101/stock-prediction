"""Triton kernel for fused two-layer MLP: x -> Linear1 -> ReLU -> Linear2 -> output.

Eliminates the intermediate hidden-wide activation tensor allocation by tiling
over the hidden dimension and accumulating partial sums in registers.

Usage:
    from pufferlib_market.kernels.fused_mlp import fused_mlp_relu, HAS_TRITON

    # x: [batch, in_dim], W1: [hidden, in_dim], b1: [hidden],
    # W2: [out_dim, hidden], b2: [out_dim]
    out = fused_mlp_relu(x, W1, b1, W2, b2)

If triton is unavailable, falls back to pure PyTorch.
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
# Triton kernel: fully fused Linear1 + bias1 + ReLU + Linear2 + bias2
# Avoids materializing the (M, D_HIDDEN) intermediate tensor.
#
# Strategy: For each output tile (BLOCK_M rows, BLOCK_D cols), iterate over
# the hidden dimension in BLOCK_H chunks:
#   1. hidden_chunk = x_tile @ W1[h:h+BLOCK_H, :].T + b1[h:h+BLOCK_H]
#   2. ReLU in-register
#   3. acc += relu_hidden @ W2[d_tile, h:h+BLOCK_H].T
# Then add b2 and store.
# ---------------------------------------------------------------------------

if HAS_TRITON:
    @triton.autotune(
        configs=[
            triton.Config({"BLOCK_M": 64,  "BLOCK_D": 64,  "BLOCK_H": 64,  "BLOCK_K": 32}, num_stages=4, num_warps=4),
            triton.Config({"BLOCK_M": 128, "BLOCK_D": 64,  "BLOCK_H": 64,  "BLOCK_K": 32}, num_stages=4, num_warps=4),
            triton.Config({"BLOCK_M": 64,  "BLOCK_D": 128, "BLOCK_H": 64,  "BLOCK_K": 32}, num_stages=4, num_warps=4),
            triton.Config({"BLOCK_M": 128, "BLOCK_D": 128, "BLOCK_H": 64,  "BLOCK_K": 32}, num_stages=4, num_warps=8),
            triton.Config({"BLOCK_M": 64,  "BLOCK_D": 64,  "BLOCK_H": 128, "BLOCK_K": 32}, num_stages=3, num_warps=4),
            triton.Config({"BLOCK_M": 128, "BLOCK_D": 64,  "BLOCK_H": 128, "BLOCK_K": 32}, num_stages=3, num_warps=4),
            triton.Config({"BLOCK_M": 64,  "BLOCK_D": 128, "BLOCK_H": 128, "BLOCK_K": 64}, num_stages=3, num_warps=4),
            triton.Config({"BLOCK_M": 128, "BLOCK_D": 128, "BLOCK_H": 128, "BLOCK_K": 64}, num_stages=3, num_warps=8),
        ],
        key=["M", "D_IN", "D_HIDDEN", "D_OUT"],
    )
    @triton.jit
    def _fused_mlp_relu_kernel(
        X_ptr,
        W1_ptr, B1_ptr,
        W2_ptr, B2_ptr,
        Out_ptr,
        M, D_IN, D_HIDDEN, D_OUT,
        stride_xm, stride_xk,
        stride_w1h, stride_w1k,
        stride_w2d, stride_w2h,
        stride_om, stride_od,
        BLOCK_M: tl.constexpr,
        BLOCK_D: tl.constexpr,
        BLOCK_H: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        """Fused Linear1 + bias + ReLU + Linear2 + bias kernel.

        Each program handles a (BLOCK_M, BLOCK_D) tile of the output.
        Accumulates in FP32, stores as BF16.
        """
        pid = tl.program_id(0)
        num_pid_d = tl.cdiv(D_OUT, BLOCK_D)
        pid_m = pid // num_pid_d
        pid_d = pid % num_pid_d

        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_d = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)

        # Output accumulator in FP32
        out_acc = tl.zeros((BLOCK_M, BLOCK_D), dtype=tl.float32)

        # Iterate over hidden dimension in BLOCK_H chunks
        for h_start in range(0, D_HIDDEN, BLOCK_H):
            offs_h = h_start + tl.arange(0, BLOCK_H)
            h_mask = offs_h < D_HIDDEN

            # --- Step 1: compute hidden_chunk = x @ W1[offs_h, :].T + b1[offs_h] ---
            hidden_acc = tl.zeros((BLOCK_M, BLOCK_H), dtype=tl.float32)

            offs_k = tl.arange(0, BLOCK_K)
            for k_start in range(0, D_IN, BLOCK_K):
                k_remaining = D_IN - k_start
                k_mask = offs_k < k_remaining

                # Load x[offs_m, k_start:k_start+BLOCK_K]
                x_ptrs = X_ptr + offs_m[:, None] * stride_xm + (k_start + offs_k[None, :]) * stride_xk
                x_tile = tl.load(
                    x_ptrs,
                    mask=(offs_m[:, None] < M) & k_mask[None, :],
                    other=0.0,
                )

                # Load W1[offs_h, k_start:k_start+BLOCK_K]
                w1_ptrs = W1_ptr + offs_h[:, None] * stride_w1h + (k_start + offs_k[None, :]) * stride_w1k
                w1_tile = tl.load(
                    w1_ptrs,
                    mask=h_mask[:, None] & k_mask[None, :],
                    other=0.0,
                )

                # hidden_acc += x_tile @ w1_tile.T  => (BLOCK_M, BLOCK_H)
                hidden_acc = tl.dot(x_tile, tl.trans(w1_tile), hidden_acc, input_precision="ieee")

            # Add bias1
            b1_ptrs = B1_ptr + offs_h
            b1_vals = tl.load(b1_ptrs, mask=h_mask, other=0.0)
            hidden_acc = hidden_acc + b1_vals[None, :]

            # --- Step 2: ReLU ---
            hidden_relu = tl.maximum(hidden_acc, 0.0)

            # --- Step 3: acc += hidden_relu @ W2[offs_d, offs_h].T ---
            # W2 shape: (D_OUT, D_HIDDEN), load W2[offs_d, offs_h]
            w2_ptrs = W2_ptr + offs_d[:, None] * stride_w2d + offs_h[None, :] * stride_w2h
            w2_tile = tl.load(
                w2_ptrs,
                mask=(offs_d[:, None] < D_OUT) & h_mask[None, :],
                other=0.0,
            )

            # hidden_relu: (BLOCK_M, BLOCK_H), w2_tile: (BLOCK_D, BLOCK_H)
            # out += hidden_relu @ w2_tile.T => (BLOCK_M, BLOCK_D)
            out_acc = tl.dot(
                hidden_relu.to(w2_tile.dtype),
                tl.trans(w2_tile),
                out_acc,
                input_precision="ieee",
            )

        # Add bias2
        b2_ptrs = B2_ptr + offs_d
        b2_vals = tl.load(b2_ptrs, mask=offs_d < D_OUT, other=0.0)
        out_acc = out_acc + b2_vals[None, :]

        # Store as BF16
        out_ptrs = Out_ptr + offs_m[:, None] * stride_om + offs_d[None, :] * stride_od
        out_mask = (offs_m[:, None] < M) & (offs_d[None, :] < D_OUT)
        tl.store(out_ptrs, out_acc.to(Out_ptr.dtype.element_ty), mask=out_mask)


def _fused_mlp_relu_triton(
    x: torch.Tensor,
    W1: torch.Tensor,
    b1: torch.Tensor,
    W2: torch.Tensor,
    b2: torch.Tensor,
) -> torch.Tensor:
    """Run fused MLP via Triton kernel. x must be 2D."""
    M, D_IN = x.shape
    D_HIDDEN = W1.shape[0]
    D_OUT = W2.shape[0]

    # Output in BF16 for tensor core efficiency
    out = torch.empty(M, D_OUT, device=x.device, dtype=torch.bfloat16)

    grid = lambda META: (
        triton.cdiv(M, META["BLOCK_M"]) * triton.cdiv(D_OUT, META["BLOCK_D"]),
    )

    _fused_mlp_relu_kernel[grid](
        x, W1, b1, W2, b2, out,
        M, D_IN, D_HIDDEN, D_OUT,
        x.stride(0), x.stride(1),
        W1.stride(0), W1.stride(1),
        W2.stride(0), W2.stride(1),
        out.stride(0), out.stride(1),
    )
    return out


def _fused_mlp_relu_fallback(
    x: torch.Tensor,
    W1: torch.Tensor,
    b1: torch.Tensor,
    W2: torch.Tensor,
    b2: torch.Tensor,
) -> torch.Tensor:
    """Pure PyTorch fallback: relu(x @ W1.T + b1) @ W2.T + b2."""
    h = F.linear(x, W1, b1)
    h = F.relu(h)
    return F.linear(h, W2, b2)


def fused_mlp_relu(
    x: torch.Tensor,
    W1: torch.Tensor,
    b1: torch.Tensor,
    W2: torch.Tensor,
    b2: torch.Tensor,
) -> torch.Tensor:
    """Compute relu(x @ W1.T + b1) @ W2.T + b2 with optional Triton fusion.

    When Triton is available and the input is on CUDA, uses the fused kernel
    that avoids materializing the full (batch, D_HIDDEN) intermediate tensor.
    Falls back to F.linear + F.relu otherwise.

    Args:
        x:  Input tensor, shape (..., D_IN). Will be reshaped to 2D.
        W1: First layer weight, shape (D_HIDDEN, D_IN).
        b1: First layer bias, shape (D_HIDDEN,).
        W2: Second layer weight, shape (D_OUT, D_HIDDEN).
        b2: Second layer bias, shape (D_OUT,).

    Returns:
        Output tensor, shape (..., D_OUT) in BF16 when using Triton,
        or matching x.dtype when using fallback.
    """
    orig_shape = x.shape
    x_2d = x.reshape(-1, x.shape[-1])

    if HAS_TRITON and x.is_cuda:
        # Cast to BF16/FP32 as required by the kernel; skip if already correct.
        def _to_bf16(t): return t if t.dtype == torch.bfloat16 and t.is_contiguous() else t.to(torch.bfloat16).contiguous()
        def _to_fp32(t): return t if t.dtype == torch.float32 and t.is_contiguous() else t.to(torch.float32).contiguous()
        out = _fused_mlp_relu_triton(
            _to_bf16(x_2d), _to_bf16(W1), _to_fp32(b1), _to_bf16(W2), _to_fp32(b2),
        )
    else:
        out = _fused_mlp_relu_fallback(x_2d, W1, b1, W2, b2)

    out_shape = orig_shape[:-1] + (out.shape[-1],)
    return out.reshape(out_shape)
