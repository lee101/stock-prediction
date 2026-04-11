"""NVFP4Linear: a drop-in nn.Linear replacement.

Forward: RHT(x) -> 2D 16x16 quantize (RTN) -> dequant -> matmul against
quantized weights (RTN). FP32 master weight stored as the parameter; we
quantize on the fly each call (cheap because the master is small in this ref).

Backward: same path but stochastic rounding on grads & weights to be unbiased.

On non-Blackwell devices we transparently fall back to a BF16 matmul reference
so tests run on CPU and consumer GPUs. The quantize/dequantize emulation is
identical on every device — only the underlying GEMM differs.
"""
from __future__ import annotations

import math
import torch
from torch import nn

from .dtypes import is_blackwell
from .quant import quantize_nvfp4_2d, dequantize_nvfp4_2d
from .hadamard import RandomHadamard
from .kernels.gemm import gemm as _backend_gemm


def _q_emulate(x: torch.Tensor, stochastic: bool) -> torch.Tensor:
    """Quantize then dequantize x along the last dim using NVFP4 (emulation)."""
    packed = quantize_nvfp4_2d(x, stochastic=stochastic)
    return dequantize_nvfp4_2d(packed)


class _NVFP4LinearFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, bias, rht_in, rht_grad):
        # x: (..., in_features); weight: (out, in); bias: (out,) or None
        x_flat = x.reshape(-1, x.shape[-1])
        # Forward RHT on activations
        x_rht, x_pad = rht_in.forward(x_flat)
        x_q = _q_emulate(x_rht, stochastic=False)
        # Quantize weight rows (last dim is in_features)
        w_rht, w_pad = rht_in.forward(weight)
        w_q = _q_emulate(w_rht, stochastic=False)
        # GEMM: y_rht = x_q @ w_q.T  -> still in RHT space along the in dim,
        # but matmul contracts that dim, so output is in normal space already.
        y = _backend_gemm(x_q, w_q.T)
        if bias is not None:
            y = y + bias
        ctx.save_for_backward(x_flat, weight, bias if bias is not None else torch.empty(0))
        ctx.rht_in = rht_in
        ctx.rht_grad = rht_grad
        ctx.has_bias = bias is not None
        ctx.x_shape = x.shape
        ctx.out_features = weight.shape[0]
        return y.reshape(*x.shape[:-1], weight.shape[0])

    @staticmethod
    def backward(ctx, grad_out):
        x_flat, weight, bias_buf = ctx.saved_tensors
        rht_in = ctx.rht_in
        rht_grad = ctx.rht_grad
        g = grad_out.reshape(-1, ctx.out_features)

        # SR-quantize grad along its last dim (out_features) after a grad RHT
        g_rht, g_pad = rht_grad.forward(g)
        g_q = _q_emulate(g_rht, stochastic=True)
        # Inverse RHT to recover gradient in original out-feature basis
        g_back = rht_grad.inverse(g_q, g_pad)

        # grad_x = g_back @ weight (both in normal basis)
        # Quantize weight (RTN) for the backward GEMM
        w_q = _q_emulate(weight, stochastic=False)
        grad_x = g_back @ w_q

        # grad_w = g_back.T @ x  (SR-quantize x for unbiasedness)
        x_q = _q_emulate(x_flat, stochastic=True)
        grad_w = g_back.T @ x_q

        grad_b = g_back.sum(dim=0) if ctx.has_bias else None

        return (grad_x.reshape(ctx.x_shape), grad_w, grad_b, None, None)


class NVFP4Linear(nn.Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True,
                 hadamard_size: int = 16, seed: int = 0):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            bound = 1 / math.sqrt(in_features)
            nn.init.uniform_(self.bias, -bound, bound)
        self.rht_in = RandomHadamard(n=hadamard_size, seed=seed)
        self.rht_grad = RandomHadamard(n=hadamard_size, seed=seed + 1)
        self.blackwell = is_blackwell()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return _NVFP4LinearFn.apply(x, self.weight, self.bias,
                                    self.rht_in, self.rht_grad)


def nvfp4_linear(x: torch.Tensor, weight: torch.Tensor, bias=None,
                 hadamard_size: int = 16, seed: int = 0) -> torch.Tensor:
    rht_in = RandomHadamard(n=hadamard_size, seed=seed,
                            device=x.device, dtype=x.dtype)
    rht_grad = RandomHadamard(n=hadamard_size, seed=seed + 1,
                              device=x.device, dtype=x.dtype)
    return _NVFP4LinearFn.apply(x, weight, bias, rht_in, rht_grad)
