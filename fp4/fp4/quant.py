"""2D 16x16 block quantization for NVFP4.

Each 16-element block (along the inner dim) gets one E4M3-ish FP scale; we use
a plain FP32 scale here for the reference implementation (the CUTLASS kernel
will use the proper E4M3 micro-scale on Blackwell). A per-tensor FP32 scale
sits on top.
"""
from __future__ import annotations

import torch

from .dtypes import NVFP4_MAX, encode_nvfp4, decode_nvfp4

BLOCK = 16


def _pad_to_block(x: torch.Tensor, dim: int, block: int = BLOCK) -> tuple[torch.Tensor, int]:
    n = x.shape[dim]
    pad = (block - n % block) % block
    if pad:
        pad_shape = list(x.shape)
        pad_shape[dim] = pad
        x = torch.cat([x, x.new_zeros(pad_shape)], dim=dim)
    return x, pad


def quantize_nvfp4_block(x: torch.Tensor, stochastic: bool = False,
                         generator: torch.Generator | None = None
                         ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    """Quantize the last dim of x in 16-element blocks.

    Returns (codes_uint8, block_scales_fp32, tensor_scale_fp32, pad).
    The dequantized value of element i within block b is:
        decode(code) * block_scale[b] * tensor_scale
    """
    orig_shape = x.shape
    x, pad = _pad_to_block(x, dim=-1)
    new_last = x.shape[-1]
    blocks = new_last // BLOCK
    x_blocks = x.reshape(*orig_shape[:-1], blocks, BLOCK)

    # per-tensor scale: keep block-amax distribution roughly in [0,1]
    amax = x_blocks.abs().amax()
    tensor_scale = (amax / NVFP4_MAX).clamp(min=1e-8)

    x_norm = x_blocks / tensor_scale  # now block amax is roughly NVFP4_MAX

    block_amax = x_norm.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8)
    block_scale = block_amax / NVFP4_MAX  # so x_norm/block_scale lives in [-NVFP4_MAX, NVFP4_MAX]

    x_scaled = x_norm / block_scale
    codes = encode_nvfp4(x_scaled, stochastic=stochastic, generator=generator)
    return codes, block_scale.squeeze(-1), tensor_scale, pad


def dequantize_nvfp4_block(codes: torch.Tensor, block_scale: torch.Tensor,
                           tensor_scale: torch.Tensor, pad: int,
                           dtype: torch.dtype = torch.float32) -> torch.Tensor:
    vals = decode_nvfp4(codes, dtype=dtype)
    out = vals * block_scale.unsqueeze(-1).to(dtype) * tensor_scale.to(dtype)
    out = out.reshape(*out.shape[:-2], -1)
    if pad:
        out = out[..., :-pad]
    return out


def quantize_nvfp4_2d(x: torch.Tensor, stochastic: bool = False,
                      generator: torch.Generator | None = None):
    """2D 16x16 block quantize on the last dim. Returns a packed dict for round-trip."""
    codes, bs, ts, pad = quantize_nvfp4_block(x, stochastic=stochastic, generator=generator)
    return {"codes": codes, "block_scale": bs, "tensor_scale": ts, "pad": pad,
            "shape": x.shape, "dtype": x.dtype}


def dequantize_nvfp4_2d(packed: dict) -> torch.Tensor:
    out = dequantize_nvfp4_block(packed["codes"], packed["block_scale"],
                                  packed["tensor_scale"], packed["pad"],
                                  dtype=packed["dtype"])
    return out.reshape(packed["shape"])
