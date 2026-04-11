"""2D 16x16 block quantization for NVFP4.

Each 16-element block (along the inner dim) gets one E4M3-ish FP scale; we use
a plain FP32 scale here for the reference implementation (the CUTLASS kernel
will use the proper E4M3 micro-scale on Blackwell). A per-tensor FP32 scale
sits on top.
"""
from __future__ import annotations

import torch

from .dtypes import NVFP4_MAX, NVFP4_POS, NVFP4_VALUES, encode_nvfp4

BLOCK = 16

# Per-(device, dtype) caches of the NVFP4 level tables.  Built outside any
# CUDA-graph capture so that later graph-captured calls can consume them via
# pure on-device ops (no host syncs, no H2D copies).
_POS_CACHE: dict[tuple[torch.device, torch.dtype], torch.Tensor] = {}
_VAL_CACHE: dict[tuple[torch.device, torch.dtype], torch.Tensor] = {}


def _get_pos(device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    key = (device, dtype)
    t = _POS_CACHE.get(key)
    if t is None:
        t = NVFP4_POS.to(device=device, dtype=dtype)
        _POS_CACHE[key] = t
    return t


def _get_vals(device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    key = (device, dtype)
    t = _VAL_CACHE.get(key)
    if t is None:
        t = NVFP4_VALUES.to(device=device, dtype=dtype)
        _VAL_CACHE[key] = t
    return t


def prime_caches(device: torch.device, dtypes: tuple[torch.dtype, ...] = (torch.float32,)) -> None:
    """Populate the per-(device, dtype) level-table caches eagerly.

    Call this before any CUDA-graph capture so that the first graph-captured
    quantize call does not need to perform a host-to-device copy of the NVFP4
    level tables.
    """
    for dt in dtypes:
        _get_pos(device, dt)
        _get_vals(device, dt)


def _encode_nvfp4_cached(x: torch.Tensor, stochastic: bool) -> torch.Tensor:
    """Graph-safe version of dtypes.encode_nvfp4 using cached level tables."""
    pos = _get_pos(x.device, x.dtype)
    sign_bit = (x < 0).to(torch.uint8) << 3
    x_abs = x.abs().clamp(max=NVFP4_MAX)
    if not stochastic:
        diffs = (x_abs.unsqueeze(-1) - pos).abs()
        idx = diffs.argmin(dim=-1)
    else:
        upper = torch.searchsorted(pos, x_abs, right=True).clamp(max=7)
        lower = (upper - 1).clamp(min=0)
        lo = pos[lower]
        hi = pos[upper]
        span = (hi - lo).clamp(min=1e-12)
        frac = ((x_abs - lo) / span).clamp(0.0, 1.0)
        r = torch.rand_like(frac)
        take_upper = (r < frac).to(torch.long)
        idx = lower + take_upper
    code = idx.to(torch.uint8) | sign_bit
    is_zero = (idx == 0)
    code = torch.where(is_zero, torch.zeros_like(code), code)
    return code


def _decode_nvfp4_cached(code: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    table = _get_vals(code.device, dtype)
    return table[code.to(torch.long)]


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
    if generator is None:
        codes = _encode_nvfp4_cached(x_scaled, stochastic=stochastic)
    else:
        codes = encode_nvfp4(x_scaled, stochastic=stochastic, generator=generator)
    return codes, block_scale.squeeze(-1), tensor_scale, pad


def dequantize_nvfp4_block(codes: torch.Tensor, block_scale: torch.Tensor,
                           tensor_scale: torch.Tensor, pad: int,
                           dtype: torch.dtype = torch.float32) -> torch.Tensor:
    vals = _decode_nvfp4_cached(codes, dtype=dtype)
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
