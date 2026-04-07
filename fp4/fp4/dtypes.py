"""NVFP4 (E2M1) data type primitives.

NVFP4 = 4-bit float, 1 sign / 2 exponent / 1 mantissa, no inf/nan, range ±6.
The 16 representable values (signed) are:
    0, ±0.5, ±1, ±1.5, ±2, ±3, ±4, ±6
This matches the CUTLASS Blackwell example 79b layout.
"""
from __future__ import annotations

import torch

# 16 levels for E2M1 (signed, no NaN/Inf). Index = 4-bit code.
# Layout: bit 3 = sign, bits 2-1 = exponent, bit 0 = mantissa.
# Standard NVFP4 / OCP-MX FP4 (e2m1) values:
NVFP4_VALUES = torch.tensor(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
     -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
    dtype=torch.float32,
)

# Positive magnitudes only (8 levels) — used for nearest search.
NVFP4_POS = torch.tensor(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0],
    dtype=torch.float32,
)

NVFP4_MAX = 6.0


def is_blackwell() -> bool:
    """True iff the current CUDA device is Blackwell (sm_100+)."""
    if not torch.cuda.is_available():
        return False
    try:
        major, _ = torch.cuda.get_device_capability()
        return major >= 10
    except Exception:
        return False


def _nearest_pos_index(x_abs: torch.Tensor) -> torch.Tensor:
    """For each |x|, return index into NVFP4_POS of the nearest level (RTN, ties to even/up)."""
    pos = NVFP4_POS.to(x_abs.device, x_abs.dtype)  # (8,)
    # distance to each level
    diffs = (x_abs.unsqueeze(-1) - pos).abs()  # (..., 8)
    return diffs.argmin(dim=-1)


def encode_nvfp4(x: torch.Tensor, stochastic: bool = False,
                 generator: torch.Generator | None = None) -> torch.Tensor:
    """Encode FP32 tensor x (already scaled to fit in [-6, 6]) into uint8 NVFP4 codes.

    Returns int8 tensor with values in [0, 15]. Inputs outside [-6, 6] are clamped.
    """
    pos = NVFP4_POS.to(x.device, x.dtype)
    sign_bit = (x < 0).to(torch.uint8) << 3
    x_abs = x.abs().clamp(max=NVFP4_MAX)

    if not stochastic:
        idx = _nearest_pos_index(x_abs)
    else:
        # Find lower level via searchsorted, then SR between lower and upper.
        # pos is sorted ascending. searchsorted gives insertion point on the right.
        upper = torch.searchsorted(pos, x_abs, right=True).clamp(max=7)
        lower = (upper - 1).clamp(min=0)
        lo = pos[lower]
        hi = pos[upper]
        span = (hi - lo).clamp(min=1e-12)
        frac = ((x_abs - lo) / span).clamp(0.0, 1.0)
        if generator is None:
            r = torch.rand_like(frac)
        else:
            r = torch.rand(frac.shape, device=frac.device, dtype=frac.dtype,
                           generator=generator)
        take_upper = (r < frac).to(torch.long)
        idx = lower + take_upper
        # When lower == upper (x_abs exactly on a level or above max), idx is fine.

    code = (idx.to(torch.uint8) | sign_bit)
    # Force -0 → +0
    is_zero = (idx == 0)
    code = torch.where(is_zero, torch.zeros_like(code), code)
    return code


def decode_nvfp4(code: torch.Tensor, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """Decode uint8 NVFP4 code tensor back to floats."""
    table = NVFP4_VALUES.to(code.device, dtype)
    return table[code.to(torch.long)]
