"""Random Hadamard Transform (RHT) for outlier suppression before NVFP4 quant."""
from __future__ import annotations

import math
import torch


def hadamard_matrix(n: int, device=None, dtype=torch.float32) -> torch.Tensor:
    """Sylvester construction. n must be a power of 2."""
    assert n > 0 and (n & (n - 1)) == 0, f"n={n} must be a power of 2"
    H = torch.ones((1, 1), device=device, dtype=dtype)
    while H.shape[0] < n:
        H = torch.cat([torch.cat([H, H], dim=1),
                       torch.cat([H, -H], dim=1)], dim=0)
    return H / math.sqrt(n)  # orthonormal


class RandomHadamard:
    """Apply x -> (x * s) @ H, with H orthonormal and s a fixed ±1 sign vector.

    Inverse: y -> (y @ H.T) * s   (since H is orthonormal and s is ±1).
    Operates on the last dimension; tensor's last dim is padded to a multiple of n.
    """

    def __init__(self, n: int = 16, seed: int = 0, device=None, dtype=torch.float32):
        self.n = n
        g = torch.Generator(device="cpu").manual_seed(seed)
        signs = (torch.randint(0, 2, (n,), generator=g, dtype=torch.int64) * 2 - 1).to(dtype)
        self.signs = signs.to(device) if device is not None else signs
        self.H = hadamard_matrix(n, device=self.signs.device, dtype=dtype)

    def to(self, device=None, dtype=None):
        if device is not None:
            self.signs = self.signs.to(device)
            self.H = self.H.to(device)
        if dtype is not None:
            self.signs = self.signs.to(dtype)
            self.H = self.H.to(dtype)
        return self

    def _ensure(self, x: torch.Tensor):
        if self.signs.device != x.device or self.signs.dtype != x.dtype:
            self.signs = self.signs.to(device=x.device, dtype=x.dtype)
            self.H = self.H.to(device=x.device, dtype=x.dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self._ensure(x)
        n = self.n
        last = x.shape[-1]
        pad = (n - last % n) % n
        if pad:
            x = torch.cat([x, x.new_zeros(*x.shape[:-1], pad)], dim=-1)
        new_last = x.shape[-1]
        groups = new_last // n
        x_blk = x.reshape(*x.shape[:-1], groups, n)
        x_blk = x_blk * self.signs
        y = x_blk @ self.H
        return y.reshape(*x.shape[:-1], new_last), pad

    def inverse(self, y: torch.Tensor, pad: int) -> torch.Tensor:
        self._ensure(y)
        n = self.n
        new_last = y.shape[-1]
        groups = new_last // n
        y_blk = y.reshape(*y.shape[:-1], groups, n)
        x_blk = y_blk @ self.H.T
        x_blk = x_blk * self.signs
        out = x_blk.reshape(*y.shape[:-1], new_last)
        if pad:
            out = out[..., :-pad]
        return out
