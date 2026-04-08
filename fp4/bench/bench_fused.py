"""Bench: 4-layer 256-dim MLP fused vs unfused, 1000 fwd+bwd passes."""
from __future__ import annotations

import time

import torch
from torch import nn
import torch.nn.functional as F

from fp4.fused_linear import FusedLinearGELU


class UnfusedMLP(nn.Module):
    def __init__(self, dim: int = 256, n_layers: int = 4) -> None:
        super().__init__()
        self.layers = nn.ModuleList([nn.Linear(dim, dim) for _ in range(n_layers)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for lin in self.layers:
            x = F.gelu(lin(x))
        return x


class FusedMLP(nn.Module):
    def __init__(self, dim: int = 256, n_layers: int = 4) -> None:
        super().__init__()
        self.layers = nn.ModuleList([FusedLinearGELU(dim, dim) for _ in range(n_layers)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for lin in self.layers:
            x = lin(x)
        return x


def time_model(model: nn.Module, x: torch.Tensor, iters: int) -> float:
    opt = torch.optim.SGD(model.parameters(), lr=1e-4)
    # warmup
    for _ in range(20):
        opt.zero_grad(set_to_none=True)
        y = model(x)
        y.sum().backward()
        opt.step()
    if x.is_cuda:
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        opt.zero_grad(set_to_none=True)
        y = model(x)
        y.sum().backward()
        opt.step()
    if x.is_cuda:
        torch.cuda.synchronize()
    return time.perf_counter() - t0


def main() -> None:
    import os

    force_cpu = os.environ.get("BENCH_FUSED_CPU") == "1"
    if force_cpu or not torch.cuda.is_available():
        if not torch.cuda.is_available():
            print("NOTE: CUDA unavailable; running on CPU (launch-overhead win will not show).")
        else:
            print("NOTE: BENCH_FUSED_CPU=1 set; running on CPU.")
        device = torch.device("cpu")
    else:
        device = torch.device("cuda")
    torch.manual_seed(0)
    dim = 256
    batch = 256
    iters = 1000

    x = torch.randn(batch, dim, device=device)

    unfused = UnfusedMLP(dim).to(device)
    fused = FusedMLP(dim).to(device)

    t_u = time_model(unfused, x, iters)
    t_f = time_model(fused, x, iters)

    sps_u = iters / t_u
    sps_f = iters / t_f
    ratio = sps_f / sps_u

    print(f"unfused: {t_u:.3f}s  ({sps_u:.1f} sps)")
    print(f"fused:   {t_f:.3f}s  ({sps_f:.1f} sps)")
    print(f"speedup: {ratio:.3f}x  (target >= 1.5x)")


if __name__ == "__main__":
    main()
