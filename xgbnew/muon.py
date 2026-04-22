"""Single-GPU Muon optimizer for 2D weight matrices.

Stripped-down from Keller Jordan et al. (modded-nanogpt
records/track_1_short/2024-12-04_ValueEmbed/train_gpt2.py) — removed the
distributed all-reduce so it works for our single-5090 tabular MLP training
loop. Semantics identical: SGD-momentum + Newton-Schulz orthogonalization of
the per-param update, bf16 compute.

Per the original warnings, Muon is ONLY for 2D parameters. Biases, LayerNorm
gains, the final 1D output (or any scalar head), and the first
embedding-style projection should use AdamW. Use ``split_params`` helper to
partition a module's parameters into the two groups.
"""
from __future__ import annotations

import torch


def zeropower_via_newtonschulz5(G: torch.Tensor, steps: int = 5, eps: float = 1e-7) -> torch.Tensor:
    """Quintic Newton-Schulz iteration that orthogonalizes G ≈ U V^T."""
    assert G.ndim == 2, "Muon param must be 2D"
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16()
    X = X / (X.norm() + eps)
    transposed = G.size(0) > G.size(1)
    if transposed:
        X = X.T
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * (A @ A)
        X = a * X + B @ X
    if transposed:
        X = X.T
    return X


class Muon(torch.optim.Optimizer):
    """Muon optimizer (single-GPU). Parameters MUST be 2D."""

    def __init__(self, params, lr: float = 0.02, momentum: float = 0.95,
                 nesterov: bool = True, ns_steps: int = 5) -> None:
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov, ns_steps=ns_steps)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            lr = group["lr"]
            momentum = group["momentum"]
            ns_steps = group["ns_steps"]
            nesterov = group["nesterov"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                assert p.ndim == 2, (
                    f"Muon param must be 2D, got shape {tuple(p.shape)} — "
                    "use AdamW for biases/LN/1D params."
                )
                g = p.grad
                state = self.state[p]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(g)
                buf = state["momentum_buffer"]
                buf.mul_(momentum).add_(g)
                g = g.add(buf, alpha=momentum) if nesterov else buf
                g = zeropower_via_newtonschulz5(g, steps=ns_steps)
                # rectangular-matrix scaling so the update RMS matches a unit
                # orthogonal matrix of equivalent fan-in.
                g = g * (max(1.0, g.size(0) / g.size(1)) ** 0.5)
                p.add_(g.type_as(p), alpha=-lr)


def split_params(module: torch.nn.Module):
    """Split module parameters into (muon_params, adam_params).

    Muon: 2D weight tensors (``.ndim == 2``).
    Adam: everything else (biases, LayerNorm/RMSNorm params, 1D outputs).
    """
    muon, adam = [], []
    for p in module.parameters():
        if not p.requires_grad:
            continue
        if p.ndim == 2:
            muon.append(p)
        else:
            adam.append(p)
    return muon, adam


__all__ = ["Muon", "zeropower_via_newtonschulz5", "split_params"]
