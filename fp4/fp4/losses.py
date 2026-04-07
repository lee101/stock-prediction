"""Constrained-MDP losses for fp4 RL trainers (Phase 4, Unit P4-3).

Clean, reusable building blocks for PPO/SAC/QR-PPO variants:

- `ppo_clipped_surrogate` — clipped policy-gradient surrogate.
- `value_loss_mse`        — scalar critic MSE.
- `entropy_bonus`         — mean entropy of a distribution (or std tensor).
- `cvar_loss`             — differentiable lower-alpha tail mean, GPU-only.
- `smooth_pnl_loss`       — second-difference penalty on log-equity paths.

All functions are fully differentiable w.r.t. their tensor inputs and avoid
any `.item()` / `.cpu()` calls so they are safe inside CUDA graphs.
"""
from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# PPO building blocks
# ---------------------------------------------------------------------------

def ppo_clipped_surrogate(
    logp_new: torch.Tensor,
    logp_old: torch.Tensor,
    adv: torch.Tensor,
    clip_eps: float = 0.2,
    normalize_adv: bool = True,
) -> torch.Tensor:
    """Return the (negated, minimisable) PPO clipped surrogate.

    `logp_new`, `logp_old`, `adv` must be the same shape (typically `[B]`).
    Advantages are normalised by default (matches `fp4.trainer._ppo_loss`).
    """
    if normalize_adv:
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
    ratio = torch.exp(logp_new - logp_old)
    surr1 = ratio * adv
    surr2 = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * adv
    return -torch.min(surr1, surr2).mean()


def value_loss_mse(v_pred: torch.Tensor, returns: torch.Tensor) -> torch.Tensor:
    """Plain MSE value loss."""
    return F.mse_loss(v_pred, returns)


def entropy_bonus(dist: Any) -> torch.Tensor:
    """Mean entropy for a distribution / std tensor.

    Accepts:
    - a `torch.distributions.Distribution` (calls `.entropy()`);
    - any object exposing `.entropy()`;
    - a raw std tensor for a diagonal Gaussian (uses the closed form
      `0.5 * log(2*pi*e*sigma^2)` summed over the last dim).
    """
    if hasattr(dist, "entropy") and callable(dist.entropy):
        return dist.entropy().mean()
    if isinstance(dist, torch.Tensor):
        # Treat as per-dim std of a diagonal Gaussian.
        const = 0.5 * math.log(2.0 * math.pi * math.e)
        ent = (torch.log(dist) + const).sum(dim=-1)
        return ent.mean()
    raise TypeError(
        f"entropy_bonus: unsupported input type {type(dist).__name__}; "
        "pass a torch.distributions.Distribution or a std tensor."
    )


# ---------------------------------------------------------------------------
# CVaR (expected shortfall) on a batch of returns
# ---------------------------------------------------------------------------

def cvar_loss(returns: torch.Tensor, alpha: float = 0.05) -> torch.Tensor:
    """Mean of the worst-`alpha` fraction of `returns` (the lower tail).

    Differentiable and GPU-only (no `.item()` / `.cpu()`). `alpha∈(0,1]`.
    `returns` may be any shape; it is flattened before the tail is taken.
    The number of tail elements is `max(1, floor(alpha * N))`.

    Note: conventionally, *loss* variables treat "lower is worse"; for PnL
    returns, `cvar_loss` returns the *mean of the worst returns*, i.e. a
    negative number for a losing tail. Minimising `-cvar_loss` (or using it
    as a constraint cost `d - cvar`) pushes the tail up.
    """
    if not (0.0 < alpha <= 1.0):
        raise ValueError(f"cvar_loss: alpha must be in (0, 1], got {alpha}")
    flat = returns.reshape(-1)
    n = flat.numel()
    if n == 0:
        return returns.sum() * 0.0  # zero, keeps grad graph
    k = max(1, int(math.floor(alpha * n)))
    # `-flat` makes the worst (most negative) returns the largest — topk
    # gives us the lower tail without a full sort. Negate back at the end.
    neg_top, _ = torch.topk(-flat, k=k, largest=True, sorted=False)
    return (-neg_top).mean()


# ---------------------------------------------------------------------------
# Smooth-PnL (second-difference penalty on log-equity paths)
# ---------------------------------------------------------------------------

def smooth_pnl_loss(log_eq: torch.Tensor) -> torch.Tensor:
    """Penalise curvature of log-equity paths.

    `log_eq` is expected to be `[B, T]` (batch of trajectories), but any
    shape `[..., T]` with `T >= 3` is accepted. Returns
    `mean((log_eq[..., t+1] - 2*log_eq[..., t] + log_eq[..., t-1])^2)`.
    A perfectly linear log-equity path (constant log-return per step) yields
    exactly zero.
    """
    if log_eq.dim() < 2:
        raise ValueError(
            f"smooth_pnl_loss: expected at least 2-d input, got shape {tuple(log_eq.shape)}"
        )
    if log_eq.shape[-1] < 3:
        # Too short to have a second difference — return a 0 that still
        # carries a grad path.
        return log_eq.sum() * 0.0
    d2 = log_eq[..., 2:] - 2.0 * log_eq[..., 1:-1] + log_eq[..., :-2]
    return (d2 * d2).mean()
