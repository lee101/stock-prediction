"""Differentiable Kelly + CVaR portfolio optimizer (PyTorch, GPU).

Maximizes expected log-growth E[log(1 + w·r)] with a soft CVaR penalty on
tail losses, subject to box bounds 0 ≤ w ≤ w_max and leverage cap
Σw ≤ L_tar. Solves via projected Adam on GPU — intended for the 30%/month
aggressive regime where the LP's linear objective and cash-constraint
budget are too conservative.

Intuition: Kelly objective is long-horizon wealth maximization → it will
concentrate aggressively in high-μ / low-σ assets and ignore diversification
that the LP's linear objective cannot reward. The CVaR penalty controls
tail risk without killing the geometric growth term.

Usage mirrors ``solve_cvar_portfolio`` so ``run_backtest`` can switch
backends via ``api="pytorch_kelly"``.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
import torch

from .optimize import OptimResult


@dataclass
class KellyConfig:
    w_max: float = 1.0
    L_tar: float = 4.0
    cvar_alpha: float = 0.95
    cvar_penalty: float = 1.0
    lr: float = 0.01
    steps: int = 2000
    l2_reg: float = 0.0          # soft ridge on weights (diversification nudge)
    turnover_penalty: float = 0.0  # penalise |w - w_prev| to reduce churn / fees
    warm_start: bool = False       # initialise from previous weights when given
    device: Optional[str] = None  # default "cuda" if available


def _pick_device(device: Optional[str]) -> torch.device:
    if device is not None:
        return torch.device(device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def differentiable_kelly_cvar(
    samples: np.ndarray,
    *,
    w_max: float = 1.0,
    L_tar: float = 4.0,
    cvar_alpha: float = 0.95,
    cvar_penalty: float = 1.0,
    lr: float = 0.01,
    steps: int = 2000,
    l2_reg: float = 0.0,
    turnover_penalty: float = 0.0,
    prev_weights: Optional[np.ndarray] = None,
    warm_start: bool = False,
    device: Optional[str] = None,
    rng_seed: int = 0,
    mean_override: Optional[np.ndarray] = None,
) -> tuple[np.ndarray, dict]:
    """Solve Kelly + CVaR portfolio on a scenario matrix.

    Parameters
    ----------
    samples : np.ndarray of shape (num_scen, n_assets)
        Per-scenario log returns per asset (one period — e.g. daily).
    mean_override : optional np.ndarray of shape (n_assets,)
        If provided, shifts the empirical per-asset mean log-return of
        ``samples`` to match this vector (adds ``override − emp_mean``
        to every scenario). Mirrors the ``mean_override`` hook in the
        LP solver so XGB/RL priors can be injected.

    Returns
    -------
    weights : np.ndarray of shape (n_assets,)
    diagnostics : dict with kelly, cvar, loss, steps_used, solve_time
    """
    dev = _pick_device(device)
    torch.manual_seed(int(rng_seed))
    n_scen, n_assets = samples.shape
    X = torch.as_tensor(samples, dtype=torch.float32, device=dev)
    if mean_override is not None:
        emp_mean = X.mean(dim=0)
        delta = torch.as_tensor(mean_override, dtype=torch.float32, device=dev) - emp_mean
        X = X + delta  # broadcast per-asset shift to every scenario

    # Init from previous weights when requested; otherwise spread leverage
    # uniformly. Warm-starting the projected optimizer materially reduces
    # needless churn across adjacent rebalance windows.
    if warm_start and prev_weights is not None:
        prev = np.asarray(prev_weights, dtype=np.float32)
        if prev.shape != (n_assets,):
            raise ValueError(f"prev_weights shape {prev.shape} != ({n_assets},)")
        prev = np.clip(prev, 0.0, float(w_max))
        prev_sum = float(prev.sum())
        if prev_sum > float(L_tar) and prev_sum > 0.0:
            prev = prev * (float(L_tar) / prev_sum)
        w0 = torch.as_tensor(prev, dtype=torch.float32, device=dev)
    else:
        w_init = float(min(w_max, max(L_tar / n_assets, 1e-4)))
        w0 = torch.full((n_assets,), w_init, device=dev)
    w = w0.clone().detach().requires_grad_(True)
    opt = torch.optim.Adam([w], lr=lr)
    prev_w_t = (
        torch.as_tensor(prev_weights, dtype=torch.float32, device=dev)
        if prev_weights is not None
        else None
    )

    # How many scenarios contribute to CVaR?
    k = max(1, int((1.0 - float(cvar_alpha)) * n_scen))

    t0 = time.time()
    last_kelly = 0.0
    last_cvar = 0.0
    last_turnover = 0.0
    for step in range(steps):
        opt.zero_grad()
        port_ret = X @ w                                     # shape (n_scen,)
        wealth = 1.0 + port_ret                              # linearised next-period wealth
        # Numerical floor: avoid log(≤0) from pathological scenarios × weights.
        wealth_safe = torch.clamp(wealth, min=1e-6)
        kelly = torch.log(wealth_safe).mean()
        worst_k = torch.topk(-port_ret, k).values            # worst losses (positive)
        cvar_val = worst_k.mean()
        l2 = (w * w).sum() if l2_reg > 0 else torch.zeros((), device=dev)
        turnover = (
            (w - prev_w_t).abs().sum()
            if prev_w_t is not None and turnover_penalty > 0.0
            else torch.zeros((), device=dev)
        )
        loss = (
            -kelly
            + cvar_penalty * cvar_val
            + l2_reg * l2
            + turnover_penalty * turnover
        )
        loss.backward()
        opt.step()
        with torch.no_grad():
            # Box project: 0 ≤ w ≤ w_max, then leverage cap Σw ≤ L_tar.
            w.clamp_(0.0, float(w_max))
            s = w.sum()
            if float(s) > float(L_tar):
                w.mul_(float(L_tar) / float(s))
        last_kelly = float(kelly.detach().cpu())
        last_cvar = float(cvar_val.detach().cpu())
        last_turnover = float(turnover.detach().cpu())
    solve_time = time.time() - t0

    weights = w.detach().cpu().numpy().astype(np.float64)
    # Hard-zero very small weights to clean up post-projection noise.
    weights = np.where(weights < 1e-6, 0.0, weights)
    diag = {
        "kelly": last_kelly,
        "cvar": last_cvar,
        "loss": -last_kelly + float(cvar_penalty) * last_cvar,
        "steps_used": int(steps),
        "solve_time": solve_time,
        "l_realized": float(weights.sum()),
        "turnover_vs_prev": last_turnover,
    }
    return weights, diag


def solve_kelly_cvar_portfolio(
    returns: pd.DataFrame,
    *,
    mean_override: Optional[np.ndarray] = None,
    num_scen: int = 2500,
    fit_type: str = "gaussian",
    kde_device: str = "CPU",  # unused here; kept for API parity
    kde_bandwidth: float = 0.01,
    risk_aversion: float = 1.0,  # alias for cvar_penalty when used via run_backtest
    confidence: float = 0.95,
    w_max: float = 1.0,
    w_min: float = 0.0,  # unused for Kelly (projection enforces 0 floor)
    c_min: float = 0.0,  # unused
    c_max: float = 1.0,  # unused
    L_tar: float = 4.0,
    cardinality: Optional[int] = None,  # unused
    api: str = "pytorch_kelly",
    rng_seed: int = 0,
    # Kelly-specific knobs (can be overridden by caller):
    kelly_lr: float = 0.01,
    kelly_steps: int = 1500,
    kelly_l2_reg: float = 0.0,
    kelly_turnover_penalty: float = 0.0,
    kelly_prev_weights: Optional[np.ndarray] = None,
    kelly_warm_start: bool = False,
    kelly_device: Optional[str] = None,
) -> OptimResult:
    """API-compatible wrapper so ``run_backtest`` can switch solvers.

    Generates scenarios the same way as the LP path, then solves with
    ``differentiable_kelly_cvar``. ``risk_aversion`` maps to the CVaR
    penalty weight.
    """
    rng = np.random.default_rng(int(rng_seed))
    emp_mean = returns.mean(axis=0).values
    if fit_type == "gaussian":
        cov = np.cov(returns.values.T)
        samples = rng.multivariate_normal(emp_mean, cov, size=int(num_scen))
    elif fit_type == "historical":
        idx = rng.integers(0, len(returns), size=int(num_scen))
        samples = returns.values[idx]
    elif fit_type == "kde":
        from sklearn.neighbors import KernelDensity
        kde = KernelDensity(kernel="gaussian", bandwidth=float(kde_bandwidth)).fit(returns.values)
        samples = kde.sample(int(num_scen))
    else:
        raise ValueError(f"Unknown fit_type: {fit_type}")

    weights, diag = differentiable_kelly_cvar(
        samples.astype(np.float32),
        w_max=float(w_max),
        L_tar=float(L_tar),
        cvar_alpha=float(confidence),
        cvar_penalty=float(risk_aversion),
        lr=float(kelly_lr),
        steps=int(kelly_steps),
        l2_reg=float(kelly_l2_reg),
        turnover_penalty=float(kelly_turnover_penalty),
        prev_weights=kelly_prev_weights,
        warm_start=bool(kelly_warm_start),
        device=kelly_device,
        rng_seed=int(rng_seed),
        mean_override=mean_override,
    )
    expected_return = float(samples.mean(axis=0) @ weights)
    tickers = list(returns.columns)
    cash = max(0.0, 1.0 - float(weights.sum()))
    return OptimResult(
        weights=weights,
        cash=cash,
        tickers=tickers,
        solve_time=diag["solve_time"],
        objective=diag["loss"],
        expected_return=expected_return,
        cvar=diag["cvar"],
        solver="pytorch_kelly",
    )


__all__ = [
    "KellyConfig",
    "differentiable_kelly_cvar",
    "solve_kelly_cvar_portfolio",
]
