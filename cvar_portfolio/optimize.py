"""Thin wrapper around cufolio.CVaR that fits our data pipeline.

Builds a CvarData from a returns matrix (KDE or Gaussian scenarios), optionally
overrides the empirical `mean` with an external expected-return vector (our
XGB/RL alpha prior), then solves the Mean-CVaR LP using CVXPY+Clarabel (CPU)
or cuOpt (GPU). Returns the optimal weight vector + diagnostics.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Optional, TYPE_CHECKING

import cvxpy as cp
import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from cufolio.cvar_data import CvarData
else:
    CvarData = Any


@dataclass
class OptimResult:
    weights: np.ndarray
    cash: float
    tickers: list[str]
    solve_time: float
    objective: float
    expected_return: float
    cvar: float
    solver: str


def _require_cufolio() -> tuple[type[Any], type[Any], type[Any]]:
    try:
        from cufolio.cvar_data import CvarData as _CvarData
        from cufolio.cvar_optimizer import CVaR as _CVaR
        from cufolio.cvar_parameters import CvarParameters as _CvarParameters
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "cufolio is required for Mean-CVaR LP solves. "
            "Install the NVIDIA cufolio package to use "
            "solve_cvar_portfolio(api='cvxpy'|'cuopt_python')."
        ) from exc
    return _CvarData, _CVaR, _CvarParameters


def _build_scenarios(
    returns: pd.DataFrame,
    num_scen: int,
    fit_type: str,
    kde_device: str,
    bandwidth: float,
    rng: np.random.Generator,
) -> CvarData:
    """Return CvarData populated with scenario matrix R of shape (n_assets, num_scen)."""
    cvar_data_cls, _, _ = _require_cufolio()
    mean = returns.mean(axis=0).values  # daily log-return mean per asset
    if fit_type == "gaussian":
        cov = np.cov(returns.values.T)
        samples = rng.multivariate_normal(mean, cov, size=num_scen)
    elif fit_type == "kde":
        if kde_device.upper() == "GPU":
            import cuml.neighbors

            kde = cuml.neighbors.KernelDensity(kernel="gaussian", bandwidth=bandwidth).fit(returns.values)
            samples = kde.sample(num_scen).get()
        else:
            from sklearn.neighbors import KernelDensity

            kde = KernelDensity(kernel="gaussian", bandwidth=bandwidth).fit(returns.values)
            samples = kde.sample(num_scen)
    elif fit_type == "historical":
        # Bootstrap rows (with replacement) to num_scen
        idx = rng.integers(0, len(returns), size=num_scen)
        samples = returns.values[idx]
    else:
        raise ValueError(f"Unknown fit_type: {fit_type}")
    R = samples.T  # (n_assets, num_scen)
    p = np.ones(num_scen) / num_scen
    return cvar_data_cls(mean=mean, R=R, p=p)


def solve_cvar_portfolio(
    returns: pd.DataFrame,
    *,
    mean_override: Optional[np.ndarray] = None,
    num_scen: int = 5000,
    fit_type: str = "kde",
    kde_device: str = "CPU",
    kde_bandwidth: float = 0.01,
    risk_aversion: float = 1.0,
    confidence: float = 0.95,
    w_max: float = 0.10,
    w_min: float = 0.0,
    c_min: float = 0.0,
    c_max: float = 1.0,
    L_tar: float = 1.0,
    cardinality: Optional[int] = None,
    api: str = "cvxpy",
    rng_seed: int = 0,
) -> OptimResult:
    """Solve a single Mean-CVaR optimization on the given returns window.

    Parameters
    ----------
    returns : pd.DataFrame (T_window x n_assets) of log returns
    mean_override : optional expected-return vector (n_assets,) to replace empirical mean.
        Use this to inject an XGB / RL alpha prior.
    w_max : per-asset cap (e.g. 0.10 → no asset > 10% of equity).
    L_tar : leverage cap on ‖w‖₁.
    cardinality : if set, MILP picks at most this many non-zero weights.
    api : "cvxpy" (Clarabel CPU) or "cuopt_python" (GPU LP/MILP).
    """
    if returns.isna().any().any():
        returns = returns.fillna(0.0)
    cvar_data_cls, cvar_cls, cvar_params_cls = _require_cufolio()
    rng = np.random.default_rng(rng_seed)
    n = returns.shape[1]
    tickers = returns.columns.tolist()
    cvar_data = _build_scenarios(returns, num_scen, fit_type, kde_device, kde_bandwidth, rng)
    if mean_override is not None:
        mo = np.asarray(mean_override, dtype=float)
        if mo.shape != (n,):
            raise ValueError(f"mean_override shape {mo.shape} != ({n},)")
        cvar_data = cvar_data_cls(mean=mo, R=cvar_data.R, p=cvar_data.p)

    returns_dict = {
        "tickers": tickers,
        "regime": {"name": "window", "range": (str(returns.index[0]), str(returns.index[-1]))},
        "covariance": np.cov(returns.values.T),
        "cvar_data": cvar_data,
    }
    params = cvar_params_cls(
        w_min=float(w_min),
        w_max=float(w_max),
        c_min=float(c_min),
        c_max=float(c_max),
        risk_aversion=float(risk_aversion),
        confidence=float(confidence),
        L_tar=float(L_tar),
        cardinality=cardinality,
    )

    if api == "cvxpy":
        api_settings = {"api": "cvxpy", "weight_constraints_type": "bounds", "cash_constraints_type": "bounds"}
    elif api == "cuopt_python":
        api_settings = {"api": "cuopt_python"}
    else:
        raise ValueError(f"Unknown api: {api}")

    t0 = time.time()
    opt = cvar_cls(returns_dict, params, api_settings=api_settings)
    if api == "cvxpy":
        solve_kwargs = {"solver": cp.CLARABEL}
    else:
        solve_kwargs = {"time_limit": 60, "log_to_console": False}
    result_df, pf = opt.solve_optimization_problem(solve_kwargs, print_results=False)
    t1 = time.time()

    w = np.asarray(pf.weights, dtype=float)
    cash = float(pf.cash)
    exp_ret = float(np.dot(cvar_data.mean, w))
    # CVaR ex-post on fitted scenarios
    losses = -(cvar_data.R.T @ w)
    var = np.quantile(losses, confidence)
    tail = losses[losses >= var]
    cvar_val = float(tail.mean()) if len(tail) else float(var)
    obj = float(risk_aversion * cvar_val - exp_ret)
    return OptimResult(
        weights=w,
        cash=cash,
        tickers=tickers,
        solve_time=t1 - t0,
        objective=obj,
        expected_return=exp_ret,
        cvar=cvar_val,
        solver=api,
    )
