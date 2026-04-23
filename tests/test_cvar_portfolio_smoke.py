"""Smoke tests for cvar_portfolio — ensure the LP solves on both backends
and the rolling backtest runs end-to-end on a toy panel."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from cvar_portfolio.backtest import run_backtest
from cvar_portfolio.optimize import solve_cvar_portfolio


def _make_toy_prices(n_days: int = 400, n_assets: int = 8, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    mu = rng.normal(0.0003, 0.0006, n_assets)
    sig = rng.uniform(0.01, 0.025, n_assets)
    shocks = rng.normal(mu, sig, size=(n_days, n_assets))
    prices = 100.0 * np.exp(np.cumsum(shocks, axis=0))
    idx = pd.date_range("2023-01-01", periods=n_days, freq="B", tz="UTC")
    cols = [f"S{i:02d}" for i in range(n_assets)]
    return pd.DataFrame(prices, index=idx, columns=cols)


def test_solve_cvxpy_smoke():
    prices = _make_toy_prices()
    returns = (np.log(prices) - np.log(prices.shift(1))).dropna()
    res = solve_cvar_portfolio(
        returns, num_scen=500, fit_type="gaussian", w_max=0.4, L_tar=1.0, api="cvxpy", rng_seed=1
    )
    assert res.weights.shape == (8,)
    assert abs(res.weights.sum() + res.cash - 1.0) < 1e-3
    assert np.all(res.weights >= -1e-6)


def test_solve_cuopt_smoke():
    pytest.importorskip("cuopt")
    prices = _make_toy_prices()
    returns = (np.log(prices) - np.log(prices.shift(1))).dropna()
    res = solve_cvar_portfolio(
        returns, num_scen=500, fit_type="gaussian", w_max=0.4, L_tar=1.0, api="cuopt_python", rng_seed=2
    )
    assert res.weights.shape == (8,)
    assert abs(res.weights.sum() + res.cash - 1.0) < 1e-3


def test_mean_override_changes_weights():
    prices = _make_toy_prices()
    returns = (np.log(prices) - np.log(prices.shift(1))).dropna()
    base = solve_cvar_portfolio(returns, num_scen=500, fit_type="gaussian", w_max=0.4, L_tar=1.0, rng_seed=3)
    alpha = np.zeros(8)
    alpha[0] = 0.01  # strong positive prior on asset 0
    biased = solve_cvar_portfolio(
        returns, num_scen=500, fit_type="gaussian", w_max=0.4, L_tar=1.0, mean_override=alpha, rng_seed=3
    )
    # biased optimizer should allocate more weight to asset 0 than base
    assert biased.weights[0] >= base.weights[0] - 1e-6
    # and specifically it should hit the per-asset cap
    assert biased.weights[0] > 0.3


def test_backtest_runs_end_to_end():
    prices = _make_toy_prices(n_days=500, n_assets=6)
    result = run_backtest(
        prices,
        fit_window=200,
        hold_days=20,
        num_scen=500,
        fit_type="gaussian",
        w_max=0.4,
        L_tar=1.0,
        cardinality=None,
        api="cvxpy",
        rng_seed=7,
    )
    assert result.summary["rebalances"] >= 10
    assert result.summary["n_days"] > 0
    assert np.isfinite(result.summary["mean_daily_log_ret"])
    # weights_history one column per asset + _cash
    assert "_cash" in result.weights_history.columns
    assert result.weights_history.shape[0] == result.summary["rebalances"]
