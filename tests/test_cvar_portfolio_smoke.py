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


def test_alpha_fn_injection_into_run_backtest():
    """alpha_fn is called each rebalance and its output becomes the LP's μ."""
    prices = _make_toy_prices(n_days=500, n_assets=6)

    calls: list[tuple] = []

    def alpha_fn(asof, tickers):
        calls.append((pd.Timestamp(asof), tuple(tickers)))
        # strongly favour asset 0 every rebalance
        v = np.zeros(len(tickers), dtype=np.float32)
        v[0] = 0.01
        return v

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
        alpha_fn=alpha_fn,
        rng_seed=11,
    )
    assert len(calls) == result.summary["rebalances"]
    # tickers match panel columns each call
    for _, tick in calls:
        assert tick == tuple(prices.columns)
    # asset 0 should dominate in most rebalances thanks to the forced alpha
    s0_col = prices.columns[0]
    assert (result.weights_history[s0_col] > 0.2).mean() > 0.5


def test_make_alpha_fn_returns_zero_on_missing_day():
    from cvar_portfolio.xgb_alpha import make_alpha_fn

    panel = pd.DataFrame({
        "date": [pd.Timestamp("2025-01-05"), pd.Timestamp("2025-01-05")],
        "symbol": ["AAA", "BBB"],
        "ensemble_score": [0.6, 0.4],
    })
    fn = make_alpha_fn(panel, k=0.01, mode="center")

    # known day matches scores
    v = fn(pd.Timestamp("2025-01-05"), ["AAA", "BBB", "CCC"])
    assert v[0] == pytest.approx(0.01 * (0.6 - 0.5))
    assert v[1] == pytest.approx(0.01 * (0.4 - 0.5))
    assert v[2] == 0.0  # missing symbol → neutral

    # distant day (>5 biz-day gap) → all zeros
    v_gone = fn(pd.Timestamp("2025-02-01"), ["AAA", "BBB"])
    assert np.all(v_gone == 0.0)


def test_make_alpha_fn_rank_mode():
    from cvar_portfolio.xgb_alpha import make_alpha_fn

    panel = pd.DataFrame({
        "date": [pd.Timestamp("2025-01-05")] * 4,
        "symbol": ["A", "B", "C", "D"],
        "ensemble_score": [0.60, 0.55, 0.48, 0.40],
    })
    fn = make_alpha_fn(panel, k=0.02, mode="rank")
    v = fn(pd.Timestamp("2025-01-05"), ["A", "B", "C", "D"])
    # A has rank 1.0 → μ = 0.02*0.5, D has rank 0.25 → μ = 0.02*(0.25-0.5)
    assert v[0] > v[1] > v[2] > v[3]
    assert v[0] == pytest.approx(0.02 * (1.0 - 0.5))
    assert v[3] == pytest.approx(0.02 * (0.25 - 0.5))


def test_universe_fn_restricts_lp_to_selected_syms():
    """universe_fn returning a subset of tickers constrains the LP to those syms."""
    prices = _make_toy_prices(n_days=500, n_assets=8)
    selected = list(prices.columns[:3])  # only first 3 can receive weight

    def universe_fn(asof, tickers):
        return selected

    result = run_backtest(
        prices,
        fit_window=200,
        hold_days=20,
        num_scen=400,
        fit_type="gaussian",
        w_max=0.5,
        L_tar=1.0,
        cardinality=None,
        api="cvxpy",
        universe_fn=universe_fn,
        rng_seed=17,
    )
    assert result.summary["rebalances"] > 0
    # excluded tickers must carry exactly zero weight every rebalance
    for t in prices.columns[3:]:
        assert (result.weights_history[t].abs() < 1e-9).all(), f"{t} nonzero"
    # at least one selected ticker holds weight somewhere
    assert (result.weights_history[selected].abs() > 1e-6).any().any()


def test_universe_fn_empty_holds_cash():
    """If universe_fn returns <2 symbols, that rebalance holds cash (no crash)."""
    prices = _make_toy_prices(n_days=500, n_assets=6)

    def universe_fn(asof, tickers):
        return []  # always empty

    result = run_backtest(
        prices,
        fit_window=200,
        hold_days=20,
        num_scen=400,
        fit_type="gaussian",
        w_max=0.4,
        L_tar=1.0,
        api="cvxpy",
        universe_fn=universe_fn,
        rng_seed=19,
    )
    # All rebalances should allocate zero weight → daily portfolio returns ≈ 0
    assert (result.portfolio_returns.abs() < 1e-9).all()
    # Weights history cash column should be 1.0 every rebalance
    assert (result.weights_history["_cash"] > 0.99).all()


def test_make_universe_fn_top_k_selection():
    from cvar_portfolio.xgb_alpha import make_universe_fn

    panel = pd.DataFrame({
        "date": [pd.Timestamp("2025-01-05")] * 4,
        "symbol": ["A", "B", "C", "D"],
        "ensemble_score": [0.70, 0.65, 0.40, 0.55],
    })
    fn = make_universe_fn(panel, top_k=2, min_score=0.0)
    out = fn(pd.Timestamp("2025-01-05"), ["A", "B", "C", "D"])
    assert out == ["A", "B"]

    # min_score filter
    fn2 = make_universe_fn(panel, top_k=4, min_score=0.60)
    assert fn2(pd.Timestamp("2025-01-05"), ["A", "B", "C", "D"]) == ["A", "B"]

    # missing day → empty list (caller holds cash)
    assert fn(pd.Timestamp("2025-03-01"), ["A", "B"]) == []


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
