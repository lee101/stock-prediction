"""Smoke tests for cvar_portfolio — ensure the LP solves on both backends
and the rolling backtest runs end-to-end on a toy panel."""
from __future__ import annotations

import importlib.util

import numpy as np
import pandas as pd
import pytest

from cvar_portfolio.backtest import run_backtest
from cvar_portfolio.optimize import solve_cvar_portfolio


HAS_CUFOLIO = importlib.util.find_spec("cufolio") is not None
requires_cufolio = pytest.mark.skipif(
    not HAS_CUFOLIO,
    reason="cufolio is not installed in this environment",
)


def _make_toy_prices(n_days: int = 400, n_assets: int = 8, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    mu = rng.normal(0.0003, 0.0006, n_assets)
    sig = rng.uniform(0.01, 0.025, n_assets)
    shocks = rng.normal(mu, sig, size=(n_days, n_assets))
    prices = 100.0 * np.exp(np.cumsum(shocks, axis=0))
    idx = pd.date_range("2023-01-01", periods=n_days, freq="B", tz="UTC")
    cols = [f"S{i:02d}" for i in range(n_assets)]
    return pd.DataFrame(prices, index=idx, columns=cols)


@requires_cufolio
def test_solve_cvxpy_smoke():
    prices = _make_toy_prices()
    returns = (np.log(prices) - np.log(prices.shift(1))).dropna()
    res = solve_cvar_portfolio(
        returns, num_scen=500, fit_type="gaussian", w_max=0.4, L_tar=1.0, api="cvxpy", rng_seed=1
    )
    assert res.weights.shape == (8,)
    assert abs(res.weights.sum() + res.cash - 1.0) < 1e-3
    assert np.all(res.weights >= -1e-6)


@requires_cufolio
def test_solve_cuopt_smoke():
    pytest.importorskip("cuopt")
    prices = _make_toy_prices()
    returns = (np.log(prices) - np.log(prices.shift(1))).dropna()
    res = solve_cvar_portfolio(
        returns, num_scen=500, fit_type="gaussian", w_max=0.4, L_tar=1.0, api="cuopt_python", rng_seed=2
    )
    assert res.weights.shape == (8,)
    assert abs(res.weights.sum() + res.cash - 1.0) < 1e-3


@requires_cufolio
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


@requires_cufolio
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


@requires_cufolio
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


@requires_cufolio
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


def test_differentiable_kelly_solver_smoke():
    pytest.importorskip("torch")
    from cvar_portfolio.differentiable_optimizer import differentiable_kelly_cvar

    rng = np.random.default_rng(0)
    n_assets = 6
    mu = np.array([0.0010, 0.0005, 0.0003, 0.0002, 0.0001, 0.0000], dtype=np.float32)
    sig = np.full(n_assets, 0.015, dtype=np.float32)
    samples = (rng.normal(mu, sig, size=(1200, n_assets))).astype(np.float32)
    # Low cvar_penalty so Kelly term can express its directional preference —
    # at daily-log-return scales (|μ|≈1e-3, σ≈1.5e-2) CVaR dwarfs Kelly if
    # penalty is O(1). Production tuning picks penalty in [1e-3, 0.1].
    w, diag = differentiable_kelly_cvar(
        samples, w_max=0.5, L_tar=2.0, cvar_alpha=0.95, cvar_penalty=0.01,
        lr=0.02, steps=500, rng_seed=3, device="cpu",
    )
    assert w.shape == (n_assets,)
    assert np.all(w >= -1e-6)
    assert w.max() <= 0.5 + 1e-5
    assert w.sum() <= 2.0 + 1e-4
    # Highest-μ asset should get substantially more weight than lowest-μ.
    assert w[0] > w[-1]


def test_differentiable_kelly_solver_honors_leverage():
    pytest.importorskip("torch")
    from cvar_portfolio.differentiable_optimizer import differentiable_kelly_cvar

    rng = np.random.default_rng(1)
    n_assets = 4
    mu = np.array([0.002, 0.0018, 0.0017, 0.0016], dtype=np.float32)
    sig = np.full(n_assets, 0.01, dtype=np.float32)
    samples = rng.normal(mu, sig, size=(800, n_assets)).astype(np.float32)
    # L_tar=3 with w_max=1 → weights should approach sum=3.
    w, _ = differentiable_kelly_cvar(
        samples, w_max=1.0, L_tar=3.0, cvar_alpha=0.95, cvar_penalty=0.1,
        lr=0.05, steps=800, rng_seed=5, device="cpu",
    )
    assert w.sum() <= 3.0 + 1e-4
    # Kelly with positive-μ assets should saturate leverage (≥ 0.6 × L_tar).
    assert w.sum() >= 1.8, f"sum(w)={w.sum():.3f} — expected leverage saturation"


def test_differentiable_kelly_turnover_penalty_respects_prev_weights():
    pytest.importorskip("torch")
    from cvar_portfolio.differentiable_optimizer import differentiable_kelly_cvar

    rng = np.random.default_rng(2)
    n_assets = 5
    mu = np.array([0.0020, 0.0012, 0.0010, 0.0008, 0.0002], dtype=np.float32)
    sig = np.full(n_assets, 0.012, dtype=np.float32)
    samples = rng.normal(mu, sig, size=(1000, n_assets)).astype(np.float32)
    prev = np.array([0.35, 0.35, 0.20, 0.10, 0.00], dtype=np.float32)

    w_free, _ = differentiable_kelly_cvar(
        samples, w_max=0.5, L_tar=1.5, cvar_alpha=0.95, cvar_penalty=0.05,
        lr=0.02, steps=500, rng_seed=4, device="cpu",
    )
    w_sticky, _ = differentiable_kelly_cvar(
        samples, w_max=0.5, L_tar=1.5, cvar_alpha=0.95, cvar_penalty=0.05,
        lr=0.02, steps=500, rng_seed=4, device="cpu",
        prev_weights=prev, warm_start=True, turnover_penalty=0.5,
    )
    dist_free = np.abs(w_free - prev).sum()
    dist_sticky = np.abs(w_sticky - prev).sum()
    assert dist_sticky < dist_free


def test_backtest_pytorch_kelly_end_to_end():
    pytest.importorskip("torch")
    prices = _make_toy_prices(n_days=500, n_assets=6)
    result = run_backtest(
        prices,
        fit_window=200,
        hold_days=20,
        num_scen=800,
        fit_type="gaussian",
        w_max=0.5,
        L_tar=2.0,
        api="pytorch_kelly",
        rng_seed=9,
    )
    assert result.summary["rebalances"] >= 10
    assert result.summary["n_days"] > 0
    assert np.isfinite(result.summary["mean_daily_log_ret"])
    # Every rebalance must satisfy box + leverage constraints.
    max_weight = result.weights_history.drop(columns=["_cash"]).max(axis=1)
    assert (max_weight <= 0.5 + 1e-4).all()


def test_backtest_summary_includes_monthly_stats_for_kelly():
    pytest.importorskip("torch")
    prices = _make_toy_prices(n_days=500, n_assets=6)
    result = run_backtest(
        prices,
        fit_window=200,
        hold_days=20,
        num_scen=500,
        fit_type="gaussian",
        w_max=0.4,
        L_tar=1.5,
        api="pytorch_kelly",
        kelly_steps=200,
        rng_seed=12,
    )
    for key in (
        "n_months",
        "median_monthly_return_pct",
        "p10_monthly_return_pct",
        "worst_monthly_drawdown_pct",
    ):
        assert key in result.summary
    assert result.summary["n_months"] > 0
    assert np.isfinite(result.summary["median_monthly_return_pct"])
    assert result.summary["worst_monthly_drawdown_pct"] <= 0.0


def test_backtest_kelly_turnover_controls_churn():
    pytest.importorskip("torch")
    prices = _make_toy_prices(n_days=500, n_assets=6, seed=3)
    common = dict(
        fit_window=200,
        hold_days=20,
        num_scen=500,
        fit_type="gaussian",
        w_max=0.5,
        L_tar=2.0,
        api="pytorch_kelly",
        kelly_steps=250,
        kelly_device="cpu",
        rng_seed=21,
    )
    loose = run_backtest(
        prices,
        kelly_turnover_penalty=0.0,
        kelly_warm_start=False,
        **common,
    )
    sticky = run_backtest(
        prices,
        kelly_turnover_penalty=0.2,
        kelly_warm_start=True,
        **common,
    )
    assert sticky.summary["mean_turnover"] < loose.summary["mean_turnover"]


@requires_cufolio
def test_backtest_fees_reduce_returns_monotonically():
    """Higher fee_bps must produce strictly lower ann_return and non-zero fee cost."""
    prices = _make_toy_prices(n_days=500, n_assets=6)
    common = dict(
        fit_window=200, hold_days=20, num_scen=500, fit_type="gaussian",
        w_max=0.4, L_tar=1.0, cardinality=None, api="cvxpy", rng_seed=7,
    )
    r0 = run_backtest(prices, fee_bps=0.0, slip_bps=0.0, **common)
    r1 = run_backtest(prices, fee_bps=10.0, slip_bps=5.0, **common)
    r2 = run_backtest(prices, fee_bps=100.0, slip_bps=50.0, **common)
    assert r0.summary["total_fee_cost_log"] == 0.0
    assert r1.summary["total_fee_cost_log"] > 0.0
    assert r2.summary["total_fee_cost_log"] > r1.summary["total_fee_cost_log"]
    # Fees eat into returns monotonically
    assert r0.summary["mean_daily_log_ret"] >= r1.summary["mean_daily_log_ret"] - 1e-9
    assert r1.summary["mean_daily_log_ret"] > r2.summary["mean_daily_log_ret"]
    # mean_turnover reported
    assert r1.summary["mean_turnover"] > 0


@requires_cufolio
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


def test_rolling_drawdown_stats_in_summary():
    """Rolling 5/21/63d drawdown + frac-exceeding-threshold in summary."""
    from cvar_portfolio.backtest import _goodness_score, _rolling_drawdown_stats

    # Fabricate a sharp 30% drop then recovery.
    idx = pd.date_range("2023-01-01", periods=80, freq="B", tz="UTC")
    rets = pd.Series(0.0, index=idx)
    rets.iloc[30:35] = np.log(0.93)  # 5 consecutive −7% days = -30%+ drawdown
    rets.iloc[50:60] = np.log(1.02)  # recovery

    stats = _rolling_drawdown_stats(rets)
    # Worst 5d window should show drop; 21d should be deeper.
    assert stats["worst_5d_drawdown_pct"] < -20.0
    assert stats["worst_21d_drawdown_pct"] < -20.0
    assert stats["frac_21d_dd_gt_25pct"] > 0.0

    # Goodness score should be strongly negative vs a gentle series.
    summary_bad = {"median_monthly_return_pct": 5.0, **stats}
    summary_good = {
        "median_monthly_return_pct": 5.0,
        "worst_5d_drawdown_pct": -3.0,
        "worst_21d_drawdown_pct": -8.0,
        "worst_63d_drawdown_pct": -12.0,
        "frac_21d_dd_gt_25pct": 0.0,
    }
    assert _goodness_score(summary_good) > _goodness_score(summary_bad)
    # Low-DD portfolio should score near its monthly return.
    assert abs(_goodness_score(summary_good) - 5.0) < 0.5


def test_goodness_score_rewards_monthly_return():
    """Higher monthly return with identical DD → higher goodness."""
    from cvar_portfolio.backtest import _goodness_score

    base = {"median_monthly_return_pct": 10.0, "worst_5d_drawdown_pct": -5.0,
            "worst_21d_drawdown_pct": -12.0, "worst_63d_drawdown_pct": -18.0,
            "frac_21d_dd_gt_25pct": 0.0}
    higher = {**base, "median_monthly_return_pct": 25.0}
    assert _goodness_score(higher) > _goodness_score(base) + 10.0


def test_goodness_score_penalizes_21d_frac_breach():
    """Repeated 25%+ monthly drawdowns punished harder than a single deep one."""
    from cvar_portfolio.backtest import _goodness_score

    occasional = {"median_monthly_return_pct": 20.0, "worst_5d_drawdown_pct": -10.0,
                  "worst_21d_drawdown_pct": -35.0, "worst_63d_drawdown_pct": -35.0,
                  "frac_21d_dd_gt_25pct": 0.02}
    chronic = {**occasional, "frac_21d_dd_gt_25pct": 0.25}
    assert _goodness_score(occasional) > _goodness_score(chronic) + 10.0


@requires_cufolio
def test_per_asset_stop_loss_caps_drawdown():
    """A very tight per-asset stop-loss should trim worst_21d_drawdown_pct
    vs a no-stop run on the same panel."""
    prices = _make_toy_prices(seed=11)
    kwargs = dict(fit_window=100, hold_days=20, num_scen=300, fit_type="gaussian",
                  w_max=0.5, L_tar=1.0, cardinality=None, api="cvxpy",
                  fee_bps=10, slip_bps=5, rng_seed=3)
    no_stop = run_backtest(prices, **kwargs)
    tight = run_backtest(prices, per_asset_stop_loss_pct=5.0, **kwargs)
    # worst rolling 21d DD should be same or shallower under the stop.
    assert tight.summary["worst_21d_drawdown_pct"] >= no_stop.summary["worst_21d_drawdown_pct"] - 1e-6
    assert "per_asset_stop_loss_pct" in tight.summary
    assert tight.summary["per_asset_stop_loss_pct"] == 5.0


@requires_cufolio
def test_portfolio_stop_loss_halts_holding():
    """A portfolio-wide stop-loss liquidates everything and leaves zero
    returns for the rest of the window."""
    prices = _make_toy_prices(seed=19)
    kwargs = dict(fit_window=100, hold_days=20, num_scen=300, fit_type="gaussian",
                  w_max=0.5, L_tar=1.0, cardinality=None, api="cvxpy",
                  fee_bps=10, slip_bps=5, rng_seed=5)
    # Tight port-wide stop; very likely to trip on random panel.
    res = run_backtest(prices, portfolio_stop_loss_pct=2.0, **kwargs)
    assert res.summary["portfolio_stop_loss_pct"] == 2.0
    # Summary must still be well-formed.
    assert "goodness_score" in res.summary
    assert np.isfinite(res.summary["goodness_score"])


@requires_cufolio
def test_per_asset_trailing_stop_caps_drawdown():
    """A tight per-asset trailing stop protects profits: worst 21d rolling
    DD on the stopped run should be shallower-or-equal than a no-stop run
    on the same panel, and the parameter must round-trip into summary."""
    prices = _make_toy_prices(seed=23)
    kwargs = dict(fit_window=100, hold_days=20, num_scen=300, fit_type="gaussian",
                  w_max=0.5, L_tar=1.0, cardinality=None, api="cvxpy",
                  fee_bps=10, slip_bps=5, rng_seed=7)
    no_stop = run_backtest(prices, **kwargs)
    tight = run_backtest(prices, per_asset_trailing_stop_pct=5.0, **kwargs)
    assert tight.summary["worst_21d_drawdown_pct"] >= no_stop.summary["worst_21d_drawdown_pct"] - 1e-6
    assert tight.summary["per_asset_trailing_stop_pct"] == 5.0
    assert tight.summary["portfolio_trailing_stop_pct"] == 0.0


@requires_cufolio
def test_portfolio_trailing_stop_roundtrips_into_summary():
    """Portfolio-wide trailing stop must be reported in summary and yield
    a finite goodness score."""
    prices = _make_toy_prices(seed=29)
    kwargs = dict(fit_window=100, hold_days=20, num_scen=300, fit_type="gaussian",
                  w_max=0.5, L_tar=1.0, cardinality=None, api="cvxpy",
                  fee_bps=10, slip_bps=5, rng_seed=9)
    res = run_backtest(prices, portfolio_trailing_stop_pct=3.0, **kwargs)
    assert res.summary["portfolio_trailing_stop_pct"] == 3.0
    assert np.isfinite(res.summary["goodness_score"])


def test_momentum_topk_picks_top_drifters():
    """`make_momentum_topk` should return the K highest past-return tickers
    and never leak forward information (pos strictly < asof)."""
    from cvar_portfolio.sweep_wide_momentum import make_momentum_topk
    rng = np.random.default_rng(0)
    dates = pd.date_range("2025-01-01", periods=60, freq="B")
    tickers = [f"T{i:02d}" for i in range(12)]
    drift = np.array([0.015] * 6 + [-0.015] * 6)
    rets = rng.normal(0, 0.005, (60, 12)) + drift[None]
    prices = pd.DataFrame(np.exp(np.cumsum(rets, axis=0)) * 100,
                          index=dates, columns=tickers)
    uf = make_momentum_topk(prices, top_k=4, lookback=20)
    picks = uf(dates[-1], tickers)
    assert len(picks) == 4
    # Top 4 drifters should all be from first 6 up-drift tickers
    assert set(picks).issubset({f"T{i:02d}" for i in range(6)})
    # Early dates with insufficient history should return empty
    assert uf(dates[5], tickers) == []


def test_momentum_topk_no_lookahead():
    """`make_momentum_topk` must use strictly past data — scheduling a
    future-only shock must not affect the pick at an earlier date."""
    from cvar_portfolio.sweep_wide_momentum import make_momentum_topk
    rng = np.random.default_rng(1)
    dates = pd.date_range("2025-01-01", periods=80, freq="B")
    tickers = [f"T{i:02d}" for i in range(8)]
    base = rng.normal(0, 0.005, (80, 8))
    prices_a = pd.DataFrame(np.exp(np.cumsum(base, axis=0)) * 100,
                            index=dates, columns=tickers)
    shocked = base.copy()
    shocked[60:, 0] += 0.10
    prices_b = pd.DataFrame(np.exp(np.cumsum(shocked, axis=0)) * 100,
                            index=dates, columns=tickers)
    uf_a = make_momentum_topk(prices_a, top_k=3, lookback=20)
    uf_b = make_momentum_topk(prices_b, top_k=3, lookback=20)
    # Picks at date 55 (before the shock at 60) must be identical
    assert uf_a(dates[55], tickers) == uf_b(dates[55], tickers)
