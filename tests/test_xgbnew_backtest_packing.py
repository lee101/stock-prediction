"""Score-weighted portfolio packing tests for xgbnew.backtest.

The DD-reduction campaign ruled out regime/vol-target knobs for top_n=1 —
each gate added neg windows by sitting out correct argmax picks. The
remaining DD lever is packing: when top_n>1, weight the picks by score so
the strongest signal still dominates PnL but the tail reduces variance.

We test the pure helper ``_allocation_weights`` directly because it's the
load-bearing contract, plus one end-to-end ``simulate()`` test that pins
``allocation_mode='equal'`` = current behaviour (regression).
"""
from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest
from xgbnew.backtest import (
    BacktestConfig,
    DayTrade,
    _allocation_weights,
    _trade_allocation_weights,
    simulate,
)


def test_allocation_mode_default_is_equal():
    cfg = BacktestConfig()
    assert cfg.allocation_mode == "equal"
    assert cfg.allocation_temp == 1.0


def test_allocation_mode_knobs_settable():
    cfg = BacktestConfig(allocation_mode="softmax", allocation_temp=0.5)
    assert cfg.allocation_mode == "softmax"
    assert cfg.allocation_temp == 0.5


def test_min_secondary_allocation_default_is_disabled():
    cfg = BacktestConfig()
    assert cfg.min_secondary_allocation == 0.0


def test_min_picks_default_preserves_confidence_gate():
    cfg = BacktestConfig()
    assert cfg.min_picks == 0


def test_equal_weights_uniform():
    w = _allocation_weights([3.0, 1.0, 0.5, -0.2], mode="equal")
    assert w.shape == (4,)
    np.testing.assert_allclose(w, 0.25)


def test_equal_weights_empty():
    w = _allocation_weights([], mode="equal")
    assert w.shape == (0,)


def test_softmax_concentrates_on_high_score():
    w = _allocation_weights([3.0, 1.0, 0.5], mode="softmax", temperature=1.0)
    np.testing.assert_allclose(w.sum(), 1.0)
    # Strictly monotone with the inputs.
    assert w[0] > w[1] > w[2]
    # Top pick should hold the majority at temp=1 on this spread.
    assert w[0] > 0.6


def test_softmax_low_temperature_is_more_concentrated():
    w_hot = _allocation_weights([1.0, 0.8, 0.6], mode="softmax", temperature=2.0)
    w_cold = _allocation_weights([1.0, 0.8, 0.6], mode="softmax", temperature=0.1)
    # Cold temp → argmax; hot temp → closer to uniform.
    assert w_cold[0] > w_hot[0]
    assert w_cold[-1] < w_hot[-1]


def test_softmax_invalid_temp_falls_back_to_one():
    w = _allocation_weights([1.0, 0.0, -1.0], mode="softmax", temperature=0.0)
    # Still a valid distribution, just rendered at temp=1.
    np.testing.assert_allclose(w.sum(), 1.0)
    assert w[0] > w[1] > w[2]


def test_softmax_nan_inputs_degenerate_to_equal():
    w = _allocation_weights([np.nan, 1.0, 0.5], mode="softmax")
    np.testing.assert_allclose(w, 1.0 / 3)


def test_score_norm_weights_proportional_to_positive_scores():
    w = _allocation_weights([3.0, 1.0, 0.0], mode="score_norm")
    np.testing.assert_allclose(w.sum(), 1.0)
    np.testing.assert_allclose(w, [0.75, 0.25, 0.0])


def test_score_norm_degenerate_all_nonpositive_falls_back_to_equal():
    w = _allocation_weights([-1.0, -2.0, 0.0], mode="score_norm")
    np.testing.assert_allclose(w, 1.0 / 3)


def test_short_allocation_scale_reduces_short_weight_under_equal_mode():
    long = DayTrade(
        symbol="LONG",
        score=0.9,
        actual_open=100.0,
        actual_close=101.0,
        entry_fill_price=100.0,
        exit_fill_price=101.0,
        spread_bps=1.0,
        commission_bps=0.0,
        fee_rate=0.0,
        fill_buffer_bps=0.0,
        leverage=1.0,
        gross_return_pct=1.0,
        net_return_pct=1.0,
        side=1,
    )
    short = DayTrade(
        symbol="SHORT",
        score=0.1,
        actual_open=100.0,
        actual_close=99.0,
        entry_fill_price=100.0,
        exit_fill_price=99.0,
        spread_bps=1.0,
        commission_bps=0.0,
        fee_rate=0.0,
        fill_buffer_bps=0.0,
        leverage=1.0,
        gross_return_pct=1.0,
        net_return_pct=1.0,
        side=-1,
    )

    weights = _trade_allocation_weights(
        [long, short],
        mode="equal",
        temperature=1.0,
        short_allocation_scale=0.5,
    )

    np.testing.assert_allclose(weights, [2.0 / 3.0, 1.0 / 3.0])


def test_min_secondary_allocation_caps_concentrated_learned_weights():
    winner = DayTrade(
        symbol="WIN",
        score=0.99,
        actual_open=100.0,
        actual_close=101.0,
        entry_fill_price=100.0,
        exit_fill_price=101.0,
        spread_bps=1.0,
        commission_bps=0.0,
        fee_rate=0.0,
        fill_buffer_bps=0.0,
        leverage=1.0,
        gross_return_pct=1.0,
        net_return_pct=1.0,
        side=1,
    )
    hedge = DayTrade(
        symbol="HEDGE",
        score=0.01,
        actual_open=100.0,
        actual_close=99.0,
        entry_fill_price=100.0,
        exit_fill_price=99.0,
        spread_bps=1.0,
        commission_bps=0.0,
        fee_rate=0.0,
        fill_buffer_bps=0.0,
        leverage=1.0,
        gross_return_pct=-1.0,
        net_return_pct=-1.0,
        side=1,
    )

    unconstrained = _trade_allocation_weights(
        [winner, hedge],
        mode="softmax",
        temperature=0.01,
        short_allocation_scale=1.0,
    )
    constrained = _trade_allocation_weights(
        [winner, hedge],
        mode="softmax",
        temperature=0.01,
        short_allocation_scale=1.0,
        min_secondary_allocation=0.20,
    )

    assert unconstrained[0] > 0.99
    np.testing.assert_allclose(constrained, [0.80, 0.20])


def test_unknown_allocation_mode_raises():
    with pytest.raises(ValueError, match="allocation_mode"):
        _allocation_weights([1.0, 0.5], mode="kelly_gauss")


def test_corr_gate_skips_redundant_same_side_pick():
    rows = []
    scores = []
    start = date(2025, 1, 2)
    ret_hist = {
        "AAA": [0.01, 0.02, 0.03, 0.04, 0.05, 0.06],
        "BBB": [0.011, 0.021, 0.031, 0.041, 0.051, 0.061],
        "CCC": [0.03, -0.02, 0.01, -0.01, 0.02, -0.03],
    }
    for d in range(6):
        day = start + timedelta(days=d)
        for sym in ("AAA", "BBB", "CCC"):
            rows.append({
                "date": day,
                "symbol": sym,
                "actual_open": 100.0,
                "actual_close": 101.0 if sym != "CCC" else 102.0,
                "spread_bps": 1.0,
                "dolvol_20d_log": np.log1p(1e9),
                "ret_1d": ret_hist[sym][d],
            })
            if d == 5:
                scores.append({"AAA": 0.90, "BBB": 0.85, "CCC": 0.80}[sym])
            else:
                scores.append(0.10)
    df = pd.DataFrame(rows)

    base_cfg = BacktestConfig(
        top_n=2,
        min_score=0.50,
        fill_buffer_bps=0.0,
        commission_bps=0.0,
        fee_rate=0.0,
        min_dollar_vol=0.0,
        max_spread_bps=100.0,
        xgb_weight=1.0,
    )
    ungated = simulate(df, model=None, config=base_cfg, precomputed_scores=pd.Series(scores, index=df.index))  # type: ignore[arg-type]
    gated = simulate(
        df,
        model=None,
        config=BacktestConfig(
            **{
                **base_cfg.__dict__,
                "corr_window_days": 6,
                "corr_min_periods": 3,
                "corr_max_signed": 0.50,
            }
        ),
        precomputed_scores=pd.Series(scores, index=df.index),
    )  # type: ignore[arg-type]

    assert [t.symbol for t in ungated.day_results[-1].trades] == ["AAA", "BBB"]
    assert [t.symbol for t in gated.day_results[-1].trades] == ["AAA", "CCC"]


# ---- simulate() end-to-end regression -------------------------------------


def _toy_dataset(n_days: int = 10, n_syms: int = 5) -> pd.DataFrame:
    """Each day's actual_close moves by a known multiple of each symbol's
    rank so we can verify that score-weighted packing shifts daily PnL.

    Columns match simulate()'s required set: date, symbol, actual_open,
    actual_close, spread_bps, + _score placeholder (supplied via
    precomputed_scores in the test).
    """
    rng = np.random.default_rng(0)
    rows = []
    start = date(2024, 1, 2)
    for d in range(n_days):
        day = start + timedelta(days=d)
        for s in range(n_syms):
            open_px = 100.0
            # Return scales with symbol index to make packing observable.
            close_px = open_px * (1.0 + (s - 2) * 0.005 + rng.normal(0, 0.001))
            rows.append({
                "date": day,
                "symbol": f"SYM{s}",
                "actual_open": open_px,
                "actual_close": close_px,
                "spread_bps": 10.0,
                "dolvol_20d_log": np.log1p(1e7),
            })
    return pd.DataFrame(rows)


def test_simulate_equal_mode_matches_legacy_mean():
    """Regression: equal weights must reproduce the pre-packing trade-loop
    behaviour (simple mean of per-trade net returns)."""
    df = _toy_dataset(n_days=6, n_syms=4)
    # Scores: symbol index descending so SYM3/SYM2 are the top picks.
    df["_score_stub"] = df["symbol"].map({f"SYM{i}": float(3 - i) for i in range(4)})
    scores = df["_score_stub"]

    cfg = BacktestConfig(
        top_n=3,
        fill_buffer_bps=0.0,
        commission_bps=0.0,
        fee_rate=0.0,
        min_dollar_vol=0.0,
        max_spread_bps=100.0,
        xgb_weight=1.0,
        allocation_mode="equal",
    )
    res = simulate(df, model=None, config=cfg, precomputed_scores=scores)  # type: ignore[arg-type]

    # Manual check: under equal mode, each day's return should be the mean
    # of the top-3 trades' net_return_pct.
    for day_res in res.day_results:
        manual = float(np.mean([t.net_return_pct for t in day_res.trades]))
        assert abs(manual - day_res.daily_return_pct) < 1e-9


def test_simulate_softmax_biases_toward_top_score():
    """Softmax packing should tilt daily PnL toward the highest-score pick's
    return vs. the equal-weighted baseline."""
    df = _toy_dataset(n_days=10, n_syms=5)
    # Give top symbol a clearly dominant score.
    score_map = {"SYM4": 5.0, "SYM3": 0.8, "SYM2": 0.4, "SYM1": 0.1, "SYM0": 0.0}
    df["_score_stub"] = df["symbol"].map(score_map)
    scores = df["_score_stub"]

    base_cfg = BacktestConfig(
        top_n=3,
        fill_buffer_bps=0.0,
        commission_bps=0.0,
        fee_rate=0.0,
        min_dollar_vol=0.0,
        max_spread_bps=100.0,
        xgb_weight=1.0,
    )
    eq_cfg = BacktestConfig(**{**base_cfg.__dict__, "allocation_mode": "equal"})
    sm_cfg = BacktestConfig(**{**base_cfg.__dict__, "allocation_mode": "softmax", "allocation_temp": 0.5})

    r_eq = simulate(df, model=None, config=eq_cfg, precomputed_scores=scores)  # type: ignore[arg-type]
    r_sm = simulate(df, model=None, config=sm_cfg, precomputed_scores=scores)  # type: ignore[arg-type]

    # SYM4's return is ~+1% (index 4: 2*0.005); mean of top3 is smaller.
    # Softmax should be *between* equal-mean and SYM4-only, and strictly
    # > equal on days SYM4 wins.
    for d_eq, d_sm in zip(r_eq.day_results, r_sm.day_results):
        top = max(d_eq.trades, key=lambda t: t.score)
        eq_ret = d_eq.daily_return_pct
        sm_ret = d_sm.daily_return_pct
        # sm should lie between equal-mean and top-only.
        low, high = sorted([eq_ret, top.net_return_pct])
        assert low - 1e-9 <= sm_ret <= high + 1e-9


def test_simulate_min_picks_forces_best_below_threshold_candidate():
    """min_picks keeps the strategy invested on low-score days, while the
    legacy confidence gate sits in cash."""
    day = date(2024, 1, 2)
    df = pd.DataFrame([
        {
            "date": day,
            "symbol": "AAA",
            "actual_open": 100.0,
            "actual_close": 101.0,
            "spread_bps": 5.0,
            "dolvol_20d_log": np.log1p(1e9),
            "_score_stub": 0.70,
        },
        {
            "date": day,
            "symbol": "BBB",
            "actual_open": 100.0,
            "actual_close": 99.0,
            "spread_bps": 5.0,
            "dolvol_20d_log": np.log1p(1e9),
            "_score_stub": 0.60,
        },
    ])
    scores = df["_score_stub"]
    base_cfg = BacktestConfig(
        top_n=2,
        min_score=0.85,
        fill_buffer_bps=0.0,
        commission_bps=0.0,
        fee_rate=0.0,
        min_dollar_vol=0.0,
        max_spread_bps=100.0,
        xgb_weight=1.0,
    )
    gated = simulate(df, model=None, config=base_cfg, precomputed_scores=scores)  # type: ignore[arg-type]
    forced_cfg = BacktestConfig(**{**base_cfg.__dict__, "min_picks": 1})
    forced = simulate(df, model=None, config=forced_cfg, precomputed_scores=scores)  # type: ignore[arg-type]

    assert len(gated.day_results) == 0
    assert len(forced.day_results) == 1
    assert [t.symbol for t in forced.day_results[0].trades] == ["AAA"]
