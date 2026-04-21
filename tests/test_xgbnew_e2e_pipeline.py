"""End-to-end smoke test for the xgbnew daily pipeline.

Exercises the full chain on synthetic CSVs:

    1. _load_symbol_csv reads OHLCV from ``trainingdata/<SYM>.csv``.
    2. build_daily_dataset produces train/val/test frames with the
       documented feature columns (+ actual_open / actual_close /
       target_oc / dolvol_20d_log / spread_bps).
    3. XGBStockModel.fit learns the synthetic signal.
    4. XGBStockModel.predict_scores on test_df returns probabilities in [0, 1].
    5. simulate() produces a BacktestResult with non-NaN monthly_return_pct,
       sortino_ratio, max_drawdown_pct, directional_accuracy, and
       time_under_water_pct.

Designed to be deterministic — the synth "signal" is that higher prior-day
close-to-close return predicts positive open-to-close today (so the tree
has something to learn), but with enough noise that the backtest doesn't
degenerate to 100%-accuracy.
"""
from __future__ import annotations

from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from xgbnew.backtest import BacktestConfig, simulate
from xgbnew.dataset import build_daily_dataset
from xgbnew.features import DAILY_FEATURE_COLS
from xgbnew.model import XGBStockModel


SYNTH_SYMBOLS = ["AAA", "BBB", "CCC", "DDD", "EEE"]


def _synth_ohlcv(n: int = 600, seed: int = 0) -> pd.DataFrame:
    """Generate a synthetic OHLCV series with learnable open-to-close signal.

    - Daily close-to-close returns are N(0, 0.015).
    - On each day there's a mild bias: if yesterday's cc_return > 0.5%,
      today's open-to-close drift is +10bps extra (creates a weak but
      learnable feature<->target link).
    """
    rng = np.random.default_rng(seed)
    cc = rng.normal(0.0, 0.015, n)
    # Carry the mild autocorrelation signal: if prev cc > +0.5%, today's
    # oc drifts up +10bps.
    oc_drift = np.where(np.concatenate([[0.0], cc[:-1] > 0.005]), 0.001, 0.0)
    price = 100.0 * np.cumprod(1.0 + cc)
    opens = price * (1.0 - 0.001 + rng.normal(0.0, 0.001, n))
    closes = opens * (1.0 + oc_drift + rng.normal(0.0, 0.008, n))
    highs = np.maximum(opens, closes) * (1.0 + np.abs(rng.normal(0.0, 0.003, n)))
    lows = np.minimum(opens, closes) * (1.0 - np.abs(rng.normal(0.0, 0.003, n)))
    vol = rng.uniform(2e6, 5e7, n)  # large dolvol so rows pass filter
    ts = pd.date_range("2021-01-04", periods=n, freq="B", tz="UTC")
    return pd.DataFrame({
        "timestamp": ts,
        "open": opens,
        "high": highs,
        "low": lows,
        "close": closes,
        "volume": vol,
    })


@pytest.fixture(scope="module")
def synth_data_root(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("synth_data")
    for i, sym in enumerate(SYNTH_SYMBOLS):
        _synth_ohlcv(seed=i).to_csv(root / f"{sym}.csv", index=False)
    return root


@pytest.fixture(scope="module")
def built_dataset(synth_data_root: Path):
    """Build train/val/test splits once per module — expensive path."""
    return build_daily_dataset(
        data_root=synth_data_root,
        symbols=SYNTH_SYMBOLS,
        train_start=date(2021, 1, 4),
        train_end=date(2022, 3, 31),
        val_start=date(2022, 4, 1),
        val_end=date(2022, 8, 31),
        test_start=date(2022, 9, 1),
        test_end=date(2023, 4, 30),
        min_dollar_vol=1e6,
    )


def test_build_daily_dataset_produces_nonempty_splits(built_dataset) -> None:
    train_df, val_df, test_df = built_dataset
    assert len(train_df) > 100
    assert len(val_df) > 10
    assert len(test_df) > 20


def test_build_daily_dataset_includes_documented_columns(built_dataset) -> None:
    train_df, _, test_df = built_dataset
    required_in_train = {"target_oc_up", "symbol", "date"} | set(DAILY_FEATURE_COLS)
    required_in_test = {
        "symbol", "date", "actual_open", "actual_close",
        "spread_bps", "dolvol_20d_log", "target_oc",
    } | set(DAILY_FEATURE_COLS)
    assert required_in_train.issubset(train_df.columns), (
        f"missing in train: {required_in_train - set(train_df.columns)}"
    )
    assert required_in_test.issubset(test_df.columns), (
        f"missing in test: {required_in_test - set(test_df.columns)}"
    )


def test_e2e_fit_predict_simulate(built_dataset) -> None:
    train_df, val_df, test_df = built_dataset
    model = XGBStockModel(n_estimators=40, max_depth=3, learning_rate=0.1)
    model.fit(train_df, DAILY_FEATURE_COLS, val_df=val_df, verbose=False)

    # predict_scores — outputs must be in [0, 1] and aligned to test_df.index
    scores = model.predict_scores(test_df)
    assert scores.between(0.0, 1.0).all()
    assert len(scores) == len(test_df)
    assert (scores.index == test_df.index).all()

    # simulate with deploy-like fees
    config = BacktestConfig(
        top_n=1,
        leverage=1.0,
        min_score=0.0,
        min_dollar_vol=1e6,
        min_vol_20d=0.0,
        xgb_weight=1.0,          # pure XGB (no chronos blend)
        fee_rate=0.0000278,      # deploy
        fill_buffer_bps=5.0,
        commission_bps=0.0,
        hold_through=False,
    )
    result = simulate(test_df, model, config, precomputed_scores=scores)

    # BacktestResult shape asserts
    assert result.total_trades >= 1
    assert np.isfinite(result.monthly_return_pct)
    assert np.isfinite(result.sortino_ratio)
    assert np.isfinite(result.sharpe_ratio)
    assert 0.0 <= result.win_rate_pct <= 100.0
    assert 0.0 <= result.directional_accuracy_pct <= 100.0
    assert result.max_drawdown_pct >= 0.0
    assert 0.0 <= result.time_under_water_pct <= 100.0
    assert result.ulcer_index >= 0.0
    assert len(result.day_results) > 0
    # Fee-aware: avg_fee_bps should be positive (since fee_rate > 0)
    assert result.avg_fee_bps > 0.0


def test_e2e_save_load_round_trip_matches_scores(
    built_dataset, tmp_path: Path,
) -> None:
    train_df, val_df, test_df = built_dataset
    model = XGBStockModel(n_estimators=30, max_depth=3).fit(
        train_df, DAILY_FEATURE_COLS, val_df=val_df, verbose=False,
    )
    pkl = tmp_path / "m.pkl"
    model.save(pkl)
    reloaded = XGBStockModel.load(pkl)
    a = model.predict_scores(test_df).values
    b = reloaded.predict_scores(test_df).values
    np.testing.assert_allclose(a, b, atol=1e-6)


def test_e2e_simulate_respects_min_score_gate(built_dataset) -> None:
    """Set ms = 1.1 (unreachable) — simulate must produce zero trades and
    pass the sim through without exploding."""
    train_df, val_df, test_df = built_dataset
    model = XGBStockModel(n_estimators=20, max_depth=3).fit(
        train_df, DAILY_FEATURE_COLS, val_df=val_df, verbose=False,
    )
    scores = model.predict_scores(test_df)
    config = BacktestConfig(
        top_n=1, leverage=1.0, min_score=1.1, min_dollar_vol=1e6,
        xgb_weight=1.0, fee_rate=0.0000278, fill_buffer_bps=5.0,
        commission_bps=0.0,
    )
    result = simulate(test_df, model, config, precomputed_scores=scores)
    assert result.total_trades == 0
    assert result.total_return_pct == 0.0
    assert result.final_equity == pytest.approx(config.initial_cash, rel=1e-6)
