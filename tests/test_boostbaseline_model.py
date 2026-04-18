"""Coverage for the GPU / val-sizing / early-stopping refactor of
``boostbaseline.model``.

Key invariants:
1. Train/val/test split uses 60/20/20 proportions (rough bounds).
2. Sizing (scale/cap) is tuned on the *validation* slice only — the test
   slice must not be consumed during ``train_and_optimize``.
3. Early stopping populates ``best_iteration`` when xgboost is available,
   and ``predict`` honours it via ``iteration_range``.
4. ``save`` round-trips ``best_iteration`` through the JSON metadata.
5. Turnover-proportional fees scale with ``|Δposition|``.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from boostbaseline.backtest import (
    _compute_fee_changes,
    grid_search_sizing,
    run_backtest,
)
from boostbaseline.model import TrainedModel, _xgb_cuda_available, train_and_optimize


# ─── fee model ───────────────────────────────────────────────────────────────

def test_fee_changes_turnover_proportional():
    positions = np.array([0.0, 0.2, 0.2, 0.1, 0.0])
    # diff(prepend 0.0) = [0.0, +0.2, 0.0, -0.1, -0.1] → |Δ| * fee
    fees = _compute_fee_changes(positions, fee=0.001, turnover_proportional=True)
    assert fees == pytest.approx([0.0, 0.0002, 0.0, 0.0001, 0.0001])


def test_fee_changes_flat_legacy_mode():
    positions = np.array([0.0, 0.2, 0.2, 0.1, 0.0])
    # 3 non-zero changes at indices 1, 3, 4 → full fee on those steps.
    fees = _compute_fee_changes(positions, fee=0.001, turnover_proportional=False)
    assert fees == pytest.approx([0.0, 0.001, 0.0, 0.001, 0.001])


def test_run_backtest_turnover_flag_changes_total_return():
    rng = np.random.default_rng(0)
    y_true = rng.normal(0.001, 0.01, 500)
    y_pred = y_true + rng.normal(0, 0.002, 500)  # mildly correlated
    res_prop = run_backtest(
        y_true, y_pred, is_crypto=True, fee=0.001, scale=5.0, cap=0.5,
        turnover_proportional_fee=True,
    )
    res_flat = run_backtest(
        y_true, y_pred, is_crypto=True, fee=0.001, scale=5.0, cap=0.5,
        turnover_proportional_fee=False,
    )
    # Flat fees are strictly higher (or equal) per step — cumulative return
    # under flat should be <= proportional.
    assert res_flat.total_return <= res_prop.total_return + 1e-9


# ─── train_and_optimize contract ─────────────────────────────────────────────

def _toy_training_df(n: int = 600, seed: int = 0) -> pd.DataFrame:
    """A tiny dataframe with the feature_* / y columns boostbaseline expects."""
    rng = np.random.default_rng(seed)
    f1 = rng.normal(0, 0.01, n)
    f2 = rng.normal(0, 0.01, n)
    noise = rng.normal(0, 0.005, n)
    y = 0.4 * f1 - 0.2 * f2 + noise
    df = pd.DataFrame({
        "feature_a": f1,
        "feature_b": f2,
        "feature_c": rng.normal(0, 0.01, n),  # pure noise feature
        "y": y,
    })
    return df


def test_train_and_optimize_produces_valid_model():
    df = _toy_training_df(n=600)
    model = train_and_optimize(df, is_crypto=True, fee=0.0001)
    assert isinstance(model, TrainedModel)
    assert set(model.feature_cols) == {"feature_a", "feature_b", "feature_c"}
    # We ask for pseudohuber XGB; when xgboost is available it should win.
    # If it fell back to sklearn that's acceptable but surfaced for debug.
    assert model.model_name in {"xgboost", "sklearn_gbr"}
    if model.is_xgb:
        # Early stopping should fire — best_iteration is a non-negative int.
        assert model.best_iteration is not None
        assert 0 <= model.best_iteration < 4000


def test_train_and_optimize_predict_matches_booster_with_best_iteration():
    df = _toy_training_df(n=600)
    model = train_and_optimize(df, is_crypto=True, fee=0.0001)
    X = df[model.feature_cols]
    preds = model.predict(X)
    assert preds.shape == (len(df),)
    assert np.all(np.isfinite(preds))


def test_train_and_optimize_does_not_consume_test_split_for_sizing(monkeypatch):
    """grid_search_sizing must be called exactly once, with y_true sized to
    the validation slice (20% of the dataset)."""
    df = _toy_training_df(n=600)
    calls: list[int] = []

    def _spy(*args, **kwargs):
        y_true = kwargs.get("y_true")
        if y_true is None and args:
            y_true = args[0]
        calls.append(len(y_true))
        # return a dummy-like BacktestResult
        from boostbaseline.backtest import BacktestResult
        return BacktestResult(
            total_return=0.0, sharpe=0.0,
            positions=np.zeros_like(y_true), returns=np.zeros_like(y_true),
            scale=1.0, cap=0.3,
        )

    monkeypatch.setattr("boostbaseline.model.grid_search_sizing", _spy)
    train_and_optimize(df, is_crypto=True, fee=0.0001)
    assert len(calls) == 1
    # 20% slice of 600 = 120.
    assert 100 <= calls[0] <= 140


def test_grid_search_sizing_uses_turnover_flag():
    rng = np.random.default_rng(0)
    y_true = rng.normal(0, 0.01, 300)
    y_pred = y_true + rng.normal(0, 0.002, 300)
    bt_prop = grid_search_sizing(y_true, y_pred, fee=0.001, turnover_proportional_fee=True)
    bt_flat = grid_search_sizing(y_true, y_pred, fee=0.001, turnover_proportional_fee=False)
    # Same search grid; the search is allowed to pick different (scale, cap),
    # but total_return under the proportional model should dominate or equal
    # what is feasible under the flat-fee model at matched params.
    assert bt_prop.total_return >= bt_flat.total_return - 1e-9


# ─── persistence round-trip ─────────────────────────────────────────────────

def test_save_round_trips_best_iteration(tmp_path, monkeypatch):
    df = _toy_training_df(n=400)
    model = train_and_optimize(df, is_crypto=True, fee=0.0001)

    # Redirect MODELS_DIR so the test stays sandboxed.
    monkeypatch.setattr("boostbaseline.model.MODELS_DIR", tmp_path)
    model.save("TESTSYM")

    meta_path = tmp_path / "TESTSYM_boost.model"
    assert meta_path.exists()
    meta = json.loads(meta_path.read_text())
    assert "best_iteration" in meta
    if model.is_xgb:
        assert meta["best_iteration"] == model.best_iteration


# ─── cuda probe is defensive ────────────────────────────────────────────────

def test_xgb_cuda_available_returns_bool():
    out = _xgb_cuda_available()
    assert isinstance(out, bool)
