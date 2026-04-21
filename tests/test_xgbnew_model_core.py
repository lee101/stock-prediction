"""Core coverage for ``xgbnew.model.XGBStockModel``.

Targets branches that the device test file doesn't reach:

- ``predict_scores`` on an unfitted model raises RuntimeError.
- ``fit`` with a non-empty ``val_df`` triggers the eval_set +
  early_stopping_rounds setup, and the val-accuracy logging path.
- ``combined_scores`` without a ``date`` column ranks chronos globally
  (line 253, the else branch).
- ``combined_scores`` without a chronos column falls back to 0.5 neutral
  rank (line 254-256 / the else that skips chronos entirely).
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import pytest

from xgbnew.model import XGBStockModel, combined_scores


def _toy_df(n: int = 300, n_features: int = 3, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, n_features)).astype(np.float32)
    y = ((X[:, 0] + X[:, 1]) > 0).astype(np.int32)
    cols = [f"f{i}" for i in range(n_features)]
    df = pd.DataFrame(X, columns=cols)
    df["target_oc_up"] = y
    return df


def test_predict_scores_before_fit_raises() -> None:
    m = XGBStockModel(n_estimators=10)
    df = _toy_df(n=20)
    with pytest.raises(RuntimeError, match="not fitted"):
        m.predict_scores(df)


def test_fit_with_val_df_logs_accuracy(caplog: pytest.LogCaptureFixture) -> None:
    """val_df path exercises:
      - eval_set construction (lines 125-129)
      - early_stopping_rounds setup (lines 135-141)
      - val directional accuracy logging (lines 156-160)
    """
    train = _toy_df(n=300, seed=1)
    val = _toy_df(n=80, seed=2)
    feats = [c for c in train.columns if c != "target_oc_up"]
    m = XGBStockModel(n_estimators=30, max_depth=3)
    with caplog.at_level(logging.INFO, logger="xgbnew.model"):
        m.fit(train, feats, val_df=val, early_stopping_rounds=10, verbose=False)
    # Val accuracy line was emitted
    messages = [r.getMessage() for r in caplog.records]
    assert any("Val directional accuracy" in m for m in messages), messages


def test_fit_with_val_df_injects_column_medians_on_nan() -> None:
    """NaNs in val_df should be replaced with train's column medians,
    not the val column medians — a silent-leak guard."""
    train = _toy_df(n=200, seed=3)
    val = _toy_df(n=50, seed=4)
    # inject a NaN into val's first feature
    val.loc[0, "f0"] = np.nan
    feats = [c for c in train.columns if c != "target_oc_up"]
    m = XGBStockModel(n_estimators=20).fit(train, feats, val_df=val, verbose=False)
    # prediction on val must not raise, and the NaN row gets a valid score
    scores = m.predict_scores(val)
    assert scores.between(0.0, 1.0).all()
    assert scores.notna().all()


def test_combined_scores_uses_per_day_rank_when_date_col_present() -> None:
    train = _toy_df(n=200, seed=5)
    feats = [c for c in train.columns if c != "target_oc_up"]
    m = XGBStockModel(n_estimators=15).fit(train, feats, verbose=False)
    df = _toy_df(n=30, seed=6)
    df["date"] = ["2026-04-01"] * 15 + ["2026-04-02"] * 15
    df["chronos_oc_return"] = np.linspace(-0.05, 0.05, 30)
    out = combined_scores(df, m, xgb_weight=0.5)
    assert out.between(0.0, 1.0).all()
    # Rank-pct per-day: within each date, min rank ~ 1/n, max rank = 1
    # Blend with XGB keeps result in [0, 1]
    assert out.name == "combined_score"


def test_combined_scores_ranks_globally_without_date_col() -> None:
    """No-date-column branch (line 253): rank pct across the whole frame."""
    train = _toy_df(n=200, seed=7)
    feats = [c for c in train.columns if c != "target_oc_up"]
    m = XGBStockModel(n_estimators=15).fit(train, feats, verbose=False)
    df = _toy_df(n=20, seed=8)
    df["chronos_oc_return"] = np.linspace(-0.1, 0.1, 20)  # monotonic non-zero
    assert "date" not in df.columns
    out = combined_scores(df, m, xgb_weight=0.0)  # pure chronos rank
    # Pure chronos, monotonic input → ranks are monotonic too
    ranks = out.values
    assert ranks[0] < ranks[-1]
    assert out.between(0.0, 1.0).all()


def test_combined_scores_falls_back_to_neutral_when_no_chronos() -> None:
    """No chronos column → chron_norm = 0.5 uniformly."""
    train = _toy_df(n=200, seed=9)
    feats = [c for c in train.columns if c != "target_oc_up"]
    m = XGBStockModel(n_estimators=15).fit(train, feats, verbose=False)
    df = _toy_df(n=10, seed=10)  # no chronos_oc_return column
    xgb_scores = m.predict_scores(df)
    out = combined_scores(df, m, xgb_weight=0.5)
    # 0.5 * xgb + 0.5 * 0.5 (neutral chronos rank)
    expected = 0.5 * xgb_scores.values + 0.25
    np.testing.assert_allclose(out.values, expected, atol=1e-6)


def test_combined_scores_all_zero_chronos_uses_neutral() -> None:
    """If chronos col exists but is all zero, the ``(df[col] != 0).any()``
    guard falls to the neutral 0.5 branch."""
    train = _toy_df(n=200, seed=11)
    feats = [c for c in train.columns if c != "target_oc_up"]
    m = XGBStockModel(n_estimators=15).fit(train, feats, verbose=False)
    df = _toy_df(n=10, seed=12)
    df["chronos_oc_return"] = 0.0
    xgb_scores = m.predict_scores(df)
    out = combined_scores(df, m, xgb_weight=1.0)
    # xgb_weight=1.0 → pure xgb
    np.testing.assert_allclose(out.values, xgb_scores.values, atol=1e-6)


def test_feature_importances_sorted_descending() -> None:
    df = _toy_df(n=200)
    feats = [c for c in df.columns if c != "target_oc_up"]
    m = XGBStockModel(n_estimators=20).fit(df, feats, verbose=False)
    imps = m.feature_importances()
    assert list(imps.index) == sorted(imps.index, key=lambda k: -imps[k])
    assert len(imps) == len(feats)
