"""Unit tests for XGBRankerStockModel."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from xgbnew.model_xgb_ranker import (
    XGBRankerStockModel,
    _decile_label_per_day,
    _group_sizes,
)


def _fake_panel(n_days: int = 60, n_syms: int = 50, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2023-01-02", periods=n_days, freq="B")
    syms = [f"S{i:03d}" for i in range(n_syms)]
    rows = []
    for d in dates:
        # Make 3 features that do carry signal: f0+f1-f2 ~ target_oc
        for s in syms:
            f0 = rng.normal()
            f1 = rng.normal()
            f2 = rng.normal()
            noise = rng.normal() * 0.5
            target_oc = 0.01 * (f0 + f1 - f2 + noise)
            rows.append({
                "date": d.date(), "symbol": s,
                "f0": f0, "f1": f1, "f2": f2,
                "target_oc": target_oc,
                "target_oc_up": int(target_oc > 0),
            })
    return pd.DataFrame(rows)


def test_decile_label_has_correct_shape_and_range():
    df = _fake_panel(n_days=10, n_syms=50)
    labels = _decile_label_per_day(df, n_deciles=10)
    assert labels.shape == (len(df),)
    assert labels.min() >= 0
    assert labels.max() <= 9


def test_decile_label_matches_within_day_rank():
    df = _fake_panel(n_days=5, n_syms=50)
    labels = _decile_label_per_day(df, n_deciles=10)
    df2 = df.assign(dec=labels)
    # Top-decile rows should have higher target_oc on average than bottom-decile
    top_mean = df2.loc[df2["dec"] >= 8, "target_oc"].mean()
    bot_mean = df2.loc[df2["dec"] <= 1, "target_oc"].mean()
    assert top_mean > bot_mean


def test_group_sizes_match_rows():
    df = _fake_panel(n_days=7, n_syms=20)
    df = df.sort_values(["date", "symbol"]).reset_index(drop=True)
    sizes = _group_sizes(df)
    assert sizes.sum() == len(df)
    assert len(sizes) == df["date"].nunique()


def test_ranker_trains_and_scores():
    df = _fake_panel(n_days=120, n_syms=40, seed=1)
    model = XGBRankerStockModel(device="cpu", n_estimators=50, max_depth=3,
                                 learning_rate=0.1, random_state=7)
    model.fit(df, feature_cols=["f0", "f1", "f2"], verbose=False)
    scores = model.predict_scores(df)
    assert len(scores) == len(df)
    assert scores.min() >= 0.0
    assert scores.max() <= 1.0

    # Score should correlate with target_oc within each day
    merged = df.assign(score=scores.values)
    corrs = []
    for _, g in merged.groupby("date"):
        c = g["score"].corr(g["target_oc"])
        if not np.isnan(c):
            corrs.append(c)
    mean_corr = float(np.mean(corrs))
    # With a noisy linear signal + 120 days training, we expect + correlation
    assert mean_corr > 0.3, f"ranker failed to learn: corr={mean_corr:.3f}"


def test_ranker_save_load_roundtrip(tmp_path):
    df = _fake_panel(n_days=60, n_syms=30, seed=2)
    m1 = XGBRankerStockModel(device="cpu", n_estimators=30, max_depth=3,
                              learning_rate=0.1, random_state=13)
    m1.fit(df, feature_cols=["f0", "f1", "f2"], verbose=False)
    s1 = m1.predict_scores(df).values

    p = tmp_path / "rnk.pkl"
    m1.save(p)
    m2 = XGBRankerStockModel.load(p)
    assert m2.feature_cols == ["f0", "f1", "f2"]
    s2 = m2.predict_scores(df).values
    np.testing.assert_allclose(s1, s2)


def test_ranker_sample_weight_modes_train():
    df = _fake_panel(n_days=60, n_syms=30, seed=4)
    for mode in ("none", "abs_target", "abs_target_sqrt"):
        m = XGBRankerStockModel(
            device="cpu", n_estimators=30, max_depth=3,
            learning_rate=0.1, random_state=5,
            sample_weight_mode=mode,
        )
        m.fit(df, feature_cols=["f0", "f1", "f2"], verbose=False)
        s = m.predict_scores(df)
        assert s.notna().all() and 0.0 <= s.min() <= s.max() <= 1.0


def test_ranker_sample_weight_roundtrip(tmp_path):
    df = _fake_panel(n_days=40, n_syms=25, seed=6)
    m1 = XGBRankerStockModel(
        device="cpu", n_estimators=20, max_depth=2,
        learning_rate=0.1, random_state=11,
        sample_weight_mode="abs_target", sample_weight_clip=0.03,
    )
    m1.fit(df, feature_cols=["f0", "f1", "f2"], verbose=False)
    p = tmp_path / "rnk_sw.pkl"
    m1.save(p)
    m2 = XGBRankerStockModel.load(p)
    assert m2.sample_weight_mode == "abs_target"
    assert m2.sample_weight_clip == 0.03
    np.testing.assert_allclose(m1.predict_scores(df).values,
                               m2.predict_scores(df).values)


def test_ranker_registry_dispatch(tmp_path):
    from xgbnew.model_registry import load_any_model

    df = _fake_panel(n_days=40, n_syms=25, seed=3)
    m = XGBRankerStockModel(device="cpu", n_estimators=20, max_depth=2,
                             learning_rate=0.1, random_state=21)
    m.fit(df, feature_cols=["f0", "f1", "f2"], verbose=False)
    p = tmp_path / "rnk_reg.pkl"
    m.save(p)

    loaded = load_any_model(p)
    assert isinstance(loaded, XGBRankerStockModel)
    assert loaded.feature_cols == ["f0", "f1", "f2"]
