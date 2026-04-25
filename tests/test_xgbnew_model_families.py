"""Parity tests for new model families against the shared contract.

Every family must:
    - fit(train_df, feature_cols) -> self
    - predict_scores(df) returns pd.Series in [0, 1] indexed like df
    - save(path) / load(path) round-trips with identical predictions
    - load_any_model dispatches by the pickle's "family" field
"""
from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


FEATURES = ["f_a", "f_b", "f_c", "f_d"]


def _toy_frame(n_rows: int = 400, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({c: rng.standard_normal(n_rows).astype(np.float32)
                       for c in FEATURES})
    # Weak-but-real signal so every family has something to fit.
    logits = 0.6 * df["f_a"].values + 0.3 * df["f_b"].values
    probs = 1.0 / (1.0 + np.exp(-logits))
    df["target_oc_up"] = (rng.random(n_rows) < probs).astype(np.int32)
    df["date"] = pd.date_range("2020-01-01", periods=n_rows, freq="1h")
    df["symbol"] = ["SYM" + str(i % 8) for i in range(n_rows)]
    return df


def _holdout_split(df: pd.DataFrame):
    # Split 80/20 in time order.
    cut = int(len(df) * 0.8)
    return df.iloc[:cut].copy(), df.iloc[cut:].copy()


# ── LightGBM ────────────────────────────────────────────────────────────────

def test_lgb_fit_predict_shape_and_range():
    pytest.importorskip("lightgbm")
    from xgbnew.model_lgb import LGBMStockModel
    df = _toy_frame()
    tr, va = _holdout_split(df)
    m = LGBMStockModel(n_estimators=50, num_leaves=15, learning_rate=0.1)
    m.fit(tr, FEATURES)
    scores = m.predict_scores(va)
    assert isinstance(scores, pd.Series)
    assert scores.index.equals(va.index)
    assert scores.between(0.0, 1.0).all()
    # Not degenerate (all-same) on a signal-carrying toy
    assert scores.std() > 1e-4


def test_lgb_save_load_roundtrip(tmp_path: Path):
    pytest.importorskip("lightgbm")
    from xgbnew.model_lgb import LGBMStockModel
    from xgbnew.model_registry import load_any_model
    df = _toy_frame()
    tr, va = _holdout_split(df)
    m = LGBMStockModel(n_estimators=40, learning_rate=0.1)
    m.fit(tr, FEATURES)
    p = tmp_path / "lgb_a.pkl"
    m.save(p)

    m2 = LGBMStockModel.load(p)
    s1 = m.predict_scores(va)
    s2 = m2.predict_scores(va)
    assert np.allclose(s1.values, s2.values, atol=1e-7)

    # dispatcher reaches the same class
    m3 = load_any_model(p)
    assert m3.__class__.__name__ == "LGBMStockModel"
    s3 = m3.predict_scores(va)
    assert np.allclose(s1.values, s3.values, atol=1e-7)


# ── CatBoost ────────────────────────────────────────────────────────────────

def test_cat_fit_predict_shape_and_range():
    pytest.importorskip("catboost")
    from xgbnew.model_cat import CatBoostStockModel
    df = _toy_frame()
    tr, va = _holdout_split(df)
    m = CatBoostStockModel(iterations=40, depth=4, learning_rate=0.1)
    m.fit(tr, FEATURES)
    scores = m.predict_scores(va)
    assert isinstance(scores, pd.Series)
    assert scores.index.equals(va.index)
    assert scores.between(0.0, 1.0).all()
    assert scores.std() > 1e-4


def test_cat_save_load_roundtrip(tmp_path: Path):
    pytest.importorskip("catboost")
    from xgbnew.model_cat import CatBoostStockModel
    from xgbnew.model_registry import load_any_model
    df = _toy_frame()
    tr, va = _holdout_split(df)
    m = CatBoostStockModel(iterations=40, depth=4, learning_rate=0.1)
    m.fit(tr, FEATURES)
    p = tmp_path / "cat_a.pkl"
    m.save(p)

    m2 = CatBoostStockModel.load(p)
    s1 = m.predict_scores(va)
    s2 = m2.predict_scores(va)
    assert np.allclose(s1.values, s2.values, atol=1e-7)

    m3 = load_any_model(p)
    assert m3.__class__.__name__ == "CatBoostStockModel"


# ── MLP (torch) ─────────────────────────────────────────────────────────────

def test_mlp_fit_predict_shape_and_range():
    from xgbnew.model_mlp import MLPStockModel
    df = _toy_frame(n_rows=800, seed=3)
    tr, va = _holdout_split(df)
    m = MLPStockModel(
        hidden_dims=(32, 16), dropout=0.1,
        epochs=5, batch_size=128, early_stop_patience=3,
        learning_rate=5e-3, random_state=7, device="cpu",
    )
    m.fit(tr, FEATURES, val_df=va, verbose=False)
    scores = m.predict_scores(va)
    assert isinstance(scores, pd.Series)
    assert scores.index.equals(va.index)
    assert scores.between(0.0, 1.0).all()
    assert scores.std() > 1e-4


def test_mlp_save_load_roundtrip(tmp_path: Path):
    from xgbnew.model_mlp import MLPStockModel
    from xgbnew.model_registry import load_any_model
    df = _toy_frame(n_rows=800, seed=4)
    tr, va = _holdout_split(df)
    m = MLPStockModel(hidden_dims=(32, 16), dropout=0.1, epochs=4,
                      batch_size=128, early_stop_patience=3,
                      learning_rate=5e-3, random_state=7, device="cpu")
    m.fit(tr, FEATURES, val_df=va, verbose=False)
    p = tmp_path / "mlp_a.pkl"
    m.save(p)

    m2 = MLPStockModel.load(p)
    s1 = m.predict_scores(va)
    s2 = m2.predict_scores(va)
    assert np.allclose(s1.values, s2.values, atol=1e-6)

    m3 = load_any_model(p)
    assert m3.__class__.__name__ == "MLPStockModel"


# ── Registry dispatch across families ───────────────────────────────────────

def test_registry_dispatch_distinguishes_families(tmp_path: Path):
    pytest.importorskip("lightgbm")
    pytest.importorskip("catboost")
    from xgbnew.model_lgb import LGBMStockModel
    from xgbnew.model_cat import CatBoostStockModel
    from xgbnew.model_mlp import MLPStockModel
    from xgbnew.model_registry import load_any_model

    df = _toy_frame(n_rows=600)
    tr, _va = _holdout_split(df)
    paths: dict[str, Path] = {}

    paths["lgb"] = tmp_path / "lgb.pkl"
    LGBMStockModel(n_estimators=30, learning_rate=0.1).fit(tr, FEATURES).save(paths["lgb"])

    paths["cat"] = tmp_path / "cat.pkl"
    CatBoostStockModel(iterations=30, depth=4, learning_rate=0.1).fit(tr, FEATURES).save(paths["cat"])

    paths["mlp"] = tmp_path / "mlp.pkl"
    MLPStockModel(hidden_dims=(16,), epochs=3, batch_size=64,
                  random_state=1, device="cpu").fit(tr, FEATURES, verbose=False).save(paths["mlp"])

    assert load_any_model(paths["lgb"]).__class__.__name__ == "LGBMStockModel"
    assert load_any_model(paths["cat"]).__class__.__name__ == "CatBoostStockModel"
    assert load_any_model(paths["mlp"]).__class__.__name__ == "MLPStockModel"


def test_registry_falls_back_to_xgb_for_legacy_pickle(tmp_path: Path):
    """A pickle without a 'family' key must dispatch to XGBStockModel."""
    from xgbnew.model import XGBStockModel
    from xgbnew.model_registry import load_any_model

    df = _toy_frame(n_rows=400)
    tr, _va = _holdout_split(df)
    m = XGBStockModel(n_estimators=30, max_depth=3, learning_rate=0.1)
    m.fit(tr, FEATURES, verbose=False)
    p = tmp_path / "legacy_xgb.pkl"
    m.save(p)  # XGBStockModel.save does NOT write a family key — legacy path

    # Sanity: legacy pickle indeed has no "family" field
    with open(p, "rb") as f:
        data = pickle.load(f)
    assert "family" not in data

    loaded = load_any_model(p)
    assert loaded.__class__.__name__ == "XGBStockModel"


def test_mismatched_family_load_raises(tmp_path: Path):
    """LGBMStockModel.load(cat_pickle) should refuse."""
    pytest.importorskip("lightgbm")
    pytest.importorskip("catboost")
    from xgbnew.model_cat import CatBoostStockModel
    from xgbnew.model_lgb import LGBMStockModel

    df = _toy_frame(n_rows=300)
    tr, _ = _holdout_split(df)
    p = tmp_path / "cat.pkl"
    CatBoostStockModel(iterations=20, depth=3, learning_rate=0.1).fit(tr, FEATURES).save(p)

    with pytest.raises(ValueError, match="family"):
        LGBMStockModel.load(p)
