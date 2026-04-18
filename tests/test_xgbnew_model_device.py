"""Device-selection coverage for ``xgbnew.model.XGBStockModel``.

Verifies the ``device`` constructor kwarg flows into XGBClassifier params
so the GPU path is reachable. An end-to-end CUDA-fit is gated by a
``torch.cuda.is_available()``-style probe: we skip when no CUDA build is
present, otherwise we train a tiny model on GPU and confirm predictions
are in [0, 1].
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from xgbnew.model import XGBStockModel


def _xgboost_has_cuda() -> bool:
    try:
        import xgboost
    except ImportError:
        return False
    info = xgboost.build_info()
    return bool(info.get("USE_CUDA"))


def _toy_df(n: int = 400, n_features: int = 4, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, n_features)).astype(np.float32)
    # Target: sign of f0 + f1 (easy signal).
    y = ((X[:, 0] + X[:, 1]) > 0).astype(np.int32)
    cols = [f"feat_{i}" for i in range(n_features)]
    df = pd.DataFrame(X, columns=cols)
    df["target_oc_up"] = y
    return df


def test_default_device_is_none_and_no_device_param():
    m = XGBStockModel(n_estimators=10)
    assert m.device is None
    # Default path must not pin any device (leave XGB to decide).
    assert m.clf.get_params().get("device") in (None, "")


def test_cuda_device_kwarg_is_forwarded_and_sets_hist_tree_method():
    m = XGBStockModel(device="cuda", n_estimators=10)
    params = m.clf.get_params()
    assert m.device == "cuda"
    assert params["device"] == "cuda"
    assert params["tree_method"] == "hist"


def test_cpu_device_string_is_forwarded_verbatim():
    m = XGBStockModel(device="cpu", n_estimators=10)
    params = m.clf.get_params()
    assert params["device"] == "cpu"


def test_explicit_tree_method_is_not_overridden_by_device():
    m = XGBStockModel(device="cuda", tree_method="approx", n_estimators=10)
    params = m.clf.get_params()
    # User explicitly asked for approx — device kwarg must not clobber it.
    assert params["tree_method"] == "approx"


def test_save_load_preserves_device(tmp_path):
    df = _toy_df(n=200)
    feats = [c for c in df.columns if c != "target_oc_up"]
    m = XGBStockModel(device="cpu", n_estimators=10).fit(df, feats, verbose=False)
    path = tmp_path / "m.pkl"
    m.save(path)
    m2 = XGBStockModel.load(path)
    assert m2.device == "cpu"
    # round-trip predictions should work too
    scores = m2.predict_scores(df)
    assert scores.between(0.0, 1.0).all()


@pytest.mark.skipif(not _xgboost_has_cuda(), reason="xgboost not built with CUDA")
def test_end_to_end_cuda_fit_and_predict():
    df = _toy_df(n=500)
    feats = [c for c in df.columns if c != "target_oc_up"]
    m = XGBStockModel(device="cuda", n_estimators=20, max_depth=3).fit(
        df, feats, verbose=False,
    )
    scores = m.predict_scores(df)
    assert scores.between(0.0, 1.0).all()
    # Easy signal — should learn it: at least 70% directional accuracy on train.
    acc = ((scores > 0.5).astype(int).values == df["target_oc_up"].values).mean()
    assert acc > 0.7
