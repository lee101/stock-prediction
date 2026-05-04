from __future__ import annotations

import numpy as np
from xgbnew import model as xgb_model
from xgbnew import model_base, model_xgb_ranker


def test_xgb_stock_model_save_uses_atomic_pickle_writer(tmp_path, monkeypatch) -> None:
    calls = []

    def write_pickle(path, payload):
        calls.append((path, payload))

    monkeypatch.setattr(xgb_model, "write_pickle_atomic", write_pickle)
    model = xgb_model.XGBStockModel.__new__(xgb_model.XGBStockModel)
    model.clf = "clf"
    model.feature_cols = ["f0", "f1"]
    model._col_medians = np.array([1.0, 2.0])
    model.device = "cpu"

    out = tmp_path / "xgb.pkl"
    model.save(out)

    assert calls == [
        (
            out,
            {
                "clf": "clf",
                "feature_cols": ["f0", "f1"],
                "col_medians": model._col_medians,
                "device": "cpu",
            },
        )
    ]


def test_base_binary_model_save_uses_atomic_pickle_writer(tmp_path, monkeypatch) -> None:
    calls = []

    def write_pickle(path, payload):
        calls.append((path, payload))

    monkeypatch.setattr(model_base, "write_pickle_atomic", write_pickle)
    model = model_base.BaseBinaryDailyModel(device="cpu")
    model.family = "test_family"
    model.clf = "clf"
    model.feature_cols = ["f0"]
    model._col_medians = np.array([3.0])

    out = tmp_path / "family.pkl"
    model.save(out)

    assert calls == [
        (
            out,
            {
                "family": "test_family",
                "clf": "clf",
                "feature_cols": ["f0"],
                "col_medians": model._col_medians,
                "device": "cpu",
                "state": None,
            },
        )
    ]


def test_xgb_ranker_save_uses_atomic_pickle_writer(tmp_path, monkeypatch) -> None:
    calls = []

    def write_pickle(path, payload):
        calls.append((path, payload))

    monkeypatch.setattr(model_xgb_ranker, "write_pickle_atomic", write_pickle)
    model = model_xgb_ranker.XGBRankerStockModel.__new__(model_xgb_ranker.XGBRankerStockModel)
    model.clf = "clf"
    model.feature_cols = ["f0"]
    model._col_medians = np.array([4.0])
    model.device = "cpu"
    model.n_deciles = 10
    model.sample_weight_mode = "abs_target"
    model.sample_weight_clip = 0.03

    out = tmp_path / "ranker.pkl"
    model.save(out)

    assert calls == [
        (
            out,
            {
                "family": "xgb_rank",
                "clf": "clf",
                "feature_cols": ["f0"],
                "col_medians": model._col_medians,
                "device": "cpu",
                "n_deciles": 10,
                "sample_weight_mode": "abs_target",
                "sample_weight_clip": 0.03,
            },
        )
    ]
