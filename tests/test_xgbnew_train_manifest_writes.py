from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
from xgbnew import train_alltrain, train_ensemble_family


def _train_frame(rows: int = 5001) -> pd.DataFrame:
    symbols = ["AAPL", "MSFT"]
    return pd.DataFrame(
        {
            "symbol": [symbols[i % len(symbols)] for i in range(rows)],
            "date": [pd.Timestamp("2025-01-02").date()] * rows,
            "target": [0.01] * rows,
        }
    )


class _FakeAlltrainModel:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.fit_calls = []

    def fit(self, df, feature_cols, *, verbose):
        self.fit_calls.append((df, list(feature_cols), verbose))

    def feature_importances(self):
        return pd.Series({"ret_1d": 0.6, "vol_20d": 0.4})

    def save(self, path):
        Path(path).write_bytes(b"model")


class _FakeFamilyModel:
    def __init__(self):
        self.fit_calls = []

    def fit(self, df, feature_cols, *, val_df=None, verbose=False):
        self.fit_calls.append((df, list(feature_cols), val_df, verbose))

    def save(self, path):
        Path(path).write_bytes(b"model")


def test_train_alltrain_writes_metadata_atomically(monkeypatch, tmp_path):
    symbols_file = tmp_path / "symbols.txt"
    symbols_file.write_text("AAPL\nMSFT\n", encoding="utf-8")
    out_path = tmp_path / "model.pkl"
    writes = []
    saved_models = []
    model = _FakeAlltrainModel()

    def save_model(model_arg, path):
        saved_models.append((model_arg, path))
        model_arg.save(path)

    monkeypatch.setattr(train_alltrain, "build_daily_dataset", lambda **_kwargs: (_train_frame(), None, None))
    monkeypatch.setattr(train_alltrain, "load_chronos_cache", lambda _path: {})
    monkeypatch.setattr(train_alltrain, "XGBStockModel", lambda **_kwargs: model)
    monkeypatch.setattr(train_alltrain, "save_model_atomic", save_model)
    monkeypatch.setattr(train_alltrain, "write_json_atomic", lambda path, payload: writes.append((path, payload)))

    rc = train_alltrain.main(
        [
            "--symbols-file",
            str(symbols_file),
            "--chronos-cache",
            str(tmp_path / "missing_chronos"),
            "--out",
            str(out_path),
        ]
    )

    assert rc == 0
    assert saved_models == [(model, out_path)]
    assert out_path.read_bytes() == b"model"
    assert len(model.fit_calls) == 1
    assert len(writes) == 1
    meta_path, payload = writes[0]
    assert meta_path == out_path.with_suffix(".json")
    assert payload["n_symbols_requested"] == 2
    assert payload["n_symbols_with_data"] == 2
    assert payload["n_rows"] == 5001
    assert payload["config"]["device"] == "cpu"
    assert payload["feature_importances_top10"] == {"ret_1d": 0.6, "vol_20d": 0.4}


def test_train_alltrain_invalid_config_fails_before_symbol_loading(
    monkeypatch,
    tmp_path,
    capsys,
):
    monkeypatch.setattr(
        train_alltrain,
        "_load_symbols",
        lambda _path: (_ for _ in ()).throw(AssertionError("symbols loaded")),
    )
    monkeypatch.setattr(
        train_alltrain,
        "build_daily_dataset",
        lambda **_kwargs: (_ for _ in ()).throw(AssertionError("data loaded")),
    )

    rc = train_alltrain.main(
        [
            "--symbols-file",
            str(tmp_path / "symbols.txt"),
            "--n-estimators",
            "0",
        ]
    )

    assert rc == 2
    assert "--n-estimators must be positive" in capsys.readouterr().err


def test_train_ensemble_family_writes_manifest_atomically(monkeypatch, tmp_path):
    symbols_file = tmp_path / "symbols.txt"
    symbols_file.write_text("AAPL\nMSFT\n", encoding="utf-8")
    out_dir = tmp_path / "family"
    writes = []
    saved_paths = []
    built_models = []

    def _build_model(_family, _seed, _device, _args):
        model = _FakeFamilyModel()
        built_models.append(model)
        return model

    def save_model(model_arg, path):
        saved_paths.append(path)
        model_arg.save(path)

    monkeypatch.setattr(train_ensemble_family, "build_daily_dataset", lambda **_kwargs: (_train_frame(), None, None))
    monkeypatch.setattr(train_ensemble_family, "load_chronos_cache", lambda _path: {})
    monkeypatch.setattr(train_ensemble_family, "_build_model", _build_model)
    monkeypatch.setattr(train_ensemble_family, "save_model_atomic", save_model)
    monkeypatch.setattr(
        train_ensemble_family,
        "write_json_atomic",
        lambda path, payload: writes.append((path, payload)),
    )

    rc = train_ensemble_family.main(
        [
            "--family",
            "xgb",
            "--symbols-file",
            str(symbols_file),
            "--chronos-cache",
            str(tmp_path / "missing_chronos"),
            "--seeds",
            "0,7",
            "--out-dir",
            str(out_dir),
        ]
    )

    assert rc == 0
    assert len(built_models) == 2
    assert saved_paths == [out_dir / "alltrain_seed0.pkl", out_dir / "alltrain_seed7.pkl"]
    assert (out_dir / "alltrain_seed0.pkl").read_bytes() == b"model"
    assert (out_dir / "alltrain_seed7.pkl").read_bytes() == b"model"
    assert len(writes) == 1
    manifest_path, payload = writes[0]
    assert manifest_path == out_dir / "alltrain_ensemble.json"
    assert payload["family"] == "xgb"
    assert payload["seeds"] == [0, 7]
    assert [model["seed"] for model in payload["models"]] == [0, 7]
    assert payload["config"]["feature_cols"]
    assert payload["n_symbols_requested"] == 2


@pytest.mark.parametrize(
    ("family", "argv", "expected"),
    [
        ("xgb", ["--seeds", ""], "--seeds"),
        ("xgb", ["--seeds", "0,7,0"], "duplicate seeds"),
        ("xgb", ["--train-start", "2025-01-02", "--train-end", "2025-01-01"], "--train-start"),
        ("xgb", ["--min-dollar-vol", "nan"], "--min-dollar-vol"),
        ("xgb", ["--val-frac", "1"], "--val-frac"),
        ("xgb", ["--n-estimators", "0"], "--n-estimators"),
        ("xgb", ["--max-depth", "0"], "--max-depth"),
        ("xgb", ["--learning-rate", "inf"], "--learning-rate"),
        ("lgb", ["--lgb-num-leaves", "0"], "--lgb-num-leaves"),
        ("mlp", ["--mlp-hidden", "256,0"], "--mlp-hidden"),
        ("mlp", ["--mlp-dropout", "1"], "--mlp-dropout"),
        ("mlp", ["--mlp-lr", "0"], "--mlp-lr"),
        ("mlp", ["--mlp-batch", "0"], "--mlp-batch"),
        ("mlp", ["--mlp-epochs", "0"], "--mlp-epochs"),
        ("mlp", ["--mlp-patience", "-1"], "--mlp-patience"),
        ("mlp", ["--mlp-weight-decay", "-0.1"], "--mlp-weight-decay"),
        ("mlp_muon", ["--muon-hidden", "0"], "--muon-hidden"),
        ("mlp_muon", ["--muon-blocks", "0"], "--muon-blocks"),
        ("mlp_muon", ["--muon-lr", "0"], "--muon-lr"),
        ("mlp_muon", ["--muon-momentum", "nan"], "--muon-momentum"),
        ("xgb_rank", ["--ranker-deciles", "1"], "--ranker-deciles"),
        ("xgb_rank", ["--ranker-sample-weight-clip", "0"], "--ranker-sample-weight-clip"),
    ],
)
def test_train_ensemble_family_invalid_config_fails_before_symbol_loading(
    monkeypatch,
    tmp_path,
    capsys,
    family,
    argv,
    expected,
):
    monkeypatch.setattr(
        train_ensemble_family,
        "_load_symbols",
        lambda _path: (_ for _ in ()).throw(AssertionError("symbols loaded")),
    )
    monkeypatch.setattr(
        train_ensemble_family,
        "build_daily_dataset",
        lambda **_kwargs: (_ for _ in ()).throw(AssertionError("data loaded")),
    )

    rc = train_ensemble_family.main(
        [
            "--family",
            family,
            "--symbols-file",
            str(tmp_path / "symbols.txt"),
            "--out-dir",
            str(tmp_path / "out"),
            *argv,
        ]
    )

    assert rc == 2
    assert expected in capsys.readouterr().err


def test_train_ensemble_family_ignores_irrelevant_family_knobs(
    monkeypatch,
    tmp_path,
):
    symbols_file = tmp_path / "symbols.txt"
    symbols_file.write_text("AAPL\nMSFT\n", encoding="utf-8")
    out_dir = tmp_path / "family"
    monkeypatch.setattr(train_ensemble_family, "build_daily_dataset", lambda **_kwargs: (_train_frame(), None, None))
    monkeypatch.setattr(train_ensemble_family, "load_chronos_cache", lambda _path: {})
    monkeypatch.setattr(train_ensemble_family, "_build_model", lambda *_args: _FakeFamilyModel())
    monkeypatch.setattr(train_ensemble_family, "write_json_atomic", lambda _path, _payload: None)

    rc = train_ensemble_family.main(
        [
            "--family",
            "xgb",
            "--symbols-file",
            str(symbols_file),
            "--out-dir",
            str(out_dir),
            "--mlp-hidden",
            "256,0",
            "--lgb-num-leaves",
            "0",
            "--ranker-deciles",
            "1",
        ]
    )

    assert rc == 0
