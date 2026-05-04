from __future__ import annotations

import json
import sys
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import pandas as pd
import pytest


REPO = Path(__file__).resolve().parents[1]
MODULE_PATH = REPO / "xgbnew" / "train_alltrain_ensemble.py"


def _load_module():
    spec = spec_from_file_location("xgbnew_train_alltrain_ensemble", MODULE_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_write_json_atomic_replaces_manifest_and_removes_temp(tmp_path: Path) -> None:
    mod = _load_module()
    manifest = tmp_path / "ensemble" / "alltrain_ensemble.json"
    manifest.parent.mkdir()
    manifest.write_text("{not-json", encoding="utf-8")

    mod._write_json_atomic(manifest, {"seeds": [0, 7], "models": []})

    assert json.loads(manifest.read_text(encoding="utf-8")) == {
        "seeds": [0, 7],
        "models": [],
    }
    assert not list(manifest.parent.glob(f".{manifest.name}.*.tmp"))


def test_parse_seed_list_rejects_duplicate_production_seed() -> None:
    mod = _load_module()

    try:
        mod._parse_seed_list("0,7,0,42")
    except ValueError as exc:
        assert "duplicate seeds are not allowed: [0]" in str(exc)
    else:
        raise AssertionError("expected duplicate production seed to fail")


def test_parse_seed_list_accepts_unique_seed_strings() -> None:
    mod = _load_module()

    assert mod._parse_seed_list("0, 7,42") == [0, 7, 42]


@pytest.mark.parametrize(
    ("argv", "expected"),
    [
        (["--n-estimators", "0"], "n_estimators must be positive"),
        (["--max-depth", "0"], "max_depth must be positive"),
        (["--learning-rate", "nan"], "learning_rate must be finite and positive"),
        (["--min-dollar-vol", "-1"], "--min-dollar-vol"),
        (["--train-start", "2025-01-02", "--train-end", "2025-01-01"], "--train-start"),
        (["--shapes", "bad"], "--shapes tuple"),
        (["--shapes", "0:5:0.03:42,400:5:0.03:7"], "--shapes tuple 1"),
    ],
)
def test_main_rejects_invalid_training_config_before_symbol_loading(
    monkeypatch,
    tmp_path: Path,
    capsys,
    argv,
    expected,
) -> None:
    mod = _load_module()
    monkeypatch.setattr(mod, "_load_symbols", lambda _path: (_ for _ in ()).throw(AssertionError))
    monkeypatch.setattr(mod, "build_daily_dataset", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError))

    rc = mod.main(
        [
            "--symbols-file", str(tmp_path / "symbols.txt"),
            *argv,
        ]
    )

    assert rc == 1
    assert expected in capsys.readouterr().err


def test_main_rejects_nonpositive_fm_n_latents_before_symbol_loading(
    monkeypatch,
    tmp_path: Path,
    capsys,
) -> None:
    mod = _load_module()
    monkeypatch.setattr(mod, "_load_symbols", lambda _path: (_ for _ in ()).throw(AssertionError))
    monkeypatch.setattr(mod, "load_fm_latents", lambda _path: (_ for _ in ()).throw(AssertionError))

    rc = mod.main(
        [
            "--symbols-file", str(tmp_path / "symbols.txt"),
            "--fm-latents-path", str(tmp_path / "latents.parquet"),
            "--fm-n-latents", "0",
        ]
    )

    assert rc == 1
    assert "--fm-n-latents must be positive" in capsys.readouterr().err


def test_main_rejects_fm_n_latents_above_artifact_before_symbol_loading(
    monkeypatch,
    tmp_path: Path,
    capsys,
) -> None:
    mod = _load_module()
    fm_df = pd.DataFrame(
        {
            "symbol": ["AAPL"],
            "date": [pd.Timestamp("2024-01-02").date()],
            "latent_0": [0.1],
            "latent_1": [0.2],
        }
    )
    monkeypatch.setattr(mod, "load_fm_latents", lambda _path: fm_df)
    monkeypatch.setattr(mod, "_load_symbols", lambda _path: (_ for _ in ()).throw(AssertionError))
    monkeypatch.setattr(mod, "build_daily_dataset", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError))

    rc = mod.main(
        [
            "--symbols-file", str(tmp_path / "symbols.txt"),
            "--fm-latents-path", str(tmp_path / "latents.parquet"),
            "--fm-n-latents", "3",
        ]
    )

    assert rc == 1
    assert "exceeds artifact latent columns (2)" in capsys.readouterr().err


def test_main_records_fm_latents_sha256_in_manifest(
    monkeypatch,
    tmp_path: Path,
) -> None:
    mod = _load_module()
    symbols_file = tmp_path / "symbols.txt"
    symbols_file.write_text("AAPL\nMSFT\n", encoding="utf-8")
    latents_path = tmp_path / "latents.parquet"
    latents_path.write_bytes(b"exact-latent-artifact")
    out_dir = tmp_path / "out"
    fm_df = pd.DataFrame(
        {
            "symbol": ["AAPL"],
            "date": [pd.Timestamp("2024-01-02").date()],
            "latent_0": [0.1],
            "latent_1": [0.2],
        }
    )
    train_df = pd.DataFrame(
        {
            "symbol": ["AAPL", "MSFT"],
            "date": [pd.Timestamp("2024-01-02").date()] * 2,
            "target": [0.01, -0.01],
        }
    )
    fit_calls = []
    saved_paths = []

    class FakeModel:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def fit(self, df, feature_cols, *, verbose):
            fit_calls.append((df.copy(), list(feature_cols), verbose))

        def save(self, path):
            Path(path).write_bytes(b"model-bytes")

    def save_model(model, path):
        saved_paths.append(path)
        model.save(path)

    monkeypatch.setattr(mod, "load_fm_latents", lambda path: fm_df)
    monkeypatch.setattr(
        mod,
        "build_daily_dataset",
        lambda **kwargs: (train_df, pd.DataFrame(), pd.DataFrame()),
    )
    monkeypatch.setattr(mod, "XGBStockModel", FakeModel)
    monkeypatch.setattr(mod, "load_chronos_cache", lambda _path: {})
    monkeypatch.setattr(mod, "save_model_atomic", save_model)

    rc = mod.main(
        [
            "--symbols-file", str(symbols_file),
            "--chronos-cache", str(tmp_path / "missing_chronos"),
            "--fm-latents-path", str(latents_path),
            "--fm-n-latents", "2",
            "--seeds", "0,7",
            "--out-dir", str(out_dir),
        ]
    )

    assert rc == 0
    manifest = json.loads((out_dir / "alltrain_ensemble.json").read_text(encoding="utf-8"))
    config = manifest["config"]
    assert config["fm_latents_path"] == str(latents_path)
    assert config["fm_latents_sha256"] == mod._file_sha256(latents_path)
    assert config["fm_n_latents"] == 2
    assert "latent_0" in config["feature_cols"]
    assert "latent_1" in config["feature_cols"]
    assert "fm_available" in config["feature_cols"]
    assert len(fit_calls) == 2
    assert saved_paths == [out_dir / "alltrain_seed0.pkl", out_dir / "alltrain_seed7.pkl"]
