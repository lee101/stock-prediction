from __future__ import annotations

import json
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
from xgbnew.backtest import PRODUCTION_STOCK_FEE_RATE


REPO = Path(__file__).resolve().parents[1]
MODULE_PATH = REPO / "xgbnew" / "eval_pretrained.py"


class _FakeModel:
    def __init__(self, feature_cols: list[str]):
        self.feature_cols = feature_cols

    def predict_scores(self, df: pd.DataFrame) -> pd.Series:
        missing = [col for col in self.feature_cols if col not in df.columns]
        if missing:
            raise AssertionError(f"missing feature columns: {missing}")
        return pd.Series(0.75, index=df.index)


def _load_module():
    spec = spec_from_file_location("xgbnew_eval_pretrained", MODULE_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _oos_frame(n_days: int = 35, n_symbols: int = 4) -> pd.DataFrame:
    days = pd.date_range("2026-01-02", periods=n_days, freq="B").date
    rows = []
    for day in days:
        for symbol_idx in range(n_symbols):
            rows.append(
                {
                    "date": day,
                    "symbol": f"S{symbol_idx}",
                    "ret_1d": 0.0,
                    "latent_0": 0.1,
                    "latent_1": 0.2,
                    "fm_available": 1.0,
                    "dollar_vol": 10_000_000.0,
                }
            )
    return pd.DataFrame(rows)


def test_eval_pretrained_default_fee_is_production_stress_fee(tmp_path: Path) -> None:
    mod = _load_module()
    model_path = tmp_path / "model.pkl"
    symbols_file = tmp_path / "symbols.txt"
    args = mod.parse_args(
        [
            "--model-path",
            str(model_path),
            "--symbols-file",
            str(symbols_file),
        ]
    )
    assert args.fee_rate == PRODUCTION_STOCK_FEE_RATE


def test_eval_pretrained_requires_fm_latents_for_fm_feature_model(
    monkeypatch,
    tmp_path: Path,
    capsys,
) -> None:
    mod = _load_module()
    model_path = tmp_path / "alltrain_seed0.pkl"
    model_path.write_bytes(b"model")
    symbols_file = tmp_path / "symbols.txt"
    symbols_file.write_text("S0\n", encoding="utf-8")
    monkeypatch.setattr(
        mod,
        "_load_pretrained",
        lambda _path: _FakeModel(["ret_1d", "latent_0", "fm_available"]),
    )
    monkeypatch.setattr(
        mod,
        "build_daily_dataset",
        lambda **_kwargs: (_ for _ in ()).throw(AssertionError("dataset built")),
    )

    rc = mod.main(
        [
            "--model-path",
            str(model_path),
            "--symbols-file",
            str(symbols_file),
        ]
    )

    assert rc == 1
    assert "require FM latents" in capsys.readouterr().err


def test_eval_pretrained_rejects_fm_n_latents_below_model_requirement(
    monkeypatch,
    tmp_path: Path,
    capsys,
) -> None:
    mod = _load_module()
    model_path = tmp_path / "alltrain_seed0.pkl"
    model_path.write_bytes(b"model")
    symbols_file = tmp_path / "symbols.txt"
    symbols_file.write_text("S0\n", encoding="utf-8")
    latents_path = tmp_path / "latents.parquet"
    latents_path.write_bytes(b"latents")
    monkeypatch.setattr(
        mod,
        "_load_pretrained",
        lambda _path: _FakeModel(["ret_1d", "latent_0", "latent_1", "fm_available"]),
    )
    monkeypatch.setattr(
        mod,
        "load_fm_latents",
        lambda _path: (_ for _ in ()).throw(AssertionError("latents loaded")),
    )

    rc = mod.main(
        [
            "--model-path",
            str(model_path),
            "--symbols-file",
            str(symbols_file),
            "--fm-latents-path",
            str(latents_path),
            "--fm-n-latents",
            "1",
        ]
    )

    assert rc == 1
    assert "smaller than model feature_cols require (2)" in capsys.readouterr().err


def test_eval_pretrained_attaches_fm_latents_and_records_provenance(
    monkeypatch,
    tmp_path: Path,
) -> None:
    mod = _load_module()
    model_path = tmp_path / "alltrain_seed0.pkl"
    model_path.write_bytes(b"model")
    symbols_file = tmp_path / "symbols.txt"
    symbols_file.write_text("S0\nS1\n", encoding="utf-8")
    latents_path = tmp_path / "latents.parquet"
    latents_path.write_bytes(b"exact-latent-artifact")
    output_path = tmp_path / "eval.json"
    fm_df = pd.DataFrame(
        {
            "symbol": ["S0"],
            "date": [pd.Timestamp("2026-01-02").date()],
            "latent_0": [0.1],
            "latent_1": [0.2],
        }
    )
    build_calls = []

    monkeypatch.setattr(
        mod,
        "_load_pretrained",
        lambda _path: _FakeModel(["ret_1d", "latent_0", "latent_1", "fm_available"]),
    )
    monkeypatch.setattr(mod, "load_fm_latents", lambda _path: fm_df)

    def _fake_build_daily_dataset(**kwargs):
        build_calls.append(kwargs)
        return pd.DataFrame(), pd.DataFrame(), _oos_frame()

    monkeypatch.setattr(mod, "build_daily_dataset", _fake_build_daily_dataset)
    monkeypatch.setattr(
        mod,
        "simulate",
        lambda *_args, **_kwargs: SimpleNamespace(
            day_results=[SimpleNamespace(day=pd.Timestamp("2026-01-02").date(), trades=[])],
            total_return_pct=10.0,
            sortino_ratio=2.0,
            max_drawdown_pct=3.0,
        ),
    )

    rc = mod.main(
        [
            "--model-path",
            str(model_path),
            "--symbols-file",
            str(symbols_file),
            "--fm-latents-path",
            str(latents_path),
            "--fm-n-latents",
            "2",
            "--output-path",
            str(output_path),
        ]
    )

    assert rc == 0
    assert build_calls
    assert build_calls[0]["fm_latents"] is fm_df
    assert build_calls[0]["fm_n_latents"] == 2
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["fm_latents_path"] == str(latents_path)
    assert payload["fm_latents_sha256"] == mod._file_sha256(latents_path)
    assert payload["fm_n_latents"] == 2
