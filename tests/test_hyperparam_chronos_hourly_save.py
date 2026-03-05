from __future__ import annotations

import json
from pathlib import Path

import hyperparam_chronos_hourly as hch
from hyperparam_chronos_hourly import Chronos2HourlyTuner


def _best_row() -> dict:
    return {
        "context_length": 1024,
        "skip_rates": [1],
        "aggregation_method": "single",
        "use_multivariate": False,
        "scaler": "meanstd",
        "price_mae": 1.0,
        "pct_return_mae": 0.01,
        "direction_accuracy": 0.5,
        "latency_s": 0.1,
    }


def _read_model_id(path: Path) -> str:
    payload = json.loads(path.read_text())
    return str(payload["config"]["model_id"])


def test_save_hyperparams_preserves_existing_model_id(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(hch, "HYPERPARAMS_DIR", tmp_path)
    existing = {
        "symbol": "NVDA",
        "model": "chronos2",
        "config": {"model_id": "chronos2_finetuned/NVDA_keep_me/finetuned-ckpt"},
    }
    (tmp_path / "NVDA.json").write_text(json.dumps(existing))

    tuner = Chronos2HourlyTuner(
        model_id="amazon/chronos-2",
        preserve_existing_model_id=True,
    )
    tuner.save_hyperparams({"NVDA": _best_row()})

    saved_path = tmp_path / "NVDA.json"
    assert _read_model_id(saved_path) == "chronos2_finetuned/NVDA_keep_me/finetuned-ckpt"
    payload = json.loads(saved_path.read_text())
    assert payload["metadata"]["model_id_source"] == "existing_hyperparams"


def test_save_hyperparams_can_disable_existing_model_id_preserve(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(hch, "HYPERPARAMS_DIR", tmp_path)
    existing = {
        "symbol": "NVDA",
        "model": "chronos2",
        "config": {"model_id": "chronos2_finetuned/NVDA_keep_me/finetuned-ckpt"},
    }
    (tmp_path / "NVDA.json").write_text(json.dumps(existing))

    tuner = Chronos2HourlyTuner(
        model_id="amazon/chronos-2",
        preserve_existing_model_id=False,
    )
    tuner.save_hyperparams({"NVDA": _best_row()})

    saved_path = tmp_path / "NVDA.json"
    assert _read_model_id(saved_path) == "amazon/chronos-2"
    payload = json.loads(saved_path.read_text())
    assert payload["metadata"]["model_id_source"] == "tuner_default"
