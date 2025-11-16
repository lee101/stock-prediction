from __future__ import annotations

import json
from pathlib import Path

import pytest

from strategytraining.symbol_sources import load_trade_stock_symbols
from src.models.chronos2_wrapper import Chronos2OHLCWrapper


def _read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def test_trade_symbols_have_chronos_assets():
    symbols = load_trade_stock_symbols()
    hyper_root = Path("hyperparams/chronos2")
    best_root = Path("hyperparams/best")
    preaug_root = Path("preaugstrategies/chronos2")

    for symbol in symbols:
        hyper_path = hyper_root / f"{symbol}.json"
        assert hyper_path.exists(), f"Missing Chronos2 hyperparams for {symbol}: {hyper_path}"
        payload = _read_json(hyper_path)
        config = payload.get("config") or {}
        assert config.get("model_id"), f"{symbol} Chronos2 config missing model_id"
        assert payload.get("validation"), f"{symbol} Chronos2 config missing validation metrics"
        assert payload.get("test"), f"{symbol} Chronos2 config missing test metrics"

        best_path = best_root / f"{symbol}.json"
        assert best_path.exists(), f"Missing best-model selection for {symbol}: {best_path}"
        best_payload = _read_json(best_path)
        assert best_payload.get("model"), f"{symbol} best-model record missing model key"

        preaug_path = preaug_root / f"{symbol}.json"
        assert preaug_path.exists(), f"Missing Chronos2 pre-augmentation for {symbol}: {preaug_path}"
        preaug_payload = _read_json(preaug_path)
        assert preaug_payload.get("best_strategy"), f"{symbol} pre-augmentation missing best_strategy"
        comparison = preaug_payload.get("comparison") or {}
        assert comparison, f"{symbol} pre-augmentation missing comparison metrics"
        assert preaug_payload.get("metadata"), f"{symbol} pre-augmentation missing metadata"


def test_chronos2_wrapper_prefers_frequency_specific_preaug(monkeypatch: pytest.MonkeyPatch):
    dummy_pipeline = type("DummyPipeline", (), {"model": object()})()
    monkeypatch.setenv("CHRONOS2_FREQUENCY", "hourly")

    wrapper = Chronos2OHLCWrapper(dummy_pipeline)
    selector = wrapper._preaug_selector  # pylint: disable=protected-access
    assert selector is not None, "Chronos2 wrapper failed to build pre-augmentation selector"

    expected_order = (
        Path("preaugstrategies") / "chronos2" / "hourly",
        Path("preaugstrategies") / "best" / "hourly",
        Path("preaugstrategies") / "chronos2",
        Path("preaugstrategies") / "best",
    )
    assert selector._best_dirs[:4] == expected_order  # pylint: disable=protected-access

    monkeypatch.delenv("CHRONOS2_FREQUENCY")
