"""Tests for scripts/chronos_wide_screener.py."""
from __future__ import annotations

import json
import sys
import unittest.mock as mock
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts.chronos_wide_screener import (
    BACKEND_CONFIGS,
    ScreenerResult,
    _load_daily_csv,
    _load_symbols_file,
)


# ─── helpers ──────────────────────────────────────────────────────────────────

def _make_daily_csv(tmp_path: Path, symbol: str = "TEST", rows: int = 300) -> Path:
    # Use recent dates so staleness filter doesn't reject them
    dates = pd.date_range(end=pd.Timestamp.now(tz="UTC"), periods=rows, freq="B")
    close = 100.0 + np.cumsum(np.random.randn(rows) * 0.5)
    df = pd.DataFrame(
        {
            "timestamp": dates.strftime("%Y-%m-%d %H:%M:%S+00:00"),
            "open": close * 0.995,
            "high": close * 1.01,
            "low": close * 0.99,
            "close": close,
            "volume": 1_000_000,
        }
    )
    path = tmp_path / f"{symbol}.csv"
    df.to_csv(path, index=False)
    return path


# ─── _load_symbols_file ────────────────────────────────────────────────────────

def test_load_symbols_file(tmp_path: Path) -> None:
    symbols_file = tmp_path / "symbols.txt"
    symbols_file.write_text("AAPL\nMSFT\n# comment\nNVDA\n  TSLA  \n")
    result = _load_symbols_file(symbols_file)
    assert result == ["AAPL", "MSFT", "NVDA", "TSLA"]


def test_load_symbols_file_empty(tmp_path: Path) -> None:
    symbols_file = tmp_path / "empty.txt"
    symbols_file.write_text("# only comments\n\n")
    result = _load_symbols_file(symbols_file)
    assert result == []


# ─── _load_daily_csv ──────────────────────────────────────────────────────────

def test_load_daily_csv_returns_dataframe(tmp_path: Path) -> None:
    _make_daily_csv(tmp_path, "TEST", rows=300)
    df = _load_daily_csv("TEST", tmp_path, context_length=128)
    assert df is not None
    assert len(df) == 128
    assert set(["timestamp", "open", "high", "low", "close"]).issubset(df.columns)


def test_load_daily_csv_too_short_returns_none(tmp_path: Path) -> None:
    _make_daily_csv(tmp_path, "SHORT", rows=10)
    result = _load_daily_csv("SHORT", tmp_path, context_length=128)
    assert result is None


def test_load_daily_csv_missing_symbol_returns_none(tmp_path: Path) -> None:
    result = _load_daily_csv("XXXYYYY", tmp_path, context_length=128)
    assert result is None


def test_load_daily_csv_searches_subdirectories(tmp_path: Path) -> None:
    sub = tmp_path / "train"
    sub.mkdir()
    _make_daily_csv(sub, "SUBTEST", rows=200)
    df = _load_daily_csv("SUBTEST", tmp_path, context_length=100)
    assert df is not None
    assert len(df) == 100


# ─── BACKEND_CONFIGS ──────────────────────────────────────────────────────────

def test_backend_configs_keys() -> None:
    for key, cfg in BACKEND_CONFIGS.items():
        assert "pipeline_backend" in cfg
        assert cfg["pipeline_backend"] in ("chronos", "cutechronos")


# ─── ScreenerResult ──────────────────────────────────────────────────────────

def test_screener_result_serializes_to_dict() -> None:
    result = ScreenerResult(
        symbol="AAPL",
        last_close=200.0,
        predicted_close=210.0,
        predicted_return_pct=5.0,
        predicted_high=215.0,
        predicted_low=205.0,
        context_rows=256,
        inference_ms=100.0,
        lora_applied=False,
        preaug_strategy=None,
    )
    from dataclasses import asdict
    d = asdict(result)
    assert d["symbol"] == "AAPL"
    assert d["predicted_return_pct"] == 5.0


# ─── main (integration, mocked) ──────────────────────────────────────────────

def _make_mock_wrapper(return_pct: float = 0.05):
    """Create a mock Chronos2OHLCWrapper that returns a fixed prediction."""

    @dataclass
    class FakeBatch:
        median: pd.DataFrame

    def predict_ohlc(df, *, symbol, prediction_length, context_length, **kw):
        last_close = float(df["close"].iloc[-1])
        pred = last_close * (1.0 + return_pct)
        dates = pd.date_range("2025-12-01", periods=1, freq="B", tz="UTC")
        med = pd.DataFrame(
            {"close": [pred], "high": [pred * 1.01], "low": [pred * 0.99], "open": [pred]},
            index=dates,
        )
        med.index.name = "timestamp"
        med.columns.name = "target_name"
        return FakeBatch(median=med)

    wrapper = mock.MagicMock()
    wrapper.predict_ohlc.side_effect = predict_ohlc
    return wrapper


def test_run_screener_ranks_correctly(tmp_path: Path) -> None:
    from scripts.chronos_wide_screener import run_screener

    _make_daily_csv(tmp_path, "HIGH", rows=300)
    _make_daily_csv(tmp_path, "LOW", rows=300)

    # Patch wrapper construction and predict to control output
    high_wrapper = _make_mock_wrapper(return_pct=0.10)
    low_wrapper = _make_mock_wrapper(return_pct=0.01)

    call_count = [0]

    def mock_predict(df, *, symbol, prediction_length, context_length, **kw):
        call_count[0] += 1
        if symbol == "HIGH":
            return high_wrapper.predict_ohlc(df, symbol=symbol, prediction_length=prediction_length, context_length=context_length)
        return low_wrapper.predict_ohlc(df, symbol=symbol, prediction_length=prediction_length, context_length=context_length)

    combined_wrapper = mock.MagicMock()
    combined_wrapper.predict_ohlc.side_effect = mock_predict

    with mock.patch("scripts.chronos_wide_screener._build_chronos_wrapper", return_value=combined_wrapper):
        results = run_screener(
            ["HIGH", "LOW"],
            data_root=tmp_path,
            backend="cute_fp32",
            context_length=128,
            device_map="cpu",
            use_multivariate=True,
        )

    assert len(results) == 2
    # Results sorted by predicted_return_pct descending
    assert results[0].symbol == "HIGH"
    assert results[0].predicted_return_pct > results[1].predicted_return_pct


def test_main_writes_output(tmp_path: Path) -> None:
    from scripts.chronos_wide_screener import main

    _make_daily_csv(tmp_path, "AAPL", rows=300)
    _make_daily_csv(tmp_path, "MSFT", rows=300)
    out_dir = tmp_path / "screener_out"

    mock_wrapper = _make_mock_wrapper(return_pct=0.03)

    with mock.patch("scripts.chronos_wide_screener._build_chronos_wrapper", return_value=mock_wrapper):
        rc = main([
            "--symbols", "AAPL,MSFT",
            "--data-root", str(tmp_path),
            "--backend", "cute_fp32",
            "--context-length", "64",
            "--top-n", "5",
            "--output-dir", str(out_dir),
        ])

    assert rc == 0
    json_files = list(out_dir.glob("screener_2*.json"))  # timestamped, not "screener_latest.json"
    assert len(json_files) == 1
    payload = json.loads(json_files[0].read_text())
    assert payload["symbols_screened"] == 2
    assert len(payload["top_candidates"]) == 2
    assert (out_dir / "screener_latest.json").exists()
