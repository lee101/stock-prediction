from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd

from marketsimlong.data import load_symbol_data
import run_large_universe_selection as mod


class _FakeForecasts:
    def __init__(self) -> None:
        self.forecasts = {
            "AAPL": type(
                "Forecast",
                (),
                {
                    "current_close": 100.0,
                    "predicted_close": 101.0,
                    "predicted_high": 102.0,
                    "predicted_low": 99.0,
                    "predicted_close_p10": 99.5,
                    "predicted_close_p90": 101.5,
                },
            )()
        }

    def get_ranked_symbols(self, metric: str = "predicted_return", ascending: bool = False):
        return [("AAPL", 0.01)]


class _FakeForecaster:
    def __init__(self, loader, forecast_config) -> None:
        self.loader = loader
        self.forecast_config = forecast_config

    def forecast_all_symbols(self, target_date, tradable):
        assert target_date == date(2026, 1, 10)
        assert tradable == ["AAPL"]
        return _FakeForecasts()


class _FakeLoader:
    def __init__(self, _config) -> None:
        frame = pd.DataFrame(
            {
                "date": pd.date_range("2025-01-01", periods=400, freq="D").date,
            }
        )
        self._data_cache = {"AAPL": frame}

    def load_all_symbols(self) -> None:
        return None

    def get_tradable_symbols_on_date(self, _target_date):
        return ["AAPL"]


def test_configure_chronos_runtime_sets_env(monkeypatch) -> None:
    monkeypatch.delenv("CHRONOS2_PIPELINE_BACKEND", raising=False)
    monkeypatch.delenv("CHRONOS_COMPILE", raising=False)
    monkeypatch.delenv("CHRONOS_COMPILE_MODE", raising=False)

    mod._configure_chronos_runtime(
        pipeline_backend="cutechronos",
        torch_compile=True,
        compile_mode="reduce-overhead",
    )

    assert mod.os.environ["CHRONOS2_PIPELINE_BACKEND"] == "cutechronos"
    assert mod.os.environ["CHRONOS_COMPILE"] == "1"
    assert mod.os.environ["CHRONOS_COMPILE_MODE"] == "reduce-overhead"


def test_main_supports_runtime_knobs_and_writes_json(tmp_path: Path, monkeypatch) -> None:
    symbols_file = tmp_path / "symbols.txt"
    symbols_file.write_text("AAPL\n", encoding="utf-8")
    output_json = tmp_path / "ranked.json"

    monkeypatch.setattr(mod, "DailyDataLoader", _FakeLoader)
    monkeypatch.setattr(mod, "Chronos2Forecaster", _FakeForecaster)

    result = mod.main(
        [
            "--date",
            "2026-01-10",
            "--symbols-file",
            str(symbols_file),
            "--data-root",
            str(tmp_path),
            "--pipeline-backend",
            "cutechronos",
            "--torch-compile",
            "--compile-mode",
            "reduce-overhead",
            "--cross-learning",
            "off",
            "--output-json",
            str(output_json),
        ]
    )

    assert result == 0
    assert mod.os.environ["CHRONOS2_PIPELINE_BACKEND"] == "cutechronos"
    assert mod.os.environ["CHRONOS_COMPILE"] == "1"
    assert mod.os.environ["CHRONOS_COMPILE_MODE"] == "reduce-overhead"
    payload = output_json.read_text(encoding="utf-8")
    assert '"symbol": "AAPL"' in payload


def test_load_symbol_data_prefers_newest_timestamp_over_file_mtime(tmp_path: Path) -> None:
    root = tmp_path / "data"
    train_dir = root / "train"
    test_dir = root / "test"
    train_dir.mkdir(parents=True)
    test_dir.mkdir(parents=True)

    fresh = pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-04-01", periods=3, freq="D", tz="UTC"),
            "open": [1.0, 1.1, 1.2],
            "high": [1.1, 1.2, 1.3],
            "low": [0.9, 1.0, 1.1],
            "close": [1.05, 1.15, 1.25],
            "volume": [1000, 1000, 1000],
        }
    )
    stale = pd.DataFrame(
        {
            "timestamp": pd.date_range("2025-12-29", periods=3, freq="D", tz="UTC"),
            "open": [2.0, 2.1, 2.2],
            "high": [2.1, 2.2, 2.3],
            "low": [1.9, 2.0, 2.1],
            "close": [2.05, 2.15, 2.25],
            "volume": [1000, 1000, 1000],
        }
    )
    fresh_path = train_dir / "AAPL.csv"
    stale_path = test_dir / "AAPL.csv"
    fresh.to_csv(fresh_path, index=False)
    stale.to_csv(stale_path, index=False)

    # Make the stale file newer on disk to prove timestamp wins over mtime.
    stale_mtime = fresh_path.stat().st_mtime + 100
    mod.os.utime(stale_path, (stale_mtime, stale_mtime))

    loaded = load_symbol_data("AAPL", root)
    assert not loaded.empty
    assert str(loaded["timestamp"].iloc[-1]) == "2026-04-03 00:00:00+00:00"
