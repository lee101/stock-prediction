from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import pytest

from stockagentcombined import simulation as sim


@dataclass
class _DummyBundle:
    bars: dict[str, object]
    _trading_days: Sequence[pd.Timestamp]

    def trading_days(self) -> list[pd.Timestamp]:
        return list(self._trading_days)


class _DummyBuilder:
    def __init__(self, *, generator, config):
        self.generator = generator
        self.config = config


class _DummyGenerator:
    pass


def _install_mocks(monkeypatch: pytest.MonkeyPatch, record: dict) -> None:
    trading_days = pd.date_range("2024-01-01", periods=5, freq="B")
    bundle = _DummyBundle(bars={"AAPL": object()}, _trading_days=trading_days)

    def fake_fetch_latest_ohlc(*, symbols, lookback_days, as_of, local_data_dir, allow_remote_download):
        record["fetch_symbols"] = tuple(symbols)
        record["fetch_lookback"] = lookback_days
        record["fetch_allow_remote"] = allow_remote_download
        record["fetch_local_dir"] = Path(local_data_dir)
        return bundle

    def fake_run_simulation(*, builder, market_frames, trading_days, starting_cash, strategies):
        record["builder"] = builder
        record["market_frames"] = market_frames
        record["trading_days"] = list(trading_days)
        record["starting_cash"] = starting_cash
        record["strategies"] = strategies
        return None

    class BuilderProxy(_DummyBuilder):
        def __init__(self, generator, config):
            super().__init__(generator=generator, config=config)
            record["config"] = config

    monkeypatch.setattr(sim, "fetch_latest_ohlc", fake_fetch_latest_ohlc)
    monkeypatch.setattr(sim, "CombinedForecastGenerator", _DummyGenerator)
    monkeypatch.setattr(sim, "CombinedPlanBuilder", BuilderProxy)
    monkeypatch.setattr(sim, "run_simulation", fake_run_simulation)


def test_main_offline_preset(monkeypatch: pytest.MonkeyPatch) -> None:
    record: dict[str, object] = {}
    _install_mocks(monkeypatch, record)

    sim.main(
        [
            "--preset",
            "offline-regression",
            "--symbols",
            "AAPL",
            "MSFT",
            "--lookback-days",
            "120",
        ]
    )

    config = record["config"]
    assert config.simulation_days == 3
    assert config.min_history == 10
    assert config.min_signal == 0.0
    assert config.error_multiplier == 0.25
    assert config.base_quantity == 10.0
    assert config.min_quantity == 1.0

    assert record["starting_cash"] == 250_000.0
    assert len(record["trading_days"]) == 3
    assert record["fetch_allow_remote"] is False
    assert record["fetch_symbols"] == ("AAPL", "MSFT")
    assert len(record["strategies"]) == 2
    assert {type(strategy).__name__ for strategy in record["strategies"]} == {"ProbeTradeStrategy", "ProfitShutdownStrategy"}


def test_main_manual_overrides(monkeypatch: pytest.MonkeyPatch) -> None:
    record: dict[str, object] = {}
    _install_mocks(monkeypatch, record)

    sim.main(
        [
            "--symbols",
            "AMD",
            "NVDA",
            "--simulation-days",
            "2",
            "--starting-cash",
            "123456",
            "--allow-remote-data",
            "--min-signal",
            "0.123",
        ]
    )

    config = record["config"]
    assert config.simulation_days == 2
    assert config.starting_cash == 123456
    assert config.min_signal == 0.123

    assert record["starting_cash"] == 123456
    assert record["fetch_allow_remote"] is True
    assert record["fetch_symbols"] == ("AMD", "NVDA")
    assert len(record["trading_days"]) == 2
