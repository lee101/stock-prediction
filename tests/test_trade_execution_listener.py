from __future__ import annotations

import json
from datetime import datetime, timezone

import pytest

from src.trade_execution_monitor import TradeEvent, TradeExecutionMonitor


def test_trade_execution_monitor_records_long_and_short(tmp_path, monkeypatch):
    def _fake_state_file(name: str, suffix: str | None = None, extension: str = ".json"):
        suffix_part = suffix or ""
        return tmp_path / f"{name}{suffix_part}{extension}"

    monkeypatch.setattr("src.trade_execution_monitor.get_state_file", _fake_state_file)

    listener = TradeExecutionMonitor(state_suffix="_unittest")
    now = datetime.now(timezone.utc)

    listener.process_event(TradeEvent(symbol="AAPL", side="buy", quantity=1.0, price=100.0, timestamp=now))
    listener.process_event(TradeEvent(symbol="AAPL", side="sell", quantity=1.0, price=110.0, timestamp=now))

    listener.process_event(TradeEvent(symbol="AAPL", side="sell", quantity=1.0, price=90.0, timestamp=now))
    listener.process_event(TradeEvent(symbol="AAPL", side="buy", quantity=1.0, price=85.0, timestamp=now))

    history_path = tmp_path / "trade_history_unittest.json"
    with history_path.open("r", encoding="utf-8") as handle:
        history = json.load(handle)

    long_key = "AAPL|buy"
    short_key = "AAPL|sell"

    assert long_key in history
    assert short_key in history

    long_entry = history[long_key][-1]
    assert pytest.approx(long_entry["pnl"], abs=1e-6) == 10.0

    short_entry = history[short_key][-1]
    assert pytest.approx(short_entry["pnl"], abs=1e-6) == 5.0
