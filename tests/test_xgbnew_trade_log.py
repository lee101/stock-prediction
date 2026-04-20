"""Unit tests for xgbnew.trade_log — per-session JSONL logger + residuals.

These cover the behaviours we rely on in live:

  - disabled=True is a no-op (TradeLogger must never raise on bad fields)
  - log() writes one JSON line per event with `ts` and `event` keys
  - non-JSON-native values (numpy, dates, dataframes dicts) round-trip
  - slippage_bps returns None on bad inputs, correct sign and magnitude
  - log file is created under the given log_dir named YYYY-MM-DD.jsonl
"""
from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import numpy as np
import pytest

from xgbnew.trade_log import TradeLogger, slippage_bps


def _read_lines(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def test_disabled_is_noop(tmp_path: Path) -> None:
    tlog = TradeLogger(log_dir=tmp_path, disabled=True)
    tlog.log("session_start", equity_pre=None)
    tlog.log("buy_submitted", symbol="AAPL", qty=1.0)
    assert tlog.path is None
    assert list(tmp_path.iterdir()) == []


def test_writes_jsonl_per_event(tmp_path: Path) -> None:
    d = date(2026, 4, 20)
    tlog = TradeLogger(log_dir=tmp_path, session_date=d)
    tlog.log("session_start", paper=True)
    tlog.log("pick", symbol="AAPL", score=0.72)
    path = tmp_path / "2026-04-20.jsonl"
    assert tlog.path == path
    lines = _read_lines(path)
    assert [r["event"] for r in lines] == ["session_start", "pick"]
    assert lines[0]["paper"] is True
    assert lines[1]["symbol"] == "AAPL"
    assert abs(lines[1]["score"] - 0.72) < 1e-9
    assert all("ts" in r for r in lines)


def test_non_native_values_round_trip(tmp_path: Path) -> None:
    tlog = TradeLogger(log_dir=tmp_path, session_date=date(2026, 4, 20))
    tlog.log(
        "scored",
        top20=[{"symbol": "AAPL", "score": np.float32(0.5),
                "per_seed_scores": [np.float32(0.4), np.float32(0.6)]}],
        when=date(2026, 4, 20),
    )
    rec = _read_lines(tlog.path)[0]
    assert rec["top20"][0]["symbol"] == "AAPL"
    assert abs(rec["top20"][0]["score"] - 0.5) < 1e-6
    assert rec["top20"][0]["per_seed_scores"] == pytest.approx([0.4, 0.6], abs=1e-6)
    assert rec["when"] == "2026-04-20"


def test_slippage_bps_signs() -> None:
    assert slippage_bps(101.0, 100.0) == pytest.approx(100.0)   # paid 100bps up
    assert slippage_bps(99.0, 100.0) == pytest.approx(-100.0)   # filled 100bps below
    assert slippage_bps(None, 100.0) is None
    assert slippage_bps(100.0, None) is None
    assert slippage_bps(100.0, 0.0) is None
    assert slippage_bps(100.0, -5.0) is None
    assert slippage_bps("not-a-number", 100.0) is None


def test_append_mode_multiple_sessions(tmp_path: Path) -> None:
    """Re-opening a logger for the same date must APPEND, not truncate."""
    d = date(2026, 4, 20)
    a = TradeLogger(log_dir=tmp_path, session_date=d)
    a.log("a_event", n=1)
    b = TradeLogger(log_dir=tmp_path, session_date=d)
    b.log("b_event", n=2)
    lines = _read_lines(a.path)
    assert [r["event"] for r in lines] == ["a_event", "b_event"]
    assert [r["n"] for r in lines] == [1, 2]
