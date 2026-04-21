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


def test_nonstring_dict_keys_trigger_coerce_fallback(tmp_path: Path) -> None:
    """A tuple key raises TypeError in json.dumps before ``default`` fires;
    the TradeLogger must catch and re-serialise via ``_coerce`` (lines 79-80,
    102-110 — the coerce path)."""
    tlog = TradeLogger(log_dir=tmp_path, session_date=date(2026, 4, 20))
    # tuple key trips json.dumps TypeError, not the default callback
    tlog.log("weird", by_pair={(1, 2): "ab", (3, 4): "cd"})
    rec = _read_lines(tlog.path)[0]
    # _coerce stringified the keys; values preserved
    assert rec["event"] == "weird"
    assert rec["by_pair"]["(1, 2)"] == "ab"
    assert rec["by_pair"]["(3, 4)"] == "cd"


def test_coerce_preserves_lists_and_stringifies_uncoercible(tmp_path: Path) -> None:
    """Force _coerce via non-string keys nested in a list, and include an
    object that isn't json-native (arbitrary class) to exercise the
    ``except TypeError: return str(obj)`` leaf of _coerce (line 110)."""
    class Weird:
        def __repr__(self) -> str:
            return "Weird<x>"

    tlog = TradeLogger(log_dir=tmp_path, session_date=date(2026, 4, 20))
    tlog.log(
        "weird",
        payload=[{(1, 2): Weird()}, [1, 2, 3]],
    )
    rec = _read_lines(tlog.path)[0]
    # list preserved, inner dict tuple key stringified, Weird str'd
    assert rec["payload"][0]["(1, 2)"] == "Weird<x>"
    assert rec["payload"][1] == [1, 2, 3]


def test_multi_element_numpy_array_falls_back_to_str(tmp_path: Path) -> None:
    """A multi-element numpy array has ``.item`` but calling it raises
    ValueError — _json_default must catch and fall through to str()
    (lines 96-98)."""
    tlog = TradeLogger(log_dir=tmp_path, session_date=date(2026, 4, 20))
    arr = np.array([1.0, 2.0, 3.0])
    tlog.log("scored", arr=arr)
    rec = _read_lines(tlog.path)[0]
    # str(arr) prints numpy repr — check shape signature is in there
    assert isinstance(rec["arr"], str)
    assert "1." in rec["arr"]


def test_log_write_failure_is_swallowed(tmp_path: Path) -> None:
    """HARD rule: trade log must never take down the trader. If the write
    fails (e.g. disk full, read-only FS), the ``except Exception: pass``
    branch (lines 86-87) keeps us running."""
    tlog = TradeLogger(log_dir=tmp_path, session_date=date(2026, 4, 20))
    # Make the target path a directory — open(…, 'ab') will raise IsADirectoryError
    assert tlog.path is not None
    tlog.path.unlink(missing_ok=True)
    tlog.path.mkdir()
    try:
        tlog.log("pick", symbol="AAPL")  # must NOT raise
    finally:
        # Cleanup the dir we created so other tests sharing tmp_path don't see it
        tlog.path.rmdir()
