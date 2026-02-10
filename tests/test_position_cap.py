"""Tests for src.position_cap — per-signal position cap tracker."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

import pytest

from src.position_cap import (
    _cap_path,
    clear_position_cap,
    get_position_cap,
    set_position_cap,
)


@pytest.fixture(autouse=True)
def _tmp_caps_dir(tmp_path, monkeypatch):
    """Redirect cap storage to a temp directory."""
    monkeypatch.setattr("src.position_cap._CAPS_DIR", tmp_path)


def test_set_and_get_cap():
    set_position_cap("SOLUSD", "buy", max_qty=5.0, buy_signal_qty=3.0)
    cap = get_position_cap("SOLUSD", "buy")
    assert cap == pytest.approx(5.0)


def test_get_returns_none_when_not_set():
    assert get_position_cap("BTCUSD", "buy") is None


def test_cap_expires_after_max_age():
    set_position_cap("SOLUSD", "buy", max_qty=5.0, buy_signal_qty=3.0)

    # Patch the stored timestamp to be 3 hours ago
    path = _cap_path("SOLUSD", "buy")
    data = json.loads(path.read_text())
    data["set_at"] = (datetime.now(timezone.utc) - timedelta(hours=3)).isoformat()
    path.write_text(json.dumps(data))

    # Default max_age is 2h, so this should be expired
    assert get_position_cap("SOLUSD", "buy") is None

    # But with a longer max_age, it should still be valid
    assert get_position_cap("SOLUSD", "buy", max_age_hours=4.0) == pytest.approx(5.0)


def test_clear_removes_all_sides():
    set_position_cap("SOLUSD", "buy", max_qty=5.0, buy_signal_qty=3.0)
    set_position_cap("SOLUSD", "sell", max_qty=2.0, buy_signal_qty=2.0)

    assert get_position_cap("SOLUSD", "buy") is not None
    assert get_position_cap("SOLUSD", "sell") is not None

    clear_position_cap("SOLUSD")

    assert get_position_cap("SOLUSD", "buy") is None
    assert get_position_cap("SOLUSD", "sell") is None


def test_different_symbols_independent():
    set_position_cap("SOLUSD", "buy", max_qty=5.0, buy_signal_qty=3.0)
    set_position_cap("BTCUSD", "buy", max_qty=0.1, buy_signal_qty=0.05)

    assert get_position_cap("SOLUSD", "buy") == pytest.approx(5.0)
    assert get_position_cap("BTCUSD", "buy") == pytest.approx(0.1)


def test_overwrite_updates_cap():
    set_position_cap("SOLUSD", "buy", max_qty=5.0, buy_signal_qty=3.0)
    assert get_position_cap("SOLUSD", "buy") == pytest.approx(5.0)

    set_position_cap("SOLUSD", "buy", max_qty=3.0, buy_signal_qty=2.0)
    assert get_position_cap("SOLUSD", "buy") == pytest.approx(3.0)


def test_corrupt_json_returns_none(tmp_path, monkeypatch):
    monkeypatch.setattr("src.position_cap._CAPS_DIR", tmp_path)
    path = _cap_path("SOLUSD", "buy")
    path.write_text("{bad json")
    assert get_position_cap("SOLUSD", "buy") is None


def test_zero_or_negative_qty_returns_none():
    set_position_cap("SOLUSD", "buy", max_qty=0.0, buy_signal_qty=0.0)
    assert get_position_cap("SOLUSD", "buy") is None

    set_position_cap("SOLUSD", "buy", max_qty=-1.0, buy_signal_qty=0.0)
    assert get_position_cap("SOLUSD", "buy") is None
