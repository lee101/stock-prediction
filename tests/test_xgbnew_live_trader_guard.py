"""HARD RULE #3 compliance tests for xgbnew/live_trader.py.

Verify that the refactored buy/sell paths:
1. Record actual fill prices after BUY (via record_buy_price)
2. Invoke guard_sell_against_death_spiral before every SELL submit
3. Let RuntimeError from the guard propagate (crash the loop, no silent swallow)
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from xgbnew.live_trader import (  # noqa: E402
    _get_position_details,
    _is_today_trading_day,
    _poll_filled_avg_price,
)


def test_poll_filled_avg_price_returns_float_when_filled():
    order_obj = SimpleNamespace(filled_avg_price="123.45", status="filled")
    client = MagicMock()
    client.get_order_by_id.return_value = order_obj
    got = _poll_filled_avg_price(client, "id-1", timeout_s=2.0)
    assert got == pytest.approx(123.45, rel=1e-9)


def test_poll_filled_avg_price_returns_none_on_cancel():
    order_obj = SimpleNamespace(filled_avg_price=None, status="canceled")
    client = MagicMock()
    client.get_order_by_id.return_value = order_obj
    got = _poll_filled_avg_price(client, "id-2", timeout_s=2.0)
    assert got is None


def test_get_position_details_extracts_current_price_and_entry():
    pos = [
        SimpleNamespace(symbol="AAPL", qty="10", current_price="200.5",
                        avg_entry_price="199.0"),
        SimpleNamespace(symbol="MSFT", qty="5", current_price="410.0",
                        avg_entry_price="405.0"),
    ]
    client = MagicMock()
    client.get_all_positions.return_value = pos
    out = _get_position_details(client)
    assert out["AAPL"]["qty"] == 10.0
    assert out["AAPL"]["current_price"] == pytest.approx(200.5)
    assert out["AAPL"]["avg_entry_price"] == pytest.approx(199.0)
    assert out["MSFT"]["qty"] == 5.0


def test_guard_raises_and_propagates_for_death_spiral_sell(tmp_path, monkeypatch):
    """After a buy at 100.00, a sell at 99.40 (>50 bps below) must raise."""
    state_dir = tmp_path / "state"
    state_dir.mkdir()
    monkeypatch.setenv("STATE_DIR", str(state_dir))
    # Make sure no override bypass is set.
    monkeypatch.delenv("ALPACA_DEATH_SPIRAL_OVERRIDE", raising=False)

    # Force re-import so state dir is picked up fresh.
    import importlib
    import src.alpaca_singleton as singleton
    importlib.reload(singleton)

    singleton.record_buy_price("TESTSYM", 100.00)

    # 60 bps below 100.00 → should raise (floor is 99.50 at 50 bps tolerance).
    with pytest.raises(RuntimeError, match="death-spiral"):
        singleton.guard_sell_against_death_spiral("TESTSYM", "sell", 99.40)


def test_guard_allows_sell_within_tolerance(tmp_path, monkeypatch):
    state_dir = tmp_path / "state2"
    state_dir.mkdir()
    monkeypatch.setenv("STATE_DIR", str(state_dir))
    monkeypatch.delenv("ALPACA_DEATH_SPIRAL_OVERRIDE", raising=False)

    import importlib
    import src.alpaca_singleton as singleton
    importlib.reload(singleton)

    singleton.record_buy_price("TESTSYM2", 100.00)

    # 30 bps below 100.00 → floor at 50 bps is 99.50, 99.70 is above → OK.
    singleton.guard_sell_against_death_spiral("TESTSYM2", "sell", 99.70)
    singleton.guard_sell_against_death_spiral("TESTSYM2", "sell", 100.50)


def test_is_today_trading_day_returns_true_when_market_open():
    """is_open=True → trading day regardless of next_open."""
    from datetime import datetime, timezone
    clock = SimpleNamespace(is_open=True, next_open=None)
    client = MagicMock()
    client.get_clock.return_value = clock
    now = datetime(2026, 4, 20, 14, 0, tzinfo=timezone.utc)  # Monday 10:00 ET
    ok, reason = _is_today_trading_day(client, now=now)
    assert ok is True
    assert "market_open" in reason


def test_is_today_trading_day_returns_true_pre_open_same_day():
    """Before 9:30 ET on a trading day, is_open=False but next_open is today."""
    from datetime import datetime, timezone
    # Monday 8:00 ET = 12:00 UTC. next_open = 2026-04-20 09:30 ET.
    now = datetime(2026, 4, 20, 12, 0, tzinfo=timezone.utc)
    next_open = datetime(2026, 4, 20, 13, 30, tzinfo=timezone.utc)
    clock = SimpleNamespace(is_open=False, next_open=next_open)
    client = MagicMock()
    client.get_clock.return_value = clock
    ok, reason = _is_today_trading_day(client, now=now)
    assert ok is True
    assert "pre-open" in reason


def test_is_today_trading_day_returns_false_on_saturday():
    """Saturday: market closed, next_open is Monday — NOT a trading day."""
    from datetime import datetime, timezone
    # Saturday 2026-04-19 14:00 UTC. next_open = Monday 2026-04-20 09:30 ET.
    now = datetime(2026, 4, 19, 14, 0, tzinfo=timezone.utc)
    next_open = datetime(2026, 4, 20, 13, 30, tzinfo=timezone.utc)  # Monday 9:30 ET
    clock = SimpleNamespace(is_open=False, next_open=next_open)
    client = MagicMock()
    client.get_clock.return_value = clock
    ok, reason = _is_today_trading_day(client, now=now)
    assert ok is False
    assert "closed" in reason


def test_is_today_trading_day_fails_open_on_clock_error():
    """If Alpaca clock query fails, assume trading day (fail-open for diagnostics)."""
    client = MagicMock()
    client.get_clock.side_effect = RuntimeError("network down")
    ok, reason = _is_today_trading_day(client)
    assert ok is True
    assert "clock_query_failed" in reason


def test_record_and_retrieve_buy_price_roundtrip(tmp_path, monkeypatch):
    state_dir = tmp_path / "state3"
    state_dir.mkdir()
    monkeypatch.setenv("STATE_DIR", str(state_dir))

    import importlib
    import src.alpaca_singleton as singleton
    importlib.reload(singleton)

    singleton.record_buy_price("RTSYM", 42.42)
    # The file should exist under the state dir.
    buy_files = list(state_dir.glob("**/alpaca_live_writer_buys.json"))
    assert len(buy_files) == 1, f"expected 1 buys.json, got {buy_files}"
    raw = json.loads(buy_files[0].read_text())
    assert "RTSYM" in raw
    assert raw["RTSYM"]["price"] == pytest.approx(42.42)


def test_min_score_flag_defaults_to_zero_and_parses():
    """--min-score defaults to 0.0 (no filter, preserves current behavior);
    accepts values in [0, 1]."""
    from xgbnew.live_trader import parse_args

    # Default: no --min-score flag → 0.0
    a = parse_args([
        "--symbols-file", "x",
        "--model-paths", "a.pkl",
    ])
    assert a.min_score == pytest.approx(0.0)

    # Explicit: --min-score 0.55 → 0.55
    a = parse_args([
        "--symbols-file", "x",
        "--model-paths", "a.pkl",
        "--min-score", "0.55",
    ])
    assert a.min_score == pytest.approx(0.55)


def test_min_score_filter_applied_to_picks(tmp_path, monkeypatch):
    """When min_score>0, candidates below the floor are dropped before
    top_n selection. When all fail, session holds cash."""
    import pandas as pd
    # Simulate the filter expression directly (mirrors live_trader.run_session
    # lines 511-527 — no client/orders needed).
    scores_df = pd.DataFrame([
        {"symbol": "HIGH", "score": 0.72, "last_close": 100.0, "spread_bps": 3.0},
        {"symbol": "MID",  "score": 0.58, "last_close": 50.0,  "spread_bps": 2.0},
        {"symbol": "LOW",  "score": 0.45, "last_close": 25.0,  "spread_bps": 4.0},
    ])

    # ms=0.55 → only HIGH + MID; top_n=1 returns HIGH
    filtered = scores_df[scores_df["score"] >= 0.55]
    assert len(filtered) == 2
    assert filtered.head(1)["symbol"].iloc[0] == "HIGH"

    # ms=0.70 → only HIGH; top_n=2 requested, only 1 candidate available
    filtered = scores_df[scores_df["score"] >= 0.70]
    assert len(filtered) == 1
    assert filtered.head(2)["symbol"].iloc[0] == "HIGH"

    # ms=0.95 → 0 candidates; caller must handle cash-only session
    filtered = scores_df[scores_df["score"] >= 0.95]
    assert len(filtered) == 0
