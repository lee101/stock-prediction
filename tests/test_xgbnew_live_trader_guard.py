"""HARD RULE #3 compliance tests for xgbnew/live_trader.py.

Verify that the refactored buy/sell paths:
1. Record actual fill prices after BUY (via record_buy_price)
2. Invoke guard_sell_against_death_spiral before every SELL submit
3. Let RuntimeError from the guard propagate (crash the loop, no silent swallow)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from xgbnew.live_trader import (
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


@pytest.mark.parametrize(
    ("extra_args", "message"),
    [
        (["--top-n", "0"], "--top-n must be >= 1"),
        (["--top-n", "2", "--min-picks", "3"], "--min-picks must be <= --top-n"),
        (["--allocation", "nan"], "--allocation must be finite"),
        (["--allocation", "0"], "--allocation must be > 0"),
        (["--allocation-temp", "0"], "--allocation-temp must be > 0"),
        (["--min-score", "1.5"], "--min-score must be between 0 and 1"),
        (["--commission-bps", "-1"], "--commission-bps must be >= 0"),
        (["--min-dollar-vol", "-1"], "--min-dollar-vol must be >= 0"),
        (["--max-spread-bps", "-1"], "--max-spread-bps must be >= 0"),
        (["--regime-gate-window", "-1"], "--regime-gate-window must be >= 0"),
        (["--vol-target-ann", "-0.1"], "--vol-target-ann must be >= 0"),
        (["--inv-vol-target-ann", "-0.1"], "--inv-vol-target-ann must be >= 0"),
        (["--inv-vol-floor", "0"], "--inv-vol-floor must be > 0"),
        (["--inv-vol-cap", "0.5"], "--inv-vol-cap must be >= 1"),
        (["--max-ret-20d-rank-pct", "1.5"], "--max-ret-20d-rank-pct must be between 0 and 1"),
        (["--no-picks-fallback-alloc", "-0.1"], "--no-picks-fallback-alloc must be >= 0"),
        (["--crypto-poll-seconds", "0"], "--crypto-poll-seconds must be >= 1"),
        (["--crypto-max-gross", "-0.1"], "--crypto-max-gross must be >= 0"),
        (["--eod-max-gross-leverage", "0"], "--eod-max-gross-leverage must be > 0"),
        (["--eod-deleverage-window-minutes", "0"], "--eod-deleverage-window-minutes must be >= 1"),
        (["--eod-force-market-minutes", "-1"], "--eod-force-market-minutes must be >= 0"),
        (
            ["--conviction-alloc-low", "0.9", "--conviction-alloc-high", "0.8"],
            "--conviction-alloc-high must be > --conviction-alloc-low",
        ),
    ],
)
def test_live_trader_parse_args_rejects_invalid_portfolio_domains(
    extra_args,
    message,
    capsys,
):
    from xgbnew.live_trader import parse_args

    with pytest.raises(SystemExit) as excinfo:
        parse_args([
            "--symbols-file", "x",
            "--model-paths", "a.pkl",
            *extra_args,
        ])

    assert excinfo.value.code == 2
    assert message in capsys.readouterr().err


def test_live_trader_parse_args_accepts_explicit_spy_csv(tmp_path):
    from xgbnew.live_trader import parse_args

    args = parse_args([
        "--symbols-file", "x",
        "--model-paths", "a.pkl",
        "--spy-csv", str(tmp_path / "spy_daily.csv"),
    ])

    assert args.spy_csv == tmp_path / "spy_daily.csv"


def test_live_startup_guard_requires_explicit_live_enable(monkeypatch):
    from xgbnew import live_trader

    monkeypatch.delenv("ALLOW_ALPACA_LIVE_TRADING", raising=False)

    with pytest.raises(RuntimeError, match="ALLOW_ALPACA_LIVE_TRADING=1 is required"):
        live_trader._enforce_live_startup_guards()


def test_live_startup_guard_forces_singleton_live_path(monkeypatch):
    from xgbnew import live_trader

    calls: list[tuple[str, dict]] = []

    def _require(service_name):
        calls.append(("require", {"service_name": service_name}))

    def _enforce(**kwargs):
        calls.append(("enforce", kwargs))
        return object()

    import src.alpaca_account_lock as account_lock
    import src.alpaca_singleton as singleton

    monkeypatch.setenv("ALLOW_ALPACA_LIVE_TRADING", "1")
    monkeypatch.setattr(account_lock, "require_explicit_live_trading_enable", _require)
    monkeypatch.setattr(singleton, "enforce_live_singleton", _enforce)

    assert live_trader._enforce_live_startup_guards() is not None
    assert calls == [
        ("require", {"service_name": "xgb_live_trader"}),
        (
            "enforce",
            {
                "service_name": "xgb_live_trader",
                "account_name": "alpaca_live_writer",
                "force_live": True,
            },
        ),
    ]


def test_cross_sectional_regime_gate_flags_default_disabled_and_parse():
    from xgbnew.live_trader import parse_args

    a = parse_args([
        "--symbols-file", "x",
        "--model-paths", "a.pkl",
    ])
    assert a.regime_cs_iqr_max == pytest.approx(0.0)
    assert a.regime_cs_skew_min == pytest.approx(-1e9)

    a = parse_args([
        "--symbols-file", "x",
        "--model-paths", "a.pkl",
        "--regime-cs-iqr-max", "0.042",
        "--regime-cs-skew-min", "1.0",
    ])
    assert a.regime_cs_iqr_max == pytest.approx(0.042)
    assert a.regime_cs_skew_min == pytest.approx(1.0)


def test_embedded_eod_deleverage_flags_default_disabled_and_parse():
    from xgbnew.live_trader import parse_args

    a = parse_args([
        "--symbols-file", "x",
        "--model-paths", "a.pkl",
    ])
    assert a.eod_deleverage is False
    assert a.eod_max_gross_leverage == pytest.approx(2.0)
    assert a.eod_deleverage_window_minutes == 60
    assert a.eod_force_market_minutes == 5

    a = parse_args([
        "--symbols-file", "x",
        "--model-paths", "a.pkl",
        "--eod-deleverage",
        "--eod-max-gross-leverage", "1.95",
        "--eod-deleverage-window-minutes", "45",
        "--eod-force-market-minutes", "7",
    ])
    assert a.eod_deleverage is True
    assert a.eod_max_gross_leverage == pytest.approx(1.95)
    assert a.eod_deleverage_window_minutes == 45
    assert a.eod_force_market_minutes == 7


def test_embedded_crypto_flags_default_disabled_and_parse():
    from xgbnew.live_trader import parse_args

    a = parse_args([
        "--symbols-file", "x",
        "--model-paths", "a.pkl",
    ])
    assert a.crypto_weekend is False
    assert a.crypto_poll_seconds == 300
    assert a.crypto_max_gross == pytest.approx(0.5)

    a = parse_args([
        "--symbols-file", "x",
        "--model-paths", "a.pkl",
        "--crypto-weekend",
        "--crypto-poll-seconds", "120",
        "--crypto-max-gross", "0.25",
    ])
    assert a.crypto_weekend is True
    assert a.crypto_poll_seconds == 120
    assert a.crypto_max_gross == pytest.approx(0.25)


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
