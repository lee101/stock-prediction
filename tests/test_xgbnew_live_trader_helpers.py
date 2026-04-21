"""Unit tests for the helpers extracted from run_session / run_session_hold_through.

Pins the behavior of the small-scope functions so refactors can't silently
change trade semantics. In particular:

- HARD RULE #3: every SELL path calls ``guard_sell_against_death_spiral``
  BEFORE ``_submit_market_order``.
- ``_execute_buys`` records buy_price for every successful submit, and when
  ``target_syms`` restricts the iteration, only those symbols are traded.
- Price fallbacks: ``current_price==0`` → ``avg_entry_price``. If both 0 the
  sell is skipped (the guard can't be invoked safely).
- Order-of-operations between the guard + the submit + the trade-log events.
"""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from xgbnew import live_trader  # noqa: E402
from xgbnew.trade_log import TradeLogger  # noqa: E402


class _CapturingTlog(TradeLogger):
    """TradeLogger that captures every event as a list of (type, fields) pairs."""

    def __init__(self):
        super().__init__(disabled=True)
        self.events: list[tuple[str, dict]] = []

    def log(self, event_type: str, **fields) -> None:  # type: ignore[override]
        self.events.append((event_type, dict(fields)))


# ── _execute_sells ────────────────────────────────────────────────────────────


def test_execute_sells_calls_guard_before_submit(monkeypatch):
    """HARD RULE #3 — guard_sell_against_death_spiral runs before _submit_market_order."""
    import src.alpaca_singleton as singleton

    events: list[tuple[str, str, float]] = []

    def _guard(sym, side, price):
        events.append(("guard", sym, float(price)))

    def _submit(client, *, symbol, qty, side):
        events.append(("submit", symbol, float(qty)))
        return SimpleNamespace(id=f"id-{symbol}")

    monkeypatch.setattr(singleton, "guard_sell_against_death_spiral", _guard)
    monkeypatch.setattr(live_trader, "_submit_market_order", _submit)

    tlog = _CapturingTlog()
    position_details = {
        "AAA": {"qty": 5.0, "current_price": 100.0, "avg_entry_price": 99.0},
    }
    live_trader._execute_sells(MagicMock(), position_details, {"AAA"}, tlog)

    assert [e[0] for e in events] == ["guard", "submit"], (
        f"guard must precede submit; got {events}"
    )
    assert events[0][1:] == ("AAA", 100.0)
    assert events[1][1:] == ("AAA", 5.0)


def test_execute_sells_skip_qty_zero(monkeypatch):
    import src.alpaca_singleton as singleton
    guard = MagicMock()
    submit = MagicMock()
    monkeypatch.setattr(singleton, "guard_sell_against_death_spiral", guard)
    monkeypatch.setattr(live_trader, "_submit_market_order", submit)

    tlog = _CapturingTlog()
    position_details = {
        "AAA": {"qty": 0.0, "current_price": 100.0, "avg_entry_price": 99.0},
    }
    live_trader._execute_sells(MagicMock(), position_details, {"AAA"}, tlog)
    guard.assert_not_called()
    submit.assert_not_called()


def test_execute_sells_falls_back_to_avg_entry_when_current_zero(monkeypatch):
    """If current_price is 0, use avg_entry_price so the guard still fires."""
    import src.alpaca_singleton as singleton
    guard = MagicMock()
    monkeypatch.setattr(singleton, "guard_sell_against_death_spiral", guard)
    monkeypatch.setattr(
        live_trader, "_submit_market_order",
        lambda *a, **k: SimpleNamespace(id="x"),
    )

    tlog = _CapturingTlog()
    position_details = {
        "AAA": {"qty": 2.0, "current_price": 0.0, "avg_entry_price": 123.45},
    }
    live_trader._execute_sells(MagicMock(), position_details, ["AAA"], tlog)
    guard.assert_called_once_with("AAA", "sell", 123.45)


def test_execute_sells_skip_when_no_price_anywhere(monkeypatch, caplog):
    """If neither current_price nor avg_entry_price is usable, skip without invoking guard."""
    import src.alpaca_singleton as singleton
    guard = MagicMock()
    submit = MagicMock()
    monkeypatch.setattr(singleton, "guard_sell_against_death_spiral", guard)
    monkeypatch.setattr(live_trader, "_submit_market_order", submit)

    tlog = _CapturingTlog()
    position_details = {
        "AAA": {"qty": 2.0, "current_price": 0.0, "avg_entry_price": 0.0},
    }
    live_trader._execute_sells(MagicMock(), position_details, ["AAA"], tlog)
    guard.assert_not_called()
    submit.assert_not_called()


def test_execute_sells_guard_exception_propagates(monkeypatch):
    """Guard raising RuntimeError must propagate — that's how the supervisor catches death spirals."""
    import src.alpaca_singleton as singleton

    def _guard_raise(sym, side, price):
        raise RuntimeError("death spiral")

    submit = MagicMock()
    monkeypatch.setattr(singleton, "guard_sell_against_death_spiral", _guard_raise)
    monkeypatch.setattr(live_trader, "_submit_market_order", submit)

    tlog = _CapturingTlog()
    position_details = {
        "AAA": {"qty": 1.0, "current_price": 100.0, "avg_entry_price": 100.0},
    }
    with pytest.raises(RuntimeError, match="death spiral"):
        live_trader._execute_sells(MagicMock(), position_details, ["AAA"], tlog)
    submit.assert_not_called()


def test_execute_sells_iterates_sorted(monkeypatch):
    """Sort order is deterministic for audit-log reproducibility."""
    import src.alpaca_singleton as singleton
    monkeypatch.setattr(singleton, "guard_sell_against_death_spiral",
                        lambda *a, **k: None)

    order = []
    monkeypatch.setattr(
        live_trader, "_submit_market_order",
        lambda c, *, symbol, qty, side: (
            order.append(symbol) or SimpleNamespace(id=f"id-{symbol}")
        ),
    )

    tlog = _CapturingTlog()
    position_details = {
        "ZZZ": {"qty": 1, "current_price": 10.0, "avg_entry_price": 10.0},
        "AAA": {"qty": 1, "current_price": 10.0, "avg_entry_price": 10.0},
        "MMM": {"qty": 1, "current_price": 10.0, "avg_entry_price": 10.0},
    }
    # Pass an unsorted iterable — helper must sort it.
    live_trader._execute_sells(MagicMock(), position_details, ["ZZZ", "AAA", "MMM"], tlog)
    assert order == ["AAA", "MMM", "ZZZ"]


# ── _execute_buys ─────────────────────────────────────────────────────────────


def test_execute_buys_records_buy_price_for_each_fill(monkeypatch):
    """Every successful BUY must call record_buy_price (HARD RULE #3 precondition)."""
    import src.alpaca_singleton as singleton

    record_calls: list[tuple[str, float]] = []
    monkeypatch.setattr(
        singleton, "record_buy_price",
        lambda sym, px: record_calls.append((sym, float(px))),
    )
    monkeypatch.setattr(
        live_trader, "_submit_market_order",
        lambda c, *, symbol, qty, side: SimpleNamespace(id=f"id-{symbol}"),
    )
    # poll returns None → fallback to last_close
    monkeypatch.setattr(
        live_trader, "_poll_filled_avg_price",
        lambda c, oid, timeout_s=30.0: None,
    )

    tlog = _CapturingTlog()
    picks = pd.DataFrame([
        {"symbol": "AAA", "score": 0.9, "last_close": 100.0, "spread_bps": 2.0},
        {"symbol": "BBB", "score": 0.8, "last_close": 50.0,  "spread_bps": 2.5},
    ])
    live_trader._execute_buys(MagicMock(), picks, buy_notional=10_000, tlog=tlog)
    # Both picks recorded at last_close fallback
    assert dict(record_calls) == {"AAA": 100.0, "BBB": 50.0}


def test_execute_buys_target_syms_filters(monkeypatch):
    """target_syms={BBB} on a picks df of {AAA, BBB} → only BBB submitted."""
    import src.alpaca_singleton as singleton
    monkeypatch.setattr(singleton, "record_buy_price", lambda *a, **k: None)

    submitted: list[str] = []
    monkeypatch.setattr(
        live_trader, "_submit_market_order",
        lambda c, *, symbol, qty, side: (
            submitted.append(symbol) or SimpleNamespace(id=f"id-{symbol}")
        ),
    )
    monkeypatch.setattr(
        live_trader, "_poll_filled_avg_price",
        lambda c, oid, timeout_s=30.0: None,
    )

    tlog = _CapturingTlog()
    picks = pd.DataFrame([
        {"symbol": "AAA", "score": 0.9, "last_close": 100.0, "spread_bps": 2.0},
        {"symbol": "BBB", "score": 0.8, "last_close": 50.0,  "spread_bps": 2.5},
    ])
    live_trader._execute_buys(MagicMock(), picks, buy_notional=10_000, tlog=tlog,
                              target_syms={"BBB"})
    assert submitted == ["BBB"]


def test_execute_buys_skip_on_nonpositive_price(monkeypatch):
    """Invalid last_close (<= 0) → skip the pick, do NOT submit."""
    import src.alpaca_singleton as singleton
    record = MagicMock()
    submit = MagicMock()
    monkeypatch.setattr(singleton, "record_buy_price", record)
    monkeypatch.setattr(live_trader, "_submit_market_order", submit)
    monkeypatch.setattr(live_trader, "_poll_filled_avg_price",
                        lambda *a, **k: None)

    tlog = _CapturingTlog()
    picks = pd.DataFrame([
        {"symbol": "AAA", "score": 0.9, "last_close": 0.0, "spread_bps": 2.0},
    ])
    live_trader._execute_buys(MagicMock(), picks, buy_notional=10_000, tlog=tlog)
    submit.assert_not_called()
    record.assert_not_called()


def test_execute_buys_uses_fill_price_when_available(monkeypatch):
    """When Alpaca returns filled_avg_price, that wins over last_close for the buy-price record."""
    import src.alpaca_singleton as singleton
    recorded: list[tuple[str, float]] = []
    monkeypatch.setattr(
        singleton, "record_buy_price",
        lambda sym, px: recorded.append((sym, float(px))),
    )
    monkeypatch.setattr(
        live_trader, "_submit_market_order",
        lambda c, *, symbol, qty, side: SimpleNamespace(id="fill-id"),
    )
    monkeypatch.setattr(
        live_trader, "_poll_filled_avg_price",
        lambda c, oid, timeout_s=30.0: 101.25,  # real fill $1.25 above last_close
    )

    tlog = _CapturingTlog()
    picks = pd.DataFrame([
        {"symbol": "AAA", "score": 0.9, "last_close": 100.0, "spread_bps": 2.0},
    ])
    live_trader._execute_buys(MagicMock(), picks, buy_notional=10_000, tlog=tlog)
    assert recorded == [("AAA", 101.25)]
    # buy_filled event records both fill_source and slippage
    buy_filled = [e for t, e in tlog.events if t == "buy_filled"]
    assert len(buy_filled) == 1
    assert buy_filled[0]["fill_source"] == "fill"
    assert buy_filled[0]["fill_price"] == pytest.approx(101.25)
    # slippage = (101.25 - 100) / 100 * 10_000 = 125 bps
    assert buy_filled[0]["slippage_bps_vs_last_close"] == pytest.approx(125.0)


def test_execute_buys_record_buy_price_exception_does_not_crash(monkeypatch):
    """If record_buy_price raises, log a warning but continue processing remaining picks."""
    import src.alpaca_singleton as singleton

    def _raise(sym, px):
        raise OSError("disk full")

    monkeypatch.setattr(singleton, "record_buy_price", _raise)

    submitted: list[str] = []
    monkeypatch.setattr(
        live_trader, "_submit_market_order",
        lambda c, *, symbol, qty, side: (
            submitted.append(symbol) or SimpleNamespace(id=f"id-{symbol}")
        ),
    )
    monkeypatch.setattr(live_trader, "_poll_filled_avg_price",
                        lambda *a, **k: None)

    tlog = _CapturingTlog()
    picks = pd.DataFrame([
        {"symbol": "AAA", "score": 0.9, "last_close": 100.0, "spread_bps": 2.0},
        {"symbol": "BBB", "score": 0.8, "last_close": 50.0, "spread_bps": 2.0},
    ])
    live_trader._execute_buys(MagicMock(), picks, buy_notional=10_000, tlog=tlog)
    # both submits went through even though record_buy_price raised on the first
    assert submitted == ["AAA", "BBB"]


# ── _wait_for_market_open ─────────────────────────────────────────────────────


def test_wait_for_market_open_no_op_if_after_open(monkeypatch):
    """If now_et is after MARKET_OPEN, _wait_until must NOT be called."""
    from datetime import datetime
    # Mock datetime.now(ET) to return 10:00 ET (after 9:30)
    fake_now = datetime(2026, 4, 21, 10, 0, tzinfo=live_trader.ET)

    class _FakeDT(datetime):
        @classmethod
        def now(cls, tz=None):  # type: ignore[override]
            return fake_now.astimezone(tz) if tz else fake_now

    monkeypatch.setattr(live_trader, "datetime", _FakeDT)

    called = MagicMock()
    monkeypatch.setattr(live_trader, "_wait_until", called)
    live_trader._wait_for_market_open()
    called.assert_not_called()


def test_wait_for_market_open_waits_if_before_open(monkeypatch):
    """If now_et is before MARKET_OPEN, _wait_until must fire."""
    from datetime import datetime
    fake_now = datetime(2026, 4, 21, 8, 30, tzinfo=live_trader.ET)

    class _FakeDT(datetime):
        @classmethod
        def now(cls, tz=None):  # type: ignore[override]
            return fake_now.astimezone(tz) if tz else fake_now

    monkeypatch.setattr(live_trader, "datetime", _FakeDT)

    called = MagicMock()
    monkeypatch.setattr(live_trader, "_wait_until", called)
    live_trader._wait_for_market_open()
    called.assert_called_once()


# ── _emit_session_start ───────────────────────────────────────────────────────


def test_emit_session_start_captures_equity_pre(monkeypatch):
    tlog = _CapturingTlog()
    args = SimpleNamespace(
        live=True, dry_run=False, top_n=1, allocation=2.0,
        min_score=0.85, min_dollar_vol=50_000_000, min_vol_20d=0.12,
    )
    client = MagicMock()
    monkeypatch.setattr(
        live_trader, "_get_account",
        lambda c: SimpleNamespace(equity="30000.50"),
    )

    eq_pre = live_trader._emit_session_start(tlog, args, client, mode="open_to_close")
    assert eq_pre == pytest.approx(30000.50)
    assert len(tlog.events) == 1
    event_type, fields = tlog.events[0]
    assert event_type == "session_start"
    assert fields["mode"] == "open_to_close"
    assert fields["paper"] is False
    assert fields["equity_pre"] == pytest.approx(30000.50)
    assert fields["top_n"] == 1
    assert fields["allocation"] == 2.0
    assert fields["min_score"] == 0.85
    assert fields["min_dollar_vol"] == 50_000_000.0
    assert fields["min_vol_20d"] == 0.12


def test_emit_session_start_dry_run_skips_equity(monkeypatch):
    """dry_run or client=None → equity_pre is None (no account query)."""
    tlog = _CapturingTlog()
    args = SimpleNamespace(
        live=False, dry_run=True, top_n=1, allocation=1.0,
        min_score=0.0, min_dollar_vol=0.0, min_vol_20d=0.0,
    )
    called = MagicMock()
    monkeypatch.setattr(live_trader, "_get_account", called)
    eq_pre = live_trader._emit_session_start(tlog, args, client=MagicMock(),
                                              mode="hold_through")
    assert eq_pre is None
    called.assert_not_called()
    assert tlog.events[0][1]["equity_pre"] is None
    assert tlog.events[0][1]["mode"] == "hold_through"


def test_emit_session_start_tolerates_account_error(monkeypatch):
    """If _get_account throws, equity_pre falls back to None gracefully."""
    tlog = _CapturingTlog()
    args = SimpleNamespace(
        live=True, dry_run=False, top_n=1, allocation=1.0,
        min_score=0.0, min_dollar_vol=0.0, min_vol_20d=0.0,
    )

    def _boom(c):
        raise RuntimeError("API down")

    monkeypatch.setattr(live_trader, "_get_account", _boom)
    eq_pre = live_trader._emit_session_start(tlog, args, MagicMock(),
                                              mode="open_to_close")
    assert eq_pre is None
    assert tlog.events[0][1]["equity_pre"] is None


# ── _announce_picks ───────────────────────────────────────────────────────────


# ── _load_models ──────────────────────────────────────────────────────────────


def test_load_models_missing_single_returns_none(tmp_path, capsys):
    """Missing single-model path must return None (caller exits 1)."""
    args = SimpleNamespace(
        model_paths="",
        model_path=tmp_path / "nonexistent.pkl",
    )
    result = live_trader._load_models(args)
    assert result is None
    err = capsys.readouterr().err
    assert "Model not found" in err


def test_load_models_missing_any_ensemble_member_returns_none(tmp_path, capsys):
    """If any member of --model-paths doesn't exist, fail fast with None."""
    good = tmp_path / "ok.pkl"
    good.write_bytes(b"x")  # content irrelevant — only `exists()` is checked
    bad = tmp_path / "missing.pkl"
    args = SimpleNamespace(
        model_paths=f"{good},{bad}",
        model_path=tmp_path / "whatever.pkl",
    )
    result = live_trader._load_models(args)
    assert result is None
    err = capsys.readouterr().err
    assert "Ensemble model not found" in err
    assert str(bad) in err


# ── _next_session_open_et ─────────────────────────────────────────────────────


def test_next_session_open_et_weekday_next_day():
    """Thursday → Friday 9:20 ET (no weekend skip)."""
    from datetime import datetime
    thu = datetime(2026, 4, 23, 16, 0, tzinfo=live_trader.ET)
    nxt = live_trader._next_session_open_et(thu)
    assert nxt.date().weekday() == 4  # Friday
    assert (nxt.hour, nxt.minute) == (9, 20)


def test_next_session_open_et_skips_weekend():
    """Friday evening → Monday 9:20 ET, skipping Sat+Sun."""
    from datetime import datetime
    fri = datetime(2026, 4, 24, 16, 0, tzinfo=live_trader.ET)
    nxt = live_trader._next_session_open_et(fri)
    assert nxt.date().weekday() == 0, f"expected Monday, got {nxt.strftime('%A')}"
    assert (nxt.hour, nxt.minute) == (9, 20)


def test_next_session_open_et_saturday_to_monday():
    """Saturday afternoon → Monday 9:20 ET."""
    from datetime import datetime
    sat = datetime(2026, 4, 25, 14, 0, tzinfo=live_trader.ET)
    nxt = live_trader._next_session_open_et(sat)
    assert nxt.date().weekday() == 0


def test_announce_picks_emits_ranked_events():
    tlog = _CapturingTlog()
    picks = pd.DataFrame([
        {"symbol": "AAA", "score": 0.9, "last_close": 100.0, "spread_bps": 2.0},
        {"symbol": "BBB", "score": 0.7, "last_close": 50.0, "spread_bps": 3.0},
    ])
    live_trader._announce_picks(picks, "2026-04-21", 2, tlog, tag="xgb-live")
    picks_events = [(t, f) for t, f in tlog.events if t == "pick"]
    assert len(picks_events) == 2
    assert picks_events[0][1]["rank"] == 1
    assert picks_events[0][1]["symbol"] == "AAA"
    assert picks_events[1][1]["rank"] == 2
    assert picks_events[1][1]["symbol"] == "BBB"
    # per_seed_scores defaults to empty list when column missing
    assert picks_events[0][1]["per_seed_scores"] == []
