"""Tests for xgbnew.live_trader hold-through rotation path.

Covers the decision table for run_session_hold_through:
  - held == picks → no orders (HOLD)
  - held ∖ picks non-empty → SELL those (with guard)
  - picks ∖ held non-empty → BUY those (with record_buy_price)
  - held ∩ picks → NO order (carry)

HARD RULE #3 invariant: every sell MUST pass through
guard_sell_against_death_spiral before submit_order.
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

from xgbnew import live_trader


def _mk_args(**over):
    base = dict(
        live=True,  # non-paper so position queries fire
        dry_run=False,
        top_n=1,
        allocation=1.0,
        min_score=0.0,
        min_dollar_vol=0.0,
        hold_through=True,
    )
    base.update(over)
    return SimpleNamespace(**base)


def _clock_open():
    return SimpleNamespace(is_open=True, next_open=None)


def _mk_pos(symbol, qty, price):
    return SimpleNamespace(
        symbol=symbol,
        qty=str(qty),
        current_price=str(price),
        avg_entry_price=str(price),
    )


def _install_picks(monkeypatch, picks_df: pd.DataFrame):
    """Stub the score/fetch chain so _score_and_pick returns picks_df."""
    monkeypatch.setattr(live_trader, "_get_latest_bars", lambda *a, **k: {})
    monkeypatch.setattr(live_trader, "_latest_stock_bid_ask", lambda *a, **k: (0.0, 0.0))
    monkeypatch.setattr(
        live_trader,
        "score_all_symbols",
        lambda *a, **k: picks_df.copy(),
    )


def test_hold_through_flag_default_off():
    a = live_trader.parse_args([
        "--symbols-file", "x",
        "--model-paths", "a.pkl",
    ])
    assert a.hold_through is False


def test_hold_through_flag_parses_true():
    a = live_trader.parse_args([
        "--symbols-file", "x",
        "--model-paths", "a.pkl",
        "--hold-through",
    ])
    assert a.hold_through is True


def test_hold_when_picks_match_held(monkeypatch):
    """held == picks → no orders. No sell, no buy, no death-spiral guard call."""
    picks = pd.DataFrame([{"symbol": "AAA", "score": 0.8,
                          "last_close": 100.0, "spread_bps": 3.0}])
    _install_picks(monkeypatch, picks)

    client = MagicMock()
    client.get_clock.return_value = _clock_open()
    client.get_all_positions.return_value = [_mk_pos("AAA", 10, 101.0)]

    monkeypatch.setattr(live_trader, "_wait_until", lambda *a, **k: None)

    submit = MagicMock()
    monkeypatch.setattr(live_trader, "_submit_limit_order", submit)

    live_trader.run_session_hold_through(
        symbols=["AAA"],
        data_root=Path("/tmp"),
        model=MagicMock(),
        client=client,
        args=_mk_args(top_n=1),
    )

    submit.assert_not_called()


def test_crypto_positions_are_not_rotated_by_stock_hold_through(monkeypatch):
    """Embedded crypto sleeve positions must not enter XGB stock rotation."""
    picks = pd.DataFrame([{"symbol": "AAA", "score": 0.8,
                          "last_close": 100.0, "spread_bps": 3.0}])
    _install_picks(monkeypatch, picks)

    client = MagicMock()
    client.get_clock.return_value = _clock_open()
    client.get_all_positions.return_value = [
        _mk_pos("AAA", 10, 101.0),
        _mk_pos("BTC/USD", 0.1, 90_000.0),
    ]
    monkeypatch.setattr(live_trader, "_wait_until", lambda *a, **k: None)
    submit = MagicMock()
    monkeypatch.setattr(live_trader, "_submit_limit_order", submit)

    live_trader.run_session_hold_through(
        symbols=["AAA"],
        data_root=Path("/tmp"),
        model=MagicMock(),
        client=client,
        args=_mk_args(top_n=1),
    )

    submit.assert_not_called()


def test_regime_gate_closed_liquidates_stock_hold_through_positions(monkeypatch):
    """SPY regime gate closed → hold-through exits stock positions and does not score."""
    client = MagicMock()
    client.get_clock.return_value = _clock_open()
    client.get_all_positions.return_value = [
        _mk_pos("AAA", 10, 101.0),
        _mk_pos("BTC/USD", 0.1, 90_000.0),
    ]

    monkeypatch.setattr(live_trader, "_live_spy_regime_gate_closed", lambda *a, **k: True)
    monkeypatch.setattr(
        live_trader,
        "_score_and_pick",
        lambda *a, **k: pytest.fail("regime gate should skip scoring"),
    )
    monkeypatch.setattr(live_trader, "_wait_until", lambda *a, **k: None)
    monkeypatch.setattr(live_trader, "_latest_stock_bid_ask", lambda *a, **k: (100.0, 102.0))

    submitted = []

    def _submit(client_arg, *, symbol, qty, side, limit_price):
        submitted.append((symbol, qty, side, limit_price))
        return SimpleNamespace(id=f"id-{symbol}-{side}", filled_avg_price=None)

    monkeypatch.setattr(live_trader, "_submit_limit_order", _submit)

    import src.alpaca_singleton as singleton
    guard_calls = []
    monkeypatch.setattr(
        singleton,
        "guard_sell_against_death_spiral",
        lambda sym, side, price: guard_calls.append((sym, side, price)),
    )

    live_trader.run_session_hold_through(
        symbols=["AAA"],
        data_root=Path("/tmp"),
        model=MagicMock(),
        client=client,
        args=_mk_args(top_n=1, regime_gate_window=20),
    )

    assert [(s, side) for s, _, side, _ in submitted] == [("AAA", "sell")]
    assert guard_calls == [("AAA", "sell", 101.0)]


def test_rotation_sells_dropped_and_buys_new(monkeypatch):
    """held={AAA}, picks={BBB} → SELL AAA (with guard), BUY BBB."""
    picks = pd.DataFrame([{"symbol": "BBB", "score": 0.9,
                          "last_close": 50.0, "spread_bps": 2.0}])
    _install_picks(monkeypatch, picks)

    client = MagicMock()
    client.get_clock.return_value = _clock_open()
    client.get_all_positions.return_value = [_mk_pos("AAA", 10, 200.0)]

    monkeypatch.setattr(live_trader, "_wait_until", lambda *a, **k: None)
    # Portfolio value 10_000 → buy_notional 10_000 for top_n=1, alloc=1.0
    monkeypatch.setattr(
        live_trader, "_get_account",
        lambda c: SimpleNamespace(portfolio_value="10000"),
    )

    submitted = []

    def _submit(client_arg, *, symbol, qty, side, limit_price):
        submitted.append((symbol, qty, side, limit_price))
        return SimpleNamespace(id=f"id-{symbol}-{side}", filled_avg_price=None)

    monkeypatch.setattr(live_trader, "_submit_limit_order", _submit)
    monkeypatch.setattr(live_trader, "_poll_filled_avg_price",
                        lambda c, oid, timeout_s=30.0: None)

    # Capture guard + record calls by patching the singleton module BEFORE
    # the inline import inside run_session_hold_through.
    import src.alpaca_singleton as singleton
    guard_calls = []
    record_calls = []

    def _guard(sym, side, price):
        guard_calls.append((sym, side, price))

    def _record(sym, price):
        record_calls.append((sym, price))

    monkeypatch.setattr(singleton, "guard_sell_against_death_spiral", _guard)
    monkeypatch.setattr(singleton, "record_buy_price", _record)

    live_trader.run_session_hold_through(
        symbols=["AAA", "BBB"],
        data_root=Path("/tmp"),
        model=MagicMock(),
        client=client,
        args=_mk_args(top_n=1),
    )

    sides = {(s, sd) for s, _, sd, _ in submitted}
    assert ("AAA", "sell") in sides, f"expected AAA sell, got {submitted}"
    assert ("BBB", "buy") in sides, f"expected BBB buy, got {submitted}"

    # Invariant: guard called for every sell before submit, with real price.
    assert any(g[0] == "AAA" and g[1] == "sell" for g in guard_calls), (
        f"guard_sell_against_death_spiral not invoked on AAA sell: {guard_calls}"
    )
    # record_buy_price captured the BUY with the last_close fallback (poll
    # returned None in this stub).
    assert any(r[0] == "BBB" and r[1] == pytest.approx(50.0) for r in record_calls), (
        f"record_buy_price not invoked on BBB buy: {record_calls}"
    )


def test_partial_overlap_keeps_intersect_sells_dropout_buys_new(monkeypatch):
    """held={AAA,BBB}, picks={BBB,CCC} → keep BBB (no trade), sell AAA, buy CCC."""
    picks = pd.DataFrame([
        {"symbol": "BBB", "score": 0.9, "last_close": 50.0, "spread_bps": 2.0},
        {"symbol": "CCC", "score": 0.85, "last_close": 20.0, "spread_bps": 2.5},
    ])
    _install_picks(monkeypatch, picks)

    client = MagicMock()
    client.get_clock.return_value = _clock_open()
    client.get_all_positions.return_value = [
        _mk_pos("AAA", 10, 200.0),
        _mk_pos("BBB", 4, 55.0),
    ]

    monkeypatch.setattr(live_trader, "_wait_until", lambda *a, **k: None)
    monkeypatch.setattr(
        live_trader, "_get_account",
        lambda c: SimpleNamespace(portfolio_value="10000"),
    )

    submitted = []

    def _submit(client_arg, *, symbol, qty, side, limit_price):
        submitted.append((symbol, qty, side, limit_price))
        return SimpleNamespace(id=f"id-{symbol}-{side}", filled_avg_price=None)

    monkeypatch.setattr(live_trader, "_submit_limit_order", _submit)
    monkeypatch.setattr(live_trader, "_poll_filled_avg_price",
                        lambda c, oid, timeout_s=30.0: None)

    import src.alpaca_singleton as singleton
    monkeypatch.setattr(singleton, "guard_sell_against_death_spiral",
                        lambda *a, **k: None)
    monkeypatch.setattr(singleton, "record_buy_price", lambda *a, **k: None)

    live_trader.run_session_hold_through(
        symbols=["AAA", "BBB", "CCC"],
        data_root=Path("/tmp"),
        model=MagicMock(),
        client=client,
        args=_mk_args(top_n=2),
    )

    syms_traded = {s for s, _, _, _ in submitted}
    assert syms_traded == {"AAA", "CCC"}, (
        f"expected only AAA sell + CCC buy (BBB carried), got {submitted}"
    )
    sides = {(s, sd) for s, _, sd, _ in submitted}
    assert ("AAA", "sell") in sides
    assert ("CCC", "buy") in sides
    # BBB should NOT be traded on either side.
    assert not any(s == "BBB" for s, _, _, _ in submitted)


def test_fresh_entry_from_flat_buys_but_no_sell(monkeypatch):
    """held={}, picks={AAA} → BUY AAA, no sell (no prior positions)."""
    picks = pd.DataFrame([{"symbol": "AAA", "score": 0.9,
                          "last_close": 100.0, "spread_bps": 3.0}])
    _install_picks(monkeypatch, picks)

    client = MagicMock()
    client.get_clock.return_value = _clock_open()
    client.get_all_positions.return_value = []

    monkeypatch.setattr(live_trader, "_wait_until", lambda *a, **k: None)
    monkeypatch.setattr(
        live_trader, "_get_account",
        lambda c: SimpleNamespace(portfolio_value="10000"),
    )

    submitted = []
    monkeypatch.setattr(
        live_trader, "_submit_limit_order",
        lambda c, *, symbol, qty, side, limit_price: (
            submitted.append((symbol, qty, side, limit_price))
            or SimpleNamespace(id="id", filled_avg_price=None)
        ),
    )
    monkeypatch.setattr(live_trader, "_poll_filled_avg_price",
                        lambda c, oid, timeout_s=30.0: None)

    import src.alpaca_singleton as singleton
    monkeypatch.setattr(singleton, "guard_sell_against_death_spiral",
                        lambda *a, **k: None)
    monkeypatch.setattr(singleton, "record_buy_price", lambda *a, **k: None)

    live_trader.run_session_hold_through(
        symbols=["AAA"],
        data_root=Path("/tmp"),
        model=MagicMock(),
        client=client,
        args=_mk_args(top_n=1),
    )

    assert submitted == [("AAA", pytest.approx(100.0), "buy", pytest.approx(100.0))], submitted


def test_hold_through_no_picks_fallback_rotates_with_scaled_notional(monkeypatch):
    """hold-through live path must match simulator no-picks fallback sizing."""
    all_scores = pd.DataFrame([
        {"symbol": "SPY", "score": 0.40, "last_close": 500.0, "spread_bps": 1.0},
    ])
    monkeypatch.setattr(
        live_trader,
        "_score_and_pick",
        lambda *a, **k: (all_scores.iloc[0:0].copy(), all_scores),
    )

    client = MagicMock()
    client.get_clock.return_value = _clock_open()
    client.get_all_positions.return_value = []
    monkeypatch.setattr(live_trader, "_wait_for_market_open", lambda: None)
    monkeypatch.setattr(
        live_trader,
        "_get_account",
        lambda _client: SimpleNamespace(portfolio_value="100000"),
    )

    captured: dict[str, float] = {}

    def _capture_buys(client, picks, buy_notional, tlog, *, notional_by_symbol=None, **kwargs):
        captured.update(notional_by_symbol or {})

    monkeypatch.setattr(live_trader, "_execute_buys", _capture_buys)
    monkeypatch.setattr(live_trader, "_execute_sells", lambda *a, **k: None)

    live_trader.run_session_hold_through(
        symbols=["SPY"],
        data_root=Path("/tmp"),
        model=MagicMock(),
        client=client,
        args=_mk_args(
            top_n=4,
            allocation=2.0,
            allocation_mode="equal",
            allocation_temp=1.0,
            min_score=0.95,
            no_picks_fallback="SPY",
            no_picks_fallback_alloc=0.25,
        ),
    )

    assert captured == {"SPY": pytest.approx(50_000.0)}


def test_gate_skips_non_trading_day(monkeypatch):
    """Saturday → Alpaca clock says closed → no score, no trade."""
    # Saturday: is_open=False, next_open in ~2 days.
    from datetime import datetime, timezone
    next_open = datetime(2026, 4, 20, 13, 30, tzinfo=timezone.utc)
    clock = SimpleNamespace(is_open=False, next_open=next_open)

    client = MagicMock()
    client.get_clock.return_value = clock

    # Must NOT be called if gate triggers.
    bars_called = []
    monkeypatch.setattr(
        live_trader, "_get_latest_bars",
        lambda *a, **k: (bars_called.append(1), {})[1],
    )
    score_called = []
    monkeypatch.setattr(
        live_trader, "score_all_symbols",
        lambda *a, **k: (score_called.append(1), pd.DataFrame())[1],
    )

    # Patch _is_today_trading_day to force False (avoids real-clock flakiness).
    monkeypatch.setattr(
        live_trader, "_is_today_trading_day",
        lambda c, now=None: (False, "weekend/holiday — market closed"),
    )

    live_trader.run_session_hold_through(
        symbols=["AAA"],
        data_root=Path("/tmp"),
        model=MagicMock(),
        client=client,
        args=_mk_args(),
    )

    assert not bars_called, "gate should have skipped bar fetch"
    assert not score_called, "gate should have skipped scoring"


def test_dry_run_scores_but_submits_nothing(monkeypatch):
    picks = pd.DataFrame([{"symbol": "AAA", "score": 0.9,
                          "last_close": 100.0, "spread_bps": 3.0}])
    _install_picks(monkeypatch, picks)

    submit = MagicMock()
    monkeypatch.setattr(live_trader, "_submit_limit_order", submit)

    client = MagicMock()
    # Gate is skipped on dry_run path.
    live_trader.run_session_hold_through(
        symbols=["AAA"],
        data_root=Path("/tmp"),
        model=MagicMock(),
        client=client,
        args=_mk_args(dry_run=True),
    )

    submit.assert_not_called()


def test_run_session_dispatches_to_hold_through(monkeypatch):
    """run_session(hold_through=True args) must delegate to run_session_hold_through."""
    called = []

    def _capture(*a, **k):
        called.append(("hold_through", a, k))

    monkeypatch.setattr(live_trader, "run_session_hold_through", _capture)

    live_trader.run_session(
        symbols=["AAA"],
        data_root=Path("/tmp"),
        model=MagicMock(),
        client=MagicMock(),
        args=_mk_args(hold_through=True),
    )
    assert len(called) == 1, f"expected 1 hold_through dispatch, got {called}"
