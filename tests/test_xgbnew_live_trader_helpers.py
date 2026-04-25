"""Unit tests for the helpers extracted from run_session / run_session_hold_through.

Pins the behavior of the small-scope functions so refactors can't silently
change trade semantics. In particular:

- HARD RULE #3: every SELL path calls ``guard_sell_against_death_spiral``
  BEFORE order submission.
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

from xgbnew import live_trader
from xgbnew.trade_log import TradeLogger


class _CapturingTlog(TradeLogger):
    """TradeLogger that captures every event as a list of (type, fields) pairs."""

    def __init__(self):
        super().__init__(disabled=True)
        self.events: list[tuple[str, dict]] = []

    def log(self, event_type: str, **fields) -> None:  # type: ignore[override]
        self.events.append((event_type, dict(fields)))


# ── cross-sectional regime gate ───────────────────────────────────────────────


def test_stock_limit_price_near_market_prefers_quote_touch(monkeypatch):
    monkeypatch.setattr(live_trader, "_latest_stock_bid_ask", lambda sym: (99.5, 100.5))

    buy_px = live_trader._stock_limit_price_near_market(
        "AAPL", side="buy", reference_price=98.0, aggressiveness_bps=15.0
    )
    sell_px = live_trader._stock_limit_price_near_market(
        "AAPL", side="sell", reference_price=102.0, aggressiveness_bps=15.0
    )

    assert buy_px == pytest.approx(100.5 * 1.0015)
    assert sell_px == pytest.approx(99.5 * (1.0 - 0.0015))


def test_stock_limit_price_near_market_falls_back_to_reference(monkeypatch):
    monkeypatch.setattr(live_trader, "_latest_stock_bid_ask", lambda sym: (0.0, 0.0))

    px = live_trader._stock_limit_price_near_market(
        "AAPL", side="buy", reference_price=101.25, aggressiveness_bps=15.0
    )

    assert px == pytest.approx(101.25)


def test_cross_sectional_regime_skew_gate_keeps_right_skew_day():
    scores_df = pd.DataFrame([
        {"symbol": "A", "score": 0.9, "ret_5d": -0.01},
        {"symbol": "B", "score": 0.8, "ret_5d": 0.00},
        {"symbol": "C", "score": 0.7, "ret_5d": 0.01},
        {"symbol": "D", "score": 0.6, "ret_5d": 0.50},
    ])
    tlog = _CapturingTlog()

    out = live_trader._apply_cross_sectional_regime_gate(
        scores_df,
        regime_cs_skew_min=1.0,
        trade_logger=tlog,
    )

    assert list(out["symbol"]) == ["A", "B", "C", "D"]
    assert tlog.events[-1][0] == "regime_cs_gate"
    assert tlog.events[-1][1]["kept"] is True
    assert tlog.events[-1][1]["cs_skew_ret5"] >= 1.0


def test_cross_sectional_regime_skew_gate_closes_left_skew_day():
    scores_df = pd.DataFrame([
        {"symbol": "A", "score": 0.9, "ret_5d": -0.50},
        {"symbol": "B", "score": 0.8, "ret_5d": -0.01},
        {"symbol": "C", "score": 0.7, "ret_5d": 0.00},
        {"symbol": "D", "score": 0.6, "ret_5d": 0.01},
    ])
    tlog = _CapturingTlog()

    out = live_trader._apply_cross_sectional_regime_gate(
        scores_df,
        regime_cs_skew_min=1.0,
        trade_logger=tlog,
    )

    assert out.empty
    assert tlog.events[-1][0] == "regime_cs_gate"
    assert tlog.events[-1][1]["kept"] is False
    assert tlog.events[-1][1]["cs_skew_ret5"] < 1.0


def test_cross_sectional_regime_iqr_gate_closes_wide_day():
    scores_df = pd.DataFrame([
        {"symbol": "A", "score": 0.9, "ret_5d": -0.30},
        {"symbol": "B", "score": 0.8, "ret_5d": 0.00},
        {"symbol": "C", "score": 0.7, "ret_5d": 0.01},
        {"symbol": "D", "score": 0.6, "ret_5d": 0.30},
    ])

    out = live_trader._apply_cross_sectional_regime_gate(
        scores_df,
        regime_cs_iqr_max=0.05,
    )

    assert out.empty


# ── embedded EOD deleverage ───────────────────────────────────────────────────


def test_eod_deleverage_tick_disabled_is_noop():
    args = SimpleNamespace(eod_deleverage=False)

    out = live_trader._eod_deleverage_tick(MagicMock(), args)

    assert out == {"action": "disabled"}


def test_eod_deleverage_tick_already_under_target(monkeypatch):
    monkeypatch.setattr(live_trader, "_minutes_to_market_close", lambda _client: 30.0)
    client = MagicMock()
    client.get_account.return_value = SimpleNamespace(equity="1000")
    client.get_all_positions.return_value = [
        SimpleNamespace(symbol="AAPL", qty="10", market_value="1500", current_price="150", side="long"),
        SimpleNamespace(symbol="BTC/USD", qty="0.1", market_value="9000", current_price="90000", side="long"),
    ]
    args = SimpleNamespace(
        eod_deleverage=True,
        eod_deleverage_window_minutes=60,
        eod_max_gross_leverage=2.0,
        eod_force_market_minutes=5,
    )

    out = live_trader._eod_deleverage_tick(client, args)

    assert out["action"] == "already_ok"
    assert out["exposure"] == pytest.approx(1500.0)
    assert out["leverage"] == pytest.approx(1.5)


def test_eod_deleverage_tick_submits_excess_equity_limit(monkeypatch):
    monkeypatch.setattr(live_trader, "_minutes_to_market_close", lambda _client: 30.0)
    submitted: list[dict] = []

    def _submit_limit(client, *, symbol, qty, side, limit_price):
        submitted.append({
            "symbol": symbol,
            "qty": qty,
            "side": side,
            "limit_price": limit_price,
        })
        return SimpleNamespace(id="order-1")

    monkeypatch.setattr(live_trader, "_submit_limit_order", _submit_limit)
    monkeypatch.setattr(live_trader, "_latest_stock_bid_ask", lambda sym: (149.0, 151.0))
    client = MagicMock()
    client.get_account.return_value = SimpleNamespace(equity="1000")
    client.get_all_positions.return_value = [
        SimpleNamespace(symbol="AAPL", qty="20", market_value="3000", current_price="150", side="long"),
        SimpleNamespace(symbol="ETH/USD", qty="1", market_value="3000", current_price="3000", side="long"),
    ]
    args = SimpleNamespace(
        eod_deleverage=True,
        eod_deleverage_window_minutes=60,
        eod_max_gross_leverage=2.0,
        eod_force_market_minutes=5,
    )

    out = live_trader._eod_deleverage_tick(client, args)

    assert out["action"] == "submitted"
    assert out["submitted"] == 1
    assert submitted[0]["symbol"] == "AAPL"
    assert submitted[0]["side"] == "sell"
    assert submitted[0]["qty"] > 0
    progress = (60.0 - 30.0) / (60.0 - 5.0)
    aggressiveness_bps = 5.0 + 20.0 * progress
    assert submitted[0]["limit_price"] == pytest.approx(149.0 * (1.0 - aggressiveness_bps / 10_000.0))


def test_eod_deleverage_tick_uses_aggressive_limit_in_force_window(monkeypatch):
    monkeypatch.setattr(live_trader, "_minutes_to_market_close", lambda _client: 3.0)
    submitted: list[dict] = []

    def _submit_limit(client, *, symbol, qty, side, limit_price):
        submitted.append({"symbol": symbol, "qty": qty, "side": side, "limit_price": limit_price})
        return SimpleNamespace(id="order-1")

    monkeypatch.setattr(live_trader, "_submit_limit_order", _submit_limit)
    monkeypatch.setattr(live_trader, "_latest_stock_bid_ask", lambda sym: (149.0, 151.0))
    client = MagicMock()
    client.get_account.return_value = SimpleNamespace(equity="1000")
    client.get_all_positions.return_value = [
        SimpleNamespace(symbol="AAPL", qty="20", market_value="3000", current_price="150", side="long"),
    ]
    args = SimpleNamespace(
        eod_deleverage=True,
        eod_deleverage_window_minutes=60,
        eod_max_gross_leverage=2.0,
        eod_force_market_minutes=5,
    )

    out = live_trader._eod_deleverage_tick(client, args)

    assert out["action"] == "submitted"
    assert out["use_market"] is False
    assert out["force_window"] is True
    assert submitted == [{
        "symbol": "AAPL",
        "qty": pytest.approx(6.6666),
        "side": "sell",
        "limit_price": pytest.approx(149.0 * (1.0 - 0.0025)),
    }]


# ── _execute_sells ────────────────────────────────────────────────────────────


def test_execute_sells_calls_guard_before_submit(monkeypatch):
    """HARD RULE #3 — guard_sell_against_death_spiral runs before order submit."""
    import src.alpaca_singleton as singleton

    events: list[tuple[str, str, float]] = []

    def _guard(sym, side, price):
        events.append(("guard", sym, float(price)))

    def _submit(client, *, symbol, qty, side, limit_price):
        events.append(("submit", symbol, float(qty)))
        return SimpleNamespace(id=f"id-{symbol}")

    monkeypatch.setattr(singleton, "guard_sell_against_death_spiral", _guard)
    monkeypatch.setattr(live_trader, "_submit_limit_order", _submit)
    monkeypatch.setattr(live_trader, "_stock_limit_price_near_market", lambda *a, **k: 99.9)

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
    monkeypatch.setattr(live_trader, "_submit_limit_order", submit)

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
        live_trader, "_submit_limit_order",
        lambda *a, **k: SimpleNamespace(id="x"),
    )
    monkeypatch.setattr(live_trader, "_stock_limit_price_near_market", lambda *a, **k: 122.95)

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
    monkeypatch.setattr(live_trader, "_submit_limit_order", submit)

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
    monkeypatch.setattr(live_trader, "_submit_limit_order", submit)

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
        live_trader, "_submit_limit_order",
        lambda c, *, symbol, qty, side, limit_price: (
            order.append(symbol) or SimpleNamespace(id=f"id-{symbol}")
        ),
    )
    monkeypatch.setattr(live_trader, "_stock_limit_price_near_market", lambda *a, **k: 9.95)

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


def test_buy_notional_by_symbol_uses_softmax_weights():
    picks = pd.DataFrame([
        {"symbol": "AAA", "score": 0.90},
        {"symbol": "BBB", "score": 0.60},
    ])

    notionals = live_trader._buy_notional_by_symbol(
        picks,
        total_notional=10_000,
        allocation_mode="softmax",
        allocation_temp=0.10,
    )

    assert sum(notionals.values()) == pytest.approx(10_000)
    assert notionals["AAA"] > notionals["BBB"]
    assert notionals["AAA"] == pytest.approx(9525.741, rel=1e-5)


def test_execute_buys_uses_per_symbol_notional_map(monkeypatch):
    import src.alpaca_singleton as singleton

    monkeypatch.setattr(singleton, "record_buy_price", lambda *a, **k: None)
    submitted: list[tuple[str, float]] = []
    monkeypatch.setattr(
        live_trader,
        "_submit_limit_order",
        lambda c, *, symbol, qty, side, limit_price: (
            submitted.append((symbol, float(qty))) or SimpleNamespace(id=f"id-{symbol}")
        ),
    )
    monkeypatch.setattr(live_trader, "_stock_limit_price_near_market", lambda *a, **k: 100.15)
    monkeypatch.setattr(live_trader, "_poll_filled_avg_price", lambda *a, **k: None)

    tlog = _CapturingTlog()
    picks = pd.DataFrame([
        {"symbol": "AAA", "score": 0.9, "last_close": 100.0, "spread_bps": 2.0},
        {"symbol": "BBB", "score": 0.8, "last_close": 50.0, "spread_bps": 2.5},
    ])
    live_trader._execute_buys(
        MagicMock(),
        picks,
        buy_notional=None,
        tlog=tlog,
        notional_by_symbol={"AAA": 20_000.0, "BBB": 5_000.0},
    )

    assert submitted == [("AAA", 200.0), ("BBB", 100.0)]
    submitted_events = [fields for event, fields in tlog.events if event == "buy_submitted"]
    assert [event["target_notional"] for event in submitted_events] == [20_000.0, 5_000.0]


def test_conviction_allocation_scale_drops_zero_conviction_picks():
    picks = pd.DataFrame([
        {"symbol": "AAA", "score": 0.50, "last_close": 100.0, "spread_bps": 1.0},
    ])
    args = SimpleNamespace(
        conviction_scaled_alloc=True,
        conviction_alloc_low=0.55,
        conviction_alloc_high=0.85,
    )
    tlog = _CapturingTlog()

    scaled_picks, scale = live_trader._conviction_allocation_scale(
        picks, picks, args, trade_logger=tlog
    )

    assert scaled_picks.empty
    assert scale == 0.0
    assert tlog.events[-1][0] == "conviction_scaled_alloc"


def test_score_and_pick_min_picks_forces_low_confidence_candidate(monkeypatch):
    scores = pd.DataFrame([
        {"symbol": "AAA", "score": 0.70, "last_close": 100.0, "spread_bps": 2.0},
        {"symbol": "BBB", "score": 0.60, "last_close": 50.0, "spread_bps": 2.5},
    ])
    monkeypatch.setattr(live_trader, "_get_latest_bars", lambda *a, **k: {})
    monkeypatch.setattr(live_trader, "score_all_symbols", lambda *a, **k: scores)

    tlog = _CapturingTlog()
    args = SimpleNamespace(
        min_dollar_vol=0.0,
        min_vol_20d=0.0,
        max_vol_20d=0.0,
        max_ret_20d_rank_pct=1.0,
        min_ret_5d_rank_pct=0.0,
        regime_cs_iqr_max=0.0,
        regime_cs_skew_min=-1e9,
        min_score=0.85,
        min_picks=1,
        top_n=2,
    )

    picks, all_scores = live_trader._score_and_pick(
        ["AAA", "BBB"], Path("."), MagicMock(), args, trade_logger=tlog
    )

    assert all_scores is scores
    assert list(picks["symbol"]) == ["AAA"]
    filtered_event = [fields for event, fields in tlog.events if event == "filtered"][-1]
    assert filtered_event["n_forced_low_confidence"] == 1
    assert filtered_event["min_picks"] == 1


def test_execute_buys_records_buy_price_for_each_fill(monkeypatch):
    """Every successful BUY must call record_buy_price (HARD RULE #3 precondition)."""
    import src.alpaca_singleton as singleton

    record_calls: list[tuple[str, float]] = []
    monkeypatch.setattr(
        singleton, "record_buy_price",
        lambda sym, px: record_calls.append((sym, float(px))),
    )
    monkeypatch.setattr(
        live_trader, "_submit_limit_order",
        lambda c, *, symbol, qty, side, limit_price: SimpleNamespace(id=f"id-{symbol}"),
    )
    monkeypatch.setattr(live_trader, "_stock_limit_price_near_market", lambda *a, **k: 100.15)
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
        live_trader, "_submit_limit_order",
        lambda c, *, symbol, qty, side, limit_price: (
            submitted.append(symbol) or SimpleNamespace(id=f"id-{symbol}")
        ),
    )
    monkeypatch.setattr(live_trader, "_stock_limit_price_near_market", lambda *a, **k: 100.15)
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
    monkeypatch.setattr(live_trader, "_submit_limit_order", submit)
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
        live_trader, "_submit_limit_order",
        lambda c, *, symbol, qty, side, limit_price: SimpleNamespace(id="fill-id"),
    )
    monkeypatch.setattr(live_trader, "_stock_limit_price_near_market", lambda *a, **k: 100.15)
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
        live_trader, "_submit_limit_order",
        lambda c, *, symbol, qty, side, limit_price: (
            submitted.append(symbol) or SimpleNamespace(id=f"id-{symbol}")
        ),
    )
    monkeypatch.setattr(live_trader, "_stock_limit_price_near_market", lambda *a, **k: 100.15)
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


# ── run_session allocation semantics ─────────────────────────────────────────


def test_run_session_fallback_sizing_is_not_divided_by_top_n(monkeypatch):
    all_scores = pd.DataFrame([
        {"symbol": "SPY", "score": 0.40, "last_close": 500.0, "spread_bps": 1.0},
    ])
    monkeypatch.setattr(
        live_trader,
        "_score_and_pick",
        lambda *a, **k: (all_scores.iloc[0:0].copy(), all_scores),
    )
    monkeypatch.setattr(live_trader, "_is_today_trading_day", lambda _client: (True, "open"))
    monkeypatch.setattr(live_trader, "_wait_for_market_open", lambda: None)
    monkeypatch.setattr(live_trader, "_wait_until", lambda *a, **k: None)
    monkeypatch.setattr(
        live_trader,
        "_get_account",
        lambda _client: SimpleNamespace(portfolio_value="100000"),
    )
    monkeypatch.setattr(live_trader, "_get_position_details", lambda _client: {})
    monkeypatch.setattr(live_trader, "_execute_sells", lambda *a, **k: None)

    captured: dict[str, float] = {}

    def _capture_buys(client, picks, buy_notional, tlog, *, notional_by_symbol=None, **kwargs):
        captured.update(notional_by_symbol or {})

    monkeypatch.setattr(live_trader, "_execute_buys", _capture_buys)

    args = SimpleNamespace(
        live=True,
        dry_run=False,
        hold_through=False,
        top_n=4,
        allocation=2.0,
        allocation_mode="equal",
        allocation_temp=1.0,
        min_score=0.95,
        min_dollar_vol=0.0,
        min_vol_20d=0.0,
        no_picks_fallback="SPY",
        no_picks_fallback_alloc=0.25,
        _trade_logger=_CapturingTlog(),
    )

    live_trader.run_session(["SPY"], Path("."), MagicMock(), MagicMock(), args)

    assert captured == {"SPY": pytest.approx(50_000.0)}


def test_run_session_scales_notional_by_conviction(monkeypatch):
    picks = pd.DataFrame([
        {"symbol": "AAA", "score": 0.70, "last_close": 100.0, "spread_bps": 1.0},
    ])
    monkeypatch.setattr(live_trader, "_score_and_pick", lambda *a, **k: (picks, picks))
    monkeypatch.setattr(live_trader, "_is_today_trading_day", lambda _client: (True, "open"))
    monkeypatch.setattr(live_trader, "_wait_for_market_open", lambda: None)
    monkeypatch.setattr(live_trader, "_wait_until", lambda *a, **k: None)
    monkeypatch.setattr(
        live_trader,
        "_get_account",
        lambda _client: SimpleNamespace(portfolio_value="100000"),
    )
    monkeypatch.setattr(live_trader, "_get_position_details", lambda _client: {})
    monkeypatch.setattr(live_trader, "_execute_sells", lambda *a, **k: None)

    captured: dict[str, float] = {}

    def _capture_buys(client, picks, buy_notional, tlog, *, notional_by_symbol=None, **kwargs):
        captured.update(notional_by_symbol or {})

    monkeypatch.setattr(live_trader, "_execute_buys", _capture_buys)

    args = SimpleNamespace(
        live=True,
        dry_run=False,
        hold_through=False,
        top_n=1,
        allocation=2.0,
        allocation_mode="equal",
        allocation_temp=1.0,
        conviction_scaled_alloc=True,
        conviction_alloc_low=0.55,
        conviction_alloc_high=0.85,
        min_score=0.0,
        min_dollar_vol=0.0,
        min_vol_20d=0.0,
        no_picks_fallback="",
        no_picks_fallback_alloc=0.0,
        _trade_logger=_CapturingTlog(),
    )

    live_trader.run_session(["AAA"], Path("."), MagicMock(), MagicMock(), args)

    expected_scale = (0.70 - 0.55) / (0.85 - 0.55)
    assert captured == {"AAA": pytest.approx(100_000.0 * 2.0 * expected_scale)}


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


def test_load_models_single_load_error_returns_none(monkeypatch, tmp_path, capsys):
    path = tmp_path / "bad.pkl"
    path.write_bytes(b"not a pickle")

    def _boom(_path):
        raise ValueError("bad pickle")

    monkeypatch.setattr(live_trader.XGBStockModel, "load", staticmethod(_boom))
    args = SimpleNamespace(
        model_paths="",
        model_path=path,
    )

    result = live_trader._load_models(args)

    assert result is None
    assert "Failed to load model: bad pickle" in capsys.readouterr().err


def test_load_models_empty_ensemble_path_list_returns_none(tmp_path, capsys):
    args = SimpleNamespace(
        model_paths=" , , ",
        model_path=tmp_path / "whatever.pkl",
    )

    result = live_trader._load_models(args)

    assert result is None
    assert "did not contain any model paths" in capsys.readouterr().err


def test_load_models_duplicate_ensemble_paths_returns_none(tmp_path, capsys):
    path = tmp_path / "alltrain_seed0.pkl"
    path.write_bytes(b"x")
    args = SimpleNamespace(
        model_paths=f"{path},{path}",
        model_path=tmp_path / "whatever.pkl",
    )

    result = live_trader._load_models(args)

    assert result is None
    assert "duplicate model paths" in capsys.readouterr().err


def test_load_models_normalized_duplicate_ensemble_paths_returns_none(
    tmp_path, monkeypatch, capsys
):
    path = tmp_path / "alltrain_seed0.pkl"
    path.write_bytes(b"x")
    monkeypatch.chdir(tmp_path)
    args = SimpleNamespace(
        model_paths=f"alltrain_seed0.pkl,{path}",
        model_path=tmp_path / "whatever.pkl",
    )

    result = live_trader._load_models(args)

    assert result is None
    assert "duplicate model paths" in capsys.readouterr().err


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


def test_load_models_rejects_mixed_ensemble_feature_contract(monkeypatch, tmp_path, capsys):
    """Direct live-trader launch must not bypass preflight's ensemble contract."""
    path0 = tmp_path / "alltrain_seed0.pkl"
    path7 = tmp_path / "alltrain_seed7.pkl"
    path0.write_bytes(b"x")
    path7.write_bytes(b"x")
    models_by_path = {
        path0: SimpleNamespace(feature_cols=["ret_1d", "ret_5d"]),
        path7: SimpleNamespace(feature_cols=["ret_5d", "ret_1d"]),
    }

    monkeypatch.setattr(
        live_trader.XGBStockModel,
        "load",
        staticmethod(lambda path: models_by_path[Path(path)]),
    )
    args = SimpleNamespace(
        model_paths=f"{path0},{path7}",
        model_path=tmp_path / "whatever.pkl",
    )

    result = live_trader._load_models(args)

    assert result is None
    err = capsys.readouterr().err
    assert "Ensemble feature_cols mismatch" in err
    assert str(path7) in err


def test_load_models_accepts_matching_ensemble_feature_contract(monkeypatch, tmp_path):
    path0 = tmp_path / "alltrain_seed0.pkl"
    path7 = tmp_path / "alltrain_seed7.pkl"
    path0.write_bytes(b"x")
    path7.write_bytes(b"x")
    models_by_path = {
        path0: SimpleNamespace(feature_cols=["ret_1d", "rank_ret_5d"]),
        path7: SimpleNamespace(feature_cols=["ret_1d", "rank_ret_5d"]),
    }

    monkeypatch.setattr(
        live_trader.XGBStockModel,
        "load",
        staticmethod(lambda path: models_by_path[Path(path)]),
    )
    args = SimpleNamespace(
        model_paths=f"{path0},{path7}",
        model_path=tmp_path / "whatever.pkl",
    )

    result = live_trader._load_models(args)

    assert result == [models_by_path[path0], models_by_path[path7]]


def test_load_models_rejects_invalid_ensemble_feature_cols(monkeypatch, tmp_path, capsys):
    path0 = tmp_path / "alltrain_seed0.pkl"
    path0.write_bytes(b"x")

    monkeypatch.setattr(
        live_trader.XGBStockModel,
        "load",
        staticmethod(lambda _path: SimpleNamespace(feature_cols=[])),
    )
    args = SimpleNamespace(
        model_paths=str(path0),
        model_path=tmp_path / "whatever.pkl",
    )

    result = live_trader._load_models(args)

    assert result is None
    assert "invalid feature_cols" in capsys.readouterr().err


def test_load_models_rejects_unsupported_single_model_features(monkeypatch, tmp_path, capsys):
    path = tmp_path / "single.pkl"
    path.write_bytes(b"x")

    monkeypatch.setattr(
        live_trader.XGBStockModel,
        "load",
        staticmethod(lambda _path: SimpleNamespace(feature_cols=["ret_1d", "future_alpha_leak"])),
    )
    args = SimpleNamespace(
        model_paths="",
        model_path=path,
    )

    result = live_trader._load_models(args)

    assert result is None
    assert "unsupported live features" in capsys.readouterr().err


def test_load_models_rejects_unsupported_ensemble_features(monkeypatch, tmp_path, capsys):
    path0 = tmp_path / "alltrain_seed0.pkl"
    path7 = tmp_path / "alltrain_seed7.pkl"
    path0.write_bytes(b"x")
    path7.write_bytes(b"x")
    models_by_path = {
        path0: SimpleNamespace(feature_cols=["ret_1d", "future_alpha_leak"]),
        path7: SimpleNamespace(feature_cols=["ret_1d", "future_alpha_leak"]),
    }

    monkeypatch.setattr(
        live_trader.XGBStockModel,
        "load",
        staticmethod(lambda path: models_by_path[Path(path)]),
    )
    args = SimpleNamespace(
        model_paths=f"{path0},{path7}",
        model_path=tmp_path / "whatever.pkl",
    )

    result = live_trader._load_models(args)

    assert result is None
    assert "unsupported live features" in capsys.readouterr().err


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
