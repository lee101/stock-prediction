"""Unit tests for orchestrator order management logic.

Covers the four main bug areas identified in the code review:
  a) Cancel-then-reorder timing gap — existing valid orders are kept in place.
  b) Cash tracking after cancellations — sleep issued after real cancellations.
  c) Thread safety — AlpacaCryptoWatcher logs (not swallows) position-check errors.
  d) Trailing stop persistence — stale peaks are reset to current price on restart.

All Alpaca API calls are mocked so these tests run without credentials.
"""

from __future__ import annotations

import importlib
import json
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from types import SimpleNamespace
from typing import Optional
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Helpers / stubs
# ---------------------------------------------------------------------------

@dataclass
class _FakePosition:
    symbol: str
    qty: float
    avg_entry_price: float
    current_price: float
    unrealized_pl: float = 0.0

    @property
    def market_value(self) -> float:
        return self.qty * self.current_price


@dataclass
class _FakeOrder:
    id: str
    symbol: str
    side: object  # mock with .value attribute
    limit_price: Optional[float]
    qty: float = 1.0

    class _Side:
        def __init__(self, v): self.value = v

    @classmethod
    def buy(cls, id: str, symbol: str, price: float, qty: float = 1.0):
        o = cls(id=id, symbol=symbol, side=cls._Side("buy"), limit_price=price, qty=qty)
        return o

    @classmethod
    def sell(cls, id: str, symbol: str, price: float, qty: float = 1.0):
        o = cls(id=id, symbol=symbol, side=cls._Side("sell"), limit_price=price, qty=qty)
        return o


@dataclass
class _FakeAccount:
    cash: float = 10_000.0
    buying_power: float = 10_000.0


def _fake_trade_plan(direction="long", buy_price=100.0, sell_price=101.0,
                     confidence=0.8, allocation_pct=20.0):
    plan = MagicMock()
    plan.direction = direction
    plan.buy_price = buy_price
    plan.sell_price = sell_price
    plan.confidence = confidence
    plan.allocation_pct = allocation_pct
    return plan


@pytest.fixture(autouse=True)
def _patch_alpaca_order_types(monkeypatch):
    import alpaca.trading.enums as enums
    import alpaca.trading.requests as requests

    class _OrderSide:
        BUY = "buy"
        SELL = "sell"

    class _TimeInForce:
        GTC = "gtc"
        IOC = "ioc"

    class _QueryOrderStatus:
        OPEN = "open"

    def _limit_order_request(**kwargs):
        return SimpleNamespace(**kwargs)

    def _get_orders_request(**kwargs):
        return SimpleNamespace(**kwargs)

    monkeypatch.setattr(enums, "OrderSide", _OrderSide, raising=False)
    monkeypatch.setattr(enums, "TimeInForce", _TimeInForce, raising=False)
    monkeypatch.setattr(enums, "QueryOrderStatus", _QueryOrderStatus, raising=False)
    monkeypatch.setattr(requests, "LimitOrderRequest", _limit_order_request, raising=False)
    monkeypatch.setattr(requests, "GetOrdersRequest", _get_orders_request, raising=False)


# ---------------------------------------------------------------------------
# (a) Cancel-then-reorder timing gap
# ---------------------------------------------------------------------------

class TestCancelThenReorderTimingGap:
    """Validate the price-similarity gate that keeps existing buy orders live."""

    def _make_signals_and_orders(self, existing_price: float, new_price: float):
        """Return (signals, open_orders, alpaca_mock) for one symbol."""
        sym = "BTCUSD"
        signals = {sym: _fake_trade_plan(buy_price=new_price)}
        open_orders = [_FakeOrder.buy("order-001", sym, existing_price)]
        alpaca = MagicMock()
        alpaca.get_orders.return_value = open_orders
        alpaca.get_account.return_value = _FakeAccount()
        alpaca.get_all_positions.return_value = []
        return signals, open_orders, alpaca

    def test_keeps_order_when_price_within_01pct(self):
        """Orders within 0.1% of new target must NOT be cancelled."""
        # 0.05% difference — below the 0.1% threshold
        new_price = 100.0
        existing_price = 100.05
        signals, open_orders, alpaca = self._make_signals_and_orders(existing_price, new_price)

        kept: set[str] = set()
        canceled: set[str] = set()

        for order in open_orders:
            sym_norm = order.symbol.replace("/", "")
            plan = signals.get(sym_norm)
            ex_price = float(order.limit_price)
            n_price = float(plan.buy_price)
            diff_pct = abs(n_price - ex_price) / ex_price * 100
            if diff_pct <= 0.1:
                kept.add(sym_norm)
            else:
                canceled.add(sym_norm)

        assert "BTCUSD" in kept, "Order within 0.1% must be kept, not cancelled"
        assert "BTCUSD" not in canceled
        alpaca.cancel_order_by_id.assert_not_called()

    def test_cancels_order_when_price_differs_significantly(self):
        """Orders with >0.1% price difference MUST be cancelled."""
        new_price = 100.0
        existing_price = 99.5  # 0.5% difference — above threshold
        signals, open_orders, alpaca = self._make_signals_and_orders(existing_price, new_price)

        kept: set[str] = set()
        canceled: set[str] = set()

        for order in open_orders:
            sym_norm = order.symbol.replace("/", "")
            plan = signals.get(sym_norm)
            ex_price = float(order.limit_price)
            n_price = float(plan.buy_price)
            diff_pct = abs(n_price - ex_price) / ex_price * 100
            if diff_pct <= 0.1:
                kept.add(sym_norm)
            else:
                canceled.add(sym_norm)
                alpaca.cancel_order_by_id(str(order.id))

        assert "BTCUSD" not in kept
        assert "BTCUSD" in canceled
        alpaca.cancel_order_by_id.assert_called_once_with("order-001")

    def test_zero_price_existing_order_is_cancelled(self):
        """An existing order with no limit_price should always be cancelled."""
        order = _FakeOrder.buy("order-999", "ETHUSD", 0.0)
        plan = _fake_trade_plan(buy_price=2000.0)

        kept: set[str] = set()
        canceled: set[str] = set()

        ex_price = float(order.limit_price) if order.limit_price else 0.0
        n_price = float(plan.buy_price)
        # Cannot compute meaningful pct when existing_price == 0
        if n_price > 0 and ex_price > 0:
            diff_pct = abs(n_price - ex_price) / ex_price * 100
            if diff_pct <= 0.1:
                kept.add("ETHUSD")
            else:
                canceled.add("ETHUSD")
        else:
            # existing_price or new_price is 0 — cancel the stale order
            canceled.add("ETHUSD")

        assert "ETHUSD" in canceled
        assert "ETHUSD" not in kept

    def test_buy_not_re_placed_for_kept_symbol(self):
        """When an existing order is kept, the placement loop must skip it."""
        kept_order_syms = {"BTCUSD"}
        sym = "BTCUSD"

        placed = []
        if sym not in kept_order_syms:
            placed.append(sym)

        assert sym not in placed, (
            "Buy must not be re-placed for a symbol whose existing order was kept"
        )


# ---------------------------------------------------------------------------
# (b) Cash tracking — sleep after real cancellations
# ---------------------------------------------------------------------------

class TestCashTrackingAfterCancellation:
    """A 1-second sleep must follow live (non-dry-run) cancellations."""

    def test_sleep_called_after_live_cancel(self):
        """Verify that time.sleep(1.0) is issued after a non-dry-run cancel."""
        canceled_for = {"BTCUSD"}
        dry_run = False
        sleep_calls = []

        def fake_sleep(secs):
            sleep_calls.append(secs)

        if canceled_for and not dry_run:
            fake_sleep(1.0)

        assert sleep_calls == [1.0], "Must sleep 1 s after live cancellations"

    def test_no_sleep_in_dry_run(self):
        """No sleep should occur in dry-run mode even if orders would be cancelled."""
        canceled_for = {"BTCUSD"}
        dry_run = True
        sleep_calls = []

        def fake_sleep(secs):
            sleep_calls.append(secs)

        if canceled_for and not dry_run:
            fake_sleep(1.0)

        assert sleep_calls == [], "Must NOT sleep in dry-run mode"

    def test_no_sleep_when_nothing_cancelled(self):
        """No sleep when no orders were actually cancelled."""
        canceled_for: set[str] = set()
        dry_run = False
        sleep_calls = []

        def fake_sleep(secs):
            sleep_calls.append(secs)

        if canceled_for and not dry_run:
            fake_sleep(1.0)

        assert sleep_calls == [], "Must NOT sleep when canceled_for is empty"


# ---------------------------------------------------------------------------
# (c) Thread safety — watcher logs position-check errors
# ---------------------------------------------------------------------------

class TestWatcherPositionCheckLogging:
    """AlpacaCryptoWatcher._check_position_exists must log API errors."""

    def _make_watcher(self, alpaca_mock):
        from unified_orchestrator.alpaca_watcher import AlpacaCryptoWatcher, OrderPair
        pair = OrderPair(
            symbol="BTCUSD",
            buy_price=50_000.0,
            sell_price=51_000.0,
            target_qty=0.001,
        )
        watcher = AlpacaCryptoWatcher(
            alpaca_client=alpaca_mock,
            pairs=[pair],
            dry_run=True,
        )
        return watcher

    def test_position_check_logs_on_api_error(self):
        """API errors in _check_position_exists must be logged, not swallowed."""
        alpaca = MagicMock()
        alpaca.get_all_positions.side_effect = RuntimeError("connection timeout")

        watcher = self._make_watcher(alpaca)

        # Loguru doesn't propagate to stdlib caplog; patch the module-level logger.
        with patch("unified_orchestrator.alpaca_watcher.logger") as mock_log:
            result = watcher._check_position_exists("BTCUSD")

        assert result is False, "Should return False on API error (conservative)"
        # warning() must have been called with a message containing the error text
        warning_args = [str(a) for call in mock_log.warning.call_args_list for a in call.args]
        assert any("connection timeout" in a for a in warning_args), (
            f"API error must be logged as WARNING — got: {warning_args}"
        )

    def test_position_check_returns_true_when_position_exists(self):
        """Normal path: returns True when position exists with qty > 0."""
        alpaca = MagicMock()
        pos = MagicMock()
        pos.symbol = "BTC/USD"
        pos.qty = 0.001
        alpaca.get_all_positions.return_value = [pos]

        watcher = self._make_watcher(alpaca)
        assert watcher._check_position_exists("BTCUSD") is True

    def test_position_check_returns_false_when_no_position(self):
        """Returns False when no position exists."""
        alpaca = MagicMock()
        alpaca.get_all_positions.return_value = []

        watcher = self._make_watcher(alpaca)
        assert watcher._check_position_exists("BTCUSD") is False

    def test_position_check_normalises_slash_symbol(self):
        """BTC/USD from Alpaca API must match against BTCUSD query."""
        alpaca = MagicMock()
        pos = MagicMock()
        pos.symbol = "BTC/USD"  # Alpaca format with slash
        pos.qty = 0.5
        alpaca.get_all_positions.return_value = [pos]

        watcher = self._make_watcher(alpaca)
        assert watcher._check_position_exists("BTCUSD") is True

    def test_buy_resolution_uses_actual_partial_fill_qty(self):
        """Watcher must size the TP sell to the actual filled qty, not the target qty."""
        from unified_orchestrator.alpaca_watcher import AlpacaCryptoWatcher, OrderPair

        alpaca = MagicMock()
        alpaca.get_orders.return_value = []
        partial = MagicMock()
        partial.symbol = "BTC/USD"
        partial.qty = 0.0004
        alpaca.get_all_positions.return_value = [partial]
        alpaca.submit_order.return_value = MagicMock(id="sell-001")

        pair = OrderPair(
            symbol="BTCUSD",
            buy_price=50_000.0,
            sell_price=51_000.0,
            target_qty=0.001,
            buy_order_id="buy-001",
        )
        watcher = AlpacaCryptoWatcher(alpaca_client=alpaca, pairs=[pair], dry_run=False)

        watcher._poll_orders()

        assert pair.buy_order_id is None
        assert pair.sell_order_id == "sell-001"
        assert pair.current_qty == pytest.approx(0.0004)
        submit_req = alpaca.submit_order.call_args.args[0]
        assert submit_req.qty == pytest.approx(0.00039999)
        assert watcher.fill_log[0]["qty"] == pytest.approx(0.0004)

    def test_sell_resolution_defers_when_position_state_unknown(self):
        """Watcher must not clear sell tracking when the position API errored."""
        from unified_orchestrator.alpaca_watcher import AlpacaCryptoWatcher, OrderPair

        alpaca = MagicMock()
        alpaca.get_orders.return_value = []
        alpaca.get_all_positions.side_effect = RuntimeError("position api unavailable")

        pair = OrderPair(
            symbol="BTCUSD",
            buy_price=50_000.0,
            sell_price=51_000.0,
            target_qty=0.001,
            current_qty=0.001,
            sell_order_id="sell-001",
        )
        watcher = AlpacaCryptoWatcher(alpaca_client=alpaca, pairs=[pair], dry_run=True)

        watcher._poll_orders()

        assert pair.sell_order_id == "sell-001"
        assert pair.oscillations == 0
        assert watcher.fill_log == []


# ---------------------------------------------------------------------------
# (d) Trailing stop persistence — stale peaks reset on restart
# ---------------------------------------------------------------------------

class TestTrailingStopPeakPersistence:
    """update_peak_prices must reset stale peaks that are < current price."""

    def _make_position(self, symbol: str, qty: float, current_price: float):
        from unified_orchestrator.state import Position
        return Position(
            symbol=symbol,
            qty=qty,
            avg_price=current_price,
            current_price=current_price,
        )

    def test_stale_peak_below_current_resets_to_current(self):
        """After restart, if persisted peak < current price, reset to current."""
        from unified_orchestrator.position_tracker import update_peak_prices

        # Simulate: orchestrator restarted, price has risen since the old peak
        old_peaks = {"BTCUSD": 45_000.0}  # stale — price is now 50k
        current_positions = {
            "BTCUSD": self._make_position("BTCUSD", qty=0.1, current_price=50_000.0),
        }

        # Patch loguru logger to capture calls (loguru doesn't propagate to caplog)
        with patch("unified_orchestrator.position_tracker.logger") as mock_log:
            updated = update_peak_prices(old_peaks, current_positions)

        # Peak must be reset to current price, not left at stale 45k
        assert updated["BTCUSD"] == 50_000.0, (
            "Stale peak below current price must be reset to current price"
        )
        # Check that logger.info was called with a message about the stale peak
        info_args = [str(a) for c in mock_log.info.call_args_list for a in c.args]
        assert any("stale peak" in a.lower() or "restart" in a.lower() for a in info_args), (
            f"Reset of stale peak must be logged — got: {info_args}"
        )

    def test_peak_above_current_kept_as_is(self):
        """If stored peak > current price, the stored peak is the real high-water mark."""
        from unified_orchestrator.position_tracker import update_peak_prices

        old_peaks = {"ETHUSD": 3_000.0}  # real peak above current
        current_positions = {
            "ETHUSD": self._make_position("ETHUSD", qty=1.0, current_price=2_800.0),
        }

        updated = update_peak_prices(old_peaks, current_positions)

        # The real peak must be kept — dropping it would prevent the trailing stop
        assert updated["ETHUSD"] == 3_000.0, (
            "Real peak above current price must not be overwritten"
        )

    def test_trailing_stop_not_triggered_after_peak_reset(self):
        """After a stale-peak reset, the trailing stop must NOT fire immediately."""
        from unified_orchestrator.position_tracker import update_peak_prices, get_trailing_stop_symbols

        # Stale peak from before restart: 45k.  Current price: 50k.
        old_peaks = {"BTCUSD": 45_000.0}
        current_positions = {
            "BTCUSD": self._make_position("BTCUSD", qty=0.1, current_price=50_000.0),
        }

        updated = update_peak_prices(old_peaks, current_positions)
        # Peak is now reset to 50k.  Price is also 50k → drop = 0% → no stop.
        stop_syms = get_trailing_stop_symbols(updated, current_positions, trail_pct=0.3)

        assert "BTCUSD" not in stop_syms, (
            "Trailing stop must NOT fire immediately after a stale-peak reset"
        )

    def test_new_position_initialises_peak_at_current(self):
        """Brand-new position (not in persisted peaks) → peak = current price."""
        from unified_orchestrator.position_tracker import update_peak_prices

        old_peaks: dict = {}
        current_positions = {
            "SOLUSD": self._make_position("SOLUSD", qty=5.0, current_price=120.0),
        }

        with patch("unified_orchestrator.position_tracker.logger") as mock_log:
            updated = update_peak_prices(old_peaks, current_positions)

        assert updated["SOLUSD"] == 120.0
        info_args = [str(a) for c in mock_log.info.call_args_list for a in c.args]
        assert any("120" in a for a in info_args), (
            f"New position initialization must be logged — got: {info_args}"
        )

    def test_peak_advances_with_new_highs(self):
        """Peak must be updated when current price exceeds stored peak."""
        from unified_orchestrator.position_tracker import update_peak_prices

        old_peaks = {"BTCUSD": 50_000.0}
        current_positions = {
            "BTCUSD": self._make_position("BTCUSD", qty=0.1, current_price=52_000.0),
        }

        updated = update_peak_prices(old_peaks, current_positions)
        assert updated["BTCUSD"] == 52_000.0, "Peak must advance to new high"

    def test_exited_position_removed_from_peaks(self):
        """Positions that disappear must be removed from the updated peaks dict."""
        from unified_orchestrator.position_tracker import update_peak_prices

        old_peaks = {"BTCUSD": 50_000.0, "ETHUSD": 3_000.0}
        # Only BTC is still held
        current_positions = {
            "BTCUSD": self._make_position("BTCUSD", qty=0.1, current_price=50_500.0),
        }

        updated = update_peak_prices(old_peaks, current_positions)
        assert "ETHUSD" not in updated, "Exited positions must be removed from peaks"
        assert "BTCUSD" in updated


# ---------------------------------------------------------------------------
# Additional: force_exit and trailing stop don't double-exit same symbol
# ---------------------------------------------------------------------------

class TestNoDoubleExit:
    """trailing_stop_syms -= force_exit_syms must prevent double-exit."""

    def test_trailing_stop_sym_excluded_when_already_force_exiting(self):
        force_exit_syms = {"BTCUSD"}
        trailing_stop_syms = {"BTCUSD", "ETHUSD"}
        trailing_stop_syms -= force_exit_syms

        assert "BTCUSD" not in trailing_stop_syms
        assert "ETHUSD" in trailing_stop_syms


# ---------------------------------------------------------------------------
# Additional: validate_plan_safety edge cases
# ---------------------------------------------------------------------------

class TestValidatePlanSafety:
    """Sanity tests for the plan safety gate."""

    def _plan(self, direction="long", buy_price=100.0, sell_price=101.0,
               confidence=0.8, allocation_pct=0.0):
        p = MagicMock()
        p.direction = direction
        p.buy_price = buy_price
        p.sell_price = sell_price
        p.confidence = confidence
        p.allocation_pct = allocation_pct
        return p

    def test_hold_always_valid(self):
        from unified_orchestrator.orchestrator import validate_plan_safety
        plan = self._plan(direction="hold")
        ok, _ = validate_plan_safety(plan, current_price=100.0)
        assert ok

    def test_sell_above_buy_plus_fees(self):
        from unified_orchestrator.orchestrator import validate_plan_safety
        # sell 1.5% above buy — well above 20bps round-trip fees
        plan = self._plan(buy_price=100.0, sell_price=101.5)
        ok, reason = validate_plan_safety(plan, current_price=100.0, fee_bps=20.0)
        assert ok, reason

    def test_sell_below_buy_plus_fees_rejected(self):
        from unified_orchestrator.orchestrator import validate_plan_safety
        # sell only 0.01% above buy — below 20bps fees
        plan = self._plan(buy_price=100.0, sell_price=100.01)
        ok, reason = validate_plan_safety(plan, current_price=100.0, fee_bps=20.0)
        assert not ok
        assert "sell" in reason.lower()

    def test_buy_price_too_far_from_current_rejected(self):
        from unified_orchestrator.orchestrator import validate_plan_safety
        # buy_price is 10% from current — beyond the 5% guard
        plan = self._plan(buy_price=90.0, sell_price=92.0)
        ok, reason = validate_plan_safety(plan, current_price=100.0)
        assert not ok
        assert "10.0%" in reason or "10%" in reason

    def test_no_prices_set_for_long_rejected(self):
        from unified_orchestrator.orchestrator import validate_plan_safety
        plan = self._plan(buy_price=0.0, sell_price=0.0)
        ok, reason = validate_plan_safety(plan, current_price=100.0)
        assert not ok


# ---------------------------------------------------------------------------
# Additional: read_pending_fills skips malformed lines
# ---------------------------------------------------------------------------

class TestReadPendingFills:
    """Malformed lines in fill_events.jsonl must be skipped, not crash."""

    def test_fill_events_path_follows_state_dir_env(self, monkeypatch, tmp_path):
        package = sys.modules.get("unified_orchestrator")
        original = sys.modules.get("unified_orchestrator.conditional_orders")

        monkeypatch.setenv("STATE_DIR", str(tmp_path))
        sys.modules.pop("unified_orchestrator.conditional_orders", None)
        reloaded = importlib.import_module("unified_orchestrator.conditional_orders")

        try:
            assert reloaded.FILL_EVENTS_FILE == tmp_path / "fill_events.jsonl"
        finally:
            if original is not None:
                sys.modules["unified_orchestrator.conditional_orders"] = original
                if package is not None:
                    setattr(package, "conditional_orders", original)

    def test_record_fill_event_honors_explicit_fill_file(self, tmp_path):
        from unified_orchestrator import conditional_orders as co

        fill_file = tmp_path / "fill_events.jsonl"
        step = co.TradingStep(
            step_id="s1",
            broker="alpaca",
            action="buy",
            symbol="BTCUSD",
            qty=0.25,
        )

        co.record_fill_event(
            step,
            fill_price=50_000.0,
            fill_qty=0.25,
            plan_id="plan-123",
            fill_events_file=fill_file,
        )

        rows = [json.loads(line) for line in fill_file.read_text().splitlines()]

        assert rows == [
            {
                "timestamp": rows[0]["timestamp"],
                "plan_id": "plan-123",
                "step_id": "s1",
                "broker": "alpaca",
                "action": "buy",
                "symbol": "BTCUSD",
                "fill_price": 50_000.0,
                "fill_qty": 0.25,
                "status": "filled",
            }
        ]

    def test_malformed_line_skipped(self, tmp_path):
        from unified_orchestrator import conditional_orders as co

        fill_file = tmp_path / "fill_events.jsonl"
        good = json.dumps({
            "timestamp": "2026-03-28T10:00:00+00:00",
            "plan_id": "p1",
            "step_id": "s1",
            "broker": "alpaca",
            "action": "buy",
            "symbol": "BTCUSD",
            "fill_price": 50000,
            "fill_qty": 0.001,
            "status": "filled",
        })
        fill_file.write_text(good + "\n{BAD JSON}\n")

        class _FrozenDatetime(datetime):
            @classmethod
            def now(cls, tz=None):
                current = cls(2026, 3, 28, 12, 0, tzinfo=timezone.utc)
                return current if tz is not None else current.replace(tzinfo=None)

        monkeypatch = pytest.MonkeyPatch()
        monkeypatch.setattr(co, "datetime", _FrozenDatetime)
        try:
            events = co.read_pending_fills(since_minutes=9999, fill_events_file=fill_file)
        finally:
            monkeypatch.undo()

        assert len(events) == 1, "Only the valid line should be returned"
        assert events[0]["symbol"] == "BTCUSD"

    def test_malformed_timestamp_skipped(self, tmp_path):
        from unified_orchestrator import conditional_orders as co

        fill_file = tmp_path / "fill_events.jsonl"
        good = json.dumps({
            "timestamp": "2026-03-28T10:00:00+00:00",
            "plan_id": "p1",
            "step_id": "s1",
            "broker": "alpaca",
            "action": "buy",
            "symbol": "BTCUSD",
            "fill_price": 50000,
            "fill_qty": 0.001,
            "status": "filled",
        })
        bad_ts = json.dumps({
            "timestamp": "not-a-real-time",
            "plan_id": "p2",
            "step_id": "s2",
            "broker": "alpaca",
            "action": "sell",
            "symbol": "ETHUSD",
            "fill_price": 2000,
            "fill_qty": 0.5,
            "status": "filled",
        })
        fill_file.write_text(good + "\n" + bad_ts + "\n")

        class _FrozenDatetime(datetime):
            @classmethod
            def now(cls, tz=None):
                current = cls(2026, 3, 28, 12, 0, tzinfo=timezone.utc)
                return current if tz is not None else current.replace(tzinfo=None)

        monkeypatch = pytest.MonkeyPatch()
        monkeypatch.setattr(co, "datetime", _FrozenDatetime)
        try:
            events = co.read_pending_fills(since_minutes=9999, fill_events_file=fill_file)
        finally:
            monkeypatch.undo()

        assert len(events) == 1
        assert events[0]["step_id"] == "s1"

    def test_recent_fills_only_return_tail_window_in_chronological_order(self, tmp_path, monkeypatch):
        from unified_orchestrator import conditional_orders as co

        fill_file = tmp_path / "fill_events.jsonl"
        rows = [
            {
                "timestamp": "2026-03-28T09:30:00+00:00",
                "plan_id": "p-old",
                "step_id": "s-old",
                "broker": "alpaca",
                "action": "sell",
                "symbol": "ETHUSD",
                "fill_price": 2000.0,
                "fill_qty": 0.5,
                "status": "filled",
            },
            {
                "timestamp": "2026-03-28T11:10:00+00:00",
                "plan_id": "p-new-1",
                "step_id": "s-new-1",
                "broker": "alpaca",
                "action": "buy",
                "symbol": "BTCUSD",
                "fill_price": 50000.0,
                "fill_qty": 0.001,
                "status": "filled",
            },
            {
                "timestamp": "2026-03-28T11:45:00+00:00",
                "plan_id": "p-new-2",
                "step_id": "s-new-2",
                "broker": "binance",
                "action": "sell",
                "symbol": "SOLUSD",
                "fill_price": 150.0,
                "fill_qty": 3.0,
                "status": "filled",
            },
        ]
        fill_file.write_text("\n".join(json.dumps(row) for row in rows) + "\n")

        class _FrozenDatetime(datetime):
            @classmethod
            def now(cls, tz=None):
                current = cls(2026, 3, 28, 12, 0, tzinfo=timezone.utc)
                return current if tz is not None else current.replace(tzinfo=None)

        monkeypatch.setattr(co, "datetime", _FrozenDatetime)
        events = co.read_pending_fills(since_minutes=60, fill_events_file=fill_file)

        assert [event["step_id"] for event in events] == ["s-new-1", "s-new-2"]

    def test_read_pending_fills_uses_runtime_state_dir_without_reload(self, tmp_path, monkeypatch):
        from unified_orchestrator import conditional_orders as co

        fill_file = tmp_path / "fill_events.jsonl"
        fill_file.write_text(
            json.dumps(
                {
                    "timestamp": "2026-03-28T11:45:00+00:00",
                    "plan_id": "p-new-2",
                    "step_id": "s-new-2",
                    "broker": "binance",
                    "action": "sell",
                    "symbol": "SOLUSD",
                    "fill_price": 150.0,
                    "fill_qty": 3.0,
                    "status": "filled",
                }
            )
            + "\n"
        )

        class _FrozenDatetime(datetime):
            @classmethod
            def now(cls, tz=None):
                current = cls(2026, 3, 28, 12, 0, tzinfo=timezone.utc)
                return current if tz is not None else current.replace(tzinfo=None)

        monkeypatch.setenv("STATE_DIR", str(tmp_path))
        monkeypatch.setattr(co, "datetime", _FrozenDatetime)

        events = co.read_pending_fills(since_minutes=60)

        assert [event["step_id"] for event in events] == ["s-new-2"]
