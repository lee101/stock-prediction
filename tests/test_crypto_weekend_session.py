"""Tests for crypto_weekend/session.py — the embedded path used by the
xgb leader process. Covers position filtering, order submission via the
TradingClient stub, buy sizing (cash-aware), and the tick state machine.
"""
from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from crypto_weekend import session as cws


def _ts(s: str) -> datetime:
    return datetime.fromisoformat(s).replace(tzinfo=timezone.utc)


# ---------- Alpaca data credentials ----------

class TestCryptoDataCredentials:
    def test_live_mode_prefers_prod_keys(self, monkeypatch):
        monkeypatch.setenv("ALP_PAPER", "0")
        monkeypatch.setenv("ALP_KEY_ID", "paper-key")
        monkeypatch.setenv("ALP_SECRET_KEY", "paper-secret")
        monkeypatch.setenv("ALP_KEY_ID_PROD", "prod-key")
        monkeypatch.setenv("ALP_SECRET_KEY_PROD", "prod-secret")

        assert cws._active_crypto_data_credentials() == ("prod-key", "prod-secret")

    def test_paper_mode_prefers_paper_keys(self, monkeypatch):
        monkeypatch.setenv("ALP_PAPER", "1")
        monkeypatch.setenv("ALP_KEY_ID", "paper-key")
        monkeypatch.setenv("ALP_SECRET_KEY", "paper-secret")
        monkeypatch.setenv("ALP_KEY_ID_PROD", "prod-key")
        monkeypatch.setenv("ALP_SECRET_KEY_PROD", "prod-secret")

        assert cws._active_crypto_data_credentials() == ("paper-key", "paper-secret")

    def test_explicit_apca_env_overrides_repo_key_names(self, monkeypatch):
        monkeypatch.setenv("ALP_PAPER", "0")
        monkeypatch.setenv("APCA_API_KEY_ID", "apca-key")
        monkeypatch.setenv("APCA_API_SECRET_KEY", "apca-secret")
        monkeypatch.setenv("ALP_KEY_ID_PROD", "prod-key")
        monkeypatch.setenv("ALP_SECRET_KEY_PROD", "prod-secret")

        assert cws._active_crypto_data_credentials() == ("apca-key", "apca-secret")


# ---------- _to_slashed ----------

class TestSlashedConversion:
    def test_compact_becomes_slashed(self):
        assert cws._to_slashed("BTCUSD") == "BTC/USD"

    def test_slashed_stays_slashed(self):
        assert cws._to_slashed("BTC/USD") == "BTC/USD"

    def test_lowercase_promoted(self):
        assert cws._to_slashed("btcusd") == "BTC/USD"


# ---------- get_crypto_positions ----------

def _fake_pos(symbol: str, mv: float, qty: float = 1.0, side: str = "long"):
    p = mock.MagicMock()
    p.symbol = symbol
    p.market_value = mv
    p.qty = qty
    p.side = side
    return p


class TestGetCryptoPositions:
    def test_filters_dust(self):
        client = mock.MagicMock()
        client.get_all_positions.return_value = [
            _fake_pos("BTCUSD", 0.0008),
            _fake_pos("ETHUSD", 5000.0),
        ]
        with mock.patch.object(cws, "log_event"):
            out = cws.get_crypto_positions(client)
        assert len(out) == 1
        assert out[0].symbol == "ETHUSD"

    def test_ignores_non_crypto_symbols(self):
        client = mock.MagicMock()
        client.get_all_positions.return_value = [
            _fake_pos("AAPL", 10000.0),
            _fake_pos("BTCUSD", 10000.0),
        ]
        with mock.patch.object(cws, "log_event"):
            out = cws.get_crypto_positions(client)
        assert len(out) == 1
        assert out[0].symbol == "BTCUSD"

    def test_handles_api_exception(self):
        client = mock.MagicMock()
        client.get_all_positions.side_effect = RuntimeError("api down")
        with mock.patch.object(cws, "log_event"):
            out = cws.get_crypto_positions(client)
        assert out == []


# ---------- do_buy (cash-aware) ----------

class _FakeAccount:
    """Plain object with real float attributes — MagicMock coerces children
    to MagicMock when accessed, which breaks `float(account.equity)`."""
    def __init__(self, equity: float, cash: float, buying_power: float | None = None):
        self.equity = equity
        self.cash = cash
        self.buying_power = buying_power if buying_power is not None else cash


def _mk_account(equity: float, cash: float, buying_power: float | None = None):
    return _FakeAccount(equity, cash, buying_power)


def _mk_picks():
    return [
        {"alpaca_symbol": "BTCUSD", "fri_close": 50_000.0},
        {"alpaca_symbol": "ETHUSD", "fri_close": 2_000.0},
    ]


class TestDoBuy:
    """In the test env `alpaca.trading.requests.LimitOrderRequest` is a
    MagicMock (see tests/conftest.py) — so we inspect the CONSTRUCTOR
    call args on `cws.LimitOrderRequest` rather than the returned object.
    """

    def test_caps_at_cash_when_stock_bot_holds_positions(self):
        client = mock.MagicMock()
        client.get_account.return_value = _mk_account(30_000.0, 5_000.0, 0.0)
        client.submit_order.return_value = mock.MagicMock(id="ORD1")
        with mock.patch.object(cws, "log_event"), \
             mock.patch.object(cws, "_crypto_limit_price",
                               side_effect=[(50_000.0, "test"), (2_000.0, "test")]), \
             mock.patch.object(cws, "LimitOrderRequest") as LOR:
            n = cws.do_buy(client, _mk_picks(), max_gross=0.5, dry_run=False)
        assert n == 2
        total_notional = 0.0
        for call in LOR.call_args_list:
            kw = call.kwargs
            total_notional += float(kw["qty"]) * (
                50_000.0 if "BTC" in kw["symbol"] else 2_000.0
            )
            assert kw["limit_price"] in (50_000.0, 2_000.0)
        assert 4900.0 <= total_notional <= 5000.0

    def test_caps_at_equity_max_gross(self):
        client = mock.MagicMock()
        client.get_account.return_value = _mk_account(30_000.0, 30_000.0)
        client.submit_order.return_value = mock.MagicMock(id="ORD1")
        with mock.patch.object(cws, "log_event"), \
             mock.patch.object(cws, "_crypto_limit_price",
                               side_effect=[(50_000.0, "test"), (2_000.0, "test")]), \
             mock.patch.object(cws, "LimitOrderRequest") as LOR:
            n = cws.do_buy(client, _mk_picks(), max_gross=0.5, dry_run=False)
        assert n == 2
        total_notional = 0.0
        for call in LOR.call_args_list:
            kw = call.kwargs
            total_notional += float(kw["qty"]) * (
                50_000.0 if "BTC" in kw["symbol"] else 2_000.0
            )
            assert kw["limit_price"] in (50_000.0, 2_000.0)
        assert 14_900.0 <= total_notional <= 15_000.0

    def test_skips_when_cash_below_min(self):
        client = mock.MagicMock()
        client.get_account.return_value = _mk_account(30_000.0, 40.0)
        with mock.patch.object(cws, "log_event"):
            n = cws.do_buy(client, _mk_picks(), max_gross=0.5, dry_run=False)
        assert n == 0
        client.submit_order.assert_not_called()

    def test_dry_run_submits_no_orders(self):
        client = mock.MagicMock()
        client.get_account.return_value = _mk_account(30_000.0, 30_000.0)
        with mock.patch.object(cws, "log_event"):
            n = cws.do_buy(client, _mk_picks(), max_gross=0.5, dry_run=True)
        assert n == 2
        client.submit_order.assert_not_called()

    def test_uses_slashed_symbol_in_order(self):
        client = mock.MagicMock()
        client.get_account.return_value = _mk_account(30_000.0, 30_000.0)
        client.submit_order.return_value = mock.MagicMock(id="ORD1")
        with mock.patch.object(cws, "log_event"), \
             mock.patch.object(cws, "_crypto_limit_price", return_value=(50_075.0, "ask")), \
             mock.patch.object(cws, "LimitOrderRequest") as LOR:
            cws.do_buy(client, [{"alpaca_symbol": "BTCUSD", "fri_close": 50_000.0}],
                       max_gross=0.5, dry_run=False)
        assert LOR.call_args.kwargs["symbol"] == "BTC/USD"
        assert LOR.call_args.kwargs["limit_price"] == 50_075.0

    def test_skips_when_no_explicit_limit_price(self):
        client = mock.MagicMock()
        client.get_account.return_value = _mk_account(30_000.0, 30_000.0)
        with mock.patch.object(cws, "log_event"), \
             mock.patch.object(cws, "_crypto_limit_price", return_value=(0.0, "none")):
            n = cws.do_buy(client, [{"alpaca_symbol": "BTCUSD", "fri_close": 50_000.0}],
                           max_gross=0.5, dry_run=False)
        assert n == 0
        client.submit_order.assert_not_called()


# ---------- do_sell ----------

class TestDoSell:
    def test_submits_sells_for_long(self):
        client = mock.MagicMock()
        client.submit_order.return_value = mock.MagicMock(id="ORD1")
        positions = [_fake_pos("BTCUSD", 5000.0, qty=0.1, side="long")]
        with mock.patch.object(cws, "log_event"), \
             mock.patch.object(cws, "_crypto_limit_price", return_value=(49_925.0, "bid")), \
             mock.patch.object(cws, "LimitOrderRequest") as LOR:
            n = cws.do_sell(client, positions, dry_run=False)
        assert n == 1
        kw = LOR.call_args.kwargs
        assert kw["symbol"] == "BTC/USD"
        assert kw["qty"] == 0.1
        assert kw["limit_price"] == 49_925.0

    def test_dry_run_submits_no_orders(self):
        client = mock.MagicMock()
        positions = [_fake_pos("BTCUSD", 5000.0, qty=0.1, side="long")]
        with mock.patch.object(cws, "log_event"):
            n = cws.do_sell(client, positions, dry_run=True)
        assert n == 1
        client.submit_order.assert_not_called()

    def test_no_positions_returns_zero(self):
        client = mock.MagicMock()
        with mock.patch.object(cws, "log_event"):
            n = cws.do_sell(client, [], dry_run=False)
        assert n == 0
        client.submit_order.assert_not_called()


# ---------- run_crypto_tick ----------

class TestRunCryptoTick:
    def setup_method(self):
        cws.reset_state_for_testing()

    def test_idle_weekday_does_nothing(self):
        client = mock.MagicMock()
        client.get_all_positions.return_value = []
        tue = _ts("2026-04-28T08:00:00")
        with mock.patch.object(cws, "log_event"):
            s = cws.run_crypto_tick(client, now=tue)
        assert s["action"] == "none"
        client.submit_order.assert_not_called()

    def test_saturday_with_no_positions_evaluates_and_buys(self):
        client = mock.MagicMock()
        client.get_all_positions.return_value = []
        client.get_account.return_value = _mk_account(30_000.0, 30_000.0)
        client.submit_order.return_value = mock.MagicMock(id="ORD1")
        sat_noon = _ts("2026-04-25T12:00:00")
        fake_picks = [{"alpaca_symbol": "BTCUSD", "fri_close": 50_000.0,
                       "passes": True, "above_sma": True, "vol_ok": True,
                       "vol_20d": 0.02, "sma_20": 45000.0, "symbol": "BTCUSD"}]
        with mock.patch.object(cws, "evaluate_signals", return_value=fake_picks), \
             mock.patch.object(cws, "_crypto_limit_price", return_value=(50_075.0, "ask")), \
             mock.patch.object(cws, "log_event") as log_event:
            s = cws.run_crypto_tick(client, now=sat_noon)
        assert s["action"] == "buy"
        assert client.submit_order.call_count == 1
        log_event.assert_any_call("tick_status", **s)

    def test_monday_early_with_positions_sells(self):
        client = mock.MagicMock()
        client.get_all_positions.return_value = [
            _fake_pos("BTCUSD", 9500.0, qty=0.19, side="long"),
        ]
        client.submit_order.return_value = mock.MagicMock(id="ORD1")
        mon_4 = _ts("2026-04-27T04:00:00")
        with mock.patch.object(cws, "_crypto_limit_price", return_value=(49_925.0, "bid")), \
             mock.patch.object(cws, "log_event"):
            s = cws.run_crypto_tick(client, now=mon_4)
        assert s["action"] == "sell"
        assert client.submit_order.call_count == 1

    def test_position_read_failure_fails_closed(self):
        client = mock.MagicMock()
        client.get_all_positions.side_effect = RuntimeError("positions down")
        sat_noon = _ts("2026-04-25T12:00:00")
        with mock.patch.object(cws, "evaluate_signals") as evaluate_signals, \
             mock.patch.object(cws, "log_event") as log_event:
            s = cws.run_crypto_tick(client, now=sat_noon)
        assert s["action"] == "skip_positions_error"
        assert s["positions_ok"] is False
        evaluate_signals.assert_not_called()
        client.submit_order.assert_not_called()
        log_event.assert_any_call("tick_status", **s)

    def test_same_day_reentry_is_noop(self):
        """Once we act on Saturday, subsequent Saturday ticks are no-ops
        until the UTC date rolls over."""
        client = mock.MagicMock()
        client.get_all_positions.return_value = []
        client.get_account.return_value = _mk_account(30_000.0, 30_000.0)
        client.submit_order.return_value = mock.MagicMock(id="ORD1")
        sat = _ts("2026-04-25T12:00:00")
        fake_picks = [{"alpaca_symbol": "BTCUSD", "fri_close": 50_000.0}]
        with mock.patch.object(cws, "evaluate_signals", return_value=fake_picks), \
             mock.patch.object(cws, "_crypto_limit_price", return_value=(50_075.0, "ask")), \
             mock.patch.object(cws, "log_event"):
            cws.run_crypto_tick(client, now=sat)
            # Simulate a position now exists — but we're idempotent on
            # last_action_date, so a no-op is correct here even if we
            # DIDN'T see the position yet on a flaky API.
            client.get_all_positions.return_value = []
            s2 = cws.run_crypto_tick(client, now=sat)
        assert s2["action"] == "none"

    def test_sunday_is_holdonly_no_trade(self):
        client = mock.MagicMock()
        client.get_all_positions.return_value = []
        sun = _ts("2026-04-26T10:00:00")
        with mock.patch.object(cws, "log_event"):
            s = cws.run_crypto_tick(client, now=sun)
        assert s["action"] == "none"

    def test_monday_post_open_is_flat_no_trade(self):
        """Monday after 12:30 UTC — no sell (missed the window), no buy."""
        client = mock.MagicMock()
        client.get_all_positions.return_value = []
        mon_15 = _ts("2026-04-27T15:00:00")
        with mock.patch.object(cws, "log_event"):
            s = cws.run_crypto_tick(client, now=mon_15)
        assert s["action"] == "none"
