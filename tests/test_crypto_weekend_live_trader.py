"""Unit tests for crypto_weekend/live_trader.py — focus on windowing logic,
symbol matching, and signal computation. Does NOT hit Alpaca or Binance.
"""
from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from crypto_weekend import live_trader as lt


def _ts(s: str) -> datetime:
    return datetime.fromisoformat(s).replace(tzinfo=timezone.utc)


# ---------- window gating ----------

class TestWindowGating:
    def test_saturday_is_hold_and_buy(self):
        sat_noon = _ts("2026-04-25T12:00:00")
        assert sat_noon.weekday() == 5
        assert lt.in_hold_window(sat_noon)
        assert lt.is_buy_trigger(sat_noon)
        assert not lt.is_sell_trigger(sat_noon)

    def test_sunday_is_hold_no_buy_no_sell(self):
        sun_10 = _ts("2026-04-26T10:00:00")
        assert sun_10.weekday() == 6
        assert lt.in_hold_window(sun_10)
        assert not lt.is_buy_trigger(sun_10)
        assert not lt.is_sell_trigger(sun_10)

    def test_monday_early_is_hold_and_sell(self):
        mon_4 = _ts("2026-04-27T04:00:00")
        assert mon_4.weekday() == 0
        assert lt.in_hold_window(mon_4)
        assert not lt.is_buy_trigger(mon_4)
        assert lt.is_sell_trigger(mon_4)

    def test_monday_noon_still_sell(self):
        mon_12 = _ts("2026-04-27T12:00:00")
        # 12:29 still eligible sell; 12:30 not
        assert lt.is_sell_trigger(mon_12)
        mon_1230 = _ts("2026-04-27T12:30:00")
        assert not lt.is_sell_trigger(mon_1230)

    def test_monday_after_open_is_flat(self):
        mon_15 = _ts("2026-04-27T15:00:00")
        assert not lt.in_hold_window(mon_15)
        assert not lt.is_buy_trigger(mon_15)
        assert not lt.is_sell_trigger(mon_15)

    def test_tuesday_is_flat(self):
        tue = _ts("2026-04-28T08:00:00")
        assert tue.weekday() == 1
        assert not lt.in_hold_window(tue)
        assert not lt.is_buy_trigger(tue)
        assert not lt.is_sell_trigger(tue)

    def test_friday_is_flat(self):
        fri = _ts("2026-04-24T20:00:00")
        assert fri.weekday() == 4
        assert not lt.in_hold_window(fri)
        assert not lt.is_buy_trigger(fri)
        assert not lt.is_sell_trigger(fri)


# ---------- symbol normalization ----------

class TestSymbolMatching:
    def test_normalize_slashed(self):
        assert lt.normalize_symbol_for_match("BTC/USD") == "BTCUSD"

    def test_normalize_compact(self):
        assert lt.normalize_symbol_for_match("BTCUSD") == "BTCUSD"

    def test_normalize_lowercase(self):
        assert lt.normalize_symbol_for_match("btc/usd") == "BTCUSD"

    def test_our_symbols_contains_canonical(self):
        compact_set = {trio[1] for trio in lt.SYMBOLS}
        assert "BTCUSD" in compact_set
        assert "ETHUSD" in compact_set
        assert "SOLUSD" in compact_set


# ---------- signal computation ----------

def _fake_closes_df(closes: list[float]) -> pd.DataFrame:
    n = len(closes)
    # Build a timestamp index ending yesterday UTC (so all bars are
    # "fully closed" relative to Saturday polling).
    end = datetime(2026, 4, 24, 12, 0, tzinfo=timezone.utc)  # Friday
    idx = pd.date_range(end=end, periods=n, freq="D", tz="UTC")
    return pd.DataFrame({"close": closes}, index=idx)


class TestSignal:
    def test_passes_when_above_sma_and_low_vol(self):
        closes = [100.0] * 20 + [100.5, 101.0, 110.0]
        sig = lt.compute_signal_from_df(_fake_closes_df(closes), symbol="BTC/USD")
        assert sig["passes"] is True
        assert sig["above_sma"] is True
        assert sig["vol_ok"] is True

    def test_fails_when_close_below_sma(self):
        closes = [100.0] * 22 + [99.0]
        sig = lt.compute_signal_from_df(_fake_closes_df(closes), symbol="BTC/USD")
        assert sig["passes"] is False
        assert sig["above_sma"] is False

    def test_fails_when_vol_too_high(self):
        rng = np.random.default_rng(42)
        closes = [100.0 + 15 * rng.choice([-1, 1]) for _ in range(22)] + [140.0]
        sig = lt.compute_signal_from_df(_fake_closes_df(closes), symbol="BTC/USD")
        assert sig["vol_ok"] is False
        assert sig["passes"] is False

    def test_not_enough_history(self):
        closes = [100.0] * 10
        sig = lt.compute_signal_from_df(_fake_closes_df(closes), symbol="BTC/USD")
        assert sig["passes"] is False
        assert sig.get("reason") == "not_enough_history"


# ---------- dust filtering ----------

class TestDustFilter:
    def _fake_pos(self, symbol: str, mv: float):
        p = mock.MagicMock()
        p.symbol = symbol
        p.market_value = mv
        return p

    def test_filters_out_sub_dollar_dust(self):
        wrap = mock.MagicMock()
        wrap.get_all_positions.return_value = [
            self._fake_pos("BTCUSD", 0.0008),     # dust
            self._fake_pos("SOLUSD", 0.000001),   # dust
            self._fake_pos("DOGEUSD", 0.006),     # not ours AND dust
        ]
        with mock.patch.object(lt, "log_event"):
            ours = lt.get_crypto_positions(wrap)
        assert ours == []

    def test_keeps_real_position(self):
        wrap = mock.MagicMock()
        wrap.get_all_positions.return_value = [
            self._fake_pos("BTCUSD", 0.0008),    # dust — drop
            self._fake_pos("ETHUSD", 500.0),     # real — keep
        ]
        with mock.patch.object(lt, "log_event"):
            ours = lt.get_crypto_positions(wrap)
        assert len(ours) == 1
        assert ours[0].symbol == "ETHUSD"

    def test_ignores_non_crypto_symbols(self):
        wrap = mock.MagicMock()
        wrap.get_all_positions.return_value = [
            self._fake_pos("AAPL", 5000.0),       # not ours
            self._fake_pos("BTC/USD", 5000.0),    # ours (slashed form still matched)
        ]
        with mock.patch.object(lt, "log_event"):
            ours = lt.get_crypto_positions(wrap)
        assert len(ours) == 1
        assert ours[0].symbol == "BTC/USD"


# ---------- evaluate_signals batch ----------

class TestEvaluateSignals:
    def test_collects_only_passing(self):
        def fake_signal(wrapper_mod, hist_sym):
            passes = hist_sym == "BTCUSD"
            return {"passes": passes, "fri_close": 100.0, "sma_20": 90.0,
                    "vol_20d": 0.02, "above_sma": passes, "vol_ok": True,
                    "symbol": hist_sym}
        with mock.patch.object(lt, "compute_signal", side_effect=fake_signal), \
             mock.patch.object(lt, "log_event"):
            picks = lt.evaluate_signals(mock.MagicMock())
        assert len(picks) == 1
        assert picks[0]["alpaca_symbol"] == "BTCUSD"


# ---------- do_buy sizing (cash-aware) ----------

def _mk_picks():
    return [
        {"alpaca_symbol": "BTCUSD", "fri_close": 50_000.0},
        {"alpaca_symbol": "ETHUSD", "fri_close": 2_000.0},
    ]


def _mk_wrapper(equity: float, cash: float, buying_power: float | None = None):
    """Build a mock alpaca_wrapper with a fake account + order capture."""
    wrap = mock.MagicMock()
    account = mock.MagicMock()
    account.equity = equity
    account.cash = cash
    account.buying_power = buying_power if buying_power is not None else cash
    wrap.alpaca_api.get_account.return_value = account
    wrap.open_order_at_price.return_value = mock.MagicMock(id="ORD1")
    # latest_data returns None so live_trader uses fri_close for price.
    wrap.latest_data = mock.MagicMock(return_value=None)
    return wrap


class TestDoBuySizing:
    def test_caps_at_cash_when_stock_bot_holds_positions(self):
        """equity=$30k, cash=$5k (stock bot at 5x lev using margin).
        Should size gross at $4950 (cash - $50 buffer), not $15k (equity * 0.5)."""
        wrap = _mk_wrapper(equity=30_000.0, cash=5_000.0, buying_power=0.0)
        with mock.patch.object(lt, "log_event"):
            n = lt.do_buy(wrap, _mk_picks(), max_gross=0.5, dry_run=False)
        assert n == 2
        calls = wrap.open_order_at_price.call_args_list
        # Two picks, each at ($5000 - $50) / 2 = $2475 notional.
        total_notional = 0.0
        for call in calls:
            sym, qty, side, price = call.args
            assert side == "buy"
            total_notional += qty * price
        assert 4900.0 <= total_notional <= 5000.0

    def test_caps_at_equity_max_gross_when_cash_is_plentiful(self):
        """equity=$30k, cash=$30k (nothing else held).
        Should size gross at equity * 0.5 = $15k, not full cash."""
        wrap = _mk_wrapper(equity=30_000.0, cash=30_000.0)
        with mock.patch.object(lt, "log_event"):
            n = lt.do_buy(wrap, _mk_picks(), max_gross=0.5, dry_run=False)
        assert n == 2
        total_notional = sum(
            call.args[1] * call.args[3]
            for call in wrap.open_order_at_price.call_args_list
        )
        assert 14_900.0 <= total_notional <= 15_000.0

    def test_skips_when_cash_below_min(self):
        """cash < $100 → skip entirely, no orders."""
        wrap = _mk_wrapper(equity=30_000.0, cash=40.0)
        with mock.patch.object(lt, "log_event"):
            n = lt.do_buy(wrap, _mk_picks(), max_gross=0.5, dry_run=False)
        assert n == 0
        wrap.open_order_at_price.assert_not_called()

    def test_dry_run_submits_no_orders(self):
        wrap = _mk_wrapper(equity=30_000.0, cash=30_000.0)
        with mock.patch.object(lt, "log_event"):
            n = lt.do_buy(wrap, _mk_picks(), max_gross=0.5, dry_run=True)
        assert n == 2
        wrap.open_order_at_price.assert_not_called()
