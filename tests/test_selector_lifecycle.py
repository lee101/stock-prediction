"""Tests for selector watcher lifecycle: orders always present, work-steal, multi-entry."""

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch, call

import pytest

from binanceexp1.trade_binance_selector import (
    SelectorState,
    _compute_edge,
    _detect_holdings,
    _handle_entry,
    _handle_exit,
    _handle_work_steal,
)


@pytest.fixture
def tmp_state(tmp_path):
    return tmp_path / "selector_state.json"


@pytest.fixture
def mock_actions():
    return {
        "BTCUSD": {
            "symbol": "BTCUSD",
            "buy_price": 69000.0,
            "sell_price": 71000.0,
            "buy_amount": 80.0,
            "sell_amount": 80.0,
            "predicted_high_p50_h1": 71500.0,
            "predicted_low_p50_h1": 68500.0,
            "predicted_close_p50_h1": 70200.0,
        },
        "ETHUSD": {
            "symbol": "ETHUSD",
            "buy_price": 2100.0,
            "sell_price": 2200.0,
            "buy_amount": 70.0,
            "sell_amount": 70.0,
            "predicted_high_p50_h1": 2250.0,
            "predicted_low_p50_h1": 2050.0,
            "predicted_close_p50_h1": 2150.0,
        },
        "SOLUSD": {
            "symbol": "SOLUSD",
            "buy_price": 86.0,
            "sell_price": 89.0,
            "buy_amount": 90.0,
            "sell_amount": 90.0,
            "predicted_high_p50_h1": 90.0,
            "predicted_low_p50_h1": 85.0,
            "predicted_close_p50_h1": 88.0,
        },
    }


SYMBOLS = ["BTCUSD", "ETHUSD", "SOLUSD"]

MOCK_RULES = MagicMock(
    min_notional=5.0,
    min_qty=0.001,
    step_size=0.001,
    tick_size=0.01,
    min_price=0.01,
)


def _patch_all():
    """Return dict of patches for all external deps."""
    return {
        "resolve_rules": patch("binanceexp1.trade_binance_selector.resolve_symbol_rules", return_value=MOCK_RULES),
        "get_free": patch("binanceexp1.trade_binance_selector.get_free_balances", return_value=(5000.0, 0.0)),
        "get_total": patch("binanceexp1.trade_binance_selector.get_total_balances", return_value=(0.0, 56.87)),
        "spawn": patch("binanceexp1.trade_binance_selector.spawn_watcher"),
        "cancel_entry": patch("binanceexp1.trade_binance_selector.cancel_entry_watchers"),
        "ensure_valid": patch("binanceexp1.trade_binance_selector._ensure_valid_levels", side_effect=lambda s, b, sp, **kw: (b, sp)),
        "enforce_spread": patch("binanceexp1.trade_binance_selector.enforce_min_spread", side_effect=lambda b, s, **kw: (b, s)),
        "enforce_gap": patch("binanceexp1.trade_binance_selector.enforce_gap", side_effect=lambda s, b, sp, **kw: (b, sp)),
        "get_fee": patch("binanceexp1.trade_binance_selector.get_fee_for_symbol", return_value=0.0),
        "build_plan": patch("binanceexp1.trade_binance_selector._build_plan"),
        "get_price": patch("binanceexp1.trade_binance_selector.binance_wrapper"),
        "compute_qty": patch("binanceexp1.trade_binance_selector.compute_order_quantities"),
    }


class TestSelectorState:
    def test_save_load_roundtrip(self, tmp_state):
        s = SelectorState(open_symbol="SOLUSD", open_ts="2026-01-01T00:00:00+00:00", open_price=87.0)
        s.save(tmp_state)
        loaded = SelectorState.load(tmp_state)
        assert loaded.open_symbol == "SOLUSD"
        assert loaded.open_price == 87.0

    def test_load_missing_file(self, tmp_state):
        s = SelectorState.load(tmp_state)
        assert s.open_symbol is None
        assert s.open_price == 0.0

    def test_hours_held(self):
        now = datetime.now(timezone.utc)
        s = SelectorState(open_symbol="SOLUSD", open_ts=now.isoformat(), open_price=87.0)
        assert s.hours_held() < 0.01


class TestComputeEdge:
    def test_positive_edge(self):
        action = {
            "buy_price": 100.0,
            "buy_amount": 80.0,
            "predicted_high_p50_h1": 105.0,
            "predicted_low_p50_h1": 98.0,
        }
        edge = _compute_edge(action, horizon=1, fee_rate=0.0, risk_weight=0.0)
        assert edge > 0

    def test_edge_with_risk_weight(self):
        action = {
            "buy_price": 100.0,
            "buy_amount": 80.0,
            "predicted_high_p50_h1": 105.0,
            "predicted_low_p50_h1": 95.0,
        }
        edge_rw0 = _compute_edge(action, horizon=1, fee_rate=0.0, risk_weight=0.0)
        edge_rw1 = _compute_edge(action, horizon=1, fee_rate=0.0, risk_weight=1.0)
        assert edge_rw0 > edge_rw1

    def test_edge_zero_buy_price(self):
        action = {"buy_price": 0, "buy_amount": 80, "predicted_high_p50_h1": 105, "predicted_low_p50_h1": 95}
        assert _compute_edge(action, horizon=1, fee_rate=0, risk_weight=0) == -999.0

    def test_edge_zero_buy_amount(self):
        action = {"buy_price": 100, "buy_amount": 0, "predicted_high_p50_h1": 105, "predicted_low_p50_h1": 95}
        assert _compute_edge(action, horizon=1, fee_rate=0, risk_weight=0) == -999.0


class TestHandleEntryMultiSymbol:
    def test_places_orders_for_all_candidates(self, tmp_state, mock_actions):
        state = SelectorState()
        mock_plan = MagicMock()
        mock_plan.buy_price = 100.0
        mock_plan.sell_price = 110.0
        mock_plan.buy_amount = 80.0

        mock_sizing = MagicMock()
        mock_sizing.buy_qty = 1.0

        def check_alloc(**kwargs):
            assert kwargs["quote_free"] == pytest.approx(5000.0 / 3, rel=0.01)
            return mock_sizing

        with (
            patch("binanceexp1.trade_binance_selector.resolve_symbol_rules", return_value=MOCK_RULES),
            patch("binanceexp1.trade_binance_selector.get_free_balances", return_value=(5000.0, 0.0)),
            patch("binanceexp1.trade_binance_selector.spawn_watcher") as mock_spawn,
            patch("binanceexp1.trade_binance_selector.cancel_entry_watchers"),
            patch("binanceexp1.trade_binance_selector.time"),
            patch("binanceexp1.trade_binance_selector._ensure_valid_levels", side_effect=lambda s, b, sp, **kw: (b, sp)),
            patch("binanceexp1.trade_binance_selector.enforce_min_spread", side_effect=lambda b, s, **kw: (b, s)),
            patch("binanceexp1.trade_binance_selector.enforce_gap", side_effect=lambda s, b, sp, **kw: (b, sp)),
            patch("binanceexp1.trade_binance_selector.get_fee_for_symbol", return_value=0.0),
            patch("binanceexp1.trade_binance_selector._build_plan", return_value=mock_plan),
            patch("binanceexp1.trade_binance_selector.compute_order_quantities", side_effect=check_alloc),
        ):
            _handle_entry(
                state, mock_actions, SYMBOLS,
                horizon=1, intensity_scale=5.0,
                price_offset_map={}, default_offset=0.0,
                min_gap_pct=0.0003, risk_weight=0.0, min_edge=0.0,
                poll_seconds=30, expiry_minutes=90,
                price_tolerance=0.0008, dry_run=False,
                state_path=tmp_state,
            )
            assert mock_spawn.call_count == 3

    def test_only_places_above_min_edge(self, tmp_state, mock_actions):
        state = SelectorState()
        with (
            patch("binanceexp1.trade_binance_selector.get_fee_for_symbol", return_value=0.0),
        ):
            _handle_entry(
                state, mock_actions, SYMBOLS,
                horizon=1, intensity_scale=5.0,
                price_offset_map={}, default_offset=0.0,
                min_gap_pct=0.0003, risk_weight=0.0, min_edge=999.0,
                poll_seconds=30, expiry_minutes=90,
                price_tolerance=0.0008, dry_run=False,
                state_path=tmp_state,
            )

    def test_saves_open_price_for_best(self, tmp_state, mock_actions):
        state = SelectorState()
        mock_plan = MagicMock()
        mock_plan.buy_price = 86.0
        mock_plan.sell_price = 89.0
        mock_plan.buy_amount = 90.0

        mock_sizing = MagicMock()
        mock_sizing.buy_qty = 1.0

        with (
            patch("binanceexp1.trade_binance_selector.resolve_symbol_rules", return_value=MOCK_RULES),
            patch("binanceexp1.trade_binance_selector.get_free_balances", return_value=(5000.0, 0.0)),
            patch("binanceexp1.trade_binance_selector.spawn_watcher"),
            patch("binanceexp1.trade_binance_selector.cancel_entry_watchers"),
            patch("binanceexp1.trade_binance_selector.time"),
            patch("binanceexp1.trade_binance_selector._ensure_valid_levels", side_effect=lambda s, b, sp, **kw: (b, sp)),
            patch("binanceexp1.trade_binance_selector.enforce_min_spread", side_effect=lambda b, s, **kw: (b, s)),
            patch("binanceexp1.trade_binance_selector.enforce_gap", side_effect=lambda s, b, sp, **kw: (b, sp)),
            patch("binanceexp1.trade_binance_selector.get_fee_for_symbol", return_value=0.0),
            patch("binanceexp1.trade_binance_selector._build_plan", return_value=mock_plan),
            patch("binanceexp1.trade_binance_selector.compute_order_quantities", return_value=mock_sizing),
        ):
            _handle_entry(
                state, mock_actions, SYMBOLS,
                horizon=1, intensity_scale=5.0,
                price_offset_map={}, default_offset=0.0,
                min_gap_pct=0.0003, risk_weight=0.0, min_edge=0.0,
                poll_seconds=30, expiry_minutes=90,
                price_tolerance=0.0008, dry_run=False,
                state_path=tmp_state,
            )
            assert state.open_price > 0


class TestHandleExit:
    def test_sell_order_always_placed(self, tmp_state, mock_actions):
        state = SelectorState(open_symbol="SOLUSD", open_ts=datetime.now(timezone.utc).isoformat(), open_price=87.0)
        mock_plan = MagicMock()
        mock_plan.sell_price = 89.0
        mock_plan.sell_amount = 90.0
        mock_plan.buy_price = 86.0
        mock_plan.buy_amount = 90.0

        sell_sizing = MagicMock()
        sell_sizing.sell_qty = 56.87
        buy_sizing = MagicMock()
        buy_sizing.buy_qty = 0.0

        with (
            patch("binanceexp1.trade_binance_selector.resolve_symbol_rules", return_value=MOCK_RULES),
            patch("binanceexp1.trade_binance_selector.get_total_balances", return_value=(0.0, 56.87)),
            patch("binanceexp1.trade_binance_selector.spawn_watcher") as mock_spawn,
            patch("binanceexp1.trade_binance_selector._ensure_valid_levels", side_effect=lambda s, b, sp, **kw: (b, sp)),
            patch("binanceexp1.trade_binance_selector.enforce_min_spread", side_effect=lambda b, s, **kw: (b, s)),
            patch("binanceexp1.trade_binance_selector.enforce_gap", side_effect=lambda s, b, sp, **kw: (b, sp)),
            patch("binanceexp1.trade_binance_selector._build_plan", return_value=mock_plan),
            patch("binanceexp1.trade_binance_selector.compute_order_quantities", side_effect=[sell_sizing, buy_sizing]),
        ):
            _handle_exit(
                state, mock_actions, SYMBOLS,
                intensity_scale=5.0, price_offset_map={}, default_offset=0.0,
                min_gap_pct=0.0003, max_hold_hours=6,
                poll_seconds=30, expiry_minutes=90,
                price_tolerance=0.0008, dry_run=False,
                state_path=tmp_state,
            )
            assert mock_spawn.call_count == 1
            spawn_call = mock_spawn.call_args[0][0]
            assert spawn_call.side == "sell"

    def test_force_close_after_max_hold(self, tmp_state, mock_actions):
        state = SelectorState(
            open_symbol="SOLUSD",
            open_ts="2020-01-01T00:00:00+00:00",
            open_price=87.0,
        )
        mock_sizing = MagicMock()
        mock_sizing.sell_qty = 56.87

        with (
            patch("binanceexp1.trade_binance_selector.resolve_symbol_rules", return_value=MOCK_RULES),
            patch("binanceexp1.trade_binance_selector.get_total_balances", return_value=(0.0, 56.87)),
            patch("binanceexp1.trade_binance_selector.spawn_watcher") as mock_spawn,
            patch("binanceexp1.trade_binance_selector._ensure_valid_levels", side_effect=lambda s, b, sp, **kw: (b, sp)),
            patch("binanceexp1.trade_binance_selector.enforce_min_spread", side_effect=lambda b, s, **kw: (b, s)),
            patch("binanceexp1.trade_binance_selector.enforce_gap", side_effect=lambda s, b, sp, **kw: (b, sp)),
            patch("binanceexp1.trade_binance_selector.binance_wrapper") as mock_bw,
            patch("binanceexp1.trade_binance_selector.compute_order_quantities", return_value=mock_sizing),
        ):
            mock_bw.get_symbol_price.return_value = 88.0
            _handle_exit(
                state, mock_actions, SYMBOLS,
                intensity_scale=5.0, price_offset_map={}, default_offset=0.0,
                min_gap_pct=0.0003, max_hold_hours=6,
                poll_seconds=30, expiry_minutes=90,
                price_tolerance=0.0008, dry_run=False,
                state_path=tmp_state,
            )
            assert mock_spawn.call_count == 1

    def test_clears_state_when_no_balance(self, tmp_state, mock_actions):
        state = SelectorState(open_symbol="SOLUSD", open_ts=datetime.now(timezone.utc).isoformat(), open_price=87.0)
        mock_plan = MagicMock()
        mock_plan.sell_price = 89.0
        mock_plan.sell_amount = 90.0
        mock_plan.buy_price = 86.0
        mock_plan.buy_amount = 90.0

        with (
            patch("binanceexp1.trade_binance_selector.resolve_symbol_rules", return_value=MOCK_RULES),
            patch("binanceexp1.trade_binance_selector.get_total_balances", return_value=(5000.0, 0.0)),
            patch("binanceexp1.trade_binance_selector.spawn_watcher"),
            patch("binanceexp1.trade_binance_selector._ensure_valid_levels", side_effect=lambda s, b, sp, **kw: (b, sp)),
            patch("binanceexp1.trade_binance_selector.enforce_min_spread", side_effect=lambda b, s, **kw: (b, s)),
            patch("binanceexp1.trade_binance_selector.enforce_gap", side_effect=lambda s, b, sp, **kw: (b, sp)),
            patch("binanceexp1.trade_binance_selector._build_plan", return_value=mock_plan),
            patch("binanceexp1.trade_binance_selector.compute_order_quantities"),
        ):
            _handle_exit(
                state, mock_actions, SYMBOLS,
                intensity_scale=5.0, price_offset_map={}, default_offset=0.0,
                min_gap_pct=0.0003, max_hold_hours=6,
                poll_seconds=30, expiry_minutes=90,
                price_tolerance=0.0008, dry_run=False,
                state_path=tmp_state,
            )
            assert state.open_symbol is None


class TestWorkSteal:
    def test_work_steal_triggers_on_profitable_position(self, tmp_state):
        state = SelectorState(open_symbol="SOLUSD", open_ts=datetime.now(timezone.utc).isoformat(), open_price=85.0)
        actions = {
            "SOLUSD": {"symbol": "SOLUSD", "buy_price": 86.0, "sell_price": 89.0, "buy_amount": 90, "sell_amount": 90,
                       "predicted_high_p50_h1": 90.0, "predicted_low_p50_h1": 84.0},
            "BTCUSD": {"symbol": "BTCUSD", "buy_price": 69000.0, "sell_price": 71000.0, "buy_amount": 80, "sell_amount": 80,
                       "predicted_high_p50_h1": 72000.0, "predicted_low_p50_h1": 68000.0},
        }
        mock_sizing = MagicMock()
        mock_sizing.sell_qty = 56.87

        with (
            patch("binanceexp1.trade_binance_selector.resolve_symbol_rules", return_value=MOCK_RULES),
            patch("binanceexp1.trade_binance_selector.get_free_balances", return_value=(0.0, 56.87)),
            patch("binanceexp1.trade_binance_selector.spawn_watcher") as mock_spawn,
            patch("binanceexp1.trade_binance_selector._ensure_valid_levels", side_effect=lambda s, b, sp, **kw: (b, sp)),
            patch("binanceexp1.trade_binance_selector.get_fee_for_symbol", return_value=0.0),
            patch("binanceexp1.trade_binance_selector.binance_wrapper") as mock_bw,
            patch("binanceexp1.trade_binance_selector.compute_order_quantities", return_value=mock_sizing),
        ):
            mock_bw.get_symbol_price.return_value = 86.0
            result = _handle_work_steal(
                state, actions, SYMBOLS,
                horizon=1, risk_weight=0.0,
                work_steal_min_profit_pct=0.0,
                work_steal_min_edge=0.0,
                work_steal_edge_margin=0.0,
                min_gap_pct=0.0003,
                poll_seconds=30, expiry_minutes=90,
                price_tolerance=0.0008, dry_run=False,
                state_path=tmp_state,
            )
            assert result is True
            assert state.open_symbol is None
            assert mock_spawn.call_count == 1

    def test_work_steal_skips_when_not_profitable(self, tmp_state):
        state = SelectorState(open_symbol="SOLUSD", open_ts=datetime.now(timezone.utc).isoformat(), open_price=90.0)
        actions = {
            "SOLUSD": {"symbol": "SOLUSD", "buy_price": 86.0, "sell_price": 89.0, "buy_amount": 90, "sell_amount": 90,
                       "predicted_high_p50_h1": 90.0, "predicted_low_p50_h1": 84.0},
            "BTCUSD": {"symbol": "BTCUSD", "buy_price": 69000.0, "sell_price": 71000.0, "buy_amount": 80, "sell_amount": 80,
                       "predicted_high_p50_h1": 72000.0, "predicted_low_p50_h1": 68000.0},
        }

        with patch("binanceexp1.trade_binance_selector.binance_wrapper") as mock_bw:
            mock_bw.get_symbol_price.return_value = 87.0
            result = _handle_work_steal(
                state, actions, SYMBOLS,
                horizon=1, risk_weight=0.0,
                work_steal_min_profit_pct=0.01,
                work_steal_min_edge=0.0,
                work_steal_edge_margin=0.0,
                min_gap_pct=0.0003,
                poll_seconds=30, expiry_minutes=90,
                price_tolerance=0.0008, dry_run=False,
                state_path=tmp_state,
            )
            assert result is False
            assert state.open_symbol == "SOLUSD"

    def test_work_steal_skips_zero_open_price(self, tmp_state):
        state = SelectorState(open_symbol="SOLUSD", open_ts=datetime.now(timezone.utc).isoformat(), open_price=0.0)
        result = _handle_work_steal(
            state, {}, SYMBOLS,
            horizon=1, risk_weight=0.0,
            work_steal_min_profit_pct=0.0,
            work_steal_min_edge=0.0,
            work_steal_edge_margin=0.0,
            min_gap_pct=0.0003,
            poll_seconds=30, expiry_minutes=90,
            price_tolerance=0.0008, dry_run=False,
            state_path=tmp_state,
        )
        assert result is False


class TestQuantizationPrecision:
    def test_quantize_total_balance_precision(self):
        from binanceneural.execution import quantize_down
        val = 56.870000000000005
        result = quantize_down(val, 0.001)
        assert result == 56.87
        assert len(str(result).split(".")[-1]) <= 3

    def test_quantize_up_precision(self):
        from binanceneural.execution import quantize_up
        val = 87.08
        result = quantize_up(val, 0.01)
        assert result == 87.08
        val2 = 87.071
        result2 = quantize_up(val2, 0.01)
        assert result2 == 87.08


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
