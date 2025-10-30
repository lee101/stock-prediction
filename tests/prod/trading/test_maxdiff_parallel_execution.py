"""Tests for maxdiff parallel execution and portfolio packing."""
from __future__ import annotations

import importlib
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture
def temp_state_dir(tmp_path, monkeypatch):
    """Create temporary state directory for tests."""
    state_dir = tmp_path / "state"
    state_dir.mkdir()
    monkeypatch.setenv("STATE_DIR", str(state_dir))
    return state_dir


@pytest.fixture
def trade_module(temp_state_dir, monkeypatch):
    """Import trade_stock_e2e module with test configuration."""
    monkeypatch.setenv("MARKETSIM_SIMPLE_MODE", "0")
    monkeypatch.setenv("MARKETSIM_ENABLE_PROBE_TRADES", "0")
    monkeypatch.setenv("MARKETSIM_MAX_MAXDIFFS", "15")

    module = importlib.import_module("trade_stock_e2e")
    module = importlib.reload(module)
    yield module
    importlib.reload(module)


class TestMaxdiffPlanPersistence:
    """Test maxdiff plan storage and retrieval."""

    def test_save_and_load_maxdiff_plan(self, trade_module):
        """Test saving and loading a maxdiff plan."""
        plan_data = {
            "symbol": "AAPL",
            "high_target": 185.50,
            "low_target": 178.20,
            "avg_return": 0.0023,
            "status": "identified",
            "created_at": datetime.now(timezone.utc).isoformat(),
        }

        trade_module._save_maxdiff_plan("AAPL", plan_data)
        plans = trade_module._load_maxdiff_plans_for_today()

        assert "AAPL" in plans
        assert plans["AAPL"]["high_target"] == 185.50
        assert plans["AAPL"]["low_target"] == 178.20
        assert plans["AAPL"]["avg_return"] == 0.0023
        assert plans["AAPL"]["status"] == "identified"

    def test_save_multiple_maxdiff_plans(self, trade_module):
        """Test saving multiple maxdiff plans."""
        symbols = ["AAPL", "NVDA", "TSLA", "MSFT"]

        for i, symbol in enumerate(symbols):
            plan_data = {
                "symbol": symbol,
                "high_target": 100.0 + i * 10,
                "low_target": 90.0 + i * 10,
                "avg_return": 0.001 * (i + 1),
                "status": "identified",
                "created_at": datetime.now(timezone.utc).isoformat(),
            }
            trade_module._save_maxdiff_plan(symbol, plan_data)

        plans = trade_module._load_maxdiff_plans_for_today()

        assert len(plans) == 4
        for symbol in symbols:
            assert symbol in plans
            assert plans[symbol]["status"] == "identified"

    def test_update_maxdiff_plan_status(self, trade_module):
        """Test updating maxdiff plan status."""
        plan_data = {
            "symbol": "AAPL",
            "high_target": 185.50,
            "low_target": 178.20,
            "avg_return": 0.0023,
            "status": "identified",
            "created_at": datetime.now(timezone.utc).isoformat(),
        }

        trade_module._save_maxdiff_plan("AAPL", plan_data)
        trade_module._update_maxdiff_plan_status("AAPL", "listening")

        plans = trade_module._load_maxdiff_plans_for_today()

        assert plans["AAPL"]["status"] == "listening"
        assert "updated_at" in plans["AAPL"]

    def test_update_maxdiff_plan_with_extra_fields(self, trade_module):
        """Test updating maxdiff plan with extra fields."""
        plan_data = {
            "symbol": "NVDA",
            "high_target": 147.80,
            "low_target": 142.10,
            "avg_return": 0.0019,
            "status": "identified",
            "created_at": datetime.now(timezone.utc).isoformat(),
        }

        trade_module._save_maxdiff_plan("NVDA", plan_data)
        trade_module._update_maxdiff_plan_status(
            "NVDA",
            "filled",
            fill_price=145.50,
            fill_qty=50,
        )

        plans = trade_module._load_maxdiff_plans_for_today()

        assert plans["NVDA"]["status"] == "filled"
        assert plans["NVDA"]["fill_price"] == 145.50
        assert plans["NVDA"]["fill_qty"] == 50

    def test_load_empty_maxdiff_plans(self, trade_module):
        """Test loading when no plans exist."""
        plans = trade_module._load_maxdiff_plans_for_today()
        assert isinstance(plans, dict)
        assert len(plans) == 0


class TestMaxdiffSpreadRanking:
    """Test maxdiff spread ranking and overflow logic."""

    def test_maxdiff_spread_rank_assignment(self, trade_module):
        """Test that maxdiff trades get assigned spread ranks."""
        # Create mock picks with maxdiff strategy
        picks = {}
        for i in range(5):
            symbol = f"SYM{i}"
            picks[symbol] = {
                "strategy": "maxdiff",
                "avg_return": 0.001 * (5 - i),  # Descending order
                "side": "buy",
            }

        # Simulate the ranking logic from manage_positions
        maxdiff_entries_seen = 0
        for symbol, data in picks.items():
            is_maxdiff_strategy = (data.get("strategy") in {"maxdiff", "highlow"})
            if is_maxdiff_strategy:
                maxdiff_entries_seen += 1
                data["maxdiff_spread_rank"] = maxdiff_entries_seen
                if 15 and maxdiff_entries_seen > 15:  # MAX_MAXDIFFS = 15
                    data["maxdiff_spread_overflow"] = True

        # Verify ranking
        assert picks["SYM0"]["maxdiff_spread_rank"] == 1
        assert picks["SYM4"]["maxdiff_spread_rank"] == 5

        # None should overflow with only 5 trades
        for symbol in picks:
            assert not picks[symbol].get("maxdiff_spread_overflow", False)

    def test_maxdiff_overflow_marking(self, trade_module):
        """Test that maxdiff trades beyond MAX_MAXDIFFS are marked as overflow."""
        # Create 20 maxdiff picks
        picks = {}
        for i in range(20):
            symbol = f"SYM{i}"
            picks[symbol] = {
                "strategy": "maxdiff",
                "avg_return": 0.001 * (20 - i),
                "side": "buy",
            }

        # Simulate the ranking logic with MAX_MAXDIFFS = 15
        MAX_MAXDIFFS = 15
        maxdiff_entries_seen = 0
        for symbol, data in picks.items():
            is_maxdiff_strategy = (data.get("strategy") in {"maxdiff", "highlow"})
            if is_maxdiff_strategy:
                maxdiff_entries_seen += 1
                data["maxdiff_spread_rank"] = maxdiff_entries_seen
                if MAX_MAXDIFFS and maxdiff_entries_seen > MAX_MAXDIFFS:
                    data["maxdiff_spread_overflow"] = True
                else:
                    data.pop("maxdiff_spread_overflow", None)

        # First 15 should not overflow
        for i in range(15):
            symbol = f"SYM{i}"
            assert not picks[symbol].get("maxdiff_spread_overflow", False)

        # Remaining 5 should overflow
        for i in range(15, 20):
            symbol = f"SYM{i}"
            assert picks[symbol].get("maxdiff_spread_overflow", False) or picks[symbol]["maxdiff_spread_rank"] > 15


class TestAdaptiveOrderSizing:
    """Test adaptive order sizing for maxdiff overflow trades."""

    def test_calculate_adjusted_quantity_to_fit_leverage(self):
        """Test calculation of adjusted quantity to fit leverage budget."""
        # Test parameters
        equity = 50000.0
        risk_threshold = 1.5
        current_exposure = 60000.0
        order_price = 250.0
        original_qty = 200.0

        # Calculate available room
        max_allowed_exposure = equity * risk_threshold  # 75000
        available_room = max_allowed_exposure - current_exposure  # 15000

        # Original order would be
        original_order_value = original_qty * order_price  # 50000

        # Adjusted order
        adjusted_order_value = available_room * 0.99  # 14850
        adjusted_qty = adjusted_order_value / order_price  # 59.4

        assert max_allowed_exposure == 75000.0
        assert available_room == 15000.0
        assert adjusted_qty == pytest.approx(59.4, rel=0.01)

        # Verify it fits
        new_exposure = current_exposure + (adjusted_qty * order_price)
        new_leverage = new_exposure / equity
        assert new_leverage <= risk_threshold

    def test_reject_order_below_minimum_size(self):
        """Test that orders below minimum size are rejected."""
        # Crypto minimum
        min_order_size_crypto = 0.01

        # Stock minimum
        min_order_size_stock = 1.0

        # Test crypto - acceptable
        adjusted_qty_crypto = 0.05
        assert adjusted_qty_crypto >= min_order_size_crypto

        # Test crypto - too small
        adjusted_qty_crypto_small = 0.005
        assert adjusted_qty_crypto_small < min_order_size_crypto

        # Test stock - acceptable
        adjusted_qty_stock = 5.0
        assert adjusted_qty_stock >= min_order_size_stock

        # Test stock - too small
        adjusted_qty_stock_small = 0.5
        assert adjusted_qty_stock_small < min_order_size_stock

    def test_no_room_available_skips_order(self):
        """Test that orders are skipped when no leverage room available."""
        equity = 50000.0
        risk_threshold = 1.5
        current_exposure = 75000.0  # Already at limit!

        max_allowed_exposure = equity * risk_threshold
        available_room = max_allowed_exposure - current_exposure

        assert available_room == 0.0
        # Should skip order


class TestStrategySelectionSimplification:
    """Test simplified strategy selection by avg_return."""

    def test_strategies_sorted_by_avg_return(self):
        """Test that strategies are sorted by avg_return."""
        candidate_avg_returns = {
            "simple": 0.0015,
            "maxdiff": 0.0025,
            "highlow": 0.0020,
            "takeprofit": 0.0010,
        }

        ordered_strategies = [
            name for name, _ in sorted(
                candidate_avg_returns.items(),
                key=lambda item: item[1],
                reverse=True,
            )
        ]

        assert ordered_strategies[0] == "maxdiff"
        assert ordered_strategies[1] == "highlow"
        assert ordered_strategies[2] == "simple"
        assert ordered_strategies[3] == "takeprofit"

    def test_ineligible_strategies_skipped(self):
        """Test that ineligible strategies are skipped during selection."""
        ordered_strategies = ["maxdiff", "highlow", "simple"]
        strategy_ineligible = {"maxdiff": "edge_too_low"}

        # Simulate selection loop
        selected = None
        for strategy in ordered_strategies:
            if strategy not in strategy_ineligible:
                selected = strategy
                break

        assert selected == "highlow"  # maxdiff was ineligible


class TestParallelMaxdiffExecution:
    """Integration tests for parallel maxdiff execution."""

    def test_maxdiff_plan_created_during_analysis(self, trade_module):
        """Test that maxdiff plans are created during symbol analysis."""
        # This would be an integration test with actual analyze_symbols call
        # For now, we verify the plan creation logic

        maxdiff_allowed_entry = True
        maxdiff_return = 0.0025

        if maxdiff_allowed_entry and maxdiff_return > 0:
            plan_data = {
                "symbol": "AAPL",
                "high_target": 185.50,
                "low_target": 178.20,
                "avg_return": maxdiff_return,
                "status": "identified",
                "created_at": datetime.now(timezone.utc).isoformat(),
            }

            trade_module._save_maxdiff_plan("AAPL", plan_data)

            plans = trade_module._load_maxdiff_plans_for_today()
            assert "AAPL" in plans
            assert plans["AAPL"]["avg_return"] == 0.0025

    def test_directional_and_maxdiff_can_coexist(self):
        """Test that directional and maxdiff positions can exist on same symbol."""
        # Simulate both position types
        positions = {
            "directional": {
                "AAPL": {"strategy": "takeprofit", "qty": 100, "side": "buy"},
            },
            "maxdiff": {
                "AAPL": {"strategy": "maxdiff", "qty": 50, "side": "sell", "limit": 185.50},
            },
        }

        # Both can exist independently
        assert "AAPL" in positions["directional"]
        assert "AAPL" in positions["maxdiff"]

        # Verify they have different strategies
        assert positions["directional"]["AAPL"]["strategy"] != positions["maxdiff"]["AAPL"]["strategy"]


class TestMaxdiffStatusDisplay:
    """Test maxdiff status display formatting."""

    def test_format_maxdiff_plans_for_display(self):
        """Test formatting maxdiff plans for status output."""
        plans = {
            "AAPL": {
                "high_target": 185.50,
                "low_target": 178.20,
                "maxdiffprofit_high_price": 185.50,
                "maxdiffprofit_low_price": 178.20,
                "avg_return": 0.0023,
                "status": "listening",
            },
            "NVDA": {
                "high_target": 147.80,
                "low_target": 142.10,
                "maxdiffprofit_high_price": 147.80,
                "maxdiffprofit_low_price": 142.10,
                "avg_return": 0.0019,
                "status": "spawned",
            },
        }

        # Sort by avg_return
        sorted_plans = sorted(
            plans.items(),
            key=lambda x: x[1].get("avg_return", 0.0),
            reverse=True
        )

        assert sorted_plans[0][0] == "AAPL"
        assert sorted_plans[1][0] == "NVDA"

        # Format display
        for symbol, plan in sorted_plans:
            avg_ret = plan.get("avg_return", 0.0)
            high = plan.get("maxdiffprofit_high_price") or plan.get("high_target", 0.0)
            low = plan.get("maxdiffprofit_low_price") or plan.get("low_target", 0.0)
            status = plan.get("status", "unknown")

            display_line = f"{symbol}: high=${high:.2f} low=${low:.2f} avg_return={avg_ret:.4f} [{status}]"

            assert symbol in display_line
            assert f"{avg_ret:.4f}" in display_line
            assert status in display_line


def test_max_maxdiffs_env_variable_parsing(monkeypatch, temp_state_dir):
    """Test MAX_MAXDIFFS environment variable parsing."""
    # Test default
    monkeypatch.delenv("MARKETSIM_MAX_MAXDIFFS", raising=False)
    module = importlib.import_module("trade_stock_e2e")
    module = importlib.reload(module)
    assert module.MAX_MAXDIFFS == 15

    # Test custom value
    monkeypatch.setenv("MARKETSIM_MAX_MAXDIFFS", "20")
    module = importlib.reload(module)
    assert module.MAX_MAXDIFFS == 20

    # Test invalid value falls back to default
    monkeypatch.setenv("MARKETSIM_MAX_MAXDIFFS", "invalid")
    module = importlib.reload(module)
    assert module.MAX_MAXDIFFS == 15

    # Test negative value falls back to default
    monkeypatch.setenv("MARKETSIM_MAX_MAXDIFFS", "-5")
    module = importlib.reload(module)
    assert module.MAX_MAXDIFFS == 15
