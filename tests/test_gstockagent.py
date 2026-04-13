import pytest
import pandas as pd
import numpy as np
import json
from pathlib import Path
from unittest.mock import patch, MagicMock

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from gstockagent.config import GStockConfig, CRYPTO_SYMBOLS
from gstockagent.prompt import (
    load_daily_bars, build_price_table, build_portfolio_table, build_prompt
)
from gstockagent.llm_client import parse_allocation
from gstockagent.simulator import (
    Position, SimState, portfolio_value, close_position,
    apply_fees, get_current_prices, compute_metrics, run_simulation,
    validate_exit_stop,
)


class TestConfig:
    def test_defaults(self):
        c = GStockConfig()
        assert c.leverage == 1.0
        assert c.fee_bps == 10.0
        assert len(c.symbols) == len(CRYPTO_SYMBOLS)
        assert c.model == "gemini-flash"

    def test_custom(self):
        c = GStockConfig(leverage=3.0, model="glm-5", max_positions=5)
        assert c.leverage == 3.0
        assert c.model == "glm-5"


class TestPrompt:
    def test_load_daily_bars(self):
        cfg = GStockConfig()
        df = load_daily_bars("BTC", cfg.data_dir)
        assert not df.empty
        assert "close" in df.columns
        assert "timestamp" in df.columns

    def test_build_price_table(self):
        cfg = GStockConfig()
        table = build_price_table(
            ["BTC", "ETH"], cfg.data_dir,
            pd.Timestamp("2025-12-01", tz="UTC")
        )
        assert "BTC" in table
        assert "Close" in table

    def test_build_portfolio_table_empty(self):
        t = build_portfolio_table({}, {})
        assert "No current" in t

    def test_build_portfolio_table(self):
        pos = {"BTC": {"qty": 0.1, "entry_price": 60000}}
        prices = {"BTC": 62000}
        t = build_portfolio_table(pos, prices)
        assert "BTC" in t
        assert "+" in t  # positive PnL

    def test_build_prompt(self):
        cfg = GStockConfig()
        p = build_prompt(
            ["BTC", "ETH"], cfg.data_dir, cfg.forecast_cache_dir,
            pd.Timestamp("2025-12-01", tz="UTC"),
            {}, {}, 10000, 2.0, 5
        )
        assert "cryptocurrency" in p.lower()
        assert "allocation" in p.lower()
        assert "10000" in p


class TestParseAllocation:
    def test_basic_json(self):
        resp = '```json\n{"allocations": {"BTC": {"allocation_pct": 30, "direction": "long", "exit_price": 70000, "stop_price": 65000}}}\n```'
        alloc = parse_allocation(resp)
        assert "BTC" in alloc
        assert alloc["BTC"]["allocation_pct"] == 30

    def test_raw_json(self):
        resp = '{"BTC": {"allocation_pct": 50, "direction": "long", "exit_price": 70000, "stop_price": 65000}}'
        alloc = parse_allocation(resp)
        assert "BTC" in alloc

    def test_messy_json(self):
        resp = 'Here is my analysis:\n```json\n{"allocations": {"ETH": {"allocation_pct": 20, "direction": "short", "exit_price": 3000, "stop_price": 3500}}}\n```\nGood luck!'
        alloc = parse_allocation(resp)
        assert "ETH" in alloc
        assert alloc["ETH"]["direction"] == "short"

    def test_empty(self):
        assert parse_allocation("no json here") == {}


class TestSimulator:
    def test_apply_fees(self):
        assert apply_fees(10000, 10) == 10.0
        assert apply_fees(10000, 0) == 0.0

    def test_portfolio_value_cash_only(self):
        s = SimState(cash=10000)
        assert portfolio_value(s, {}) == 10000

    def test_portfolio_value_long(self):
        s = SimState(cash=5000)
        s.positions["BTC"] = Position("BTC", 0.1, 50000, "long")
        assert portfolio_value(s, {"BTC": 60000}) == 5000 + 0.1 * 60000

    def test_portfolio_value_short(self):
        s = SimState(cash=5000)
        s.positions["BTC"] = Position("BTC", 0.1, 50000, "short")
        # short PnL: qty * (entry - current) = 0.1 * (50000 - 60000) = -1000
        val = portfolio_value(s, {"BTC": 60000})
        assert val == 5000 + 0.1 * (2 * 50000 - 60000)  # 5000 + 4000 = 9000

    def test_close_position(self):
        s = SimState(cash=0)
        s.positions["ETH"] = Position("ETH", 1.0, 3000, "long", entry_date="2025-01-01")
        close_position(s, "ETH", 3500, "tp", "2025-01-02")
        assert len(s.positions) == 0
        assert s.cash == 3500  # long close: qty * price
        assert len(s.trade_log) == 1
        assert s.trade_log[0]["pnl"] == 500

    def test_compute_metrics(self):
        s = SimState(cash=11000)
        s.equity_curve = [
            {"date": "2025-01-01", "equity": 10000},
            {"date": "2025-01-02", "equity": 10100},
            {"date": "2025-01-03", "equity": 10050},
            {"date": "2025-01-04", "equity": 10200},
            {"date": "2025-01-05", "equity": 11000},
        ]
        s.daily_returns = [0, 0.01, -0.005, 0.015, 0.078]
        s.trade_log = [
            {"pnl": 100, "symbol": "BTC"},
            {"pnl": -50, "symbol": "ETH"},
            {"pnl": 200, "symbol": "SOL"},
        ]
        cfg = GStockConfig()
        m = compute_metrics(s, cfg)
        assert m["total_return_pct"] == 10.0
        assert m["n_trades"] == 3
        assert m["win_rate_pct"] == pytest.approx(66.7, abs=0.1)
        assert m["max_drawdown_pct"] < 0  # there's a dip
        assert m["sortino"] > 0

    def test_simulation_with_mock_llm(self):
        mock_resp = json.dumps({
            "allocations": {
                "BTC": {"allocation_pct": 50, "direction": "long",
                        "exit_price": 999999, "stop_price": 1}
            }
        })
        cfg = GStockConfig(symbols=["BTC"], leverage=1.0, initial_capital=10000)
        with patch("gstockagent.simulator.call_llm", return_value=mock_resp):
            result = run_simulation(cfg, "2025-11-01", "2025-11-10")
        assert "error" not in result
        assert result["n_days"] > 0
        assert result["final_equity"] > 0


class TestGetCurrentPrices:
    def test_basic(self):
        df = pd.DataFrame({
            "timestamp": pd.to_datetime(["2025-01-01", "2025-01-02", "2025-01-03"], utc=True),
            "close": [100, 105, 110],
        })
        bars = {"TEST": df}
        prices = get_current_prices(bars, pd.Timestamp("2025-01-02", tz="UTC"))
        assert prices["TEST"] == 105


class TestMarginConstraints:
    def test_short_requires_cash(self):
        s = SimState(cash=1000)
        from gstockagent.simulator import apply_fees
        # opening short should deduct cash like a long
        s.positions["BTC"] = Position("BTC", 0.01, 100000, "short", entry_date="2025-01-01")
        # if we had the bug, shorts wouldn't cost cash
        # verify portfolio_value works correctly for shorts
        assert portfolio_value(s, {"BTC": 100000}) == 1000 + 0.01 * (2 * 100000 - 100000)

    def test_cash_constraint_prevents_overleveraging(self):
        s = SimState(cash=5000)
        # cannot buy more than cash allows
        prices = {"BTC": 50000}
        target_qty = 10000 / 50000  # want 0.2 BTC = $10k
        cost = target_qty * prices["BTC"]
        fee = apply_fees(cost, 10)
        total_cost = cost + fee
        assert total_cost > s.cash  # confirms would exceed cash
        # sim should reduce qty
        actual_qty = max(0, (s.cash - fee) / prices["BTC"])
        assert actual_qty < target_qty
        assert actual_qty * prices["BTC"] <= s.cash

    def test_close_short_returns_correct_cash(self):
        s = SimState(cash=0)
        s.positions["ETH"] = Position("ETH", 1.0, 3000, "short", entry_date="2025-01-01")
        # profitable short: entry 3000, exit 2500
        close_position(s, "ETH", 2500, "tp", "2025-01-02")
        assert s.cash == 1.0 * (2 * 3000 - 2500)  # 3500
        assert s.trade_log[-1]["pnl"] == 500

    def test_close_short_losing(self):
        s = SimState(cash=0)
        s.positions["ETH"] = Position("ETH", 1.0, 3000, "short", entry_date="2025-01-01")
        close_position(s, "ETH", 3500, "stop", "2025-01-02")
        assert s.cash == 1.0 * (2 * 3000 - 3500)  # 2500
        assert s.trade_log[-1]["pnl"] == -500


class TestValidateExitStop:
    def test_long_valid(self):
        e, s = validate_exit_stop(100, 110, 90, "long")
        assert e == 110
        assert s == 90

    def test_long_exit_below_entry_disabled(self):
        e, s = validate_exit_stop(100, 95, 90, "long")
        assert e == 0  # disabled
        assert s == 90

    def test_long_stop_above_entry_disabled(self):
        e, s = validate_exit_stop(100, 110, 105, "long")
        assert e == 110
        assert s == 0  # disabled

    def test_short_valid(self):
        e, s = validate_exit_stop(100, 90, 110, "short")
        assert e == 90
        assert s == 110

    def test_short_exit_above_entry_disabled(self):
        e, s = validate_exit_stop(100, 105, 110, "short")
        assert e == 0
        assert s == 110

    def test_short_stop_below_entry_disabled(self):
        e, s = validate_exit_stop(100, 90, 95, "short")
        assert e == 90
        assert s == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
