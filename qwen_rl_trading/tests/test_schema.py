"""Tests for qwen_rl_trading.schema."""
import json

import numpy as np
import pandas as pd
import pytest

from qwen_rl_trading.schema import (
    MarketRegime,
    SymbolPlan,
    TradeAction,
    TradingPlan,
    plan_to_sim_actions,
    validate_plan,
)


def _sample_plan_dict(**overrides):
    base = {
        "plans": [
            {
                "symbol": "BTCUSD",
                "action": "LONG",
                "allocation_pct": 40.0,
                "entry_price": 67500.0,
                "stop_loss": 66800.0,
                "take_profit": 68850.0,
                "max_hold_hours": 6,
                "confidence": 0.8,
                "reasoning": "strong momentum",
            }
        ],
        "cash_reserve_pct": 20.0,
        "market_regime": "trending_up",
    }
    base.update(overrides)
    return base


def _sample_forward_bars(symbols=("BTCUSD",), n_hours=6):
    rows = []
    base_ts = pd.Timestamp("2025-01-01", tz="UTC")
    for sym in symbols:
        for h in range(n_hours):
            rows.append({
                "timestamp": base_ts + pd.Timedelta(hours=h + 1),
                "symbol": sym,
                "open": 67500 + h * 10,
                "high": 67600 + h * 10,
                "low": 67400 + h * 10,
                "close": 67550 + h * 10,
                "volume": 1000.0,
            })
    return pd.DataFrame(rows)


class TestValidatePlan:
    def test_valid_json(self):
        plan = validate_plan(json.dumps(_sample_plan_dict()))
        assert plan is not None
        assert len(plan.plans) == 1
        assert plan.plans[0].action == TradeAction.LONG
        assert plan.cash_reserve_pct == 20.0

    def test_with_code_block(self):
        text = "```json\n" + json.dumps(_sample_plan_dict()) + "\n```"
        plan = validate_plan(text)
        assert plan is not None

    def test_with_surrounding_text(self):
        text = "Here is my plan:\n" + json.dumps(_sample_plan_dict()) + "\nEnd of plan."
        plan = validate_plan(text)
        assert plan is not None

    def test_malformed_returns_none(self):
        assert validate_plan("not json at all") is None
        assert validate_plan("{invalid json}") is None
        assert validate_plan("") is None

    def test_missing_required_fields(self):
        # plans with bad entry_price
        bad = _sample_plan_dict()
        bad["plans"][0]["entry_price"] = -1
        assert validate_plan(json.dumps(bad)) is None

    def test_empty_plans_list(self):
        plan = validate_plan(json.dumps({"plans": [], "cash_reserve_pct": 100, "market_regime": "ranging"}))
        assert plan is not None
        assert len(plan.plans) == 0


class TestSymbolPlan:
    def test_allocation_bounds(self):
        with pytest.raises(Exception):
            SymbolPlan(symbol="BTC", action=TradeAction.LONG, allocation_pct=150,
                       entry_price=100, stop_loss=90, take_profit=110)

    def test_confidence_bounds(self):
        with pytest.raises(Exception):
            SymbolPlan(symbol="BTC", action=TradeAction.LONG, allocation_pct=50,
                       entry_price=100, stop_loss=90, take_profit=110, confidence=1.5)


class TestMarketRegime:
    def test_all_values(self):
        for v in ["trending_up", "trending_down", "ranging", "high_volatility"]:
            assert MarketRegime(v).value == v


class TestPlanToSimActions:
    def test_long_produces_buy_actions(self):
        plan = TradingPlan.model_validate(_sample_plan_dict())
        bars = _sample_forward_bars()
        actions = plan_to_sim_actions(plan, bars)
        assert not actions.empty
        assert (actions["buy_price"] > 0).any()
        assert (actions["buy_amount"] > 0).any()

    def test_short_produces_sell_actions(self):
        d = _sample_plan_dict()
        d["plans"][0]["action"] = "SHORT"
        plan = TradingPlan.model_validate(d)
        bars = _sample_forward_bars()
        actions = plan_to_sim_actions(plan, bars)
        assert not actions.empty
        assert (actions["sell_price"] > 0).any()

    def test_flat_produces_empty(self):
        d = _sample_plan_dict()
        d["plans"][0]["action"] = "FLAT"
        plan = TradingPlan.model_validate(d)
        bars = _sample_forward_bars()
        actions = plan_to_sim_actions(plan, bars)
        assert actions.empty

    def test_multi_symbol(self):
        d = _sample_plan_dict()
        d["plans"].append({
            "symbol": "ETHUSD", "action": "SHORT", "allocation_pct": 30,
            "entry_price": 3500, "stop_loss": 3600, "take_profit": 3350,
            "max_hold_hours": 4, "confidence": 0.7, "reasoning": "weak",
        })
        plan = TradingPlan.model_validate(d)
        bars = _sample_forward_bars(symbols=("BTCUSD", "ETHUSD"))
        actions = plan_to_sim_actions(plan, bars)
        assert not actions.empty
        assert set(actions["symbol"].unique()) == {"BTCUSD", "ETHUSD"}

    def test_max_hold_limits_rows(self):
        d = _sample_plan_dict()
        d["plans"][0]["max_hold_hours"] = 3
        plan = TradingPlan.model_validate(d)
        bars = _sample_forward_bars(n_hours=10)
        actions = plan_to_sim_actions(plan, bars)
        assert len(actions) == 3

    def test_empty_plan(self):
        plan = TradingPlan(plans=[], cash_reserve_pct=100)
        bars = _sample_forward_bars()
        actions = plan_to_sim_actions(plan, bars)
        assert actions.empty
