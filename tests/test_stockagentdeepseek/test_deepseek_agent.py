import json
from datetime import datetime, timezone, date

import pandas as pd
import pytest

from stockagent.agentsimulator.data_models import AccountPosition, AccountSnapshot
from stockagent.agentsimulator.market_data import MarketDataBundle
from stockagent.agentsimulator import prompt_builder as stateful_prompt_builder
from stockagentdeepseek.agent import simulate_deepseek_plan


@pytest.fixture(autouse=True)
def _patch_state_loader(monkeypatch):
    monkeypatch.setattr(stateful_prompt_builder, "load_all_state", lambda *_, **__: {})
    yield


def _sample_market_bundle() -> MarketDataBundle:
    index = pd.date_range("2025-01-01", periods=3, freq="D", tz="UTC")
    frame = pd.DataFrame(
        {
            "open": [110.0, 112.0, 111.0],
            "close": [112.0, 113.5, 114.0],
            "high": [112.0, 114.0, 115.0],
            "low": [109.0, 110.5, 110.0],
        },
        index=index,
    )
    return MarketDataBundle(
        bars={"AAPL": frame},
        lookback_days=3,
        as_of=index[-1].to_pydatetime(),
    )


def test_simulate_deepseek_plan_produces_expected_pnl(monkeypatch):
    plan_payload = {
        "target_date": "2025-01-02",
        "instructions": [
            {
                "symbol": "AAPL",
                "action": "buy",
                "quantity": 5,
                "execution_session": "market_open",
                "entry_price": 110.0,
                "exit_price": 114.0,
                "exit_reason": "initial position",
                "notes": "increase exposure",
            },
            {
                "symbol": "AAPL",
                "action": "sell",
                "quantity": 5,
                "execution_session": "market_close",
                "entry_price": 110.0,
                "exit_price": 114.0,
                "exit_reason": "close for profit",
                "notes": "close position",
            },
        ],
        "risk_notes": "Focus on momentum while keeping exposure bounded.",
        "focus_symbols": ["AAPL"],
        "stop_trading_symbols": [],
        "execution_window": "market_open",
        "metadata": {"capital_allocation_plan": "Allocate 100% to AAPL for the session."},
    }
    plan_json = json.dumps(plan_payload)

    monkeypatch.setattr(
        "stockagentdeepseek.agent.call_deepseek_chat",
        lambda *_, **__: plan_json,
    )

    snapshot = AccountSnapshot(
        equity=10_000.0,
        cash=8_000.0,
        buying_power=12_000.0,
        timestamp=datetime(2025, 1, 1, tzinfo=timezone.utc),
        positions=[
            AccountPosition(
                symbol="AAPL",
                quantity=0.0,
                side="flat",
                market_value=0.0,
                avg_entry_price=0.0,
                unrealized_pl=0.0,
                unrealized_plpc=0.0,
            )
        ],
    )

    result = simulate_deepseek_plan(
        market_data=_sample_market_bundle(),
        account_snapshot=snapshot,
        target_date=date(2025, 1, 2),
    )

    assert result.plan.instructions[0].action.value == "buy"
    assert result.plan.instructions[1].action.value == "sell"

    simulation = result.simulation
    assert simulation.realized_pnl == pytest.approx(19.715, rel=1e-4)
    assert simulation.total_fees == pytest.approx(0.56, rel=1e-4)
    assert simulation.ending_cash == pytest.approx(snapshot.cash + 19.44, rel=1e-4)
