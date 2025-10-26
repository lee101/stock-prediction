import json
from datetime import datetime, timezone, date

import pandas as pd
import pytest

from evaltests.baseline_pnl_extract import patched_deepseek_response, offline_alpaca_state
from stockagent.agentsimulator.data_models import AccountPosition, AccountSnapshot
from stockagent.agentsimulator.market_data import MarketDataBundle
from stockagentdeepseek_combinedmaxdiff.agent import simulate_deepseek_combined_maxdiff_plan
from stockagentdeepseek_neural.forecaster import NeuralForecast, ModelForecastSummary


@pytest.fixture()
def sample_bundle() -> MarketDataBundle:
    index = pd.date_range("2025-01-01", periods=3, freq="D", tz="UTC")
    frame = pd.DataFrame(
        {
            "open": [110.0, 112.0, 111.0],
            "high": [112.0, 114.0, 115.0],
            "low": [109.0, 110.0, 110.0],
            "close": [112.0, 113.5, 114.5],
        },
        index=index,
    )
    return MarketDataBundle(
        bars={"AAPL": frame, "BTCUSD": frame},
        lookback_days=3,
        as_of=index[-1].to_pydatetime(),
    )


@pytest.fixture()
def sample_snapshot() -> AccountSnapshot:
    return AccountSnapshot(
        equity=20_000.0,
        cash=15_000.0,
        buying_power=20_000.0,
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


def _build_forecasts(symbols):
    summary = ModelForecastSummary(
        model="test-model",
        config_name="baseline",
        average_price_mae=0.5,
        forecasts={"next_close": 114.0, "expected_return": 0.02},
    )
    return {
        symbol: NeuralForecast(
            symbol=symbol,
            combined={"next_close": 114.0, "expected_return": 0.02},
            best_model="test-model",
            selection_source="unit-test",
            model_summaries={"test-model": summary},
        )
        for symbol in symbols
    }


def test_combined_maxdiff_generates_metrics(sample_bundle, sample_snapshot):
    plan_payload = {
        "target_date": "2025-01-02",
        "instructions": [
            {
                "symbol": "AAPL",
                "action": "buy",
                "quantity": 10,
                "execution_session": "market_open",
                "entry_price": 112.0,
                "exit_price": 114.0,
                "exit_reason": "enter position",
                "notes": "plan trade",
            },
            {
                "symbol": "AAPL",
                "action": "exit",
                "quantity": 10,
                "execution_session": "market_close",
                "entry_price": None,
                "exit_price": 114.0,
                "exit_reason": "close position",
                "notes": "flatten",
            },
        ],
        "metadata": {"capital_allocation_plan": "All in AAPL"},
    }

    forecasts = _build_forecasts(["AAPL"])

    generator = _DummyGenerator()

    with patched_deepseek_response(plan_payload), offline_alpaca_state():
        result = simulate_deepseek_combined_maxdiff_plan(
            market_data=sample_bundle,
            account_snapshot=sample_snapshot,
            target_date=date(2025, 1, 2),
            symbols=["AAPL"],
            forecasts=forecasts,
            generator=generator,
            calibration_window=5,
        )

    assert result.plan.instructions[0].symbol == "AAPL"
    assert result.simulation.realized_pnl >= 0
    assert "net_pnl" in result.summary
    assert "annual_return_equity_pct" in result.summary
    assert "annual_return_crypto_pct" not in result.summary
    assert any(key.endswith("calibrated_expected_move_pct") for key in result.calibration)


def test_combined_maxdiff_crypto_annualisation(sample_bundle, sample_snapshot):
    plan_payload = {
        "target_date": "2025-01-02",
        "instructions": [
            {
                "symbol": "BTCUSD",
                "action": "buy",
                "quantity": 1.5,
                "execution_session": "market_open",
                "entry_price": 112.0,
                "exit_price": 114.0,
                "exit_reason": "enter position",
                "notes": "crypto plan",
            },
            {
                "symbol": "BTCUSD",
                "action": "exit",
                "quantity": 1.5,
                "execution_session": "market_close",
                "entry_price": None,
                "exit_price": 114.0,
                "exit_reason": "close position",
                "notes": "flatten",
            },
        ],
        "metadata": {"capital_allocation_plan": "Crypto focus"},
    }

    forecasts = _build_forecasts(["BTCUSD"])

    generator = _DummyGenerator()

    with patched_deepseek_response(plan_payload), offline_alpaca_state():
        result = simulate_deepseek_combined_maxdiff_plan(
            market_data=sample_bundle,
            account_snapshot=sample_snapshot,
            target_date=date(2025, 1, 2),
            symbols=["BTCUSD"],
            forecasts=forecasts,
            generator=generator,
            calibration_window=5,
        )

    assert result.plan.instructions[0].symbol == "BTCUSD"
    assert "annual_return_crypto_pct" in result.summary
    assert "annual_return_equity_pct" not in result.summary
    assert any(key.endswith("calibrated_expected_move_pct") for key in result.calibration)
class _DummyCombinedForecast:
    def __init__(self, close_price: float):
        self.combined = {"close": close_price}


class _DummyGenerator:
    def __init__(self, bump: float = 0.01):
        self.bump = bump

    def generate_for_symbol(self, symbol, *, prediction_length, historical_frame):
        last_close = float(historical_frame.iloc[-1]["close"])
        return _DummyCombinedForecast(last_close * (1.0 + self.bump))
