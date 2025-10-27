from __future__ import annotations

import pandas as pd
import numpy as np

from stockagent.agentsimulator.market_data import MarketDataBundle
from stockagent.agentsimulator import ExecutionSession, PlanActionType

from stockagentcombined.forecaster import CombinedForecast, ErrorBreakdown, ModelForecast
from stockagentcombined.simulation import SimulationConfig, build_trading_plans


class StubGenerator:
    def __init__(self, price_mae: float = 1.0, return_scale: float = 0.02):
        self.price_mae = price_mae
        self.return_scale = return_scale

    def generate_for_symbol(self, symbol: str, *, prediction_length: int, historical_frame: pd.DataFrame):
        last_row = historical_frame.iloc[-1]
        last_open = float(last_row["open"])
        last_close = float(last_row["close"])
        scale = 1.0 + self.return_scale
        combined_prices = {
            "open": last_open * scale,
            "high": last_close * (1.0 + self.return_scale * 1.5),
            "low": last_close * (1.0 - self.return_scale * 0.5),
            "close": last_close * scale,
        }
        breakdown = ErrorBreakdown(price_mae=self.price_mae, pct_return_mae=0.01, latency_s=1.0)
        model_forecast = ModelForecast(
            symbol=symbol,
            model="toto",
            config_name="stub",
            config={},
            validation=breakdown,
            test=breakdown,
            average_price_mae=self.price_mae,
            average_pct_return_mae=0.01,
            forecasts=combined_prices,
        )
        return CombinedForecast(
            symbol=symbol,
            model_forecasts={"toto": model_forecast},
            combined=combined_prices,
            weights={"toto": 1.0},
            best_model="toto",
            selection_source="stub",
        )


def _make_market_bundle(symbol: str, periods: int = 8) -> MarketDataBundle:
    dates = pd.date_range("2024-01-01", periods=periods, freq="1D")
    frame = pd.DataFrame(
        {
            "timestamp": dates,
            "open": np.linspace(100, 100 + periods - 1, periods),
            "high": np.linspace(101, 101 + periods - 1, periods),
            "low": np.linspace(99, 99 + periods - 1, periods),
            "close": np.linspace(100, 100 + periods - 1, periods),
            "volume": np.linspace(1_000_000, 1_000_000 + 10_000 * periods, periods),
        }
    )
    bars = {symbol: frame.set_index("timestamp")}
    return MarketDataBundle(bars=bars, lookback_days=periods, as_of=dates[-1].to_pydatetime())


def test_build_trading_plans_generates_instructions():
    generator = StubGenerator(price_mae=1.0, return_scale=0.02)
    market_data = _make_market_bundle("AAPL", periods=6)
    config = SimulationConfig(
        symbols=["AAPL"],
        lookback_days=6,
        simulation_days=2,
        starting_cash=100_000.0,
        min_history=3,
        min_signal=0.001,
        error_multiplier=1.5,
        base_quantity=10.0,
        max_quantity_multiplier=3.0,
        min_quantity=1.0,
    )

    plans = build_trading_plans(
        generator=generator,
        market_data=market_data,
        config=config,
    )

    assert len(plans) == 2
    for plan in plans:
        assert plan.instructions, "Expected at least one instruction per plan"
        entry = plan.instructions[0]
        assert entry.action == PlanActionType.BUY
        assert entry.quantity >= config.min_quantity
        assert "pred_return" in (entry.notes or "")
        assert len(plan.instructions) >= 2
        exit_instruction = plan.instructions[1]
        assert exit_instruction.action == PlanActionType.EXIT
        assert exit_instruction.execution_session == ExecutionSession.MARKET_CLOSE
        assert plan.metadata.get("generated_by") == "stockagentcombined"
