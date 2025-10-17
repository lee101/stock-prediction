from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from loguru import logger
import pandas as pd

from stockagent.constants import DEFAULT_SYMBOLS
from stockagent.agentsimulator import (
    AgentSimulator,
    AccountSnapshot,
    BaseRiskStrategy,
    MarketDataBundle,
    ProbeTradeStrategy,
    ProfitShutdownStrategy,
    SimulationResult,
    TradingPlan,
    fetch_latest_ohlc,
)

from .agentsimulator import CombinedPlanBuilder, SimulationConfig, build_daily_plans
from .forecaster import CombinedForecastGenerator


def build_trading_plans(
    *,
    generator: CombinedForecastGenerator,
    market_data: MarketDataBundle,
    config: SimulationConfig,
) -> list[TradingPlan]:
    builder = CombinedPlanBuilder(generator=generator, config=config)
    if config.symbols is not None:
        market_frames: Mapping[str, pd.DataFrame] = {
            symbol: market_data.bars.get(symbol, pd.DataFrame()) for symbol in config.symbols
        }
    else:
        market_frames = market_data.bars

    trading_days = list(market_data.trading_days())
    if not trading_days:
        return []
    if config.simulation_days > 0:
        trading_days = trading_days[-config.simulation_days :]

    return build_daily_plans(
        builder=builder,
        market_frames=market_frames,
        trading_days=trading_days,
    )


def run_simulation(
    *,
    builder: CombinedPlanBuilder,
    market_frames: Mapping[str, pd.DataFrame],
    trading_days: Sequence[pd.Timestamp],
    starting_cash: float,
    strategies: Sequence[BaseRiskStrategy] | None = None,
) -> SimulationResult | None:
    plans = build_daily_plans(
        builder=builder,
        market_frames=market_frames,
        trading_days=trading_days,
    )
    if not plans:
        logger.warning("No plans generated; aborting simulation.")
        return None

    snapshot = AccountSnapshot(
        equity=starting_cash,
        cash=starting_cash,
        buying_power=None,
        timestamp=datetime.now(timezone.utc),
        positions=[],
    )

    bundle = MarketDataBundle(
        bars={symbol: frame.copy() for symbol, frame in market_frames.items()},
        lookback_days=0,
        as_of=datetime.now(timezone.utc),
    )

    simulator = AgentSimulator(
        market_data=bundle,
        starting_cash=starting_cash,
        account_snapshot=snapshot,
    )
    strategy_list = list(strategies) if strategies is not None else []
    result = simulator.simulate(plans, strategies=strategy_list)
    logger.info(
        "Simulation complete: equity=%s realized=%s unrealized=%s",
        result.ending_equity,
        result.realized_pnl,
        result.unrealized_pnl,
    )
    return result


def main(args: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Run stockagentcombined simulation.")
    parser.add_argument("--symbols", nargs="+", default=DEFAULT_SYMBOLS, help="Symbols to simulate.")
    parser.add_argument("--lookback-days", type=int, default=120)
    parser.add_argument("--simulation-days", type=int, default=5)
    parser.add_argument("--starting-cash", type=float, default=1_000_000.0)
    parser.add_argument("--local-data-dir", type=Path, default=Path("trainingdata"))
    parser.add_argument("--allow-remote-data", action="store_true")
    parsed = parser.parse_args(args)

    config = SimulationConfig(
        symbols=parsed.symbols,
        lookback_days=parsed.lookback_days,
    )

    bundle = fetch_latest_ohlc(
        symbols=config.symbols,
        lookback_days=config.lookback_days,
        as_of=datetime.now(timezone.utc),
        local_data_dir=parsed.local_data_dir,
        allow_remote_download=parsed.allow_remote_data,
    )
    market_frames = bundle.bars
    trading_days = list(bundle.trading_days())[-parsed.simulation_days :]

    generator = CombinedForecastGenerator()
    builder = CombinedPlanBuilder(generator=generator, config=config)
    strategies: list[BaseRiskStrategy] = [
        ProbeTradeStrategy(),
        ProfitShutdownStrategy(),
    ]

    run_simulation(
        builder=builder,
        market_frames=market_frames,
        trading_days=trading_days,
        starting_cash=parsed.starting_cash,
        strategies=strategies,
    )


if __name__ == "__main__":  # pragma: no cover
    main()
