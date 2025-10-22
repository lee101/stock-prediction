from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from loguru import logger

from stockagent.agentsimulator import (
    AccountPosition,
    AccountSnapshot,
    AgentSimulator,
    SimulationResult,
    TradingPlan,
    fetch_latest_ohlc,
)
from stockagent.constants import DEFAULT_SYMBOLS

from ..config import OptimizationConfig, PipelineConfig
from ..optimizer import CostAwareOptimizer
from ..pipeline import AllocationPipeline, AllocationResult
from stockagentcombined.forecaster import CombinedForecastGenerator
from .forecast_adapter import CombinedForecastAdapter
from .plan_builder import PipelinePlanBuilder, PipelineSimulationConfig


@dataclass
class RunnerConfig:
    symbols: Sequence[str] = tuple(DEFAULT_SYMBOLS)
    lookback_days: int = 252
    simulation_days: int = 10
    starting_cash: float = 1_000_000.0
    local_data_dir: Path | None = Path("trainingdata")
    allow_remote_data: bool = False


@dataclass(frozen=True)
class PipelineSimulationResult:
    simulator: AgentSimulator
    simulation: SimulationResult
    plans: Tuple[TradingPlan, ...]
    allocations: Tuple[AllocationResult, ...]


def _positions_from_weights(
    *,
    weights: Dict[str, float],
    prices: Dict[str, float],
    nav: float,
) -> Dict[str, float]:
    positions: Dict[str, float] = {}
    for symbol, weight in weights.items():
        price = prices.get(symbol)
        if price is None or not np.isfinite(price) or price <= 0:
            continue
        positions[symbol] = (weight * nav) / price
    return positions


def _snapshot_from_positions(
    *,
    positions: Dict[str, float],
    prices: Dict[str, float],
    nav: float,
) -> AccountSnapshot:
    account_positions: List[AccountPosition] = []
    equity = nav
    for symbol, qty in positions.items():
        price = prices.get(symbol, 0.0)
        market_value = qty * price
        side = "short" if qty < 0 else "long"
        account_positions.append(
            AccountPosition(
                symbol=symbol,
                quantity=float(abs(qty)),
                side=side,
                market_value=float(abs(market_value)),
                avg_entry_price=float(price),
                unrealized_pl=0.0,
                unrealized_plpc=0.0,
            )
        )
    return AccountSnapshot(
        equity=equity,
        cash=max(nav - sum(abs(qty) * prices.get(symbol, 0.0) for symbol, qty in positions.items()), 0.0),
        buying_power=None,
        timestamp=datetime.utcnow(),
        positions=account_positions,
    )


def run_pipeline_simulation(
    *,
    runner_config: RunnerConfig,
    optimisation_config: OptimizationConfig,
    pipeline_config: PipelineConfig,
) -> Optional[PipelineSimulationResult]:
    bundle = fetch_latest_ohlc(
        symbols=runner_config.symbols,
        lookback_days=runner_config.lookback_days,
        as_of=datetime.utcnow(),
        local_data_dir=runner_config.local_data_dir,
        allow_remote_download=runner_config.allow_remote_data,
    )
    trading_days = list(bundle.trading_days())[-runner_config.simulation_days :]
    if not trading_days:
        logger.warning("No trading days available for simulation")
        return None

    optimizer = CostAwareOptimizer(optimisation_config)
    pipeline = AllocationPipeline(
        optimisation_config=optimisation_config,
        pipeline_config=pipeline_config,
        optimizer=optimizer,
    )
    forecast_adapter = CombinedForecastAdapter(generator=CombinedForecastGenerator())
    builder = PipelinePlanBuilder(
        pipeline=pipeline,
        forecast_adapter=forecast_adapter,
        pipeline_config=PipelineSimulationConfig(symbols=runner_config.symbols),
        pipeline_params=pipeline_config,
    )

    plans: List[TradingPlan] = []
    allocations: List[AllocationResult] = []
    positions: Dict[str, float] = {}
    nav = runner_config.starting_cash
    for timestamp in trading_days:
        prices = {symbol: float(frame.loc[:timestamp].iloc[-1]["close"]) for symbol, frame in bundle.bars.items() if symbol in runner_config.symbols and not frame.empty}
        snapshot = _snapshot_from_positions(positions=positions, prices=prices, nav=nav)
        plan = builder.build_for_day(
            target_timestamp=timestamp,
            market_frames=bundle.bars,
            account_snapshot=snapshot,
        )
        if plan is None or builder.last_allocation is None:
            continue
        plans.append(plan)
        allocations.append(builder.last_allocation)
        positions = _positions_from_weights(
            weights={symbol: weight for symbol, weight in zip(builder.last_allocation.universe, builder.last_allocation.weights)},
            prices=prices,
            nav=nav,
        )

    if not plans:
        logger.warning("Pipeline simulation produced no plans")
        return None

    simulator = AgentSimulator(
        market_data=type("Bundle", (), {"get_symbol_bars": bundle.bars.get})(),
        starting_cash=runner_config.starting_cash,
        account_snapshot=_snapshot_from_positions(positions={}, prices={}, nav=runner_config.starting_cash),
    )
    simulation_result = simulator.simulate(plans)
    return PipelineSimulationResult(
        simulator=simulator,
        simulation=simulation_result,
        plans=tuple(plans),
        allocations=tuple(allocations),
    )
