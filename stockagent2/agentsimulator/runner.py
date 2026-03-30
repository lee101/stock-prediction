from __future__ import annotations

from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from loguru import logger

from stockagent.agentsimulator import (
    AccountPosition,
    AccountSnapshot,
    AgentSimulator,
    SimulationResult,
    TradingPlan,
    default_local_data_dir,
    default_use_fallback_data_dirs,
    fetch_latest_ohlc,
)
from stockagent.constants import DEFAULT_SYMBOLS

from ..config import OptimizationConfig, PipelineConfig
from ..optimizer import CostAwareOptimizer
from ..pipeline import AllocationPipeline, AllocationResult
from stockagentcombined.forecaster import CombinedForecastGenerator
from .forecast_adapter import CombinedForecastAdapter
from .plan_builder import (
    PipelinePlanBuildDiagnostics,
    PipelinePlanBuilder,
    PipelineSimulationConfig,
)


@dataclass
class RunnerConfig:
    symbols: Sequence[str] = tuple(DEFAULT_SYMBOLS)
    lookback_days: int = 252
    simulation_days: int = 10
    starting_cash: float = 1_000_000.0
    local_data_dir: Path | None = field(default_factory=default_local_data_dir)
    allow_remote_data: bool = False
    use_fallback_data_dirs: bool = field(default_factory=default_use_fallback_data_dirs)


@dataclass(frozen=True)
class PipelineMarketDataSummary:
    symbols_requested: Tuple[str, ...]
    loaded_symbols: Tuple[str, ...]
    empty_symbols: Tuple[str, ...]
    bars_per_symbol: Dict[str, int]
    latest_bar_dates: Dict[str, str]
    trading_day_count: int
    first_trading_day: str | None
    last_trading_day: str | None


@dataclass(frozen=True)
class PipelineSimulationResult:
    simulator: AgentSimulator
    simulation: SimulationResult
    plans: Tuple[TradingPlan, ...]
    allocations: Tuple[AllocationResult, ...]
    market_data_summary: PipelineMarketDataSummary


@dataclass(frozen=True)
class PipelineSimulationAttempt:
    result: Optional[PipelineSimulationResult]
    market_data_summary: PipelineMarketDataSummary
    build_diagnostics: Tuple[PipelinePlanBuildDiagnostics, ...]
    failure_reason: str | None = None


@dataclass(frozen=True)
class _BarsMarketDataAdapter:
    bars: Mapping[str, pd.DataFrame]

    def get_symbol_bars(self, symbol: str) -> pd.DataFrame:
        return self.bars.get(symbol.upper(), pd.DataFrame()).copy()


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
        timestamp=datetime.now(timezone.utc),
        positions=account_positions,
    )


def _build_close_price_snapshots(
    *,
    bars: Dict[str, pd.DataFrame],
    symbols: Sequence[str],
    trading_days: Sequence[pd.Timestamp],
) -> Dict[pd.Timestamp, Dict[str, float]]:
    if len(trading_days) == 0:
        return {}

    target_index = pd.DatetimeIndex(trading_days)
    requested_symbols = {str(symbol).upper() for symbol in symbols}
    snapshots: Dict[pd.Timestamp, Dict[str, float]] = {pd.Timestamp(timestamp): {} for timestamp in target_index}

    for symbol, frame in bars.items():
        if symbol not in requested_symbols or frame.empty or "close" not in frame.columns:
            continue

        ordered = frame.sort_index()
        close_series = pd.to_numeric(ordered["close"], errors="coerce")
        if not close_series.index.is_unique:
            close_series = close_series.groupby(level=0).last()
        aligned = close_series.reindex(target_index, method="ffill").dropna()

        for timestamp, price in aligned.items():
            snapshots[pd.Timestamp(timestamp)][symbol] = float(price)

    return snapshots


def _summarise_market_data(
    *,
    bars: Dict[str, pd.DataFrame],
    symbols: Sequence[str],
    trading_days: Sequence[pd.Timestamp],
) -> PipelineMarketDataSummary:
    requested_symbols = tuple(str(symbol).upper() for symbol in symbols)
    loaded_symbols = tuple(
        symbol
        for symbol in requested_symbols
        if symbol in bars and not bars[symbol].empty
    )
    empty_symbols = tuple(symbol for symbol in requested_symbols if symbol not in loaded_symbols)
    bars_per_symbol = {
        symbol: int(len(bars.get(symbol, pd.DataFrame())))
        for symbol in requested_symbols
    }
    latest_bar_dates = {
        symbol: pd.Timestamp(bars[symbol].index[-1]).date().isoformat()
        for symbol in loaded_symbols
        if not bars[symbol].empty and len(bars[symbol].index) > 0
    }
    first_trading_day = pd.Timestamp(trading_days[0]).date().isoformat() if trading_days else None
    last_trading_day = pd.Timestamp(trading_days[-1]).date().isoformat() if trading_days else None
    return PipelineMarketDataSummary(
        symbols_requested=requested_symbols,
        loaded_symbols=loaded_symbols,
        empty_symbols=empty_symbols,
        bars_per_symbol=bars_per_symbol,
        latest_bar_dates=latest_bar_dates,
        trading_day_count=len(trading_days),
        first_trading_day=first_trading_day,
        last_trading_day=last_trading_day,
    )


def run_pipeline_simulation_with_diagnostics(
    *,
    runner_config: RunnerConfig,
    optimisation_config: OptimizationConfig,
    pipeline_config: PipelineConfig,
    simulation_config: PipelineSimulationConfig | None = None,
) -> PipelineSimulationAttempt:
    config = replace(simulation_config) if simulation_config is not None else PipelineSimulationConfig()
    symbols = config.symbols if config.symbols is not None else runner_config.symbols
    config.symbols = tuple(str(symbol).upper() for symbol in symbols)

    bundle = fetch_latest_ohlc(
        symbols=config.symbols,
        lookback_days=runner_config.lookback_days,
        as_of=datetime.now(timezone.utc),
        local_data_dir=runner_config.local_data_dir,
        allow_remote_download=runner_config.allow_remote_data,
        use_fallback_data_dirs=runner_config.use_fallback_data_dirs,
    )
    trading_days = list(bundle.trading_days())[-runner_config.simulation_days :]
    market_data_summary = _summarise_market_data(
        bars=bundle.bars,
        symbols=config.symbols,
        trading_days=trading_days,
    )
    logger.info(
        "Pipeline market data coverage: {} to {} across {}/{} loaded symbols; empty_symbols={}",
        market_data_summary.first_trading_day or "n/a",
        market_data_summary.last_trading_day or "n/a",
        len(market_data_summary.loaded_symbols),
        len(market_data_summary.symbols_requested),
        list(market_data_summary.empty_symbols),
    )
    if not trading_days:
        logger.warning(
            "No trading days available for simulation; latest_bar_dates={}",
            market_data_summary.latest_bar_dates,
        )
        return PipelineSimulationAttempt(
            result=None,
            market_data_summary=market_data_summary,
            build_diagnostics=(),
            failure_reason="No trading days available for simulation.",
        )

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
        pipeline_config=config,
        pipeline_params=pipeline_config,
    )
    close_price_snapshots = _build_close_price_snapshots(
        bars=bundle.bars,
        symbols=config.symbols,
        trading_days=trading_days,
    )

    plans: List[TradingPlan] = []
    allocations: List[AllocationResult] = []
    build_diagnostics: List[PipelinePlanBuildDiagnostics] = []
    positions: Dict[str, float] = {}
    nav = runner_config.starting_cash
    for timestamp in trading_days:
        prices = close_price_snapshots.get(pd.Timestamp(timestamp), {})
        snapshot = _snapshot_from_positions(positions=positions, prices=prices, nav=nav)
        plan = builder.build_for_day(
            target_timestamp=timestamp,
            market_frames=bundle.bars,
            account_snapshot=snapshot,
        )
        last_build_diagnostics = getattr(builder, "last_build_diagnostics", None)
        if last_build_diagnostics is not None:
            build_diagnostics.append(last_build_diagnostics)
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
        logger.warning(
            "Pipeline simulation produced no plans; statuses={}",
            [diagnostic.status for diagnostic in build_diagnostics],
        )
        return PipelineSimulationAttempt(
            result=None,
            market_data_summary=market_data_summary,
            build_diagnostics=tuple(build_diagnostics),
            failure_reason="Pipeline simulation produced no trading plans.",
        )

    simulator = AgentSimulator(
        market_data=_BarsMarketDataAdapter(bars=bundle.bars),
        starting_cash=runner_config.starting_cash,
        account_snapshot=_snapshot_from_positions(positions={}, prices={}, nav=runner_config.starting_cash),
    )
    simulation_result = simulator.simulate(plans)
    return PipelineSimulationAttempt(
        result=PipelineSimulationResult(
            simulator=simulator,
            simulation=simulation_result,
            plans=tuple(plans),
            allocations=tuple(allocations),
            market_data_summary=market_data_summary,
        ),
        market_data_summary=market_data_summary,
        build_diagnostics=tuple(build_diagnostics),
    )


def run_pipeline_simulation(
    *,
    runner_config: RunnerConfig,
    optimisation_config: OptimizationConfig,
    pipeline_config: PipelineConfig,
    simulation_config: PipelineSimulationConfig | None = None,
) -> Optional[PipelineSimulationResult]:
    return run_pipeline_simulation_with_diagnostics(
        runner_config=runner_config,
        optimisation_config=optimisation_config,
        pipeline_config=pipeline_config,
        simulation_config=simulation_config,
    ).result
