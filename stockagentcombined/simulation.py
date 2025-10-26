from __future__ import annotations

import argparse
from dataclasses import dataclass, fields
from collections.abc import Callable, Mapping, Sequence
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


StrategyFactory = Callable[[], BaseRiskStrategy]


@dataclass(frozen=True)
class SimulationPreset:
    description: str
    config_overrides: dict[str, object]
    starting_cash: float | None = None
    allow_remote_data: bool | None = None
    strategy_names: tuple[str, ...] | None = None


STRATEGY_FACTORIES: dict[str, StrategyFactory] = {
    "probe-trade": ProbeTradeStrategy,
    "profit-shutdown": ProfitShutdownStrategy,
}

DEFAULT_STRATEGIES: tuple[str, ...] = ("probe-trade", "profit-shutdown")

SIMULATION_PRESETS: dict[str, SimulationPreset] = {
    "offline-regression": SimulationPreset(
        description=(
            "Replicates the offline regression sanity-check from the README "
            "(AAPL/MSFT, three trading days, tighter thresholds)."
        ),
        config_overrides={
            "simulation_days": 3,
            "min_history": 10,
            "min_signal": 0.0,
            "error_multiplier": 0.25,
            "base_quantity": 10.0,
            "min_quantity": 1.0,
        },
        starting_cash=250_000.0,
        allow_remote_data=False,
        strategy_names=DEFAULT_STRATEGIES,
    ),
}


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
    parser.add_argument(
        "--preset",
        choices=sorted(SIMULATION_PRESETS),
        help="Optional preset that seeds the CLI defaults (use --list-presets to inspect).",
    )
    parser.add_argument("--list-presets", action="store_true", help="List available presets and exit.")
    parser.add_argument("--symbols", nargs="+", help="Symbols to simulate.")
    parser.add_argument("--lookback-days", type=int)
    parser.add_argument("--simulation-days", type=int)
    parser.add_argument("--starting-cash", type=float)
    parser.add_argument("--min-history", type=int)
    parser.add_argument("--min-signal", type=float)
    parser.add_argument("--error-multiplier", type=float)
    parser.add_argument("--base-quantity", type=float)
    parser.add_argument("--max-quantity-multiplier", type=float)
    parser.add_argument("--min-quantity", type=float)
    parser.add_argument("--allow-short", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--local-data-dir", type=Path)
    parser.add_argument("--allow-remote-data", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument(
        "--strategy",
        dest="strategy_names",
        action="append",
        choices=sorted(STRATEGY_FACTORIES),
        help="Risk strategy to include. Repeat for multiple. Defaults to probe-trade and profit-shutdown.",
        metavar="NAME",
    )
    parsed = parser.parse_args(args)

    if parsed.list_presets:
        lines = [f"{name}: {SIMULATION_PRESETS[name].description}" for name in sorted(SIMULATION_PRESETS)]
        parser.exit(status=0, message="\n".join(lines) + "\n")

    preset = SIMULATION_PRESETS.get(parsed.preset) if parsed.preset else None
    config_defaults = SimulationConfig()
    config_kwargs: dict[str, object] = {field.name: getattr(config_defaults, field.name) for field in fields(SimulationConfig)}
    if preset is not None:
        config_kwargs.update(preset.config_overrides)

    symbols_obj = tuple(parsed.symbols) if parsed.symbols is not None else config_kwargs.get("symbols")
    if symbols_obj is None:
        symbols = tuple(DEFAULT_SYMBOLS)
    elif isinstance(symbols_obj, (str, bytes)):
        symbols = (str(symbols_obj),)
    elif isinstance(symbols_obj, Sequence):
        symbols = tuple(symbols_obj)
    else:
        symbols = tuple(DEFAULT_SYMBOLS)
    config_kwargs["symbols"] = symbols

    if parsed.lookback_days is not None:
        config_kwargs["lookback_days"] = parsed.lookback_days
    if parsed.simulation_days is not None:
        config_kwargs["simulation_days"] = parsed.simulation_days
    if parsed.starting_cash is not None:
        config_kwargs["starting_cash"] = parsed.starting_cash
    elif preset is not None and preset.starting_cash is not None:
        config_kwargs["starting_cash"] = preset.starting_cash
    if parsed.min_history is not None:
        config_kwargs["min_history"] = parsed.min_history
    if parsed.min_signal is not None:
        config_kwargs["min_signal"] = parsed.min_signal
    if parsed.error_multiplier is not None:
        config_kwargs["error_multiplier"] = parsed.error_multiplier
    if parsed.base_quantity is not None:
        config_kwargs["base_quantity"] = parsed.base_quantity
    if parsed.max_quantity_multiplier is not None:
        config_kwargs["max_quantity_multiplier"] = parsed.max_quantity_multiplier
    if parsed.min_quantity is not None:
        config_kwargs["min_quantity"] = parsed.min_quantity
    if parsed.allow_short is not None:
        config_kwargs["allow_short"] = parsed.allow_short

    simulation_config = SimulationConfig(**config_kwargs)

    strategy_names: Sequence[str] | None = parsed.strategy_names
    if not strategy_names and preset is not None:
        strategy_names = preset.strategy_names
    if not strategy_names:
        strategy_names = DEFAULT_STRATEGIES
    strategies: list[BaseRiskStrategy] = [_build_strategy(name) for name in strategy_names]

    allow_remote_data = parsed.allow_remote_data
    if allow_remote_data is None and preset is not None and preset.allow_remote_data is not None:
        allow_remote_data = preset.allow_remote_data
    if allow_remote_data is None:
        allow_remote_data = False

    local_data_dir = parsed.local_data_dir if parsed.local_data_dir is not None else Path("trainingdata")

    bundle = fetch_latest_ohlc(
        symbols=simulation_config.symbols,
        lookback_days=simulation_config.lookback_days,
        as_of=datetime.now(timezone.utc),
        local_data_dir=local_data_dir,
        allow_remote_download=allow_remote_data,
    )
    market_frames = bundle.bars
    trading_days = list(bundle.trading_days())
    if simulation_config.simulation_days > 0:
        trading_days = trading_days[-simulation_config.simulation_days :]

    generator = CombinedForecastGenerator()
    builder = CombinedPlanBuilder(generator=generator, config=simulation_config)

    run_simulation(
        builder=builder,
        market_frames=market_frames,
        trading_days=trading_days,
        starting_cash=simulation_config.starting_cash,
        strategies=strategies,
    )


def _build_strategy(name: str) -> BaseRiskStrategy:
    factory = STRATEGY_FACTORIES.get(name)
    if factory is None:
        raise ValueError(f"Unknown strategy '{name}'")
    return factory()


if __name__ == "__main__":  # pragma: no cover
    main()
