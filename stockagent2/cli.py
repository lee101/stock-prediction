from __future__ import annotations

import argparse
import json
import sys
from dataclasses import fields
from pathlib import Path
from types import ModuleType
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple, Union, cast

tomllib: ModuleType | None = None

try:
    import tomllib  # type: ignore[attr-defined]
except ModuleNotFoundError:  # pragma: no cover - Python <3.11 fallback
    tomllib = None

from stockagent2.agentsimulator.runner import (
    PipelineSimulationConfig,
    PipelineSimulationResult,
    RunnerConfig,
    run_pipeline_simulation,
)
from stockagent2.config import OptimizationConfig, PipelineConfig


JSONLike = Mapping[str, Any]


def _load_overrides(path: Optional[Path]) -> Dict[str, Any]:
    if path is None:
        return {}
    if not path.exists():
        raise FileNotFoundError(f"Config file {path} does not exist")
    suffix = path.suffix.lower()
    data: Mapping[str, Any]
    if suffix == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
    elif suffix in (".toml", ".tml"):
        if tomllib is None:  # pragma: no cover - defensive branch
            raise RuntimeError("tomllib module unavailable; cannot parse TOML configuration.")
        data = cast(Mapping[str, Any], tomllib.loads(path.read_text(encoding="utf-8")))
    else:
        raise ValueError(f"Unsupported config format {path.suffix!r}; expected .json or .toml.")
    if not isinstance(data, Mapping):
        raise ValueError(f"Configuration file {path} must contain a mapping/object at the top level")
    return dict(data)


def _symbol_tuple(value: Any) -> Tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, (list, tuple, set)):
        return tuple(str(item).upper() for item in value)
    if isinstance(value, str):
        if not value.strip():
            return ()
        parts = [part.strip() for part in value.replace(",", " ").split() if part.strip()]
        return tuple(part.upper() for part in parts)
    raise ValueError(f"Unsupported symbols payload: {value!r}")


def _normalise_runner_field(name: str, value: Any) -> Any:
    if value is None:
        return None
    if name == "symbols":
        return _symbol_tuple(value)
    if name in {"lookback_days", "simulation_days"}:
        return int(value)
    if name == "starting_cash":
        return float(value)
    if name == "local_data_dir":
        return Path(value)
    if name == "allow_remote_data":
        return bool(value)
    return value


def _normalise_optimisation_field(name: str, value: Any) -> Any:
    if value is None:
        return None
    if name == "sector_exposure_limits":
        if not isinstance(value, Mapping):
            raise ValueError("sector_exposure_limits must be a mapping of sector -> limit")
        return {str(key).upper(): float(val) for key, val in value.items()}
    return float(value)


def _normalise_pipeline_field(name: str, value: Any) -> Any:
    if value is None:
        return None
    if name == "annualisation_periods":
        return int(value)
    if name == "apply_confidence_to_mu":
        return bool(value)
    if name == "default_market_caps":
        if value is None:
            return None
        if not isinstance(value, Mapping):
            raise ValueError("default_market_caps must be a mapping of symbol -> market cap")
        return {str(key).upper(): float(val) for key, val in value.items()}
    return float(value)


def _normalise_simulation_field(name: str, value: Any) -> Any:
    if value is None:
        return None
    if name == "symbols":
        return _symbol_tuple(value)
    if name in {"lookback_days", "sample_count", "llm_horizon_days"}:
        return int(value)
    return float(value)


def _load_dataclass_defaults(cls):
    instance = cls()  # type: ignore[call-arg]
    return {field.name: getattr(instance, field.name) for field in fields(cls)}


def _build_config(
    cls,
    *,
    file_overrides: Mapping[str, Any],
    cli_overrides: Mapping[str, Any],
    normaliser,
):
    defaults = _load_dataclass_defaults(cls)
    field_names = set(defaults.keys())
    merged: Dict[str, Any] = dict(defaults)
    for source in (file_overrides, cli_overrides):
        for key, value in source.items():
            if key not in field_names:
                raise ValueError(f"Unknown field {key!r} for {cls.__name__}")
            normalised = normaliser(key, value)
            if normalised is not None:
                merged[key] = normalised
    return cls(**merged)


def _serialise_value(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, tuple):
        return [_serialise_value(item) for item in value]
    if isinstance(value, list):
        return [_serialise_value(item) for item in value]
    if isinstance(value, Mapping):
        return {str(key): _serialise_value(val) for key, val in value.items()}
    return value


def _serialise_dataclass(instance) -> Dict[str, Any]:
    payload: Dict[str, Any] = {}
    for field in fields(instance.__class__):
        payload[field.name] = _serialise_value(getattr(instance, field.name))
    return payload


def _parse_kv_pairs(items: Optional[Sequence[str]]) -> Dict[str, float]:
    result: Dict[str, float] = {}
    if not items:
        return result
    for item in items:
        if "=" not in item:
            raise ValueError(f"Expected KEY=VALUE pair, received {item!r}")
        key, raw_value = item.split("=", 1)
        key = key.strip().upper()
        if not key:
            raise ValueError(f"Missing key in {item!r}")
        try:
            value = float(raw_value)
        except ValueError as exc:
            raise ValueError(f"Invalid numeric value in {item!r}") from exc
        result[key] = value
    return result


def _format_currency(value: float) -> str:
    return f"${value:,.2f}"


def _summarise_result(
    result: PipelineSimulationResult,
    *,
    paper: bool,
    runner: RunnerConfig,
    optimisation: OptimizationConfig,
    pipeline: PipelineConfig,
    simulation_cfg: PipelineSimulationConfig,
) -> Dict[str, Any]:
    simulation = result.simulation
    allocations = [
        {
            "universe": list(allocation.universe),
            "weights": [float(weight) for weight in allocation.weights],
        }
        for allocation in result.allocations
    ]
    summary: Dict[str, Any] = {
        "trading_mode": "paper" if paper else "live",
        "paper": paper,
        "plans_generated": len(result.plans),
        "trades_executed": len(result.simulator.trade_log),
        "runner": _serialise_dataclass(runner),
        "optimisation": _serialise_dataclass(optimisation),
        "pipeline": _serialise_dataclass(pipeline),
        "simulation_config": _serialise_dataclass(simulation_cfg),
        "simulation": {
            "starting_cash": simulation.starting_cash,
            "ending_cash": simulation.ending_cash,
            "ending_equity": simulation.ending_equity,
            "realized_pnl": simulation.realized_pnl,
            "unrealized_pnl": simulation.unrealized_pnl,
            "total_fees": simulation.total_fees,
        },
        "allocation_count": len(result.allocations),
        "last_allocation": allocations[-1] if allocations else None,
    }
    return summary


def _emit_text_summary(summary: Mapping[str, Any]) -> str:
    runner = summary["runner"]
    simulation_cfg = summary["simulation_config"]
    simulation = summary["simulation"]
    symbols = runner.get("symbols", [])
    if isinstance(symbols, tuple):
        symbols = list(symbols)
    lines = [
        f"Trading mode: {summary['trading_mode']}",
        f"Symbols: {', '.join(symbols) if symbols else 'n/a'}",
        f"Lookback days: {runner.get('lookback_days')}",
        f"Simulation days: {runner.get('simulation_days')}",
        f"Plans generated: {summary['plans_generated']}",
        f"Trades executed: {summary['trades_executed']}",
    ]

    starting_cash = float(simulation["starting_cash"])
    ending_cash = float(simulation["ending_cash"])
    ending_equity = float(simulation["ending_equity"])
    realized = float(simulation["realized_pnl"])
    unrealized = float(simulation["unrealized_pnl"])
    fees = float(simulation["total_fees"])

    lines.extend(
        [
            f"Starting cash: {_format_currency(starting_cash)}",
            (
                "Ending equity: "
                f"{_format_currency(ending_equity)} "
                f"(cash {_format_currency(ending_cash)}, "
                f"realized {_format_currency(realized)}, "
                f"unrealized {_format_currency(unrealized)}, "
                f"fees {_format_currency(fees)})"
            ),
            f"Sample count: {simulation_cfg.get('sample_count')}",
            f"LLM horizon days: {simulation_cfg.get('llm_horizon_days')}",
        ]
    )

    last_allocation = summary.get("last_allocation")
    if last_allocation:
        weights = [round(float(value), 5) for value in last_allocation.get("weights", [])]
        lines.append(f"Last allocation weights: {weights}")
        universe = last_allocation.get("universe", [])
        lines.append(f"Last allocation universe: {universe}")

    return "\n".join(lines)


def _write_output(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _write_json_output(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _handle_pipeline_simulation(args: argparse.Namespace) -> int:
    runner_cli: Dict[str, Any] = {}
    if args.symbols:
        runner_cli["symbols"] = args.symbols
    if args.lookback_days is not None:
        runner_cli["lookback_days"] = args.lookback_days
    if args.simulation_days is not None:
        runner_cli["simulation_days"] = args.simulation_days
    if args.starting_cash is not None:
        runner_cli["starting_cash"] = args.starting_cash
    if args.local_data_dir is not None:
        runner_cli["local_data_dir"] = args.local_data_dir
    if args.allow_remote_data is not None:
        runner_cli["allow_remote_data"] = args.allow_remote_data

    optimisation_cli: Dict[str, Any] = {}
    if args.net_exposure_target is not None:
        optimisation_cli["net_exposure_target"] = args.net_exposure_target
    if args.gross_exposure_limit is not None:
        optimisation_cli["gross_exposure_limit"] = args.gross_exposure_limit
    if args.long_cap is not None:
        optimisation_cli["long_cap"] = args.long_cap
    if args.short_cap is not None:
        optimisation_cli["short_cap"] = args.short_cap
    if args.transaction_cost_bps is not None:
        optimisation_cli["transaction_cost_bps"] = args.transaction_cost_bps
    if args.turnover_penalty_bps is not None:
        optimisation_cli["turnover_penalty_bps"] = args.turnover_penalty_bps
    if args.optimiser_risk_aversion is not None:
        optimisation_cli["risk_aversion"] = args.optimiser_risk_aversion
    if args.min_weight is not None:
        optimisation_cli["min_weight"] = args.min_weight
    if args.max_weight is not None:
        optimisation_cli["max_weight"] = args.max_weight
    sector_limits = _parse_kv_pairs(args.sector_limit)
    if sector_limits:
        optimisation_cli["sector_exposure_limits"] = sector_limits

    pipeline_cli: Dict[str, Any] = {}
    if args.tau is not None:
        pipeline_cli["tau"] = args.tau
    if args.shrinkage is not None:
        pipeline_cli["shrinkage"] = args.shrinkage
    if args.min_confidence is not None:
        pipeline_cli["min_confidence"] = args.min_confidence
    if args.annualisation_periods is not None:
        pipeline_cli["annualisation_periods"] = args.annualisation_periods
    if args.chronos_weight is not None:
        pipeline_cli["chronos_weight"] = args.chronos_weight
    if args.timesfm_weight is not None:
        pipeline_cli["timesfm_weight"] = args.timesfm_weight
    if args.pipeline_risk_aversion is not None:
        pipeline_cli["risk_aversion"] = args.pipeline_risk_aversion
    if args.market_prior_weight is not None:
        pipeline_cli["market_prior_weight"] = args.market_prior_weight
    if args.apply_confidence_to_mu is not None:
        pipeline_cli["apply_confidence_to_mu"] = args.apply_confidence_to_mu
    market_caps = _parse_kv_pairs(args.default_market_cap)
    if market_caps:
        pipeline_cli["default_market_caps"] = market_caps

    simulation_cli: Dict[str, Any] = {}
    if args.sim_symbols:
        simulation_cli["symbols"] = args.sim_symbols
    if args.sample_count is not None:
        simulation_cli["sample_count"] = args.sample_count
    if args.min_trade_value is not None:
        simulation_cli["min_trade_value"] = args.min_trade_value
    if args.min_volatility is not None:
        simulation_cli["min_volatility"] = args.min_volatility
    if args.confidence_floor is not None:
        simulation_cli["confidence_floor"] = args.confidence_floor
    if args.confidence_ceiling is not None:
        simulation_cli["confidence_ceiling"] = args.confidence_ceiling
    if args.llm_horizon_days is not None:
        simulation_cli["llm_horizon_days"] = args.llm_horizon_days

    runner = _build_config(
        RunnerConfig,
        file_overrides=_load_overrides(args.runner_config),
        cli_overrides=runner_cli,
        normaliser=_normalise_runner_field,
    )
    optimisation = _build_config(
        OptimizationConfig,
        file_overrides=_load_overrides(args.optimisation_config),
        cli_overrides=optimisation_cli,
        normaliser=_normalise_optimisation_field,
    )
    pipeline_cfg = _build_config(
        PipelineConfig,
        file_overrides=_load_overrides(args.pipeline_config),
        cli_overrides=pipeline_cli,
        normaliser=_normalise_pipeline_field,
    )
    simulation_cfg = _build_config(
        PipelineSimulationConfig,
        file_overrides=_load_overrides(args.simulation_config),
        cli_overrides=simulation_cli,
        normaliser=_normalise_simulation_field,
    )
    if not simulation_cfg.symbols:
        simulation_cfg.symbols = runner.symbols

    result = run_pipeline_simulation(
        runner_config=runner,
        optimisation_config=optimisation,
        pipeline_config=pipeline_cfg,
        simulation_config=simulation_cfg,
    )
    if result is None:
        print("Pipeline simulation produced no trading plans (check data availability and configuration).", file=sys.stderr)
        return 1

    summary = _summarise_result(
        result,
        paper=args.paper,
        runner=runner,
        optimisation=optimisation,
        pipeline=pipeline_cfg,
        simulation_cfg=simulation_cfg,
    )

    if args.summary_format == "json":
        output_payload = summary
        text_output = json.dumps(summary, indent=2, sort_keys=True)
    else:
        output_payload = summary
        text_output = _emit_text_summary(summary)

    if not args.quiet:
        print(text_output)

    if args.summary_output is not None:
        if args.summary_format == "json":
            _write_json_output(args.summary_output, output_payload)
        else:
            _write_output(args.summary_output, text_output)

    if args.plans_output is not None:
        plan_payload = [plan.to_dict() for plan in result.plans]
        _write_json_output(args.plans_output, plan_payload)

    if args.trades_output is not None:
        _write_json_output(args.trades_output, result.simulation.trades)

    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="stockagent2 command suite")
    subparsers = parser.add_subparsers(dest="command")

    pipeline_parser = subparsers.add_parser(
        "pipeline-sim",
        help="Run the stockagent2 allocation pipeline over recent market data.",
    )

    pipeline_parser.add_argument("--symbols", nargs="+", help="Symbols for runner configuration (defaults to production universe).")
    pipeline_parser.add_argument("--lookback-days", type=int, help="Historical lookback window for market data.")
    pipeline_parser.add_argument("--simulation-days", type=int, help="Number of trading days to simulate.")
    pipeline_parser.add_argument("--starting-cash", type=float, help="Starting cash balance for the simulated account.")
    pipeline_parser.add_argument("--local-data-dir", type=Path, help="Optional override for cached OHLC data directory.")
    pipeline_parser.add_argument(
        "--allow-remote-data",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Permit remote OHLC fetch when local cache misses occur.",
    )
    pipeline_parser.add_argument("--runner-config", type=Path, help="Path to JSON/TOML file with RunnerConfig overrides.")
    pipeline_parser.add_argument("--optimisation-config", type=Path, help="Path to JSON/TOML file with OptimizationConfig overrides.")
    pipeline_parser.add_argument("--pipeline-config", type=Path, help="Path to JSON/TOML file with PipelineConfig overrides.")
    pipeline_parser.add_argument("--simulation-config", type=Path, help="Path to JSON/TOML file with PipelineSimulationConfig overrides.")

    pipeline_parser.add_argument("--net-exposure-target", type=float, help="Net exposure target (OptimizationConfig).")
    pipeline_parser.add_argument("--gross-exposure-limit", type=float, help="Gross exposure cap (OptimizationConfig).")
    pipeline_parser.add_argument("--long-cap", type=float, help="Maximum individual long weight (OptimizationConfig).")
    pipeline_parser.add_argument("--short-cap", type=float, help="Maximum individual short weight (OptimizationConfig).")
    pipeline_parser.add_argument("--transaction-cost-bps", type=float, help="Transaction cost penalty in basis points.")
    pipeline_parser.add_argument("--turnover-penalty-bps", type=float, help="Turnover penalty in basis points.")
    pipeline_parser.add_argument("--optimiser-risk-aversion", type=float, help="Risk aversion parameter for optimiser.")
    pipeline_parser.add_argument("--min-weight", type=float, help="Minimum weight bound.")
    pipeline_parser.add_argument("--max-weight", type=float, help="Maximum weight bound.")
    pipeline_parser.add_argument(
        "--sector-limit",
        action="append",
        metavar="SECTOR=LIMIT",
        help="Sector exposure limit override (repeatable).",
    )

    pipeline_parser.add_argument("--tau", type=float, help="Black–Litterman tau parameter.")
    pipeline_parser.add_argument("--shrinkage", type=float, help="Linear covariance shrinkage coefficient.")
    pipeline_parser.add_argument("--min-confidence", type=float, help="Minimum LLM confidence floor.")
    pipeline_parser.add_argument("--annualisation-periods", type=int, help="Trading periods per year for scaling.")
    pipeline_parser.add_argument("--chronos-weight", type=float, help="Weight assigned to Chronos forecasts.")
    pipeline_parser.add_argument("--timesfm-weight", type=float, help="Weight assigned to TimesFM forecasts.")
    pipeline_parser.add_argument("--pipeline-risk-aversion", type=float, help="Black–Litterman risk aversion parameter.")
    pipeline_parser.add_argument("--market-prior-weight", type=float, help="Weight assigned to the market equilibrium prior.")
    pipeline_parser.add_argument(
        "--apply-confidence-to-mu",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Apply LLM confidence scores when adjusting posterior mean.",
    )
    pipeline_parser.add_argument(
        "--default-market-cap",
        action="append",
        metavar="SYMBOL=CAP",
        help="Default market cap override (repeatable).",
    )

    pipeline_parser.add_argument("--sim-symbols", nargs="+", help="Override symbols for the plan builder (defaults to runner symbols).")
    pipeline_parser.add_argument("--sample-count", type=int, help="Monte Carlo sample count for forecasts.")
    pipeline_parser.add_argument("--min-trade-value", type=float, help="Minimum trade value filter for generated instructions.")
    pipeline_parser.add_argument("--min-volatility", type=float, help="Minimum volatility floor used for confidence estimation.")
    pipeline_parser.add_argument("--confidence-floor", type=float, help="Lower bound for generated LLM confidence scores.")
    pipeline_parser.add_argument("--confidence-ceiling", type=float, help="Upper bound for generated LLM confidence scores.")
    pipeline_parser.add_argument("--llm_horizon_days", dest="llm_horizon_days", type=int, help="Horizon (days) used when synthesising LLM views.")

    mode_group = pipeline_parser.add_mutually_exclusive_group()
    mode_group.add_argument("--paper", dest="paper", action="store_true", default=True, help="Tag run as paper trading (default).")
    mode_group.add_argument("--live", dest="paper", action="store_false", help="Tag run as live trading.")

    pipeline_parser.add_argument(
        "--summary-format",
        choices=("text", "json"),
        default="text",
        help="Format for CLI summary output.",
    )
    pipeline_parser.add_argument("--summary-output", type=Path, help="Optional path to write summary output.")
    pipeline_parser.add_argument("--plans-output", type=Path, help="Optional path to write generated trading plans (JSON).")
    pipeline_parser.add_argument("--trades-output", type=Path, help="Optional path to write executed trade log (JSON).")
    pipeline_parser.add_argument("--quiet", action="store_true", help="Suppress stdout summary (use with --summary-output).")

    pipeline_parser.set_defaults(handler=_handle_pipeline_simulation)

    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if not getattr(args, "command", None):
        parser.print_help()
        return 0
    handler = getattr(args, "handler", None)
    if handler is None:
        parser.error("Command handler not configured.")
    try:
        return handler(args)
    except Exception as exc:  # pragma: no cover - defensive fallback
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
