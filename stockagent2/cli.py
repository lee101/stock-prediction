from __future__ import annotations

import argparse
import json
import os
import shlex
import sys
import tempfile
from dataclasses import fields
from pathlib import Path
from types import ModuleType
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple, cast

from stockagent.agentsimulator import normalize_market_symbol, resolve_local_data_dirs

tomllib: ModuleType | None = None

try:
    import tomllib  # type: ignore[attr-defined]
except ModuleNotFoundError:  # pragma: no cover - Python <3.11 fallback
    tomllib = None

from stockagent2.agentsimulator.runner import (
    PipelinePlanBuildDiagnostics,
    PipelineSimulationConfig,
    PipelineSimulationResult,
    RunnerConfig,
    run_pipeline_simulation_with_diagnostics,
)
from stockagent2.config import OptimizationConfig, PipelineConfig


JSONLike = Mapping[str, Any]
EFFECTIVE_ARGS_JSON_SUFFIX = ".effective_args.json"
EFFECTIVE_ARGS_TXT_SUFFIX = ".effective_args.txt"
PLANS_JSON_SUFFIX = ".plans.json"
TRADES_JSON_SUFFIX = ".trades.json"


class ArgsFileParser(argparse.ArgumentParser):
    """ArgumentParser with shell-style @args file support."""

    def convert_arg_line_to_args(self, arg_line: str) -> Iterable[str]:
        line = arg_line.strip()
        if not line or line.startswith("#"):
            return []
        return shlex.split(line)


def _preferred_option_string(action: argparse.Action) -> str | None:
    long_options = [option for option in action.option_strings if option.startswith("--")]
    if long_options:
        return max(long_options, key=len)
    if action.option_strings:
        return action.option_strings[-1]
    return None


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
        return tuple(normalize_market_symbol(str(item)) for item in value)
    if isinstance(value, str):
        if not value.strip():
            return ()
        parts = [part.strip() for part in value.replace(",", " ").split() if part.strip()]
        return tuple(normalize_market_symbol(part) for part in parts)
    raise ValueError(f"Unsupported symbols payload: {value!r}")


def _parse_bool_like(value: Any, *, field_name: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and value in (0, 1):
        return bool(value)
    if isinstance(value, str):
        normalised = value.strip().lower()
        if normalised in {"1", "true", "yes", "on"}:
            return True
        if normalised in {"0", "false", "no", "off"}:
            return False
    raise ValueError(f"{field_name} must be a boolean-compatible value, got {value!r}")


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
        return _parse_bool_like(value, field_name=name)
    if name == "use_fallback_data_dirs":
        return _parse_bool_like(value, field_name=name)
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
        return _parse_bool_like(value, field_name=name)
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
    if name in {
        "lookback_days",
        "sample_count",
        "llm_horizon_days",
        "history_min_period_divisor",
        "min_view_half_life_days",
        "max_view_half_life_days",
        "rng_seed",
    }:
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


def _build_data_setup_summary(runner: RunnerConfig) -> Dict[str, Any]:
    search_dirs = [
        str(path)
        for path in resolve_local_data_dirs(
            local_data_dir=runner.local_data_dir,
            use_fallback_data_dirs=runner.use_fallback_data_dirs,
        )
    ]
    return {
        "local_data_dir": str(runner.local_data_dir) if runner.local_data_dir is not None else None,
        "use_fallback_data_dirs": runner.use_fallback_data_dirs,
        "allow_remote_data": runner.allow_remote_data,
        "data_search_dirs": search_dirs,
    }


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
    build_diagnostics: Sequence[PipelinePlanBuildDiagnostics] = (),
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
        "data_setup": _build_data_setup_summary(runner),
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
        "market_data": _serialise_dataclass(result.market_data_summary),
        "plan_build_diagnostics": [_serialise_dataclass(item) for item in build_diagnostics],
        "allocation_count": len(result.allocations),
        "last_allocation": allocations[-1] if allocations else None,
    }
    return summary


def _emit_text_summary(summary: Mapping[str, Any]) -> str:
    data_setup = cast(Mapping[str, Any], summary["data_setup"])
    runner = summary["runner"]
    simulation_cfg = summary["simulation_config"]
    simulation = summary["simulation"]
    market_data = summary.get("market_data", {})
    build_diagnostics = cast(Sequence[Mapping[str, Any]], summary.get("plan_build_diagnostics", []))
    symbols = runner.get("symbols", [])
    if isinstance(symbols, tuple):
        symbols = list(symbols)
    lines = [
        f"Trading mode: {summary['trading_mode']}",
        f"Symbols: {', '.join(symbols) if symbols else 'n/a'}",
        f"Local data dir: {data_setup.get('local_data_dir') or 'n/a'}",
        f"Fallback data dirs: {'enabled' if data_setup.get('use_fallback_data_dirs') else 'disabled'}",
        f"Remote fetch on miss: {'enabled' if data_setup.get('allow_remote_data') else 'disabled'}",
        f"Lookback days: {runner.get('lookback_days')}",
        f"Simulation days: {runner.get('simulation_days')}",
        f"Plans generated: {summary['plans_generated']}",
        f"Trades executed: {summary['trades_executed']}",
    ]
    search_dirs = data_setup.get("data_search_dirs", [])
    if search_dirs:
        lines.append(f"Data search order: {', '.join(search_dirs)}")

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
            f"History divisor: {simulation_cfg.get('history_min_period_divisor')}",
            f"Secondary sample scale: {simulation_cfg.get('secondary_sample_scale')}",
            f"Sample return clip: {simulation_cfg.get('sample_return_clip')}",
        ]
    )
    if market_data:
        loaded_symbols = market_data.get("loaded_symbols", [])
        requested_symbols = market_data.get("symbols_requested", [])
        first_day = market_data.get("first_trading_day")
        last_day = market_data.get("last_trading_day")
        if first_day and last_day:
            lines.append(
                "Data coverage: "
                f"{first_day} to {last_day} across {len(loaded_symbols)}/{len(requested_symbols)} loaded symbols"
            )
        empty_symbols = market_data.get("empty_symbols", [])
        if empty_symbols:
            lines.append(f"Empty symbols: {', '.join(empty_symbols)}")
    if build_diagnostics:
        status_counts: Dict[str, int] = {}
        for item in build_diagnostics:
            status = str(item.get("status", "unknown"))
            status_counts[status] = status_counts.get(status, 0) + 1
        rendered = ", ".join(f"{status}={count}" for status, count in sorted(status_counts.items()))
        lines.append(f"Plan build statuses: {rendered}")

    last_allocation = summary.get("last_allocation")
    if last_allocation:
        weights = [round(float(value), 5) for value in last_allocation.get("weights", [])]
        lines.append(f"Last allocation weights: {weights}")
        universe = last_allocation.get("universe", [])
        lines.append(f"Last allocation universe: {universe}")

    if summary.get("effective_args_path"):
        lines.append(f"Effective args JSON: {summary['effective_args_path']}")
    if summary.get("effective_args_cli_path"):
        lines.append(f"Effective args file: {summary['effective_args_cli_path']}")
    if summary.get("effective_args_warning"):
        lines.append(f"Effective args warning: {summary['effective_args_warning']}")
    if summary.get("plans_output_path"):
        lines.append(f"Plans JSON: {summary['plans_output_path']}")
    if summary.get("trades_output_path"):
        lines.append(f"Trades JSON: {summary['trades_output_path']}")
    if summary.get("plans_output_warning"):
        lines.append(f"Plans output warning: {summary['plans_output_warning']}")
    if summary.get("trades_output_warning"):
        lines.append(f"Trades output warning: {summary['trades_output_warning']}")

    return "\n".join(lines)


def _summarise_run_plan(
    *,
    paper: bool,
    runner: RunnerConfig,
    optimisation: OptimizationConfig,
    pipeline: PipelineConfig,
    simulation_cfg: PipelineSimulationConfig,
) -> Dict[str, Any]:
    return {
        "mode": "describe-run",
        "trading_mode": "paper" if paper else "live",
        "paper": paper,
        "data_setup": _build_data_setup_summary(runner),
        "runner": _serialise_dataclass(runner),
        "optimisation": _serialise_dataclass(optimisation),
        "pipeline": _serialise_dataclass(pipeline),
        "simulation_config": _serialise_dataclass(simulation_cfg),
    }


def _summarise_no_plan_result(
    *,
    paper: bool,
    runner: RunnerConfig,
    optimisation: OptimizationConfig,
    pipeline: PipelineConfig,
    simulation_cfg: PipelineSimulationConfig,
    market_data_summary: Mapping[str, Any] | None = None,
    build_diagnostics: Sequence[PipelinePlanBuildDiagnostics] = (),
    failure_reason: str | None = None,
) -> Dict[str, Any]:
    summary = _summarise_run_plan(
        paper=paper,
        runner=runner,
        optimisation=optimisation,
        pipeline=pipeline,
        simulation_cfg=simulation_cfg,
    )
    summary["mode"] = "no-plans"
    summary["failure_reason"] = failure_reason or "Pipeline simulation produced no trading plans."
    if market_data_summary is not None:
        summary["market_data"] = market_data_summary
    if build_diagnostics:
        summary["plan_build_diagnostics"] = [_serialise_dataclass(item) for item in build_diagnostics]
    summary["next_steps"] = [
        "Check local market data coverage and empty symbols with --describe-run.",
        "Relax trade filters or minimum trade value if the pipeline is too selective.",
        "Enable --allow-remote-data if local caches are incomplete.",
    ]
    return summary


def _emit_text_run_plan(summary: Mapping[str, Any]) -> str:
    data_setup = cast(Mapping[str, Any], summary["data_setup"])
    runner = cast(Mapping[str, Any], summary["runner"])
    simulation_cfg = cast(Mapping[str, Any], summary["simulation_config"])
    optimisation = cast(Mapping[str, Any], summary["optimisation"])
    pipeline = cast(Mapping[str, Any], summary["pipeline"])
    symbols = runner.get("symbols", [])
    if isinstance(symbols, tuple):
        symbols = list(symbols)
    sim_symbols = simulation_cfg.get("symbols", [])
    if isinstance(sim_symbols, tuple):
        sim_symbols = list(sim_symbols)
    lines = [
        f"Mode: {summary['mode']}",
        f"Trading mode: {summary['trading_mode']}",
        f"Runner symbols: {', '.join(symbols) if symbols else 'n/a'}",
        f"Simulation symbols: {', '.join(sim_symbols) if sim_symbols else 'n/a'}",
        f"Local data dir: {data_setup.get('local_data_dir') or 'n/a'}",
        f"Fallback data dirs: {'enabled' if data_setup.get('use_fallback_data_dirs') else 'disabled'}",
        f"Remote fetch on miss: {'enabled' if data_setup.get('allow_remote_data') else 'disabled'}",
        f"Lookback days: {runner.get('lookback_days')}",
        f"Simulation days: {runner.get('simulation_days')}",
        f"Starting cash: {_format_currency(float(runner.get('starting_cash', 0.0)))}",
        f"Sample count: {simulation_cfg.get('sample_count')}",
        f"LLM horizon days: {simulation_cfg.get('llm_horizon_days')}",
        f"History divisor: {simulation_cfg.get('history_min_period_divisor')}",
        f"Secondary sample scale: {simulation_cfg.get('secondary_sample_scale')}",
        f"Sample return clip: {simulation_cfg.get('sample_return_clip')}",
        f"Net exposure target: {optimisation.get('net_exposure_target')}",
        f"Gross exposure limit: {optimisation.get('gross_exposure_limit')}",
        f"Chronos / TimesFM weights: {pipeline.get('chronos_weight')} / {pipeline.get('timesfm_weight')}",
    ]
    search_dirs = data_setup.get("data_search_dirs", [])
    if search_dirs:
        lines.append(f"Data search order: {', '.join(search_dirs)}")
    if summary.get("effective_args_path"):
        lines.append(f"Effective args JSON: {summary['effective_args_path']}")
    if summary.get("effective_args_cli_path"):
        lines.append(f"Effective args file: {summary['effective_args_cli_path']}")
    if summary.get("effective_args_warning"):
        lines.append(f"Effective args warning: {summary['effective_args_warning']}")
    return "\n".join(lines)


def _emit_text_no_plan_summary(summary: Mapping[str, Any]) -> str:
    lines = _emit_text_run_plan(summary).splitlines()
    lines.append(f"Failure: {summary['failure_reason']}")
    market_data = cast(Mapping[str, Any], summary.get("market_data", {}))
    if market_data:
        loaded_symbols = market_data.get("loaded_symbols", [])
        requested_symbols = market_data.get("symbols_requested", [])
        first_day = market_data.get("first_trading_day")
        last_day = market_data.get("last_trading_day")
        if first_day and last_day:
            lines.append(
                "Data coverage: "
                f"{first_day} to {last_day} across {len(loaded_symbols)}/{len(requested_symbols)} loaded symbols"
            )
        empty_symbols = market_data.get("empty_symbols", [])
        if empty_symbols:
            lines.append(f"Empty symbols: {', '.join(empty_symbols)}")
    build_diagnostics = cast(Sequence[Mapping[str, Any]], summary.get("plan_build_diagnostics", []))
    if build_diagnostics:
        lines.append("Plan build diagnostics:")
        for item in build_diagnostics:
            lines.append(
                f"- {item.get('target_date')}: {item.get('status')} "
                f"(history {len(item.get('symbols_with_history', []))}/{len(item.get('symbols_considered', []))}, "
                f"forecasts {len(item.get('forecasted_symbols', []))}, "
                f"instructions {item.get('generated_instruction_count', 0)})"
            )
            failures = cast(Mapping[str, str], item.get("forecast_failure_reasons", {}))
            if failures:
                rendered = ", ".join(f"{symbol}={reason}" for symbol, reason in sorted(failures.items()))
                lines.append(f"  Forecast failures: {rendered}")
            skipped = item.get("skipped_min_trade_symbols", [])
            if skipped:
                lines.append(f"  Below min trade value: {', '.join(skipped)}")
            missing_prices = item.get("missing_price_symbols", [])
            if missing_prices:
                lines.append(f"  Missing price inputs: {', '.join(missing_prices)}")
    next_steps = summary.get("next_steps", [])
    if next_steps:
        lines.append("Next steps:")
        for step in next_steps:
            lines.append(f"- {step}")
    return "\n".join(lines)


def _write_output(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    preserved_mode: int | None = None
    try:
        preserved_mode = path.stat().st_mode & 0o777
    except FileNotFoundError:
        preserved_mode = None

    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.tmp.", dir=str(path.parent))
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        if preserved_mode is not None:
            os.chmod(tmp_path, preserved_mode)
        os.replace(tmp_path, path)
    except Exception:
        try:
            tmp_path.unlink(missing_ok=True)
        except OSError:
            pass
        raise


def _write_json_output(path: Path, payload: Any) -> None:
    _write_output(path, json.dumps(payload, indent=2, sort_keys=True))


def _render_effective_args_file(parser: argparse.ArgumentParser, args: argparse.Namespace) -> str:
    actions_by_dest: dict[str, list[argparse.Action]] = {}
    for action in parser._actions:
        if not action.option_strings or action.dest in {"help", "_parser_ref"}:
            continue
        actions_by_dest.setdefault(action.dest, []).append(action)

    lines = [
        f"# Re-run with: python -m stockagent2.cli @{Path('run').with_suffix(EFFECTIVE_ARGS_TXT_SUFFIX).name}",
    ]
    for dest, actions in actions_by_dest.items():
        if not hasattr(args, dest):
            continue
        value = getattr(args, dest)
        if value is None:
            continue

        bool_actions = [
            action for action in actions if isinstance(action, (argparse._StoreTrueAction, argparse._StoreFalseAction))
        ]
        if bool_actions:
            chosen = next((action for action in bool_actions if getattr(action, "const", None) == value), None)
            if chosen is None:
                continue
            option = _preferred_option_string(chosen)
            if option is not None:
                lines.append(option)
            continue

        action = actions[-1]
        option = _preferred_option_string(action)
        if option is None:
            continue
        if isinstance(value, Path):
            rendered_value = str(value)
        elif isinstance(value, (list, tuple)):
            rendered_value = " ".join(shlex.quote(str(item)) for item in value)
            lines.append(f"{option} {rendered_value}")
            continue
        else:
            rendered_value = shlex.quote(str(value))
        lines.append(f"{option} {rendered_value}")
    return "\n".join(lines) + "\n"


def _effective_args_paths(summary_output: Path) -> tuple[Path, Path]:
    base = summary_output.with_suffix("")
    return (
        base.with_name(base.name + EFFECTIVE_ARGS_JSON_SUFFIX),
        base.with_name(base.name + EFFECTIVE_ARGS_TXT_SUFFIX),
    )


def _sidecar_output_path(summary_output: Path, suffix: str) -> Path:
    base = summary_output.with_suffix("")
    return base.with_name(base.name + suffix)


def _maybe_write_effective_args_artifacts(
    parser: argparse.ArgumentParser,
    args: argparse.Namespace,
    *,
    summary_output: Path | None,
) -> tuple[Path | None, Path | None, str | None]:
    if summary_output is None:
        return None, None, None
    json_path, txt_path = _effective_args_paths(summary_output)
    try:
        effective_args = {
            key: (str(value) if isinstance(value, Path) else value)
            for key, value in vars(args).items()
            if key not in {"_parser_ref", "handler"} and not callable(value)
        }
        _write_json_output(json_path, effective_args)
        _write_output(txt_path, _render_effective_args_file(parser, args))
        return json_path, txt_path, None
    except Exception as exc:  # pragma: no cover - defensive
        return None, None, f"Failed to write effective args artifacts: {exc}"


def _write_json_artifact(
    *,
    path: Path | None,
    payload: Any,
    best_effort: bool,
) -> str | None:
    if path is None:
        return None
    try:
        _write_json_output(path, payload)
        return None
    except Exception as exc:
        if not best_effort:
            raise
        return f"Failed to write {path.name}: {exc}"


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
    if args.use_fallback_data_dirs is not None:
        runner_cli["use_fallback_data_dirs"] = args.use_fallback_data_dirs

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
    parser = cast(argparse.ArgumentParser, args._parser_ref)

    if args.describe_run:
        summary = _summarise_run_plan(
            paper=args.paper,
            runner=runner,
            optimisation=optimisation,
            pipeline=pipeline_cfg,
            simulation_cfg=simulation_cfg,
        )
        effective_args_path, effective_args_cli_path, effective_args_warning = _maybe_write_effective_args_artifacts(
            parser,
            args,
            summary_output=args.summary_output,
        )
        if effective_args_path is not None:
            summary["effective_args_path"] = str(effective_args_path)
        if effective_args_cli_path is not None:
            summary["effective_args_cli_path"] = str(effective_args_cli_path)
        if effective_args_warning is not None:
            summary["effective_args_warning"] = effective_args_warning
        if args.summary_format == "json":
            output_payload = summary
            text_output = json.dumps(summary, indent=2, sort_keys=True)
        else:
            output_payload = summary
            text_output = _emit_text_run_plan(summary)
        if not args.quiet:
            print(text_output)
        if args.summary_output is not None:
            if args.summary_format == "json":
                _write_json_output(args.summary_output, output_payload)
            else:
                _write_output(args.summary_output, text_output)
        return 0

    attempt = run_pipeline_simulation_with_diagnostics(
        runner_config=runner,
        optimisation_config=optimisation,
        pipeline_config=pipeline_cfg,
        simulation_config=simulation_cfg,
    )
    if attempt.result is None:
        summary = _summarise_no_plan_result(
            paper=args.paper,
            runner=runner,
            optimisation=optimisation,
            pipeline=pipeline_cfg,
            simulation_cfg=simulation_cfg,
            market_data_summary=_serialise_dataclass(attempt.market_data_summary),
            build_diagnostics=attempt.build_diagnostics,
            failure_reason=attempt.failure_reason,
        )
        effective_args_path, effective_args_cli_path, effective_args_warning = _maybe_write_effective_args_artifacts(
            parser,
            args,
            summary_output=args.summary_output,
        )
        if effective_args_path is not None:
            summary["effective_args_path"] = str(effective_args_path)
        if effective_args_cli_path is not None:
            summary["effective_args_cli_path"] = str(effective_args_cli_path)
        if effective_args_warning is not None:
            summary["effective_args_warning"] = effective_args_warning

        if args.summary_format == "json":
            output_payload = summary
            text_output = json.dumps(summary, indent=2, sort_keys=True)
        else:
            output_payload = summary
            text_output = _emit_text_no_plan_summary(summary)

        if args.summary_output is not None:
            if args.summary_format == "json":
                _write_json_output(args.summary_output, output_payload)
            else:
                _write_output(args.summary_output, text_output)

        if not args.quiet:
            print(text_output, file=sys.stderr)
        return 1

    result = attempt.result
    summary = _summarise_result(
        result,
        paper=args.paper,
        runner=runner,
        optimisation=optimisation,
        pipeline=pipeline_cfg,
        simulation_cfg=simulation_cfg,
        build_diagnostics=attempt.build_diagnostics,
    )
    plan_payload = [plan.to_dict() for plan in result.plans]
    trades_payload = result.simulation.trades

    derived_plans_output = None
    derived_trades_output = None
    if args.summary_output is not None:
        if args.plans_output is None:
            derived_plans_output = _sidecar_output_path(args.summary_output, PLANS_JSON_SUFFIX)
        if args.trades_output is None:
            derived_trades_output = _sidecar_output_path(args.summary_output, TRADES_JSON_SUFFIX)

    plans_output_path = args.plans_output or derived_plans_output
    trades_output_path = args.trades_output or derived_trades_output

    plans_output_warning = _write_json_artifact(
        path=plans_output_path,
        payload=plan_payload,
        best_effort=args.plans_output is None,
    )
    trades_output_warning = _write_json_artifact(
        path=trades_output_path,
        payload=trades_payload,
        best_effort=args.trades_output is None,
    )
    if plans_output_path is not None:
        summary["plans_output_path"] = str(plans_output_path)
    if trades_output_path is not None:
        summary["trades_output_path"] = str(trades_output_path)
    if plans_output_warning is not None:
        summary["plans_output_warning"] = plans_output_warning
    if trades_output_warning is not None:
        summary["trades_output_warning"] = trades_output_warning

    effective_args_path, effective_args_cli_path, effective_args_warning = _maybe_write_effective_args_artifacts(
        parser,
        args,
        summary_output=args.summary_output,
    )
    if effective_args_path is not None:
        summary["effective_args_path"] = str(effective_args_path)
    if effective_args_cli_path is not None:
        summary["effective_args_cli_path"] = str(effective_args_cli_path)
    if effective_args_warning is not None:
        summary["effective_args_warning"] = effective_args_warning

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

    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = ArgsFileParser(
        description="stockagent2 command suite",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        fromfile_prefix_chars="@",
        epilog="Long commands can be stored in an @args.txt file; lines support shell-style quoting and # comments.",
    )
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
    pipeline_parser.add_argument(
        "--fallback-data-dirs",
        dest="use_fallback_data_dirs",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Search built-in fallback OHLC directories after the primary local data dir.",
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
    pipeline_parser.add_argument(
        "--describe-run",
        action="store_true",
        help="Print the resolved run plan and exit without loading data or simulating trades.",
    )
    pipeline_parser.add_argument("--quiet", action="store_true", help="Suppress stdout summary (use with --summary-output).")

    pipeline_parser.set_defaults(handler=_handle_pipeline_simulation, _parser_ref=pipeline_parser)

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
