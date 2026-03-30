"""Structured reporting helpers for binance_worksteal CLI tools."""
from __future__ import annotations

import argparse
import json
import shlex
import sys
from collections.abc import Mapping, Sequence as SequenceABC
from contextlib import nullcontext, redirect_stdout
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np
import pandas as pd

from binance_worksteal.cli import build_partial_universe_hint, build_run_warnings, build_symbol_selection_cli_args
from binance_worksteal.config_io import (
    build_worksteal_config_override_mapping,
    build_worksteal_config_from_result,
    default_best_config_path,
    default_best_overrides_path,
    write_worksteal_config_file_or_warn,
    write_worksteal_config_overrides_file_or_warn,
)
from binance_worksteal.io_utils import write_text_atomic


SummaryRunner = Callable[[], tuple[int, dict[str, Any] | None]]


def json_ready(value: Any) -> Any:
    if is_dataclass(value):
        return json_ready(asdict(value))
    if isinstance(value, dict):
        return {str(k): json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [json_ready(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, pd.DataFrame):
        return [json_ready(row) for row in value.to_dict(orient="records")]
    if isinstance(value, pd.Series):
        return json_ready(value.to_dict())
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def build_summary_payload(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "summary_schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        **payload,
    }


def add_summary_run_status(payload: dict[str, Any], exit_code: int) -> dict[str, Any]:
    status_payload = dict(payload)
    status_payload.setdefault("exit_code", int(exit_code))
    status_payload.setdefault("status", "success" if exit_code == 0 else "error")
    return status_payload


def build_cli_error_summary(
    *,
    tool: str,
    error: str,
    error_type: str | None = None,
    data_dir: str | None = None,
    config_file: str | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload = {
        "tool": tool,
        "error": error,
        "config_file": config_file,
    }
    if error_type is not None:
        payload["error_type"] = error_type
    if data_dir is not None:
        payload["data_dir"] = data_dir
    if extra:
        payload.update(extra)
    return payload


def build_symbol_run_summary(
    *,
    tool: str,
    data_dir: str,
    symbol_source: str,
    symbols: list[str],
    load_summary: dict[str, Any],
    data_coverage: dict[str, Any] | None = None,
    config_file: str | None,
    config: Any | None = None,
    require_full_universe: bool = False,
) -> dict[str, Any]:
    missing_symbol_count = int(load_summary.get("missing_symbol_count", 0))
    payload = {
        "tool": tool,
        "data_dir": data_dir,
        "symbol_source": symbol_source,
        "requested_symbol_count": len(symbols),
        "symbol_count": len(symbols),
        "symbols": list(symbols),
        "loaded_symbol_count": load_summary.get("loaded_symbol_count", 0),
        "loaded_symbols": load_summary.get("loaded_symbols", []),
        "missing_symbol_count": missing_symbol_count,
        "missing_symbols": load_summary.get("missing_symbols", []),
        "universe_complete": bool(load_summary.get("universe_complete", missing_symbol_count == 0)),
        "require_full_universe": bool(require_full_universe),
        "config_file": config_file,
        "warnings": build_run_warnings(load_summary=load_summary, data_coverage=data_coverage),
    }
    partial_universe_hint = load_summary.get("partial_universe_hint") or build_partial_universe_hint(load_summary)
    if partial_universe_hint:
        payload["partial_universe_hint"] = partial_universe_hint
    strict_retry_command = load_summary.get("strict_retry_command")
    if strict_retry_command:
        payload["strict_retry_command"] = strict_retry_command
    if data_coverage is not None:
        payload["data_coverage"] = data_coverage
    if config is not None:
        payload["config"] = config
    return payload


def render_summary_json(payload: dict[str, Any]) -> str:
    return json.dumps(json_ready(build_summary_payload(payload)), indent=2, sort_keys=True) + "\n"


def summary_json_writes_to_stdout(path: str | Path | None) -> bool:
    return str(path) == "-"


def redirect_non_summary_output(path: str | Path | None):
    if summary_json_writes_to_stdout(path):
        return redirect_stdout(sys.stderr)
    return nullcontext()


def write_summary_json(path: str | Path, payload: dict[str, Any]) -> Path | None:
    body = render_summary_json(payload)
    if str(path) == "-":
        sys.stdout.write(body)
        return None
    return write_text_atomic(path, body, encoding="utf-8")


def safe_write_summary_json(path: str | Path, payload: dict[str, Any]) -> tuple[Path | None, str | None]:
    try:
        return write_summary_json(path, payload), None
    except (OSError, TypeError, ValueError, OverflowError) as exc:
        return None, str(exc)


def write_summary_json_or_warn(
    path: str | Path,
    payload: dict[str, Any],
    *,
    announce_write: bool = True,
) -> Path | None:
    written_path, summary_error = safe_write_summary_json(path, payload)
    if summary_error:
        stream = sys.stderr if summary_json_writes_to_stdout(path) else sys.stdout
        print(f"WARN: failed to write summary JSON to {path}: {summary_error}", file=stream)
        return None
    if summary_json_writes_to_stdout(path):
        return None
    if announce_write:
        print(f"Wrote summary JSON to {written_path}")
    return written_path


def default_sidecar_json_path(output_path: str | Path) -> Path:
    path = Path(output_path)
    if path.suffix:
        return path.with_name(f"{path.stem}.summary.json")
    return path.with_name(f"{path.name}.summary.json")


def default_next_steps_script_path(output_path: str | Path) -> Path:
    path = Path(output_path)
    stem = path.stem if path.suffix else path.name
    return path.with_name(f"{stem}.next_steps.sh")


def add_summary_json_arg(
    parser: argparse.ArgumentParser,
    *,
    default: str | None = None,
    defaults_to_sidecar: bool = False,
) -> None:
    help_text = "Optional path to write a structured JSON run summary."
    if defaults_to_sidecar:
        help_text += " Defaults to a sidecar next to --output."
    help_text += " Use '-' to print JSON to stdout."
    parser.add_argument("--summary-json", default=default, help=help_text)


def add_preview_run_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--preview-run",
        action="store_true",
        help="Print the resolved run plan and exit before loading data or executing the tool.",
    )


def _build_symbol_selection_summary_payload(
    *,
    tool: str,
    symbol_source: str,
    symbols: Sequence[str],
    data_dir: str | None,
    config_file: str | None,
    requested_symbol_count: int | None = None,
) -> dict[str, Any]:
    payload = {
        "tool": tool,
        "symbol_source": symbol_source,
        "requested_symbol_count": len(symbols) if requested_symbol_count is None else int(requested_symbol_count),
        "symbol_count": len(symbols),
        "symbols": list(symbols),
        "config_file": config_file,
    }
    if data_dir is not None:
        payload["data_dir"] = data_dir
    return payload


def build_preview_run_summary(
    *,
    tool: str,
    data_dir: str | None,
    symbol_source: str,
    symbols: Sequence[str],
    config_file: str | None,
    requested_symbol_count: int | None = None,
    config: Any | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload = {
        **_build_symbol_selection_summary_payload(
            tool=tool,
            symbol_source=symbol_source,
            symbols=symbols,
            data_dir=data_dir,
            config_file=config_file,
            requested_symbol_count=requested_symbol_count,
        ),
        "preview_only": True,
    }
    if config is not None:
        payload["config"] = config
    if extra:
        payload.update(extra)
    return payload


def build_symbol_listing_summary(
    *,
    tool: str,
    symbol_source: str,
    symbols: Sequence[str],
    data_dir: str | None = None,
    config_file: str | None = None,
    requested_symbol_count: int | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload = {
        **_build_symbol_selection_summary_payload(
            tool=tool,
            symbol_source=symbol_source,
            symbols=symbols,
            data_dir=data_dir,
            config_file=config_file,
            requested_symbol_count=requested_symbol_count,
        ),
        "list_symbols_only": True,
    }
    if extra:
        payload.update(extra)
    return payload


def _format_preview_value(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, (list, tuple, set)):
        rendered = [str(item) for item in value]
        if not rendered:
            return "-"
        if len(rendered) > 8:
            preview = ", ".join(rendered[:8])
            return f"{preview} (+{len(rendered) - 8} more)"
        return ", ".join(rendered)
    if isinstance(value, dict):
        return json.dumps(json_ready(value), sort_keys=True)
    return str(value)


def print_run_preview(
    *,
    tool: str,
    sections: Sequence[tuple[str, Sequence[tuple[str, Any]]]],
) -> None:
    print(f"{tool} run preview:")
    for title, fields in sections:
        print(f"{title}:")
        for label, value in fields:
            print(f"  {label}: {_format_preview_value(value)}")


def render_command_preview(module: str, argv: Sequence[str]) -> str:
    return shlex.join(["python", "-m", module, *[str(arg) for arg in argv]])


def build_sweep_follow_up_commands(
    *,
    config_file: str,
    data_dir: str,
    symbol_args: Sequence[str],
    start_date: str,
    end_date: str,
    eval_days: int | None = None,
    eval_windows: int | None = None,
    eval_start_date: str | None = None,
    eval_end_date: str | None = None,
) -> list[dict[str, Any]]:
    commands = []

    backtest_argv = [
        "--config-file",
        config_file,
        "--data-dir",
        data_dir,
        *symbol_args,
        "--start",
        start_date,
        "--end",
        end_date,
    ]
    commands.append(
        {
            "name": "backtest",
            "description": "Replay the recommended config over the sweep date span.",
            "module": "binance_worksteal.backtest",
            "argv": backtest_argv,
            "command": render_command_preview("binance_worksteal.backtest", backtest_argv),
        }
    )

    audit_argv = [
        "--config-file",
        config_file,
        "--data-dir",
        data_dir,
        *symbol_args,
        "--start",
        start_date,
        "--end",
        end_date,
    ]
    commands.append(
        {
            "name": "sim_vs_live_audit",
            "description": "Check whether the recommended config survives the parity audit.",
            "module": "binance_worksteal.sim_vs_live_audit",
            "argv": audit_argv,
            "command": render_command_preview("binance_worksteal.sim_vs_live_audit", audit_argv),
        }
    )

    if eval_start_date is not None and eval_end_date is not None:
        evaluate_argv = [
            "--config-file",
            config_file,
            "--data-dir",
            data_dir,
            *symbol_args,
            "--start",
            str(eval_start_date),
            "--end",
            str(eval_end_date),
        ]
        commands.append(
            {
                "name": "evaluate_symbols",
                "description": "Measure which symbols help or hurt the recommended config over the fixed date range.",
                "module": "binance_worksteal.evaluate_symbols",
                "argv": evaluate_argv,
                "command": render_command_preview("binance_worksteal.evaluate_symbols", evaluate_argv),
            }
        )
    elif eval_days is not None and eval_windows is not None:
        evaluate_argv = [
            "--config-file",
            config_file,
            "--data-dir",
            data_dir,
            *symbol_args,
            "--days",
            str(eval_days),
            "--windows",
            str(eval_windows),
        ]
        commands.append(
            {
                "name": "evaluate_symbols",
                "description": "Measure which symbols help or hurt the recommended config.",
                "module": "binance_worksteal.evaluate_symbols",
                "argv": evaluate_argv,
                "command": render_command_preview("binance_worksteal.evaluate_symbols", evaluate_argv),
            }
        )

    return commands


def render_next_steps_script(commands: Sequence[dict[str, Any]]) -> str:
    lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        "",
        "# Generated by binance_worksteal sweep follow-up recommendations.",
        "",
    ]
    for item in commands:
        lines.append(f"# {item['name']}: {item['description']}")
        lines.append(str(item["command"]))
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def write_next_steps_script_or_warn(
    output_csv: str | Path,
    commands: Sequence[dict[str, Any]],
    *,
    announce_write: bool = True,
) -> Path | None:
    if not commands:
        return None
    script_path = default_next_steps_script_path(output_csv)
    try:
        write_text_atomic(
            script_path,
            render_next_steps_script(commands),
            encoding="utf-8",
            mode=0o755,
        )
    except OSError as exc:
        print(f"WARN: failed to write next steps shell script to {script_path}: {exc}")
        return None
    if announce_write:
        print(f"Wrote next steps shell script to {script_path}")
    return script_path


def prepare_sweep_recommendation_artifacts(
    *,
    ranked_results: Sequence[dict[str, Any]],
    base_config: Any,
    swept_fields: Sequence[str],
    output_csv: str | Path,
    data_dir: str,
    symbols_arg: Sequence[str] | None,
    universe_file: str | None,
    start_date: str,
    end_date: str,
    eval_days: int | None = None,
    eval_windows: int | None = None,
    eval_start_date: str | None = None,
    eval_end_date: str | None = None,
) -> dict[str, Any]:
    if not ranked_results:
        return {
            "recommended_config": None,
            "recommended_config_file": None,
            "recommended_overrides": None,
            "recommended_overrides_file": None,
            "follow_up_config_file": None,
            "follow_up_config_kind": None,
            "next_steps_script_file": None,
            "next_steps": [],
        }

    best_config = build_worksteal_config_from_result(
        base_config=base_config,
        result_row=ranked_results[0],
        swept_fields=swept_fields,
    )
    written_best_config_path = write_worksteal_config_file_or_warn(
        default_best_config_path(output_csv),
        best_config,
        announce_write=False,
    )
    best_config_path = str(written_best_config_path) if written_best_config_path is not None else None
    recommended_overrides = write_worksteal_config_overrides_file_or_warn(
        default_best_overrides_path(output_csv),
        best_config,
        announce_write=False,
    )
    recommended_overrides_path = str(recommended_overrides) if recommended_overrides is not None else None
    recommended_overrides_payload = build_worksteal_config_override_mapping(best_config)
    follow_up_config_path = recommended_overrides_path or best_config_path
    follow_up_config_kind = "standalone_overrides" if recommended_overrides_path is not None else (
        "full_config" if best_config_path is not None else None
    )
    follow_up_commands: list[dict[str, Any]] = []
    next_steps_script_path: str | None = None
    if follow_up_config_path is not None:
        symbol_args = build_symbol_selection_cli_args(
            symbols_arg=symbols_arg,
            universe_file=universe_file,
        )
        follow_up_commands = build_sweep_follow_up_commands(
            config_file=follow_up_config_path,
            data_dir=data_dir,
            symbol_args=symbol_args,
            start_date=start_date,
            end_date=end_date,
            eval_days=eval_days,
            eval_windows=eval_windows,
            eval_start_date=eval_start_date,
            eval_end_date=eval_end_date,
        )
        print_follow_up_commands(follow_up_commands)
        written_next_steps_script = write_next_steps_script_or_warn(
            output_csv,
            follow_up_commands,
            announce_write=False,
        )
        next_steps_script_path = str(written_next_steps_script) if written_next_steps_script is not None else None

    return {
        "recommended_config": best_config,
        "recommended_config_file": best_config_path,
        "recommended_overrides": recommended_overrides_payload,
        "recommended_overrides_file": recommended_overrides_path,
        "follow_up_config_file": follow_up_config_path,
        "follow_up_config_kind": follow_up_config_kind,
        "next_steps_script_file": next_steps_script_path,
        "next_steps": follow_up_commands,
    }


def build_summary_json_artifact(path: str | Path | None) -> dict[str, str] | None:
    if path is None or summary_json_writes_to_stdout(path):
        return None
    return {
        "name": "summary_json",
        "path": str(path),
        "description": "Structured JSON run summary.",
    }


def build_sweep_artifact_manifest(
    *,
    recommendation: dict[str, Any],
    output_csv: str | Path | None = None,
    include_output_csv: bool = False,
    summary_json_file: str | Path | None = None,
) -> list[dict[str, str]]:
    artifacts: list[dict[str, str]] = []
    summary_artifact = build_summary_json_artifact(summary_json_file)
    if summary_artifact is not None:
        artifacts.append(summary_artifact)
    if include_output_csv and output_csv is not None:
        artifacts.append(
            {
                "name": "results_csv",
                "path": str(output_csv),
                "description": "Sweep results CSV.",
            }
        )
    artifact_specs = [
        (
            "recommended_config",
            recommendation.get("recommended_config_file"),
            "Full best-config YAML snapshot.",
        ),
        (
            "recommended_overrides",
            recommendation.get("recommended_overrides_file"),
            "Minimal best-config overrides YAML.",
        ),
        (
            "next_steps_script",
            recommendation.get("next_steps_script_file"),
            "Runnable shell script with suggested follow-up commands.",
        ),
    ]
    for name, path_value, description in artifact_specs:
        if not path_value:
            continue
        artifacts.append(
            {
                "name": name,
                "path": str(path_value),
                "description": description,
            }
        )
    return artifacts


def print_artifact_manifest(
    artifacts: Sequence[dict[str, Any]],
    *,
    title: str = "Artifacts",
) -> None:
    if not artifacts:
        return
    print(f"\n{title}:")
    for item in artifacts:
        print(f"  {item['name']}: {item['path']}")


def print_reproduce_command(command: str | None) -> None:
    if not command:
        return
    print("\nReproduce:")
    print(f"  {command}")


def print_warning_summary(warnings: Any) -> None:
    if not isinstance(warnings, SequenceABC) or isinstance(warnings, (str, bytes)):
        return
    lines = [str(item).strip() for item in warnings if str(item).strip()]
    if not lines:
        return
    print("\nWarnings:")
    for line in lines:
        print(f"  - {line}")


def announce_sweep_artifacts(
    *,
    recommendation: dict[str, Any],
    output_csv: str | Path,
    summary_json_file: str | Path | None,
    module: str,
    argv: Sequence[str],
    include_output_csv: bool,
    warnings: Sequence[str] | None = None,
) -> list[dict[str, str]]:
    artifacts = build_sweep_artifact_manifest(
        recommendation=recommendation,
        output_csv=output_csv,
        include_output_csv=include_output_csv,
        summary_json_file=summary_json_file,
    )
    print_artifact_manifest(artifacts, title="Generated artifacts")
    if include_output_csv:
        print_reproduce_command(render_command_preview(module, argv))
    print_warning_summary(warnings)
    return artifacts


def empty_sweep_recommendation() -> dict[str, Any]:
    return {
        "recommended_config": None,
        "recommended_config_file": None,
        "recommended_overrides": None,
        "recommended_overrides_file": None,
        "follow_up_config_file": None,
        "follow_up_config_kind": None,
        "next_steps_script_file": None,
        "next_steps": [],
    }


def build_sweep_run_summary(
    *,
    tool: str,
    data_dir: str,
    symbol_source: str,
    symbols: Sequence[str],
    load_summary: dict[str, Any],
    data_coverage: dict[str, Any] | None,
    config_file: str | None,
    base_config: Any,
    output_csv: str | Path,
    swept_fields: Sequence[str],
    windows: Sequence[tuple[str, str]],
    results_count: int,
    recommendation: dict[str, Any],
    best_result: dict[str, Any] | None,
    top_results: Sequence[dict[str, Any]],
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload = {
        **build_symbol_run_summary(
            tool=tool,
            data_dir=data_dir,
            symbol_source=symbol_source,
            symbols=list(symbols),
            load_summary=load_summary,
            data_coverage=data_coverage,
            config_file=config_file,
            config=base_config,
        ),
        "output_csv": str(output_csv),
        "base_config": base_config,
        "swept_fields": list(swept_fields),
        "windows": [{"start_date": start, "end_date": end} for start, end in windows],
        "results_count": int(results_count),
        "recommended_config_file": recommendation["recommended_config_file"],
        "recommended_config": recommendation["recommended_config"],
        "recommended_overrides_file": recommendation["recommended_overrides_file"],
        "recommended_overrides": recommendation["recommended_overrides"],
        "follow_up_config_file": recommendation["follow_up_config_file"],
        "follow_up_config_kind": recommendation["follow_up_config_kind"],
        "next_steps_script_file": recommendation["next_steps_script_file"],
        "next_steps": recommendation["next_steps"],
        "artifacts": build_sweep_artifact_manifest(
            recommendation=recommendation,
            output_csv=output_csv,
            include_output_csv=results_count > 0,
        ),
        "best_result": best_result,
        "top_results": list(top_results),
    }
    if extra:
        payload.update(extra)
    return payload


def build_empty_sweep_run_summary(
    *,
    tool: str,
    data_dir: str,
    symbol_source: str,
    symbols: Sequence[str],
    load_summary: dict[str, Any],
    data_coverage: dict[str, Any] | None,
    config_file: str | None,
    base_config: Any,
    output_csv: str | Path,
    swept_fields: Sequence[str],
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return build_sweep_run_summary(
        tool=tool,
        data_dir=data_dir,
        symbol_source=symbol_source,
        symbols=symbols,
        load_summary=load_summary,
        data_coverage=data_coverage,
        config_file=config_file,
        base_config=base_config,
        output_csv=output_csv,
        swept_fields=swept_fields,
        windows=[],
        results_count=0,
        recommendation=empty_sweep_recommendation(),
        best_result=None,
        top_results=[],
        extra=extra,
    )


def print_follow_up_commands(commands: Sequence[dict[str, Any]]) -> None:
    if not commands:
        return
    print("\nNext steps:")
    for item in commands:
        print(f"  {item['name']}: {item['description']}")
        print(f"    {item['command']}")


def _build_invocation_summary(*, module: str, argv: Sequence[str]) -> dict[str, Any]:
    rendered_argv = [str(arg) for arg in argv]
    return {
        "module": module,
        "argv": rendered_argv,
        "command": render_command_preview(module, rendered_argv),
        "cwd": str(Path.cwd()),
    }


def _build_summary_contract_error(*, module: str | None, message: str, error_type: str) -> dict[str, Any]:
    return build_cli_error_summary(
        tool=module.rsplit(".", 1)[-1] if module else "cli",
        error=message,
        error_type=error_type,
    )


def _prepare_summary_payload(
    payload: Mapping[str, Any],
    *,
    summary_path: str | Path | None,
    module: str | None,
    argv: Sequence[str] | None,
) -> dict[str, Any]:
    prepared_payload = dict(payload)
    if module is not None and argv is not None:
        prepared_payload = {
            **prepared_payload,
            "invocation": _build_invocation_summary(module=module, argv=argv),
        }
    summary_artifact = build_summary_json_artifact(summary_path)
    if summary_artifact is not None:
        artifacts = prepared_payload.get("artifacts")
        if isinstance(artifacts, list):
            if not any(
                isinstance(item, dict) and item.get("name") == "summary_json" for item in artifacts
            ):
                prepared_payload = {
                    **prepared_payload,
                    "artifacts": [summary_artifact, *artifacts],
                }
        else:
            prepared_payload = {
                **prepared_payload,
                "artifacts": [summary_artifact],
            }
        prepared_payload = {
            **prepared_payload,
            "summary_json_file": str(summary_path),
        }
    return prepared_payload


def run_with_optional_summary(
    summary_path: str | Path | None,
    runner: SummaryRunner,
    *,
    module: str | None = None,
    argv: Sequence[str] | None = None,
    announce_summary_write_on_error: bool = True,
    announce_summary_write_on_success: bool = True,
    announce_artifact_manifest_on_success: bool = False,
) -> int:
    try:
        with redirect_non_summary_output(summary_path):
            rc, payload = runner()
    except Exception as exc:
        message = str(exc).strip() or exc.__class__.__name__
        payload = _build_summary_contract_error(
            module=module,
            message=f"ERROR: {message}",
            error_type=exc.__class__.__name__,
        )
        rc = 1
        with redirect_non_summary_output(summary_path):
            print(payload["error"])
    if summary_path and payload is not None:
        if not isinstance(payload, Mapping):
            payload = _build_summary_contract_error(
                module=module,
                message=(
                    f"ERROR: runner returned {type(payload).__name__} summary payload, expected a mapping"
                ),
                error_type="TypeError",
            )
            rc = 1
            with redirect_non_summary_output(summary_path):
                print(payload["error"])
        payload = _prepare_summary_payload(
            payload,
            summary_path=summary_path,
            module=module,
            argv=argv,
        )
        final_payload = add_summary_run_status(payload, rc)
        announce_error_manifest = rc != 0 and announce_summary_write_on_error
        announce_write = False if announce_error_manifest else (
            announce_summary_write_on_error
            if rc != 0
            else announce_summary_write_on_success and not announce_artifact_manifest_on_success
        )
        written_path = write_summary_json_or_warn(
            summary_path,
            final_payload,
            announce_write=announce_write,
        )
        if written_path is not None and (
            (rc == 0 and announce_artifact_manifest_on_success) or announce_error_manifest
        ):
            artifacts = final_payload.get("artifacts")
            if isinstance(artifacts, list):
                print_artifact_manifest(
                    artifacts,
                    title="Generated artifacts" if rc == 0 else "Diagnostic artifacts",
                )
            invocation = final_payload.get("invocation")
            command = invocation.get("command") if isinstance(invocation, Mapping) else None
            print_reproduce_command(command)
            print_warning_summary(final_payload.get("warnings"))
    return rc
