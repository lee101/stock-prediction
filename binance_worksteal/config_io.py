"""Config file helpers for binance_worksteal."""
from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, fields, replace
from pathlib import Path
from typing import Any

import yaml

from binance_worksteal.io_utils import write_text_atomic
from binance_worksteal.strategy import WorkStealConfig


_WORKSTEAL_CONFIG_FIELDS = {field.name for field in fields(WorkStealConfig)}
_WORKSTEAL_CONFIG_FIELD_ORDER = [field.name for field in fields(WorkStealConfig)]
ConfigFlagMappingValue = str | tuple[str, str]
ConfigExplanationAdjuster = Callable[[dict[str, Any]], None]


def _config_yaml_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _config_yaml_ready(inner) for key, inner in value.items()}
    if isinstance(value, tuple):
        return [_config_yaml_ready(inner) for inner in value]
    if isinstance(value, list):
        return [_config_yaml_ready(inner) for inner in value]
    return value


def _normalize_worksteal_overrides(overrides: Mapping[str, Any]) -> dict[str, Any]:
    normalized = dict(overrides)
    if "dip_pct_fallback" in normalized and normalized["dip_pct_fallback"] is not None:
        normalized["dip_pct_fallback"] = tuple(normalized["dip_pct_fallback"])
    return normalized


def load_worksteal_config_overrides(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    try:
        with config_path.open("r", encoding="utf-8") as handle:
            payload = yaml.safe_load(handle)
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"Config file not found: {config_path}") from exc
    except yaml.YAMLError as exc:
        raise ValueError(f"Invalid config file format: {config_path}") from exc

    if payload is None:
        raise ValueError(f"Config file is empty: {config_path}")
    if not isinstance(payload, dict):
        raise ValueError(f"Config file must contain a mapping: {config_path}")

    raw_overrides = payload.get("config", payload)
    if not isinstance(raw_overrides, dict):
        raise ValueError(f"Config file 'config' value must be a mapping: {config_path}")

    unknown = sorted(set(raw_overrides) - _WORKSTEAL_CONFIG_FIELDS)
    if unknown:
        joined = ", ".join(unknown)
        raise ValueError(f"Unsupported WorkStealConfig fields in {config_path}: {joined}")
    return dict(raw_overrides)


def apply_worksteal_config_overrides(
    base_config: WorkStealConfig,
    overrides: dict[str, Any],
) -> WorkStealConfig:
    if not overrides:
        return base_config
    return replace(base_config, **_normalize_worksteal_overrides(overrides))


def load_worksteal_config(
    path: str | Path,
    *,
    base_config: WorkStealConfig | None = None,
) -> WorkStealConfig:
    config = base_config or WorkStealConfig()
    overrides = load_worksteal_config_overrides(path)
    return apply_worksteal_config_overrides(config, overrides)


def add_config_file_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--config-file",
        default=None,
        help="Optional YAML/JSON config file with WorkStealConfig fields under top-level keys or a 'config' mapping.",
    )


def add_print_config_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--print-config",
        action="store_true",
        help="Print the resolved WorkStealConfig as YAML after applying defaults, --config-file, and CLI overrides, then exit.",
    )


def add_explain_config_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--explain-config",
        action="store_true",
        help="Print the resolved WorkStealConfig plus source/provenance for each override, then exit.",
    )


def render_worksteal_config_yaml(config: WorkStealConfig) -> str:
    payload = {"config": _config_yaml_ready(asdict(config))}
    return yaml.safe_dump(payload, sort_keys=True)


def build_worksteal_config_override_mapping(
    config: WorkStealConfig,
    *,
    base_config: WorkStealConfig | None = None,
) -> dict[str, Any]:
    baseline = asdict(base_config or WorkStealConfig())
    effective = asdict(config)
    overrides = {
        field_name: _config_yaml_ready(effective[field_name])
        for field_name in _WORKSTEAL_CONFIG_FIELD_ORDER
        if effective[field_name] != baseline[field_name]
    }
    return overrides


def render_worksteal_config_overrides_yaml(
    config: WorkStealConfig,
    *,
    base_config: WorkStealConfig | None = None,
) -> str:
    payload = {
        "config": build_worksteal_config_override_mapping(
            config,
            base_config=base_config,
        )
    }
    return yaml.safe_dump(payload, sort_keys=True)


def print_worksteal_config(config: WorkStealConfig) -> None:
    print(render_worksteal_config_yaml(config), end="")


def build_worksteal_config_from_result(
    *,
    base_config: WorkStealConfig,
    result_row: Mapping[str, Any],
    swept_fields: Sequence[str],
) -> WorkStealConfig:
    overrides = {
        field_name: result_row[field_name]
        for field_name in swept_fields
        if field_name in result_row
    }
    return replace(base_config, **_normalize_worksteal_overrides(overrides))


def default_best_config_path(output_path: str | Path) -> Path:
    path = Path(output_path)
    if path.suffix:
        return path.with_name(f"{path.stem}.best_config.yaml")
    return path.with_name(f"{path.name}.best_config.yaml")


def default_best_overrides_path(output_path: str | Path) -> Path:
    path = Path(output_path)
    if path.suffix:
        return path.with_name(f"{path.stem}.best_overrides.yaml")
    return path.with_name(f"{path.name}.best_overrides.yaml")


def write_worksteal_config_file(path: str | Path, config: WorkStealConfig) -> Path:
    return write_text_atomic(path, render_worksteal_config_yaml(config), encoding="utf-8")


def write_worksteal_config_overrides_file(
    path: str | Path,
    config: WorkStealConfig,
    *,
    base_config: WorkStealConfig | None = None,
) -> Path:
    return write_text_atomic(
        path,
        render_worksteal_config_overrides_yaml(config, base_config=base_config),
        encoding="utf-8",
    )


def write_worksteal_config_file_or_warn(
    path: str | Path,
    config: WorkStealConfig,
    *,
    announce_write: bool = True,
) -> Path | None:
    try:
        written_path = write_worksteal_config_file(path, config)
    except (OSError, TypeError, ValueError) as exc:
        print(f"WARN: failed to write recommended config YAML to {path}: {exc}")
        return None
    if announce_write:
        print(f"Wrote recommended config YAML to {written_path}")
    return written_path


def write_worksteal_config_overrides_file_or_warn(
    path: str | Path,
    config: WorkStealConfig,
    *,
    base_config: WorkStealConfig | None = None,
    announce_write: bool = True,
) -> Path | None:
    try:
        written_path = write_worksteal_config_overrides_file(
            path,
            config,
            base_config=base_config,
        )
    except (OSError, TypeError, ValueError) as exc:
        print(f"WARN: failed to write recommended overrides YAML to {path}: {exc}")
        return None
    if announce_write:
        print(f"Wrote recommended overrides YAML to {written_path}")
    return written_path


def argv_has_flag(raw_argv: Sequence[str], flag: str) -> bool:
    return any(arg == flag or str(arg).startswith(f"{flag}=") for arg in raw_argv)


def collect_worksteal_cli_overrides(
    args: argparse.Namespace,
    raw_argv: Sequence[str],
    flag_to_field: Mapping[str, ConfigFlagMappingValue],
) -> dict[str, Any]:
    overrides: dict[str, Any] = {}
    for flag, target in flag_to_field.items():
        if not argv_has_flag(raw_argv, flag):
            continue
        if isinstance(target, tuple):
            field_name, arg_name = target
        else:
            field_name = target
            arg_name = flag.lstrip("-").replace("-", "_")
        overrides[field_name] = getattr(args, arg_name)
    return _normalize_worksteal_overrides(overrides)


def resolve_worksteal_config_inputs(
    *,
    base_config: WorkStealConfig,
    config_file: str | Path | None,
    args: argparse.Namespace,
    raw_argv: Sequence[str] | None,
    flag_to_field: Mapping[str, ConfigFlagMappingValue],
) -> tuple[WorkStealConfig, dict[str, Any], dict[str, Any]]:
    config_file_overrides: dict[str, Any] = {}
    cli_overrides: dict[str, Any] = {}
    config = base_config
    if config_file:
        config_file_overrides = _normalize_worksteal_overrides(load_worksteal_config_overrides(config_file))
        config = apply_worksteal_config_overrides(config, config_file_overrides)
    if raw_argv is None:
        return config, config_file_overrides, cli_overrides
    cli_overrides = collect_worksteal_cli_overrides(args, raw_argv, flag_to_field)
    config = apply_worksteal_config_overrides(config, cli_overrides)
    return config, config_file_overrides, cli_overrides


def build_worksteal_config_explanation(
    *,
    base_config: WorkStealConfig,
    config_file: str | Path | None,
    args: argparse.Namespace,
    raw_argv: Sequence[str] | None,
    flag_to_field: Mapping[str, ConfigFlagMappingValue],
) -> dict[str, Any]:
    config, config_file_overrides, cli_overrides = resolve_worksteal_config_inputs(
        base_config=base_config,
        config_file=config_file,
        args=args,
        raw_argv=raw_argv,
        flag_to_field=flag_to_field,
    )
    base_values = asdict(base_config)
    effective_values = asdict(config)

    changed_fields: dict[str, Any] = {}
    sources: dict[str, str] = {}
    for field_name in _WORKSTEAL_CONFIG_FIELD_ORDER:
        if field_name in cli_overrides:
            source = "cli"
        elif field_name in config_file_overrides:
            source = "config_file"
        else:
            source = "default"
        sources[field_name] = source
        if source == "default" and effective_values[field_name] == base_values[field_name]:
            continue
        change = {
            "source": source,
            "value": _config_yaml_ready(effective_values[field_name]),
            "default": _config_yaml_ready(base_values[field_name]),
        }
        if field_name in config_file_overrides:
            change["config_file_value"] = _config_yaml_ready(config_file_overrides[field_name])
        if field_name in cli_overrides:
            change["cli_value"] = _config_yaml_ready(cli_overrides[field_name])
        changed_fields[field_name] = change

    return {
        "config": _config_yaml_ready(effective_values),
        "changed_fields": changed_fields,
        "sources": sources,
        "config_file": str(config_file) if config_file else None,
        "config_file_overrides": _config_yaml_ready(config_file_overrides),
        "cli_overrides": _config_yaml_ready(cli_overrides),
    }


def render_worksteal_config_explanation_yaml(explanation: dict[str, Any]) -> str:
    return yaml.safe_dump(_config_yaml_ready(explanation), sort_keys=False)


def print_worksteal_config_explanation(explanation: dict[str, Any]) -> None:
    print(render_worksteal_config_explanation_yaml(explanation), end="")


def build_worksteal_config_from_args(
    *,
    base_config: WorkStealConfig,
    config_file: str | Path | None,
    args: argparse.Namespace,
    raw_argv: Sequence[str] | None,
    flag_to_field: Mapping[str, ConfigFlagMappingValue],
) -> WorkStealConfig:
    config, _, _ = resolve_worksteal_config_inputs(
        base_config=base_config,
        config_file=config_file,
        args=args,
        raw_argv=raw_argv,
        flag_to_field=flag_to_field,
    )
    return config


def maybe_handle_worksteal_config_output(
    *,
    args: argparse.Namespace,
    build_config: Callable[[], WorkStealConfig],
    base_config: WorkStealConfig,
    config_file: str | Path | None,
    raw_argv: Sequence[str] | None,
    flag_to_field: Mapping[str, ConfigFlagMappingValue],
    explain_adjuster: ConfigExplanationAdjuster | None = None,
) -> int | None:
    if not (getattr(args, "print_config", False) or getattr(args, "explain_config", False)):
        return None

    try:
        config = build_config()
    except (FileNotFoundError, OSError, ValueError) as exc:
        print(f"ERROR: {exc}")
        return 1

    if getattr(args, "explain_config", False):
        explanation = build_worksteal_config_explanation(
            base_config=base_config,
            config_file=config_file,
            args=args,
            raw_argv=raw_argv,
            flag_to_field=flag_to_field,
        )
        if explain_adjuster is not None:
            explain_adjuster(explanation)
        print_worksteal_config_explanation(explanation)
        return 0

    print_worksteal_config(config)
    return 0
