"""Shared CLI helpers for binance_worksteal entrypoints."""
from __future__ import annotations

import argparse
import shlex
from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import Any

import pandas as pd

from binance_worksteal.universe import get_symbols, load_universe


def _normalize_symbols(symbols: Iterable[str]) -> list[str]:
    normalized: list[str] = []
    seen: set[str] = set()
    for symbol in symbols:
        value = str(symbol).strip().upper()
        if not value or value in seen:
            continue
        seen.add(value)
        normalized.append(value)
    return normalized


def add_symbol_selection_args(
    parser: argparse.ArgumentParser,
    *,
    symbols_help: str = "Override symbol list (takes precedence over universe file).",
    universe_help: str = "YAML universe config file.",
    include_list_symbols: bool = True,
) -> None:
    parser.add_argument("--universe-file", default=None, help=universe_help)
    parser.add_argument("--symbols", nargs="+", default=None, help=symbols_help)
    if include_list_symbols:
        parser.add_argument(
            "--list-symbols",
            action="store_true",
            help="Print the resolved symbol list and exit.",
        )


def add_require_full_universe_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--require-full-universe",
        "--strict-symbols",
        dest="require_full_universe",
        action="store_true",
        help="Fail if any requested symbols are missing data instead of continuing with a partial universe.",
    )


def add_date_range_args(
    parser: argparse.ArgumentParser,
    *,
    start_dest: str = "start_date",
    end_dest: str = "end_date",
    include_days: bool = False,
    days_default: int = 30,
    days_help: str = "Use the last N days when no explicit start/end range is provided.",
) -> None:
    parser.add_argument("--start-date", "--start", dest=start_dest, default=None)
    parser.add_argument("--end-date", "--end", dest=end_dest, default=None)
    if include_days:
        parser.add_argument("--days", type=int, default=days_default, help=days_help)


def resolve_cli_symbols(
    *,
    symbols_arg: Sequence[str] | None,
    universe_file: str | None,
    default_symbols: Iterable[str],
) -> tuple[list[str], str]:
    if symbols_arg:
        symbols = _normalize_symbols(symbols_arg)
        if not symbols:
            raise ValueError("No valid symbols were provided via --symbols.")
        return symbols, "command line --symbols"
    if universe_file:
        universe = load_universe(universe_file)
        symbols = _normalize_symbols(get_symbols(universe))
        if not symbols:
            raise ValueError(f"Universe file {universe_file} did not resolve to any symbols.")
        return symbols, f"universe file {universe_file}"
    symbols = _normalize_symbols(default_symbols)
    if not symbols:
        raise ValueError("No default symbols are configured.")
    return symbols, "built-in default universe"


def build_cli_error(exc: Exception) -> dict[str, str]:
    message = str(exc).strip() or exc.__class__.__name__
    return {
        "error": f"ERROR: {message}",
        "error_type": exc.__class__.__name__,
    }


def resolve_cli_symbols_with_error(
    *,
    symbols_arg: Sequence[str] | None,
    universe_file: str | None,
    default_symbols: Iterable[str],
) -> tuple[tuple[list[str], str] | None, dict[str, str] | None]:
    try:
        resolved = resolve_cli_symbols(
            symbols_arg=symbols_arg,
            universe_file=universe_file,
            default_symbols=default_symbols,
        )
    except (FileNotFoundError, OSError, ValueError) as exc:
        return None, build_cli_error(exc)
    return resolved, None


def resolve_cli_symbols_or_print_error(
    *,
    symbols_arg: Sequence[str] | None,
    universe_file: str | None,
    default_symbols: Iterable[str],
) -> tuple[list[str], str] | None:
    resolved, error = resolve_cli_symbols_with_error(
        symbols_arg=symbols_arg,
        universe_file=universe_file,
        default_symbols=default_symbols,
    )
    if error is not None:
        print(error["error"])
        return None
    return resolved


def print_resolved_symbols(symbols: Sequence[str], source: str) -> None:
    print(f"Resolved {len(symbols)} symbols from {source}:")
    for symbol in symbols:
        print(symbol)


def build_symbol_selection_cli_args(
    *,
    symbols_arg: Sequence[str] | None,
    universe_file: str | None,
) -> list[str]:
    if symbols_arg:
        symbols = _normalize_symbols(symbols_arg)
        if not symbols:
            return []
        return ["--symbols", *symbols]
    if universe_file:
        return ["--universe-file", str(universe_file)]
    return []


def validate_date_range_with_error(
    *,
    start_date: str | None,
    end_date: str | None,
    require_pair: bool = False,
) -> tuple[tuple[str | None, str | None] | None, dict[str, str] | None]:
    if require_pair and bool(start_date) != bool(end_date):
        return None, {
            "error": "ERROR: --start/--start-date and --end/--end-date must be provided together.",
            "error_type": "ValueError",
        }

    try:
        start_ts = _parse_cli_date(start_date, flag_label="--start/--start-date") if start_date else None
        end_ts = _parse_cli_date(end_date, flag_label="--end/--end-date") if end_date else None
    except ValueError as exc:
        return None, build_cli_error(exc)

    if start_ts is not None and end_ts is not None and start_ts > end_ts:
        return None, {
            "error": "ERROR: --start/--start-date must be on or before --end/--end-date.",
            "error_type": "ValueError",
        }

    return (start_date, end_date), None


def resolve_paired_date_range_with_error(
    *,
    start_date: str | None,
    end_date: str | None,
) -> tuple[tuple[str, str] | None, dict[str, str] | None]:
    validated, error = validate_date_range_with_error(
        start_date=start_date,
        end_date=end_date,
        require_pair=True,
    )
    if error is not None:
        return None, error
    if validated is None or validated == (None, None):
        return None, None
    validated_start, validated_end = validated
    if validated_start is None or validated_end is None:
        return None, None
    return (validated_start, validated_end), None


def resolve_paired_date_range_or_print_error(
    *,
    start_date: str | None,
    end_date: str | None,
) -> tuple[str, str] | None:
    validated, error = resolve_paired_date_range_with_error(
        start_date=start_date,
        end_date=end_date,
    )
    if error is not None:
        print(error["error"])
        return None
    return validated


def _parse_cli_date(value: str, *, flag_label: str) -> pd.Timestamp:
    try:
        parsed = pd.Timestamp(value, tz="UTC")
    except Exception as exc:
        raise ValueError(f"Invalid {flag_label} value: {value!r}") from exc
    if pd.isna(parsed):
        raise ValueError(f"Invalid {flag_label} value: {value!r}")
    return parsed


def validate_date_range_or_print_error(
    *,
    start_date: str | None,
    end_date: str | None,
    require_pair: bool = False,
) -> tuple[str | None, str | None] | None:
    validated, error = validate_date_range_with_error(
        start_date=start_date,
        end_date=end_date,
        require_pair=require_pair,
    )
    if error is not None:
        print(error["error"])
        return None
    return validated


def summarize_loaded_symbols(
    requested_symbols: Sequence[str],
    loaded_symbols: Iterable[str],
) -> dict[str, object]:
    requested = _normalize_symbols(requested_symbols)
    loaded_set = set(_normalize_symbols(loaded_symbols))
    requested_set = set(requested)
    loaded = [symbol for symbol in requested if symbol in loaded_set]
    missing = [symbol for symbol in requested if symbol not in loaded_set]
    unexpected = [symbol for symbol in _normalize_symbols(loaded_symbols) if symbol not in requested_set]
    summary = {
        "requested_symbol_count": len(requested),
        "loaded_symbol_count": len(loaded),
        "missing_symbol_count": len(missing),
        "requested_symbols": requested,
        "loaded_symbols": loaded,
        "missing_symbols": missing,
        "unexpected_loaded_symbols": unexpected,
        "universe_complete": len(missing) == 0,
    }
    partial_universe_hint = build_partial_universe_hint(summary)
    if partial_universe_hint:
        summary["partial_universe_hint"] = partial_universe_hint
    return summary


def _format_missing_symbol_preview(
    missing_symbols: Sequence[str],
    *,
    max_missing_preview: int = 10,
) -> tuple[str, str]:
    preview = ", ".join(missing_symbols[:max_missing_preview])
    remaining = len(missing_symbols) - min(len(missing_symbols), max_missing_preview)
    suffix = f" (+{remaining} more)" if remaining > 0 else ""
    return preview, suffix


def build_load_summary_warnings(
    load_summary: Mapping[str, object] | None,
    *,
    max_missing_preview: int = 10,
) -> list[str]:
    if not load_summary:
        return []
    missing = list(load_summary.get("missing_symbols", []))
    if not missing:
        return []
    preview, suffix = _format_missing_symbol_preview(missing, max_missing_preview=max_missing_preview)
    noun = "symbol" if len(missing) == 1 else "symbols"
    return [f"missing data for {len(missing)} {noun}: {preview}{suffix}"]


def build_data_coverage_warnings(
    data_coverage: Mapping[str, object] | None,
    *,
    max_symbol_preview: int = 10,
) -> list[str]:
    if not data_coverage:
        return []
    start_date = data_coverage.get("coverage_start_date")
    end_date = data_coverage.get("coverage_end_date")
    if not start_date or not end_date:
        return []
    range_label = str(data_coverage.get("coverage_range_label") or "requested range")
    warnings: list[str] = []

    missing_start = list(data_coverage.get("symbols_missing_range_start", []))
    if missing_start:
        preview, suffix = _format_missing_symbol_preview(missing_start, max_missing_preview=max_symbol_preview)
        noun = "symbol starts" if len(missing_start) == 1 else "symbols start"
        warnings.append(f"{len(missing_start)} {noun} after {range_label} start: {preview}{suffix}")

    missing_end = list(data_coverage.get("symbols_missing_range_end", []))
    if missing_end:
        preview, suffix = _format_missing_symbol_preview(missing_end, max_missing_preview=max_symbol_preview)
        noun = "symbol ends" if len(missing_end) == 1 else "symbols end"
        warnings.append(f"{len(missing_end)} {noun} before {range_label} end: {preview}{suffix}")

    return warnings


def build_run_warnings(
    *,
    load_summary: Mapping[str, object] | None = None,
    data_coverage: Mapping[str, object] | None = None,
) -> list[str]:
    return [
        *build_load_summary_warnings(load_summary),
        *build_data_coverage_warnings(data_coverage),
    ]


def build_require_full_universe_error(
    load_summary: dict[str, object],
    *,
    flag_name: str = "--require-full-universe",
    max_missing_preview: int = 10,
) -> str | None:
    missing = list(load_summary.get("missing_symbols", []))
    if not missing:
        return None
    preview, suffix = _format_missing_symbol_preview(missing, max_missing_preview=max_missing_preview)
    noun = "symbol" if len(missing) == 1 else "symbols"
    return f"ERROR: {flag_name} found missing data for {len(missing)} {noun}: {preview}{suffix}"


def build_strict_retry_command(
    *,
    module: str,
    argv: Sequence[str],
    flag_name: str = "--require-full-universe",
) -> str:
    rendered_argv = [str(arg) for arg in argv]
    if flag_name not in rendered_argv and "--strict-symbols" not in rendered_argv:
        rendered_argv.append(flag_name)
    return shlex.join(["python", "-m", module, *rendered_argv])


def build_partial_universe_hint(
    load_summary: dict[str, object],
    *,
    flag_name: str = "--require-full-universe",
    strict_retry_command: str | None = None,
) -> str | None:
    missing = list(load_summary.get("missing_symbols", []))
    loaded_count = int(load_summary.get("loaded_symbol_count", 0))
    if not missing or loaded_count <= 0:
        return None
    hint = f"Hint: rerun with {flag_name} to fail instead of continuing with a partial universe."
    if strict_retry_command:
        return hint + f"\nRetry command: {strict_retry_command}"
    return hint


def require_full_universe_or_print_error(
    *,
    require_full_universe: bool,
    load_summary: dict[str, object],
    flag_name: str = "--require-full-universe",
    max_missing_preview: int = 10,
) -> str | None:
    if not require_full_universe:
        return None
    error = build_require_full_universe_error(
        load_summary,
        flag_name=flag_name,
        max_missing_preview=max_missing_preview,
    )
    if error:
        print(error)
    return error


def print_loaded_symbol_summary(
    requested_symbols: Sequence[str],
    loaded_symbols: Iterable[str],
    *,
    max_missing_preview: int = 10,
    strict_retry_command: str | None = None,
) -> dict[str, object]:
    summary = summarize_loaded_symbols(requested_symbols, loaded_symbols)
    if strict_retry_command and summary.get("missing_symbols") and summary.get("loaded_symbol_count", 0):
        summary["strict_retry_command"] = strict_retry_command
        summary["partial_universe_hint"] = build_partial_universe_hint(
            summary,
            strict_retry_command=strict_retry_command,
        )
    requested_count = int(summary["requested_symbol_count"])
    loaded_count = int(summary["loaded_symbol_count"])
    missing = list(summary["missing_symbols"])

    print(f"Loaded {loaded_count}/{requested_count} symbols with data")
    if missing:
        for warning in build_load_summary_warnings(summary, max_missing_preview=max_missing_preview):
            print(f"WARN: {warning}")
        partial_universe_hint = summary.get("partial_universe_hint")
        if partial_universe_hint:
            print(partial_universe_hint)
    return summary


def build_load_bars_failure(exc: Exception) -> dict[str, str]:
    return build_cli_error(exc)


def load_bars_with_summary(
    *,
    data_dir: str,
    requested_symbols: Sequence[str],
    load_bars: Callable[[str, Sequence[str]], dict[str, Any]],
    loading_message: str | None = None,
    no_data_message: str | None = "ERROR: No data loaded",
    return_summary_on_empty: bool = False,
    return_failure_on_error: bool = False,
    strict_retry_command: str | None = None,
) -> tuple[dict[str, Any], dict[str, object]] | tuple[dict[str, Any], dict[str, object], dict[str, str] | None] | None:
    def build_failure_result(exc: Exception):
        load_failure = build_load_bars_failure(exc)
        print(load_failure["error"])
        if return_failure_on_error:
            failure_summary = summarize_loaded_symbols(requested_symbols, ())
            if strict_retry_command:
                failure_summary["strict_retry_command"] = strict_retry_command
            return {}, failure_summary, load_failure
        return None

    if loading_message:
        print(loading_message)
    try:
        loaded_bars = load_bars(data_dir, requested_symbols)
    except Exception as exc:
        return build_failure_result(exc)
    if not isinstance(loaded_bars, Mapping):
        return build_failure_result(
            TypeError(
                f"load_bars returned {type(loaded_bars).__name__}, expected a symbol->bars mapping"
            )
        )
    load_summary = print_loaded_symbol_summary(
        requested_symbols,
        loaded_bars,
        strict_retry_command=strict_retry_command,
    )
    if not loaded_bars:
        if no_data_message:
            print(no_data_message)
        if return_summary_on_empty:
            if return_failure_on_error:
                return loaded_bars, load_summary, None
            return loaded_bars, load_summary
        return None
    if return_failure_on_error and return_summary_on_empty:
        return loaded_bars, load_summary, None
    return loaded_bars, load_summary


def summarize_data_coverage(
    loaded_bars: dict[str, object],
    *,
    start_date: str | None = None,
    end_date: str | None = None,
) -> dict[str, object]:
    if not loaded_bars:
        return {
            "loaded_symbol_count": 0,
            "overall_start_date": None,
            "overall_end_date": None,
            "symbols_covering_range_count": 0,
            "symbols_missing_range_start_count": 0,
            "symbols_missing_range_end_count": 0,
            "symbols_missing_range_start": [],
            "symbols_missing_range_end": [],
        }

    requested_start = pd.Timestamp(start_date, tz="UTC") if start_date else None
    requested_end = pd.Timestamp(end_date, tz="UTC") if end_date else None
    symbol_ranges: dict[str, tuple[pd.Timestamp, pd.Timestamp]] = {}

    for symbol, bars in loaded_bars.items():
        if not hasattr(bars, "__getitem__"):
            continue
        try:
            timestamps = pd.to_datetime(bars["timestamp"], utc=True, errors="coerce")
        except Exception:
            continue
        timestamps = timestamps.dropna()
        if len(timestamps) == 0:
            continue
        symbol_ranges[str(symbol)] = (timestamps.min(), timestamps.max())

    if not symbol_ranges:
        return {
            "loaded_symbol_count": 0,
            "overall_start_date": None,
            "overall_end_date": None,
            "symbols_covering_range_count": 0,
            "symbols_missing_range_start_count": 0,
            "symbols_missing_range_end_count": 0,
            "symbols_missing_range_start": [],
            "symbols_missing_range_end": [],
        }

    overall_start = min(start for start, _end in symbol_ranges.values())
    overall_end = max(end for _start, end in symbol_ranges.values())
    late_start_symbols = []
    early_end_symbols = []
    fully_covering = 0
    for symbol, (symbol_start, symbol_end) in symbol_ranges.items():
        starts_ok = requested_start is None or symbol_start <= requested_start
        ends_ok = requested_end is None or symbol_end >= requested_end
        if starts_ok and ends_ok:
            fully_covering += 1
        else:
            if not starts_ok:
                late_start_symbols.append(symbol)
            if not ends_ok:
                early_end_symbols.append(symbol)

    return {
        "loaded_symbol_count": len(symbol_ranges),
        "overall_start_date": overall_start.date().isoformat(),
        "overall_end_date": overall_end.date().isoformat(),
        "symbols_covering_range_count": fully_covering,
        "symbols_missing_range_start_count": len(late_start_symbols),
        "symbols_missing_range_end_count": len(early_end_symbols),
        "symbols_missing_range_start": sorted(late_start_symbols),
        "symbols_missing_range_end": sorted(early_end_symbols),
    }


def print_data_coverage_summary(
    loaded_bars: dict[str, object],
    *,
    start_date: str | None = None,
    end_date: str | None = None,
    range_label: str = "requested range",
    max_symbol_preview: int = 10,
) -> dict[str, object]:
    summary = summarize_data_coverage(
        loaded_bars,
        start_date=start_date,
        end_date=end_date,
    )
    summary["coverage_range_label"] = range_label
    summary["coverage_start_date"] = start_date
    summary["coverage_end_date"] = end_date
    loaded_count = int(summary["loaded_symbol_count"])
    overall_start = summary["overall_start_date"]
    overall_end = summary["overall_end_date"]

    if loaded_count <= 0 or overall_start is None or overall_end is None:
        print("Data coverage: unavailable")
        return summary

    print(f"Data coverage: {overall_start} to {overall_end} across {loaded_count} loaded symbols")
    if start_date and end_date:
        covered = int(summary["symbols_covering_range_count"])
        print(
            f"{range_label.capitalize()} coverage: {covered}/{loaded_count} loaded symbols fully cover "
            f"{start_date} to {end_date}"
        )

        for warning in build_data_coverage_warnings(summary, max_symbol_preview=max_symbol_preview):
            print(f"WARN: {warning}")

    return summary


def print_window_span_coverage_summary(
    loaded_bars: dict[str, object],
    windows: Sequence[tuple[str, str]],
    *,
    range_label: str = "window span",
) -> dict[str, object]:
    if not windows:
        return print_data_coverage_summary(loaded_bars)
    coverage_start = min(start for start, _end in windows)
    coverage_end = max(end for _start, end in windows)
    return print_data_coverage_summary(
        loaded_bars,
        start_date=coverage_start,
        end_date=coverage_end,
        range_label=range_label,
    )
