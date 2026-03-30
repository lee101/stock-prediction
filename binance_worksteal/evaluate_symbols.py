#!/usr/bin/env python3
"""Per-symbol contribution evaluator for work-stealing strategy."""
from __future__ import annotations

import argparse
import sys
from dataclasses import asdict
from dataclasses import replace
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from binance_worksteal.cli import (
    build_cli_error,
    add_date_range_args,
    add_require_full_universe_arg,
    build_require_full_universe_error,
    build_run_warnings,
    load_bars_with_summary,
    add_symbol_selection_args,
    print_window_span_coverage_summary,
    print_resolved_symbols,
    resolve_cli_symbols_with_error,
    summarize_loaded_symbols,
    validate_date_range_with_error,
)
from binance_worksteal.config_io import (
    add_config_file_arg,
    add_explain_config_arg,
    add_print_config_arg,
    build_worksteal_config_from_args,
    maybe_handle_worksteal_config_output,
)
from binance_worksteal.eval_diagnostics import format_eval_failure
from binance_worksteal.reporting import (
    add_preview_run_arg,
    add_summary_json_arg,
    build_cli_error_summary,
    build_preview_run_summary,
    build_symbol_listing_summary,
    build_symbol_run_summary,
    print_run_preview,
    run_with_optional_summary,
)
from binance_worksteal.windowing import compute_rolling_windows
from binance_worksteal.strategy import (
    WorkStealConfig,
    compute_avg_hold_days_from_trades,
    load_daily_bars,
    prepare_backtest_bars,
    run_worksteal_backtest,
)
from binance_worksteal.backtest import FULL_UNIVERSE
from binance_worksteal.csim.compat import config_supports_csim, summarize_csim_incompatibility


PRODUCTION_CONFIG = WorkStealConfig(
    dip_pct=0.20,
    profit_target_pct=0.15,
    stop_loss_pct=0.10,
    sma_filter_period=20,
    trailing_stop_pct=0.03,
    max_positions=5,
    max_hold_days=14,
)


_csim_fn = None
_csim_checked = False
_csim_incompatibility_warned = False
_csim_runtime_failure_warned = False

EVALUATE_CONFIG_FLAG_TO_FIELD = {
    "--dip-pct": "dip_pct",
    "--profit-target": "profit_target_pct",
    "--stop-loss": "stop_loss_pct",
    "--sma-filter": "sma_filter_period",
    "--trailing-stop": "trailing_stop_pct",
    "--max-positions": "max_positions",
    "--max-hold-days": "max_hold_days",
}


def _build_evaluation_failure(
    *,
    stage: str,
    symbol: str | None,
    config: WorkStealConfig,
    start_date: str | None,
    end_date: str | None,
    exc: Exception,
    engine: str = "Python backtest",
) -> dict[str, object]:
    context = f"evaluate_symbols {stage}"
    if symbol:
        context = f"{context} {symbol}"
    message = format_eval_failure(
        context,
        engine,
        config,
        start_date,
        end_date,
        exc,
    )
    return {
        "stage": stage,
        "symbol": symbol,
        "start_date": start_date,
        "end_date": end_date,
        "engine": engine,
        "error_type": exc.__class__.__name__,
        "error": str(exc),
        "message": message,
    }


def _normalize_requested_symbol_list(symbols: Optional[List[str]]) -> list[str]:
    return list(summarize_loaded_symbols(symbols or [], ())["requested_symbols"])

def _split_candidate_symbol_list(
    base_symbols: Sequence[str],
    candidate_symbols: Optional[List[str]],
) -> tuple[list[str], list[str]]:
    base_requested = _normalize_requested_symbol_list(list(base_symbols))
    normalized_candidates = _normalize_requested_symbol_list(candidate_symbols)
    base_set = set(base_requested)
    seen_effective = set(base_requested)
    seen_ignored: set[str] = set()
    effective: list[str] = []
    ignored: list[str] = []
    for symbol in normalized_candidates:
        if symbol in base_set:
            if symbol not in seen_ignored:
                ignored.append(symbol)
                seen_ignored.add(symbol)
            continue
        if symbol in seen_effective:
            continue
        seen_effective.add(symbol)
        effective.append(symbol)
    return effective, ignored


def _print_symbol_group_load_summary(label: str, summary: dict[str, object]) -> None:
    requested_count = int(summary["requested_symbol_count"])
    loaded_count = int(summary["loaded_symbol_count"])
    print(f"{label}: {loaded_count}/{requested_count} with data")
    missing = list(summary["missing_symbols"])
    if not missing:
        return
    preview = ", ".join(missing[:10])
    remaining = len(missing) - min(len(missing), 10)
    suffix = f" (+{remaining} more)" if remaining > 0 else ""
    noun = "symbol" if len(missing) == 1 else "symbols"
    print(f"WARN: {len(missing)} missing from {label.lower()}: {preview}{suffix} ({noun})")


def _build_evaluation_metadata(
    *,
    windows: list[tuple[str, str]],
    window_mode: str,
    requested_window_days: int | None,
    requested_window_count: int,
    start_date: str | None,
    end_date: str | None,
    avg_full_return: float,
    avg_full_sortino: float,
    base_symbols: list[str],
    eval_symbols: list[str],
    base_load_summary: dict[str, object],
    candidate_load_summary: dict[str, object],
    requested_candidate_symbols: list[str],
    ignored_candidate_symbols: list[str] | None,
    data_coverage: dict[str, object] | None,
    evaluation_failures: list[dict[str, object]] | None = None,
    skipped_symbols: list[str] | None = None,
    overall_universe_complete: bool | None = None,
) -> dict[str, object]:
    failures = evaluation_failures or []
    skipped = skipped_symbols or []
    ignored_candidates = ignored_candidate_symbols or []
    base_missing = int(base_load_summary.get("missing_symbol_count", 0))
    candidate_missing = int(candidate_load_summary.get("missing_symbol_count", 0))
    warnings = build_run_warnings(load_summary=base_load_summary, data_coverage=data_coverage)
    candidate_missing_symbols = list(candidate_load_summary.get("missing_symbols", []))
    if candidate_missing_symbols:
        preview = ", ".join(candidate_missing_symbols[:10])
        remaining = len(candidate_missing_symbols) - min(len(candidate_missing_symbols), 10)
        suffix = f" (+{remaining} more)" if remaining > 0 else ""
        noun = "symbol" if len(candidate_missing_symbols) == 1 else "symbols"
        warnings.append(
            f"missing data for {len(candidate_missing_symbols)} candidate {noun}: {preview}{suffix}"
        )
    return {
        "windows": windows,
        "window_mode": window_mode,
        "requested_window_days": requested_window_days,
        "requested_window_count": requested_window_count,
        "start_date": start_date,
        "end_date": end_date,
        "avg_full_return": avg_full_return,
        "avg_full_sortino": avg_full_sortino,
        "base_symbol_count": len(base_symbols),
        "evaluated_symbol_count": len(eval_symbols),
        "loaded_symbol_count": base_load_summary["loaded_symbol_count"],
        "loaded_symbols": base_load_summary["loaded_symbols"],
        "missing_symbol_count": base_missing,
        "missing_symbols": base_load_summary["missing_symbols"],
        "base_universe_complete": bool(base_load_summary.get("universe_complete", base_missing == 0)),
        "candidate_symbol_count": candidate_load_summary["requested_symbol_count"],
        "candidate_loaded_symbol_count": candidate_load_summary["loaded_symbol_count"],
        "candidate_loaded_symbols": candidate_load_summary["loaded_symbols"],
        "candidate_missing_symbol_count": candidate_missing,
        "candidate_missing_symbols": candidate_load_summary["missing_symbols"],
        "candidate_universe_complete": bool(candidate_load_summary.get("universe_complete", candidate_missing == 0)),
        "universe_complete": bool(
            overall_universe_complete
            if overall_universe_complete is not None
            else (base_missing == 0 and candidate_missing == 0)
        ),
        "candidate_symbols": requested_candidate_symbols,
        "ignored_candidate_symbols": ignored_candidates,
        "ignored_candidate_symbol_count": len(ignored_candidates),
        "data_coverage": data_coverage,
        "warnings": warnings,
        "evaluation_failure_count": len(failures),
        "evaluation_failures": failures,
        "skipped_symbols": skipped,
    }


def _build_empty_evaluation_result(
    error: str,
    *,
    window_mode: str,
    requested_window_days: int | None,
    requested_window_count: int,
    start_date: str | None,
    end_date: str | None,
    base_symbols: list[str],
    eval_symbols: list[str],
    base_load_summary: dict[str, object],
    candidate_load_summary: dict[str, object],
    requested_candidate_symbols: list[str],
    ignored_candidate_symbols: list[str] | None,
    data_coverage: dict[str, object] | None,
    overall_universe_complete: bool | None = None,
    windows: list[tuple[str, str]] | None = None,
    evaluation_failures: list[dict[str, object]] | None = None,
    skipped_symbols: list[str] | None = None,
    extra_metadata: dict[str, object] | None = None,
) -> tuple[str, list[dict], dict[str, object]]:
    metadata = _build_evaluation_metadata(
        windows=windows or [],
        window_mode=window_mode,
        requested_window_days=requested_window_days,
        requested_window_count=requested_window_count,
        start_date=start_date,
        end_date=end_date,
        avg_full_return=0.0,
        avg_full_sortino=0.0,
        base_symbols=base_symbols,
        eval_symbols=eval_symbols,
        base_load_summary=base_load_summary,
        candidate_load_summary=candidate_load_summary,
        requested_candidate_symbols=requested_candidate_symbols,
        ignored_candidate_symbols=ignored_candidate_symbols,
        data_coverage=data_coverage,
        evaluation_failures=evaluation_failures,
        skipped_symbols=skipped_symbols,
        overall_universe_complete=overall_universe_complete,
    )
    if extra_metadata:
        metadata.update(extra_metadata)
    return error, [], metadata


def _resolve_evaluation_windows(
    all_bars: Dict[str, pd.DataFrame],
    *,
    window_days: int,
    n_windows: int,
    start_date: str | None = None,
    end_date: str | None = None,
) -> tuple[list[tuple[str, str]], str]:
    if bool(start_date) != bool(end_date):
        raise ValueError("start_date and end_date must be provided together")
    if start_date and end_date:
        return [(start_date, end_date)], "fixed"
    return compute_rolling_windows(all_bars, window_days=window_days, n_windows=n_windows), "rolling"


def build_evaluate_cli_default_config(args: argparse.Namespace) -> WorkStealConfig:
    return WorkStealConfig(
        dip_pct=args.dip_pct,
        profit_target_pct=args.profit_target,
        stop_loss_pct=args.stop_loss,
        sma_filter_period=args.sma_filter,
        trailing_stop_pct=args.trailing_stop,
        max_positions=args.max_positions,
        max_hold_days=args.max_hold_days,
    )


def build_evaluate_config(args: argparse.Namespace, raw_argv: list[str]) -> WorkStealConfig:
    return build_worksteal_config_from_args(
        base_config=build_evaluate_cli_default_config(args),
        config_file=args.config_file,
        args=args,
        raw_argv=raw_argv,
        flag_to_field=EVALUATE_CONFIG_FLAG_TO_FIELD,
    )

def _get_csim_fn():
    global _csim_fn, _csim_checked
    if not _csim_checked:
        _csim_checked = True
        try:
            from binance_worksteal.csim.fast_worksteal import run_worksteal_backtest_fast
            _csim_fn = run_worksteal_backtest_fast
        except Exception:
            _csim_fn = None
    return _csim_fn


def _run_backtest(
    all_bars: Dict[str, pd.DataFrame],
    config: WorkStealConfig,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    use_csim: bool = True,
    prepared_bars: Optional[dict] = None,
) -> dict:
    global _csim_incompatibility_warned, _csim_runtime_failure_warned

    fast_fn = _get_csim_fn() if use_csim and config_supports_csim(config) else None
    if use_csim and fast_fn is None and not config_supports_csim(config) and not _csim_incompatibility_warned:
        issues = summarize_csim_incompatibility(config)
        print(f"WARN: evaluate_symbols disabled C sim for unsupported config features: {issues}")
        _csim_incompatibility_warned = True
    if fast_fn is not None:
        try:
            return fast_fn(all_bars, config, start_date=start_date, end_date=end_date)
        except Exception as exc:
            if not _csim_runtime_failure_warned:
                message = format_eval_failure(
                    "evaluate_symbols",
                    "C sim",
                    config,
                    start_date,
                    end_date,
                    exc,
                )
                print(f"WARN: {message}; falling back to Python backtest")
                _csim_runtime_failure_warned = True
    equity_df, trades, metrics = run_worksteal_backtest(
        all_bars, config, start_date=start_date, end_date=end_date, prepared_bars=prepared_bars,
    )
    return metrics


def _filter_bars(all_bars: Dict[str, pd.DataFrame], symbols: List[str]) -> Dict[str, pd.DataFrame]:
    return {s: all_bars[s] for s in symbols if s in all_bars}


def evaluate_standalone(
    all_bars: Dict[str, pd.DataFrame],
    symbols: List[str],
    config: WorkStealConfig,
    windows: List[Tuple[str, str]],
    use_csim: bool = True,  # unused, always uses Python sim for trade details
    failures: Optional[List[dict[str, object]]] = None,
    prepared_bars: Optional[dict] = None,
) -> Dict[str, dict]:
    single_config = replace(config, max_positions=1)
    results = {}
    for sym in symbols:
        if sym not in all_bars:
            continue
        window_metrics = []
        all_hold_days = []
        symbol_bars = {sym: all_bars[sym]}
        symbol_prepared_bars = _filter_bars(prepared_bars, [sym]) if prepared_bars is not None else None
        symbol_failed = False
        for start, end in windows:
            try:
                _eq, trades, m = run_worksteal_backtest(
                    symbol_bars, single_config,
                    start_date=start, end_date=end,
                    prepared_bars=symbol_prepared_bars,
                )
            except Exception as exc:
                failure = _build_evaluation_failure(
                    stage="standalone",
                    symbol=sym,
                    config=single_config,
                    start_date=start,
                    end_date=end,
                    exc=exc,
                )
                print(f"WARN: {failure['message']}; skipping symbol")
                if failures is not None:
                    failures.append(failure)
                symbol_failed = True
                break
            window_metrics.append(m)
            all_hold_days.append(compute_avg_hold_days_from_trades(trades))
        if symbol_failed:
            continue
        avg_return = float(np.mean([m.get("total_return_pct", 0.0) for m in window_metrics]))
        avg_sortino = float(np.mean([m.get("sortino", 0.0) for m in window_metrics]))
        avg_trades = float(np.mean([m.get("n_trades", m.get("total_trades", 0)) for m in window_metrics]))
        avg_hold = float(np.mean([h for h in all_hold_days if h > 0])) if any(h > 0 for h in all_hold_days) else 0.0
        results[sym] = {
            "standalone_return": avg_return,
            "standalone_sortino": avg_sortino,
            "n_trades": avg_trades,
            "avg_hold_days": avg_hold,
            "per_window": window_metrics,
        }
    return results


def evaluate_leave_one_out(
    all_bars: Dict[str, pd.DataFrame],
    symbols: List[str],
    config: WorkStealConfig,
    windows: List[Tuple[str, str]],
    full_sortinos: List[float],
    use_csim: bool = True,
    failures: Optional[List[dict[str, object]]] = None,
    prepared_bars: Optional[dict] = None,
) -> Dict[str, dict]:
    results = {}
    for sym in symbols:
        if sym not in all_bars:
            continue
        remaining = [s for s in symbols if s != sym and s in all_bars]
        if not remaining:
            continue
        subset_bars = _filter_bars(all_bars, remaining)
        subset_prepared_bars = _filter_bars(prepared_bars, remaining) if prepared_bars is not None else None
        window_marginals = []
        symbol_failed = False
        for wi, (start, end) in enumerate(windows):
            try:
                m = _run_backtest(
                    subset_bars,
                    config,
                    start_date=start,
                    end_date=end,
                    use_csim=use_csim,
                    prepared_bars=subset_prepared_bars,
                )
            except Exception as exc:
                failure = _build_evaluation_failure(
                    stage="leave_one_out",
                    symbol=sym,
                    config=config,
                    start_date=start,
                    end_date=end,
                    exc=exc,
                )
                print(f"WARN: {failure['message']}; skipping symbol")
                if failures is not None:
                    failures.append(failure)
                symbol_failed = True
                break
            without_sortino = m.get("sortino", 0.0)
            marginal = full_sortinos[wi] - without_sortino
            window_marginals.append(marginal)
        if symbol_failed:
            continue
        results[sym] = {
            "marginal_contribution": float(np.mean(window_marginals)),
            "per_window_marginal": window_marginals,
        }
    return results


def evaluate_full_universe(
    all_bars: Dict[str, pd.DataFrame],
    symbols: List[str],
    config: WorkStealConfig,
    windows: List[Tuple[str, str]],
    use_csim: bool = True,
    prepared_bars: Optional[dict] = None,
) -> Tuple[List[float], List[dict]]:
    full_bars = _filter_bars(all_bars, symbols)
    full_prepared_bars = _filter_bars(prepared_bars, symbols) if prepared_bars is not None else None
    sortinos = []
    metrics_list = []
    for start, end in windows:
        m = _run_backtest(
            full_bars,
            config,
            start_date=start,
            end_date=end,
            use_csim=use_csim,
            prepared_bars=full_prepared_bars,
        )
        sortinos.append(m.get("sortino", 0.0))
        metrics_list.append(m)
    return sortinos, metrics_list


def format_results_table(
    symbols: List[str],
    standalone: Dict[str, dict],
    leave_one_out: Dict[str, dict],
) -> str:
    rows = []
    for sym in symbols:
        sa = standalone.get(sym, {})
        loo = leave_one_out.get(sym, {})
        rows.append({
            "symbol": sym,
            "standalone_return": sa.get("standalone_return", 0.0),
            "standalone_sortino": sa.get("standalone_sortino", 0.0),
            "marginal_contribution": loo.get("marginal_contribution", 0.0),
            "n_trades": sa.get("n_trades", 0.0),
            "avg_hold_days": sa.get("avg_hold_days", 0.0),
        })
    rows.sort(key=lambda r: r["marginal_contribution"], reverse=True)

    header = f"{'Symbol':<12} {'StandRet%':>10} {'StandSort':>10} {'Marginal':>10} {'Trades':>8} {'AvgHold':>8}"
    sep = "-" * len(header)
    lines = [sep, header, sep]
    for r in rows:
        lines.append(
            f"{r['symbol']:<12} {r['standalone_return']:>10.2f} {r['standalone_sortino']:>10.2f} "
            f"{r['marginal_contribution']:>10.3f} {r['n_trades']:>8.1f} {r['avg_hold_days']:>8.1f}"
        )
    lines.append(sep)
    return "\n".join(lines)


def run_evaluation(
    data_dir: str,
    symbols: List[str],
    config: WorkStealConfig,
    window_days: int = 60,
    n_windows: int = 3,
    start_date: str | None = None,
    end_date: str | None = None,
    candidate_symbols: Optional[List[str]] = None,
    use_csim: bool = True,
    return_metadata: bool = False,
    require_full_universe: bool = False,
) -> Tuple[str, List[dict]] | Tuple[str, List[dict], dict]:
    base_requested = _normalize_requested_symbol_list(symbols)
    requested_candidate_symbols, ignored_candidate_symbols = _split_candidate_symbol_list(base_requested, candidate_symbols)

    all_symbols = [*base_requested, *requested_candidate_symbols]
    initial_window_mode = "fixed" if start_date and end_date else "rolling"
    initial_requested_window_days = None if initial_window_mode == "fixed" else window_days
    initial_requested_window_count = 1 if initial_window_mode == "fixed" else n_windows

    loaded = load_bars_with_summary(
        data_dir=data_dir,
        requested_symbols=all_symbols,
        load_bars=load_daily_bars,
        no_data_message=None,
        return_failure_on_error=True,
    )
    if loaded is None:
        empty_base_load_summary = summarize_loaded_symbols(base_requested, ())
        empty_candidate_load_summary = summarize_loaded_symbols(requested_candidate_symbols, ())
        empty_overall_load_summary = summarize_loaded_symbols(all_symbols, ())
        result = _build_empty_evaluation_result(
            "ERROR: No data loaded",
            window_mode=initial_window_mode,
            requested_window_days=initial_requested_window_days,
            requested_window_count=initial_requested_window_count,
            start_date=start_date,
            end_date=end_date,
            base_symbols=[],
            eval_symbols=[],
            base_load_summary=empty_base_load_summary,
            candidate_load_summary=empty_candidate_load_summary,
            requested_candidate_symbols=requested_candidate_symbols,
            ignored_candidate_symbols=ignored_candidate_symbols,
            data_coverage=None,
            overall_universe_complete=bool(empty_overall_load_summary.get("universe_complete", False)),
        )
        return result if return_metadata else result[:2]
    load_failure = None
    if len(loaded) == 3:
        all_bars, overall_load_summary, load_failure = loaded
    else:
        all_bars, overall_load_summary = loaded
    if load_failure is not None:
        empty_base_load_summary = summarize_loaded_symbols(base_requested, ())
        empty_candidate_load_summary = summarize_loaded_symbols(requested_candidate_symbols, ())
        empty_overall_load_summary = summarize_loaded_symbols(all_symbols, ())
        result = _build_empty_evaluation_result(
            load_failure["error"],
            window_mode=initial_window_mode,
            requested_window_days=initial_requested_window_days,
            requested_window_count=initial_requested_window_count,
            start_date=start_date,
            end_date=end_date,
            base_symbols=[],
            eval_symbols=[],
            base_load_summary=empty_base_load_summary,
            candidate_load_summary=empty_candidate_load_summary,
            requested_candidate_symbols=requested_candidate_symbols,
            ignored_candidate_symbols=ignored_candidate_symbols,
            data_coverage=None,
            overall_universe_complete=bool(empty_overall_load_summary.get("universe_complete", False)),
            extra_metadata={"load_failure": load_failure},
        )
        return result if return_metadata else result[:2]
    prepared_all_bars = prepare_backtest_bars(all_bars)
    base_load_summary = summarize_loaded_symbols(base_requested, all_bars.keys())
    candidate_load_summary = summarize_loaded_symbols(requested_candidate_symbols, all_bars.keys())
    if requested_candidate_symbols:
        _print_symbol_group_load_summary("Base symbols", base_load_summary)
        _print_symbol_group_load_summary("Candidate symbols", candidate_load_summary)
    base_symbols = list(base_load_summary["loaded_symbols"])
    base_bars = _filter_bars(all_bars, base_symbols)

    if require_full_universe and not bool(overall_load_summary.get("universe_complete", False)):
        error_message = build_require_full_universe_error(overall_load_summary) or "ERROR: --require-full-universe found missing data"
        result = _build_empty_evaluation_result(
            error_message,
            window_mode=initial_window_mode,
            requested_window_days=initial_requested_window_days,
            requested_window_count=initial_requested_window_count,
            start_date=start_date,
            end_date=end_date,
            base_symbols=list(base_load_summary["loaded_symbols"]),
            eval_symbols=list(base_load_summary["loaded_symbols"]),
            base_load_summary=base_load_summary,
            candidate_load_summary=candidate_load_summary,
            requested_candidate_symbols=requested_candidate_symbols,
            ignored_candidate_symbols=ignored_candidate_symbols,
            data_coverage=None,
            overall_universe_complete=bool(overall_load_summary.get("universe_complete", False)),
        )
        return result if return_metadata else result[:2]

    if not base_symbols:
        result = _build_empty_evaluation_result(
            "ERROR: None of the requested base symbols have data",
            window_mode=initial_window_mode,
            requested_window_days=initial_requested_window_days,
            requested_window_count=initial_requested_window_count,
            start_date=start_date,
            end_date=end_date,
            base_symbols=[],
            eval_symbols=[],
            base_load_summary=base_load_summary,
            candidate_load_summary=candidate_load_summary,
            requested_candidate_symbols=requested_candidate_symbols,
            ignored_candidate_symbols=ignored_candidate_symbols,
            data_coverage=None,
            overall_universe_complete=bool(overall_load_summary.get("universe_complete", False)),
        )
        return result if return_metadata else result[:2]

    windows, window_mode = _resolve_evaluation_windows(
        base_bars,
        window_days=window_days,
        n_windows=n_windows,
        start_date=start_date,
        end_date=end_date,
    )
    resolved_requested_window_days = None if window_mode == "fixed" else window_days
    resolved_requested_window_count = 1 if window_mode == "fixed" else n_windows
    if not windows:
        result = _build_empty_evaluation_result(
            "ERROR: Not enough data for rolling windows" if window_mode == "rolling" else "ERROR: Invalid fixed evaluation window",
            window_mode=window_mode,
            requested_window_days=resolved_requested_window_days,
            requested_window_count=resolved_requested_window_count,
            start_date=start_date,
            end_date=end_date,
            base_symbols=base_symbols,
            eval_symbols=base_symbols,
            base_load_summary=base_load_summary,
            candidate_load_summary=candidate_load_summary,
            requested_candidate_symbols=requested_candidate_symbols,
            ignored_candidate_symbols=ignored_candidate_symbols,
            data_coverage=None,
            overall_universe_complete=bool(overall_load_summary.get("universe_complete", False)),
        )
        return result if return_metadata else result[:2]
    if window_mode == "rolling" and len(windows) < n_windows:
        print(
            f"WARN: only {len(windows)}/{n_windows} rolling windows of {window_days} days "
            "fit within loaded data coverage"
        )
    print(f"Windows: {windows}")
    data_coverage = print_window_span_coverage_summary(
        base_bars,
        windows,
        range_label="requested range" if window_mode == "fixed" else "window span",
    )

    eval_symbols = [s for s in all_symbols if s in all_bars]
    evaluation_failures: list[dict[str, object]] = []

    print(f"Running full universe backtest ({len(base_symbols)} symbols)...")
    try:
        full_sortinos, full_metrics = evaluate_full_universe(
            all_bars, base_symbols, config, windows, use_csim=use_csim, prepared_bars=prepared_all_bars,
        )
    except Exception as exc:
        failure = _build_evaluation_failure(
            stage="full_universe",
            symbol=None,
            config=config,
            start_date=windows[-1][0] if windows else None,
            end_date=windows[0][1] if windows else None,
            exc=exc,
        )
        result = _build_empty_evaluation_result(
            f"ERROR: {failure['message']}",
            windows=windows,
            window_mode=window_mode,
            requested_window_days=resolved_requested_window_days,
            requested_window_count=resolved_requested_window_count,
            start_date=start_date,
            end_date=end_date,
            base_symbols=base_symbols,
            eval_symbols=eval_symbols,
            base_load_summary=base_load_summary,
            candidate_load_summary=candidate_load_summary,
            requested_candidate_symbols=requested_candidate_symbols,
            ignored_candidate_symbols=ignored_candidate_symbols,
            data_coverage=data_coverage,
            evaluation_failures=[failure],
        )
        return result if return_metadata else result[:2]
    avg_full_sortino = float(np.mean(full_sortinos))
    avg_full_return = float(np.mean([m.get("total_return_pct", 0.0) for m in full_metrics]))
    print(f"Full universe: avg return={avg_full_return:.2f}%, avg sortino={avg_full_sortino:.2f}")

    print(f"Running standalone evaluation for {len(eval_symbols)} symbols...")
    standalone = evaluate_standalone(
        all_bars,
        eval_symbols,
        config,
        windows,
        use_csim=use_csim,
        failures=evaluation_failures,
        prepared_bars=prepared_all_bars,
    )

    print(f"Running leave-one-out for {len(base_symbols)} symbols...")
    loo = evaluate_leave_one_out(
        all_bars,
        base_symbols,
        config,
        windows,
        full_sortinos,
        use_csim=use_csim,
        failures=evaluation_failures,
        prepared_bars=prepared_all_bars,
    )

    if requested_candidate_symbols:
        for cs in requested_candidate_symbols:
            if cs not in all_bars:
                continue
            expanded = base_symbols + [cs]
            expanded_bars = _filter_bars(all_bars, expanded)
            expanded_prepared_bars = _filter_bars(prepared_all_bars, expanded)
            marginals = []
            candidate_failed = False
            for wi, (start, end) in enumerate(windows):
                try:
                    m = _run_backtest(
                        expanded_bars,
                        config,
                        start_date=start,
                        end_date=end,
                        use_csim=use_csim,
                        prepared_bars=expanded_prepared_bars,
                    )
                except Exception as exc:
                    failure = _build_evaluation_failure(
                        stage="candidate",
                        symbol=cs,
                        config=config,
                        start_date=start,
                        end_date=end,
                        exc=exc,
                    )
                    print(f"WARN: {failure['message']}; skipping symbol")
                    evaluation_failures.append(failure)
                    candidate_failed = True
                    break
                marginals.append(m.get("sortino", 0.0) - full_sortinos[wi])
            if candidate_failed:
                continue
            loo[cs] = {
                "marginal_contribution": float(np.mean(marginals)),
                "per_window_marginal": marginals,
            }

    failed_symbols = {str(item["symbol"]) for item in evaluation_failures if item.get("symbol")}
    reported_symbols = [sym for sym in eval_symbols if sym not in failed_symbols]
    table = format_results_table(reported_symbols, standalone, loo)

    result_rows = []
    for sym in reported_symbols:
        sa = standalone.get(sym, {})
        lo = loo.get(sym, {})
        result_rows.append({
            "symbol": sym,
            "standalone_return": sa.get("standalone_return", 0.0),
            "standalone_sortino": sa.get("standalone_sortino", 0.0),
            "marginal_contribution": lo.get("marginal_contribution", 0.0),
            "n_trades": sa.get("n_trades", 0.0),
            "avg_hold_days": sa.get("avg_hold_days", 0.0),
        })

    summary = [
        f"\nFull Universe ({len(base_symbols)} symbols):",
        f"  Avg Return: {avg_full_return:.2f}%",
        f"  Avg Sortino: {avg_full_sortino:.2f}",
        f"  Window: {start_date} to {end_date}" if window_mode == "fixed" else f"  Windows: {len(windows)}x{window_days}d",
        "",
        table,
    ]
    if requested_candidate_symbols:
        candidate_symbol_set = set(requested_candidate_symbols)
        cands = [r for r in result_rows if r["symbol"] in candidate_symbol_set]
        if cands:
            summary.append("\nCandidate symbols (positive marginal = improves universe):")
            for c in sorted(cands, key=lambda x: x["marginal_contribution"], reverse=True):
                sign = "+" if c["marginal_contribution"] >= 0 else ""
                summary.append(
                    f"  {c['symbol']:<12} marginal={sign}{c['marginal_contribution']:.3f} "
                    f"standalone_sort={c['standalone_sortino']:.2f}"
                )
    if ignored_candidate_symbols:
        summary.append(
            "\nIgnored candidate symbols already present in base universe: "
            + ", ".join(ignored_candidate_symbols)
        )
    skipped_symbols = sorted(failed_symbols)
    if skipped_symbols:
        preview = ", ".join(skipped_symbols[:10])
        remaining = len(skipped_symbols) - min(len(skipped_symbols), 10)
        suffix = f" (+{remaining} more)" if remaining > 0 else ""
        summary.append(
            f"\nSkipped {len(skipped_symbols)} symbols due to evaluation failures: {preview}{suffix}"
        )

    metadata = _build_evaluation_metadata(
        windows=windows,
        window_mode=window_mode,
        requested_window_days=resolved_requested_window_days,
        requested_window_count=resolved_requested_window_count,
        start_date=start_date,
        end_date=end_date,
        avg_full_return=avg_full_return,
        avg_full_sortino=avg_full_sortino,
        base_symbols=base_symbols,
        eval_symbols=eval_symbols,
        base_load_summary=base_load_summary,
        candidate_load_summary=candidate_load_summary,
        requested_candidate_symbols=requested_candidate_symbols,
        ignored_candidate_symbols=ignored_candidate_symbols,
        data_coverage=data_coverage,
        evaluation_failures=evaluation_failures,
        skipped_symbols=skipped_symbols,
    )
    result = ("\n".join(summary), result_rows, metadata)
    return result if return_metadata else result[:2]


def main(argv: list[str] | None = None):
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(description="Evaluate per-symbol contribution to work-stealing strategy")
    parser.add_argument("--data-dir", default="trainingdata/train")
    add_symbol_selection_args(parser)
    add_require_full_universe_arg(parser)
    add_date_range_args(parser, start_dest="start_date", end_dest="end_date")
    parser.add_argument("--candidate-symbols", nargs="+", default=None)
    parser.add_argument("--days", type=int, default=60)
    parser.add_argument("--windows", "--n-windows", dest="windows", type=int, default=3)
    parser.add_argument("--dip-pct", type=float, default=0.20)
    parser.add_argument("--profit-target", type=float, default=0.15)
    parser.add_argument("--stop-loss", type=float, default=0.10)
    parser.add_argument("--sma-filter", type=int, default=20)
    parser.add_argument("--trailing-stop", type=float, default=0.03)
    parser.add_argument("--max-positions", type=int, default=5)
    parser.add_argument("--max-hold-days", type=int, default=14)
    parser.add_argument("--no-csim", action="store_true")
    add_config_file_arg(parser)
    add_print_config_arg(parser)
    add_explain_config_arg(parser)
    add_preview_run_arg(parser)
    add_summary_json_arg(parser)
    args = parser.parse_args(raw_argv)

    config_output_rc = maybe_handle_worksteal_config_output(
        args=args,
        build_config=lambda: build_evaluate_config(args, raw_argv),
        base_config=build_evaluate_cli_default_config(args),
        config_file=args.config_file,
        raw_argv=raw_argv,
        flag_to_field=EVALUATE_CONFIG_FLAG_TO_FIELD,
    )
    if config_output_rc is not None:
        return config_output_rc

    def build_summary_payload(
        *,
        output: str,
        rows: list[dict],
        metadata: dict,
        config: WorkStealConfig,
        symbol_source: str,
        symbols: list[str],
    ) -> dict:
        return {
            **build_symbol_run_summary(
                tool="evaluate_symbols",
                data_dir=args.data_dir,
                symbol_source=symbol_source,
                symbols=symbols,
                load_summary={
                    "loaded_symbol_count": metadata.get("loaded_symbol_count", 0),
                    "loaded_symbols": metadata.get("loaded_symbols", []),
                    "missing_symbol_count": metadata.get("missing_symbol_count", 0),
                    "missing_symbols": metadata.get("missing_symbols", []),
                },
                data_coverage=metadata.get("data_coverage"),
                config_file=args.config_file,
                config=asdict(config),
                require_full_universe=args.require_full_universe,
            ),
            "candidate_symbols": metadata.get("candidate_symbols", []),
            "candidate_symbol_count": metadata.get("candidate_symbol_count", 0),
            "ignored_candidate_symbols": metadata.get("ignored_candidate_symbols", []),
            "ignored_candidate_symbol_count": metadata.get("ignored_candidate_symbol_count", 0),
            "candidate_loaded_symbol_count": metadata.get("candidate_loaded_symbol_count", 0),
            "candidate_loaded_symbols": metadata.get("candidate_loaded_symbols", []),
            "candidate_missing_symbol_count": metadata.get("candidate_missing_symbol_count", 0),
            "candidate_missing_symbols": metadata.get("candidate_missing_symbols", []),
            "base_universe_complete": metadata.get("base_universe_complete", metadata.get("missing_symbol_count", 0) == 0),
            "candidate_universe_complete": metadata.get("candidate_universe_complete", metadata.get("candidate_missing_symbol_count", 0) == 0),
            "universe_complete": metadata.get("universe_complete", metadata.get("missing_symbol_count", 0) == 0 and metadata.get("candidate_missing_symbol_count", 0) == 0),
            "window_mode": metadata.get("window_mode", "rolling"),
            "windows": metadata.get("windows", []),
            "requested_window_days": metadata.get("requested_window_days"),
            "requested_window_count": metadata.get("requested_window_count", 0),
            "start_date": metadata.get("start_date"),
            "end_date": metadata.get("end_date"),
            "summary_text": output,
            "rows": rows,
            "avg_full_return": metadata.get("avg_full_return", 0.0),
            "avg_full_sortino": metadata.get("avg_full_sortino", 0.0),
            "base_symbol_count": metadata.get("base_symbol_count", 0),
            "evaluated_symbol_count": metadata.get("evaluated_symbol_count", 0),
            "evaluation_failure_count": metadata.get("evaluation_failure_count", 0),
            "evaluation_failures": metadata.get("evaluation_failures", []),
            "skipped_symbols": metadata.get("skipped_symbols", []),
            "use_csim": not args.no_csim,
        }

    def build_main_error_summary(
        *,
        error: str,
        error_type: str,
        list_symbols_only: bool = False,
        preview_only: bool = False,
        candidate_symbols: Sequence[str] | None = None,
        ignored_candidate_symbols: Sequence[str] | None = None,
    ) -> dict:
        extra: dict[str, object] = {}
        if list_symbols_only:
            extra["list_symbols_only"] = True
        if preview_only:
            extra["preview_only"] = True
        if candidate_symbols is not None or ignored_candidate_symbols is not None:
            candidate_list = list(candidate_symbols or [])
            ignored_list = list(ignored_candidate_symbols or [])
            extra.update(
                {
                    "candidate_symbols": candidate_list,
                    "candidate_symbol_count": len(candidate_list),
                    "candidate_symbol_source": "command line --candidate-symbols" if candidate_list else None,
                    "ignored_candidate_symbols": ignored_list,
                    "ignored_candidate_symbol_count": len(ignored_list),
                }
            )
        return build_cli_error_summary(
            tool="evaluate_symbols",
            error=error,
            error_type=error_type,
            data_dir=args.data_dir,
            config_file=args.config_file,
            extra=extra or None,
        )

    def finalize_run_output(
        *,
        output: str,
        rows: list[dict],
        metadata: dict,
        config: WorkStealConfig,
        symbol_source: str,
        symbols: list[str],
    ) -> tuple[int, dict | None]:
        exit_code = 0
        if output.startswith("ERROR:"):
            exit_code = 1
        elif not rows:
            print("ERROR: No valid symbol evaluation results")
            output = f"{output}\nERROR: No valid symbol evaluation results"
            exit_code = 1
        if not args.summary_json:
            return exit_code, None
        return exit_code, build_summary_payload(
            output=output,
            rows=rows,
            metadata=metadata,
            config=config,
            symbol_source=symbol_source,
            symbols=symbols,
        )

    def run():
        requested_candidate_args = _normalize_requested_symbol_list(args.candidate_symbols)
        validated_range, date_error = validate_date_range_with_error(
            start_date=args.start_date,
            end_date=args.end_date,
            require_pair=True,
        )
        if date_error is not None:
            print(date_error["error"])
            return 1, build_main_error_summary(
                error=date_error["error"],
                error_type=date_error["error_type"],
                list_symbols_only=args.list_symbols,
                preview_only=args.preview_run,
                candidate_symbols=requested_candidate_args,
                ignored_candidate_symbols=[],
            )
        args.start_date, args.end_date = validated_range

        resolved, symbol_error = resolve_cli_symbols_with_error(
            symbols_arg=args.symbols,
            universe_file=args.universe_file,
            default_symbols=FULL_UNIVERSE,
        )
        if symbol_error is not None:
            print(symbol_error["error"])
            return 1, build_main_error_summary(
                error=symbol_error["error"],
                error_type=symbol_error["error_type"],
                list_symbols_only=args.list_symbols,
                preview_only=args.preview_run,
                candidate_symbols=requested_candidate_args,
                ignored_candidate_symbols=[],
            )
        symbols, symbol_source = resolved
        candidate_symbols, ignored_candidate_symbols = _split_candidate_symbol_list(symbols, args.candidate_symbols)
        if args.list_symbols:
            print_resolved_symbols(symbols, symbol_source)
            if candidate_symbols:
                print_resolved_symbols(candidate_symbols, "command line --candidate-symbols")
            if ignored_candidate_symbols:
                print(
                    "Ignored "
                    f"{len(ignored_candidate_symbols)} candidate symbols already in base universe: "
                    + ", ".join(ignored_candidate_symbols)
                )
            return 0, build_symbol_listing_summary(
                tool="evaluate_symbols",
                data_dir=args.data_dir,
                symbol_source=symbol_source,
                symbols=symbols,
                config_file=args.config_file,
                extra={
                    "candidate_symbols": candidate_symbols,
                    "candidate_symbol_count": len(candidate_symbols),
                    "candidate_symbol_source": "command line --candidate-symbols" if candidate_symbols else None,
                    "ignored_candidate_symbols": ignored_candidate_symbols,
                    "ignored_candidate_symbol_count": len(ignored_candidate_symbols),
                },
            )

        try:
            config = build_evaluate_config(args, raw_argv)
        except (FileNotFoundError, OSError, ValueError) as exc:
            config_error = build_cli_error(exc)
            print(config_error["error"])
            return 1, build_main_error_summary(
                error=config_error["error"],
                error_type=config_error["error_type"],
                preview_only=args.preview_run,
                candidate_symbols=candidate_symbols,
                ignored_candidate_symbols=ignored_candidate_symbols,
            )
        if args.preview_run:
            date_mode = "fixed_range" if args.start_date and args.end_date else "rolling_windows"
            extra = {
                "date_mode": date_mode,
                "window_mode": "fixed" if date_mode == "fixed_range" else "rolling",
                "start_date": args.start_date,
                "end_date": args.end_date,
                "days": None if date_mode == "fixed_range" else args.days,
                "requested_window_days": None if date_mode == "fixed_range" else args.days,
                "requested_window_count": 1 if date_mode == "fixed_range" else args.windows,
                "candidate_symbols": candidate_symbols,
                "candidate_symbol_count": len(candidate_symbols),
                "ignored_candidate_symbols": ignored_candidate_symbols,
                "ignored_candidate_symbol_count": len(ignored_candidate_symbols),
                "use_csim": not args.no_csim,
                "summary_json": args.summary_json,
                "require_full_universe": args.require_full_universe,
            }
            print_run_preview(
                tool="evaluate_symbols",
                sections=[
                    (
                        "Inputs",
                        (
                            ("data_dir", args.data_dir),
                            ("symbol_source", symbol_source),
                            ("base_symbol_count", len(symbols)),
                            ("base_symbols", symbols),
                            ("candidate_symbol_count", len(candidate_symbols)),
                            ("candidate_symbols", candidate_symbols),
                            ("ignored_candidate_symbol_count", len(ignored_candidate_symbols)),
                            ("ignored_candidate_symbols", ignored_candidate_symbols),
                        ),
                    ),
                    (
                        "Execution",
                        (
                            ("date_mode", date_mode),
                            ("start_date", args.start_date),
                            ("end_date", args.end_date),
                            ("days", extra["days"]),
                            ("requested_windows", extra["requested_window_count"]),
                            ("use_csim", extra["use_csim"]),
                            ("require_full_universe", args.require_full_universe),
                            ("summary_json", args.summary_json),
                        ),
                    ),
                    (
                        "Config",
                        (
                            ("config_file", args.config_file),
                            ("max_positions", config.max_positions),
                            ("max_hold_days", config.max_hold_days),
                            ("sma_filter_period", config.sma_filter_period),
                        ),
                    ),
                ],
            )
            return 0, build_preview_run_summary(
                tool="evaluate_symbols",
                data_dir=args.data_dir,
                symbol_source=symbol_source,
                symbols=symbols,
                config_file=args.config_file,
                config=config,
                extra=extra,
            )
        if args.config_file:
            print(f"Loaded config overrides from {args.config_file}")
        print(f"Using {len(symbols)} symbols from {symbol_source}")

        output, rows, metadata = run_evaluation(
            data_dir=args.data_dir,
            symbols=symbols,
            config=config,
            window_days=args.days,
            n_windows=args.windows,
            start_date=args.start_date,
            end_date=args.end_date,
            candidate_symbols=candidate_symbols,
            use_csim=not args.no_csim,
            return_metadata=True,
            require_full_universe=args.require_full_universe,
        )
        print(output)
        return finalize_run_output(
            output=output,
            rows=rows,
            metadata=metadata,
            config=config,
            symbol_source=symbol_source,
            symbols=symbols,
        )

    return run_with_optional_summary(
        args.summary_json,
        run,
        module="binance_worksteal.evaluate_symbols",
        argv=raw_argv,
        announce_artifact_manifest_on_success=True,
    )


if __name__ == "__main__":
    sys.exit(main())
