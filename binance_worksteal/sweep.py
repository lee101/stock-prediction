#!/usr/bin/env python3
"""Auto-research sweep for work-stealing strategy parameters.

Supports:
- Multi-window evaluation for robustness (worst-case Sortino selection)
- Production-realistic fill model via --realistic flag
- Optional C simulator for 10x speedup via --use-csim
- Dated CSV output with full metrics
"""
from __future__ import annotations

import argparse
import sys
import time
from dataclasses import replace
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from binance_worksteal.cli import (
    build_cli_error,
    add_date_range_args,
    add_require_full_universe_arg,
    load_bars_with_summary,
    build_strict_retry_command,
    add_symbol_selection_args,
    print_window_span_coverage_summary,
    print_resolved_symbols,
    require_full_universe_or_print_error,
    resolve_cli_symbols_with_error,
    resolve_paired_date_range_with_error,
)
from binance_worksteal.config_io import (
    add_config_file_arg,
    add_explain_config_arg,
    add_print_config_arg,
    argv_has_flag,
    build_worksteal_config_from_args,
    maybe_handle_worksteal_config_output,
)
from binance_worksteal.eval_diagnostics import format_eval_failure
from binance_worksteal.reporting import (
    add_preview_run_arg,
    add_summary_json_arg,
    announce_sweep_artifacts,
    build_cli_error_summary,
    build_empty_sweep_run_summary,
    build_preview_run_summary,
    build_sweep_run_summary,
    empty_sweep_recommendation,
    build_symbol_listing_summary,
    default_sidecar_json_path,
    print_run_preview,
    prepare_sweep_recommendation_artifacts,
    run_with_optional_summary,
)
from binance_worksteal.grid_sampling import cartesian_product_size, sample_cartesian_product
from binance_worksteal.windowing import compute_rolling_windows
from binance_worksteal.strategy import (
    WorkStealConfig, count_completed_trades, load_daily_bars, prepare_backtest_bars, run_worksteal_backtest,
)
from binance_worksteal.backtest import FULL_UNIVERSE
from binance_worksteal.csim.compat import config_supports_csim, summarize_csim_incompatibility


SWEEP_GRID = {
    "dip_pct": [0.05, 0.07, 0.10, 0.15, 0.20],
    "proximity_pct": [0.01, 0.02, 0.03, 0.05],
    "profit_target_pct": [0.03, 0.05, 0.07, 0.10, 0.15, 0.20],
    "stop_loss_pct": [0.05, 0.08, 0.10, 0.15, 0.20],
    "max_positions": [3, 5, 7, 10],
    "max_hold_days": [7, 14, 21, 30],
    "lookback_days": [10, 20, 30],
    "ref_price_method": ["high", "sma", "close"],
    "max_leverage": [1.0, 2.0, 3.0, 5.0],
    "enable_shorts": [False, True],
    "trailing_stop_pct": [0.0, 0.03, 0.05],
    "sma_filter_period": [0, 10, 20],
    "market_breadth_filter": [0.0, 0.6, 0.7, 0.8],
}

SWEEP_CONFIG_FLAG_TO_FIELD = {
    "--cash": ("initial_cash", "cash"),
}


def build_windows(all_bars, window_days=60, n_windows=3):
    return compute_rolling_windows(all_bars, window_days=window_days, n_windows=n_windows)


def _try_load_csim():
    try:
        from binance_worksteal.csim.fast_worksteal import run_worksteal_backtest_fast
        return run_worksteal_backtest_fast
    except Exception:
        return None


def build_sweep_cli_default_config(args: argparse.Namespace) -> WorkStealConfig:
    return WorkStealConfig(initial_cash=args.cash)


def build_sweep_base_config(args: argparse.Namespace, raw_argv: list[str]) -> WorkStealConfig:
    config = build_worksteal_config_from_args(
        base_config=build_sweep_cli_default_config(args),
        config_file=args.config_file,
        args=args,
        raw_argv=raw_argv,
        flag_to_field=SWEEP_CONFIG_FLAG_TO_FIELD,
    )
    if argv_has_flag(raw_argv, "--realistic"):
        config = replace(config, realistic_fill=True, daily_checkpoint_only=True)
    return config


def summarize_execution_mode(config: WorkStealConfig) -> dict[str, object]:
    realistic_fill = bool(config.realistic_fill)
    daily_checkpoint_only = bool(config.daily_checkpoint_only)
    if realistic_fill and daily_checkpoint_only:
        label = "REALISTIC (touch-fill + next-bar execution)"
    elif realistic_fill:
        label = "TOUCH_FILL_ONLY"
    elif daily_checkpoint_only:
        label = "NEXT_BAR_ONLY"
    else:
        label = "DEFAULT"
    return {
        "label": label,
        "realistic_fill": realistic_fill,
        "daily_checkpoint_only": daily_checkpoint_only,
        "realistic": realistic_fill and daily_checkpoint_only,
    }


def _apply_realistic_flag_explanation(explanation: dict[str, object]) -> None:
    changed_fields = explanation["changed_fields"]
    sources = explanation["sources"]
    cli_overrides = explanation["cli_overrides"]
    rendered_config = explanation["config"]
    config_file_overrides = explanation["config_file_overrides"]
    for field_name in ("realistic_fill", "daily_checkpoint_only"):
        change = changed_fields.get(field_name, {"default": False})
        change["source"] = "cli"
        change["value"] = True
        change["cli_value"] = True
        if field_name in config_file_overrides:
            change["config_file_value"] = config_file_overrides[field_name]
        changed_fields[field_name] = change
        sources[field_name] = "cli"
        cli_overrides[field_name] = True
        rendered_config[field_name] = True


def eval_config_single_window(
    all_bars,
    config,
    start_date,
    end_date,
    prepared_bars=None,
    use_csim_fn=None,
    warn_csim_failure=None,
    report_backtest_failure=None,
):
    if use_csim_fn is not None and config_supports_csim(config):
        try:
            metrics = use_csim_fn(
                all_bars, config, start_date=start_date, end_date=end_date,
            )
        except Exception as exc:
            if warn_csim_failure is not None:
                warn_csim_failure(
                    format_eval_failure("sweep", "C sim", config, start_date, end_date, exc)
                )
        else:
            if not metrics:
                return None
            metrics["n_trades"] = metrics.get("total_trades", 0)
            return metrics
    try:
        equity_df, trades, metrics = run_worksteal_backtest(
            all_bars, config, start_date=start_date, end_date=end_date, prepared_bars=prepared_bars,
        )
    except Exception as exc:
        if report_backtest_failure is not None:
            report_backtest_failure(
                format_eval_failure("sweep", "Python backtest", config, start_date, end_date, exc)
            )
        return None
    if not metrics:
        return None
    metrics["n_trades"] = metrics.get(
        "n_trades",
        count_completed_trades(trades),
    )
    return metrics


def eval_config_multi_window(
    all_bars,
    config,
    windows,
    prepared_bars=None,
    use_csim_fn=None,
    warn_csim_failure=None,
    report_backtest_failure=None,
):
    window_metrics = []
    for start, end in windows:
        m = eval_config_single_window(
            all_bars,
            config,
            start,
            end,
            prepared_bars,
            use_csim_fn,
            warn_csim_failure=warn_csim_failure,
            report_backtest_failure=report_backtest_failure,
        )
        if m is None:
            return None
        window_metrics.append(m)

    sortinos = [m.get("sortino", 0) for m in window_metrics]
    returns = [m.get("total_return_pct", 0) for m in window_metrics]
    drawdowns = [m.get("max_drawdown_pct", 0) for m in window_metrics]
    n_trades_list = [m.get("n_trades", 0) for m in window_metrics]
    win_rates = [m.get("win_rate", 0) for m in window_metrics]

    combined = {
        "mean_sortino": float(np.mean(sortinos)),
        "min_sortino": float(np.min(sortinos)),
        "max_sortino": float(np.max(sortinos)),
        "mean_return_pct": float(np.mean(returns)),
        "min_return_pct": float(np.min(returns)),
        "max_return_pct": float(np.max(returns)),
        "mean_drawdown_pct": float(np.mean(drawdowns)),
        "worst_drawdown_pct": float(np.min(drawdowns)),
        "mean_n_trades": float(np.mean(n_trades_list)),
        "total_n_trades": int(np.sum(n_trades_list)),
        "mean_win_rate": float(np.mean(win_rates)),
        "n_windows": len(windows),
    }
    for i, m in enumerate(window_metrics):
        combined[f"w{i}_sortino"] = m.get("sortino", 0)
        combined[f"w{i}_return_pct"] = m.get("total_return_pct", 0)
        combined[f"w{i}_drawdown_pct"] = m.get("max_drawdown_pct", 0)
        combined[f"w{i}_n_trades"] = m.get("n_trades", 0)
    return combined


def run_sweep(
    all_bars, windows, output_csv,
    max_trials=500,
    cash: float | None = None,
    realistic: bool | None = None,
    use_csim=False,
    base_config: WorkStealConfig | None = None,
    return_metadata: bool = False,
):
    template_config = base_config or WorkStealConfig()
    if cash is not None:
        template_config = replace(template_config, initial_cash=cash)
    if realistic is not None:
        template_config = replace(
            template_config,
            realistic_fill=bool(realistic),
            daily_checkpoint_only=bool(realistic),
        )
    mode_summary = summarize_execution_mode(template_config)
    csim_fn = _try_load_csim() if use_csim else None
    prepared_bars = prepare_backtest_bars(all_bars)
    warned_incompatible_csim = False
    warned_runtime_csim_failure = False
    incompatible_csim_issues = ""
    skipped_backtest_failures = 0
    logged_backtest_failures = 0
    failure_samples: list[str] = []
    csim_runtime_failure_samples: list[str] = []
    csim_runtime_failure_count = 0
    if use_csim and csim_fn is None:
        print("WARN: --use-csim requested but C lib not available, falling back to Python")

    def warn_csim_failure(message):
        nonlocal warned_runtime_csim_failure, csim_runtime_failure_count
        csim_runtime_failure_count += 1
        if len(csim_runtime_failure_samples) < 5:
            csim_runtime_failure_samples.append(message)
        if warned_runtime_csim_failure:
            return
        warned_runtime_csim_failure = True
        print(f"WARN: {message}; falling back to Python for this window")

    def report_backtest_failure(message):
        nonlocal skipped_backtest_failures, logged_backtest_failures
        skipped_backtest_failures += 1
        if len(failure_samples) < 10:
            failure_samples.append(message)
        if logged_backtest_failures >= 3:
            return
        logged_backtest_failures += 1
        print(f"WARN: {message}")

    keys = list(SWEEP_GRID.keys())
    values = list(SWEEP_GRID.values())
    total_combos, combos = sample_cartesian_product(values, max_trials=max_trials, seed=42)

    print(f"Sweep: {len(combos)} configs (from {total_combos} total)")
    print(f"Windows: {len(windows)}")
    for i, (s, e) in enumerate(windows):
        print(f"  W{i}: {s} to {e}")
    if mode_summary["label"] != "DEFAULT":
        print(f"Mode: {mode_summary['label']}")

    results = []
    best_min_sortino = -999
    best_config = None
    t0 = time.time()

    for i, combo in enumerate(combos):
        params = dict(zip(keys, combo))
        config = replace(template_config, **params)

        config_csim_fn = csim_fn if config_supports_csim(config) else None
        if csim_fn is not None and config_csim_fn is None and not warned_incompatible_csim:
            incompatible_csim_issues = summarize_csim_incompatibility(config)
            print(f"WARN: falling back to Python backtest for unsupported C sim features: {incompatible_csim_issues}")
            warned_incompatible_csim = True

        multi = eval_config_multi_window(
            all_bars,
            config,
            windows,
            prepared_bars,
            config_csim_fn,
            warn_csim_failure=warn_csim_failure,
            report_backtest_failure=report_backtest_failure,
        )
        if multi is None:
            continue

        row = {**params, **multi}
        results.append(row)

        min_sort = multi["min_sortino"]
        mean_sort = multi["mean_sortino"]
        mean_ret = multi["mean_return_pct"]
        worst_dd = multi["worst_drawdown_pct"]
        mean_wr = multi["mean_win_rate"]
        total_tr = multi["total_n_trades"]

        if min_sort > best_min_sortino:
            best_min_sortino = min_sort
            best_config = params
            marker = " ***BEST***"
        else:
            marker = ""

        if (i + 1) % 25 == 0 or marker:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed if elapsed > 0 else 0
            lev = params["max_leverage"]
            sh = "S" if params["enable_shorts"] else "L"
            sma = params["sma_filter_period"]
            print(f"  [{i+1}/{len(combos)} {rate:.1f}/s] "
                  f"min_sort={min_sort:6.2f} mean_sort={mean_sort:6.2f} "
                  f"ret={mean_ret:6.2f}% dd={worst_dd:6.2f}% "
                  f"wr={mean_wr:5.1f}% tr={total_tr:3d} "
                  f"dip={params['dip_pct']:.0%} tp={params['profit_target_pct']:.0%} "
                  f"sl={params['stop_loss_pct']:.0%} pos={params['max_positions']} "
                  f"lev={lev:.0f}x {sh} sma={sma}{marker}")

    if skipped_backtest_failures:
        print(
            "WARN: skipped "
            f"{skipped_backtest_failures} failed Python window evaluations during sweep"
        )
        suppressed = skipped_backtest_failures - logged_backtest_failures
        if suppressed > 0:
            print(f"WARN: suppressed {suppressed} additional failure details")

    if results:
        df = pd.DataFrame(results)
        df = df.sort_values("min_sortino", ascending=False)
        df.to_csv(output_csv, index=False)
        print(f"\nSaved {len(results)} results to {output_csv}")
        elapsed = time.time() - t0
        print(f"Total time: {elapsed:.1f}s ({len(results)/elapsed:.1f} configs/s)")

        print(f"\nTop 10 by WORST-CASE Sortino:")
        for rank, (_, r) in enumerate(df.head(10).iterrows(), 1):
            sh = "Y" if r.get("enable_shorts", False) else "N"
            print(f"  #{rank} min_sort={r['min_sortino']:6.2f} mean_sort={r['mean_sortino']:6.2f} "
                  f"ret={r['mean_return_pct']:7.2f}% dd={r['worst_drawdown_pct']:7.2f}% "
                  f"wr={r['mean_win_rate']:5.1f}% tr={r['total_n_trades']:4.0f} | "
                  f"dip={r['dip_pct']:.0%} tp={r['profit_target_pct']:.0%} "
                  f"sl={r['stop_loss_pct']:.0%} pos={r['max_positions']:.0f} "
                  f"hold={r['max_hold_days']:.0f} look={r['lookback_days']:.0f} "
                  f"ref={r['ref_price_method']} lev={r['max_leverage']:.0f}x {sh} "
                  f"sma={r.get('sma_filter_period',0):.0f} "
                  f"mb={r.get('market_breadth_filter',0):.1f}")
            for w in range(int(r.get("n_windows", 0))):
                ws = r.get(f"w{w}_sortino", 0)
                wr_ = r.get(f"w{w}_return_pct", 0)
                wd = r.get(f"w{w}_drawdown_pct", 0)
                wt = r.get(f"w{w}_n_trades", 0)
                print(f"       W{w}: sort={ws:6.2f} ret={wr_:6.2f}% dd={wd:6.2f}% tr={wt:.0f}")

    if best_config:
        print(f"\nRecommended production config (best worst-case Sortino={best_min_sortino:.2f}):")
        for k, v in best_config.items():
            print(f"  {k}: {v}")
        print(f"\nRationale: selected by worst-case Sortino across {len(windows)} "
              f"non-overlapping {windows[0][0]}..{windows[-1][1]} windows. "
              f"This minimizes regime-dependent overfitting.")
    metadata = {
        "skipped_backtest_failure_count": skipped_backtest_failures,
        "backtest_failure_samples": failure_samples,
        "suppressed_backtest_failure_count": max(skipped_backtest_failures - len(failure_samples), 0),
        "c_sim_requested": bool(use_csim),
        "c_sim_available": csim_fn is not None,
        "c_sim_incompatibility_detected": bool(warned_incompatible_csim),
        "c_sim_incompatibility_issues": incompatible_csim_issues or None,
        "c_sim_runtime_fallback_count": csim_runtime_failure_count,
        "c_sim_runtime_fallback_samples": csim_runtime_failure_samples,
    }
    if return_metadata:
        return results, metadata
    return results


def main(argv: list[str] | None = None):
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(description="Hyperparameter sweep for the Binance worksteal strategy.")
    parser.add_argument("--data-dir", default="trainingdata/train")
    add_symbol_selection_args(parser)
    add_require_full_universe_arg(parser)
    add_date_range_args(
        parser,
        start_dest="start_date",
        end_dest="end_date",
        include_days=True,
        days_default=60,
    )
    parser.add_argument("--n-windows", "--windows", dest="n_windows", type=int, default=3)
    parser.add_argument("--n-trials", type=int, default=500)
    parser.add_argument("--cash", type=float, default=10000.0)
    parser.add_argument("--output", default=None)
    add_config_file_arg(parser)
    add_print_config_arg(parser)
    add_explain_config_arg(parser)
    add_preview_run_arg(parser)
    add_summary_json_arg(parser, defaults_to_sidecar=True)
    parser.add_argument("--realistic", action="store_true",
                        help="Strict fill model: only fill when low <= buy_target")
    parser.add_argument("--use-csim", action="store_true",
                        help="Use C simulator for ~10x speedup")
    args = parser.parse_args(raw_argv)

    if args.output is None:
        tag = datetime.now().strftime("%Y%m%d")
        mode = "_realistic" if args.realistic else ""
        args.output = f"binance_worksteal/sweep_results_{tag}{mode}.csv"

    config_output_rc = maybe_handle_worksteal_config_output(
        args=args,
        build_config=lambda: build_sweep_base_config(args, raw_argv),
        base_config=build_sweep_cli_default_config(args),
        config_file=args.config_file,
        raw_argv=raw_argv,
        flag_to_field=SWEEP_CONFIG_FLAG_TO_FIELD,
        explain_adjuster=_apply_realistic_flag_explanation if argv_has_flag(raw_argv, "--realistic") else None,
    )
    if config_output_rc is not None:
        return config_output_rc

    summary_path = args.summary_json
    if summary_path is None and not args.list_symbols and not args.preview_run:
        summary_path = str(default_sidecar_json_path(args.output))

    def run():
        explicit_window, date_error = resolve_paired_date_range_with_error(
            start_date=args.start_date,
            end_date=args.end_date,
        )
        if date_error is not None:
            print(date_error["error"])
            error_extra: dict[str, object] = {"output_csv": args.output}
            if args.preview_run:
                error_extra["preview_only"] = True
            return 1, build_cli_error_summary(
                tool="sweep",
                error=date_error["error"],
                error_type=date_error["error_type"],
                data_dir=args.data_dir,
                config_file=args.config_file,
                extra=error_extra,
            )

        resolved, symbol_error = resolve_cli_symbols_with_error(
            symbols_arg=args.symbols,
            universe_file=args.universe_file,
            default_symbols=FULL_UNIVERSE,
        )
        if symbol_error is not None:
            print(symbol_error["error"])
            error_extra: dict[str, object] = {"output_csv": args.output}
            if args.list_symbols:
                error_extra["list_symbols_only"] = True
            if args.preview_run:
                error_extra["preview_only"] = True
            return 1, build_cli_error_summary(
                tool="sweep",
                error=symbol_error["error"],
                error_type=symbol_error["error_type"],
                data_dir=args.data_dir,
                config_file=args.config_file,
                extra=error_extra,
            )
        symbols, symbol_source = resolved
        if args.list_symbols:
            print_resolved_symbols(symbols, symbol_source)
            return 0, build_symbol_listing_summary(
                tool="sweep",
                data_dir=args.data_dir,
                symbol_source=symbol_source,
                symbols=symbols,
                config_file=args.config_file,
                extra={"output_csv": args.output},
            )

        try:
            base_config = build_sweep_base_config(args, raw_argv)
        except (FileNotFoundError, OSError, ValueError) as exc:
            config_error = build_cli_error(exc)
            print(config_error["error"])
            error_extra: dict[str, object] = {"output_csv": args.output}
            if args.preview_run:
                error_extra["preview_only"] = True
            return 1, build_cli_error_summary(
                tool="sweep",
                error=config_error["error"],
                error_type=config_error["error_type"],
                data_dir=args.data_dir,
                config_file=args.config_file,
                extra=error_extra,
            )
        if args.preview_run:
            mode_summary = summarize_execution_mode(base_config)
            date_mode = "fixed_range" if explicit_window is not None else "rolling_windows"
            total_grid_size = cartesian_product_size(list(SWEEP_GRID.values()))
            extra = {
                "date_mode": date_mode,
                "start_date": explicit_window[0] if explicit_window is not None else None,
                "end_date": explicit_window[1] if explicit_window is not None else None,
                "days": None if explicit_window is not None else args.days,
                "requested_window_count": 1 if explicit_window is not None else args.n_windows,
                "n_trials_requested": args.n_trials,
                "total_grid_size": total_grid_size,
                "output_csv": args.output,
                "summary_json": summary_path,
                "use_csim_requested": bool(args.use_csim),
                "require_full_universe": bool(args.require_full_universe),
                "execution_mode": mode_summary["label"],
                "realistic_fill": mode_summary["realistic_fill"],
                "daily_checkpoint_only": mode_summary["daily_checkpoint_only"],
            }
            print_run_preview(
                tool="sweep",
                sections=[
                    (
                        "Inputs",
                        (
                            ("data_dir", args.data_dir),
                            ("symbol_source", symbol_source),
                            ("symbol_count", len(symbols)),
                            ("symbols", symbols),
                        ),
                    ),
                    (
                        "Execution",
                        (
                            ("date_mode", date_mode),
                            ("start_date", extra["start_date"]),
                            ("end_date", extra["end_date"]),
                            ("days", extra["days"]),
                            ("requested_windows", extra["requested_window_count"]),
                            ("n_trials", args.n_trials),
                            ("grid_size", total_grid_size),
                            ("use_csim", args.use_csim),
                            ("execution_mode", mode_summary["label"]),
                            ("require_full_universe", args.require_full_universe),
                        ),
                    ),
                    (
                        "Outputs",
                        (
                            ("output_csv", args.output),
                            ("summary_json", summary_path),
                            ("config_file", args.config_file),
                        ),
                    ),
                ],
            )
            return 0, build_preview_run_summary(
                tool="sweep",
                data_dir=args.data_dir,
                symbol_source=symbol_source,
                symbols=symbols,
                config_file=args.config_file,
                config=base_config,
                extra=extra,
            )
        if args.config_file:
            print(f"Loaded config overrides from {args.config_file}")

        print(f"Using {len(symbols)} symbols from {symbol_source}")
        all_bars, load_summary, load_failure = load_bars_with_summary(
            data_dir=args.data_dir,
            requested_symbols=symbols,
            load_bars=load_daily_bars,
            loading_message=f"Loading {len(symbols)} symbols from {args.data_dir}",
            no_data_message="ERROR: No data",
            return_summary_on_empty=True,
            return_failure_on_error=True,
            strict_retry_command=build_strict_retry_command(module="binance_worksteal.sweep", argv=raw_argv),
        )
        if load_failure is not None:
            mode_summary = summarize_execution_mode(base_config)
            return 1, build_empty_sweep_run_summary(
                tool="sweep",
                data_dir=args.data_dir,
                symbol_source=symbol_source,
                symbols=symbols,
                load_summary=load_summary,
                data_coverage=None,
                config_file=args.config_file,
                base_config=base_config,
                output_csv=args.output,
                swept_fields=sorted(SWEEP_GRID),
                extra={
                    "n_trials_requested": args.n_trials,
                    "execution_mode": mode_summary["label"],
                    "realistic": mode_summary["realistic"],
                    "realistic_fill": mode_summary["realistic_fill"],
                    "daily_checkpoint_only": mode_summary["daily_checkpoint_only"],
                    "use_csim_requested": bool(args.use_csim),
                    "error": load_failure["error"],
                    "load_failure": load_failure,
                    "require_full_universe": bool(args.require_full_universe),
                },
            )
        if not all_bars:
            mode_summary = summarize_execution_mode(base_config)
            return 1, build_empty_sweep_run_summary(
                tool="sweep",
                data_dir=args.data_dir,
                symbol_source=symbol_source,
                symbols=symbols,
                load_summary=load_summary,
                data_coverage=None,
                config_file=args.config_file,
                base_config=base_config,
                output_csv=args.output,
                swept_fields=sorted(SWEEP_GRID),
                extra={
                    "n_trials_requested": args.n_trials,
                    "execution_mode": mode_summary["label"],
                    "realistic": mode_summary["realistic"],
                    "realistic_fill": mode_summary["realistic_fill"],
                    "daily_checkpoint_only": mode_summary["daily_checkpoint_only"],
                    "use_csim_requested": bool(args.use_csim),
                    "error": "ERROR: No data",
                    "require_full_universe": bool(args.require_full_universe),
                },
            )

        missing_data_error = require_full_universe_or_print_error(
            require_full_universe=args.require_full_universe,
            load_summary=load_summary,
        )
        if missing_data_error is not None:
            mode_summary = summarize_execution_mode(base_config)
            return 1, build_empty_sweep_run_summary(
                tool="sweep",
                data_dir=args.data_dir,
                symbol_source=symbol_source,
                symbols=symbols,
                load_summary=load_summary,
                data_coverage=None,
                config_file=args.config_file,
                base_config=base_config,
                output_csv=args.output,
                swept_fields=sorted(SWEEP_GRID),
                extra={
                    "n_trials_requested": args.n_trials,
                    "execution_mode": mode_summary["label"],
                    "realistic": mode_summary["realistic"],
                    "realistic_fill": mode_summary["realistic_fill"],
                    "daily_checkpoint_only": mode_summary["daily_checkpoint_only"],
                    "use_csim_requested": bool(args.use_csim),
                    "error": missing_data_error,
                    "require_full_universe": bool(args.require_full_universe),
                },
            )

        if explicit_window is not None:
            windows = [explicit_window]
        else:
            windows = build_windows(all_bars, window_days=args.days, n_windows=args.n_windows)
            if not windows:
                error = "ERROR: Not enough data for rolling windows"
                print(error)
                mode_summary = summarize_execution_mode(base_config)
                return 1, build_empty_sweep_run_summary(
                    tool="sweep",
                    data_dir=args.data_dir,
                    symbol_source=symbol_source,
                    symbols=symbols,
                    load_summary=load_summary,
                    data_coverage=None,
                    config_file=args.config_file,
                    base_config=base_config,
                    output_csv=args.output,
                    swept_fields=sorted(SWEEP_GRID),
                    extra={
                        "n_trials_requested": args.n_trials,
                        "requested_window_count": args.n_windows,
                        "window_days": args.days,
                        "execution_mode": mode_summary["label"],
                        "realistic": mode_summary["realistic"],
                        "realistic_fill": mode_summary["realistic_fill"],
                        "daily_checkpoint_only": mode_summary["daily_checkpoint_only"],
                        "use_csim_requested": bool(args.use_csim),
                        "error": error,
                        "require_full_universe": bool(args.require_full_universe),
                    },
                )
            if len(windows) < args.n_windows:
                print(
                    f"WARN: only {len(windows)}/{args.n_windows} rolling windows of {args.days} days "
                    "fit within loaded data coverage"
                )
        data_coverage = print_window_span_coverage_summary(all_bars, windows)

        sweep_output = run_sweep(
            all_bars, windows, args.output,
            max_trials=args.n_trials,
            use_csim=args.use_csim,
            base_config=base_config,
            return_metadata=True,
        )
        if isinstance(sweep_output, tuple) and len(sweep_output) == 2:
            results, sweep_metadata = sweep_output
        else:
            results = sweep_output
            sweep_metadata = {
                "skipped_backtest_failure_count": 0,
                "backtest_failure_samples": [],
                "suppressed_backtest_failure_count": 0,
                "c_sim_requested": bool(args.use_csim),
                "c_sim_available": False,
                "c_sim_incompatibility_detected": False,
                "c_sim_incompatibility_issues": None,
                "c_sim_runtime_fallback_count": 0,
                "c_sim_runtime_fallback_samples": [],
            }
        mode_summary = summarize_execution_mode(base_config)
        ranked = sorted(results, key=lambda row: row.get("min_sortino", float("-inf")), reverse=True)
        if ranked:
            recommendation = prepare_sweep_recommendation_artifacts(
                ranked_results=ranked,
                base_config=base_config,
                swept_fields=sorted(SWEEP_GRID),
                output_csv=args.output,
                data_dir=args.data_dir,
                symbols_arg=args.symbols,
                universe_file=args.universe_file,
                start_date=min(start for start, _end in windows),
                end_date=max(end for _start, end in windows),
                eval_days=args.days if explicit_window is None else None,
                eval_windows=args.n_windows if explicit_window is None else None,
                eval_start_date=explicit_window[0] if explicit_window is not None else None,
                eval_end_date=explicit_window[1] if explicit_window is not None else None,
            )
        else:
            print("ERROR: No valid sweep results")
            recommendation = empty_sweep_recommendation()
        summary_payload = build_sweep_run_summary(
            tool="sweep",
            data_dir=args.data_dir,
            symbol_source=symbol_source,
            symbols=symbols,
            load_summary=load_summary,
            data_coverage=data_coverage,
            config_file=args.config_file,
            base_config=base_config,
            output_csv=args.output,
            swept_fields=sorted(SWEEP_GRID),
            windows=windows,
            results_count=len(results),
            recommendation=recommendation,
            best_result=ranked[0] if ranked else None,
            top_results=ranked[:5],
            extra={
                "n_trials_requested": args.n_trials,
                "execution_mode": mode_summary["label"],
                "realistic": mode_summary["realistic"],
                "realistic_fill": mode_summary["realistic_fill"],
                "daily_checkpoint_only": mode_summary["daily_checkpoint_only"],
                "use_csim_requested": bool(args.use_csim),
                "skipped_backtest_failure_count": sweep_metadata["skipped_backtest_failure_count"],
                "backtest_failure_samples": sweep_metadata["backtest_failure_samples"],
                "suppressed_backtest_failure_count": sweep_metadata["suppressed_backtest_failure_count"],
                "c_sim_available": sweep_metadata["c_sim_available"],
                "c_sim_incompatibility_detected": sweep_metadata["c_sim_incompatibility_detected"],
                "c_sim_incompatibility_issues": sweep_metadata["c_sim_incompatibility_issues"],
                "c_sim_runtime_fallback_count": sweep_metadata["c_sim_runtime_fallback_count"],
                "c_sim_runtime_fallback_samples": sweep_metadata["c_sim_runtime_fallback_samples"],
            },
        )
        announce_sweep_artifacts(
            recommendation=recommendation,
            output_csv=args.output,
            summary_json_file=summary_path,
            module="binance_worksteal.sweep",
            argv=raw_argv,
            include_output_csv=bool(ranked),
            warnings=summary_payload.get("warnings"),
        )
        return (0 if results else 1), summary_payload

    return run_with_optional_summary(
        summary_path,
        run,
        module="binance_worksteal.sweep",
        argv=raw_argv,
        announce_summary_write_on_error=args.summary_json is not None,
        announce_summary_write_on_success=False,
        announce_artifact_manifest_on_success=bool(
            args.summary_json is not None and (args.list_symbols or args.preview_run)
        ),
    )


if __name__ == "__main__":
    sys.exit(main())
