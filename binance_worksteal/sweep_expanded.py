#!/usr/bin/env python3
"""Expanded hyperparameter sweep for work-stealing strategy.

Focused grid exploring dip thresholds, leverage, position counts,
trailing stops, profit targets, and stop losses. Uses C simulator
batch mode when available for ~10x speedup.

Multi-window evaluation with safety score ranking.
"""
from __future__ import annotations

import argparse
import sys
import time
from dataclasses import replace
from datetime import datetime
from multiprocessing import Pool, cpu_count
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
    WorkStealConfig,
    compute_avg_hold_days_from_trades,
    count_completed_trades,
    load_daily_bars,
    prepare_backtest_bars,
    run_worksteal_backtest,
)
from binance_worksteal.backtest import FULL_UNIVERSE
from binance_worksteal.csim.compat import (
    assert_csim_compatible_configs,
)

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

SWEEP_GRID = {
    "dip_pct": [0.08, 0.10, 0.12, 0.15, 0.20],
    "max_leverage": [1.0, 2.0, 3.0, 5.0],
    "max_positions": [5, 7, 10, 15],
    "trailing_stop_pct": [0.02, 0.03, 0.05],
    "profit_target_pct": [0.10, 0.15, 0.20],
    "stop_loss_pct": [0.08, 0.10, 0.15],
}

EXPANDED_SWEEP_CONFIG_FLAG_TO_FIELD = {
    "--cash": ("initial_cash", "cash"),
}


def generate_grid(grid=None, max_trials=None, seed=42):
    grid = grid or SWEEP_GRID
    keys = list(grid.keys())
    values = list(grid.values())
    capped_trials = max_trials or None
    _total, combos = sample_cartesian_product(values, max_trials=capped_trials, seed=seed)
    return keys, combos


def build_expanded_sweep_cli_default_config(args: argparse.Namespace) -> WorkStealConfig:
    return WorkStealConfig(
        initial_cash=args.cash,
        sma_filter_period=20,
        lookback_days=20,
        max_hold_days=14,
        proximity_pct=0.03,
        ref_price_method="high",
    )


def build_expanded_sweep_base_config(args: argparse.Namespace, raw_argv: list[str]) -> WorkStealConfig:
    return build_worksteal_config_from_args(
        base_config=build_expanded_sweep_cli_default_config(args),
        config_file=args.config_file,
        args=args,
        raw_argv=raw_argv,
        flag_to_field=EXPANDED_SWEEP_CONFIG_FLAG_TO_FIELD,
    )


def combo_to_config(
    keys,
    combo,
    cash: float | None = None,
    base_config: WorkStealConfig | None = None,
):
    params = dict(zip(keys, combo))
    template_config = base_config or WorkStealConfig(
        sma_filter_period=20,
        lookback_days=20,
        max_hold_days=14,
        proximity_pct=0.03,
        ref_price_method="high",
    )
    if cash is not None:
        template_config = replace(template_config, initial_cash=cash)
    return replace(template_config, **params)


def compute_safety_score(mean_sortino, max_drawdown_pct):
    dd_abs = max(abs(max_drawdown_pct), 0.01)
    return mean_sortino / dd_abs


def build_windows(all_bars, window_days=60, n_windows=3):
    return compute_rolling_windows(all_bars, window_days=window_days, n_windows=n_windows)


def _try_load_csim_batch():
    try:
        from binance_worksteal.csim.fast_worksteal import run_worksteal_batch_fast
        return run_worksteal_batch_fast
    except Exception:
        return None


def eval_config_single_window_python(
    all_bars,
    config,
    start_date,
    end_date,
    prepared_bars=None,
    report_backtest_failure=None,
):
    try:
        equity_df, trades, metrics = run_worksteal_backtest(
            all_bars, config, start_date=start_date, end_date=end_date, prepared_bars=prepared_bars,
        )
    except Exception as exc:
        if report_backtest_failure is not None:
            report_backtest_failure(
                format_eval_failure(
                    "sweep_expanded",
                    "Python backtest",
                    config,
                    start_date,
                    end_date,
                    exc,
                )
            )
        return None
    if not metrics:
        return None
    metrics["n_trades"] = metrics.get("n_trades", count_completed_trades(trades))
    metrics["avg_hold_days"] = compute_avg_hold_days_from_trades(trades)
    return metrics


def eval_config_multi_window_python(all_bars, config, windows, prepared_bars=None, report_backtest_failure=None):
    window_metrics = []
    for start, end in windows:
        m = eval_config_single_window_python(
            all_bars,
            config,
            start,
            end,
            prepared_bars=prepared_bars,
            report_backtest_failure=report_backtest_failure,
        )
        if m is None:
            return None
        window_metrics.append(m)
    return _aggregate_window_metrics(window_metrics)


def _aggregate_window_metrics(window_metrics):
    sortinos = [m.get("sortino", 0) for m in window_metrics]
    returns = [m.get("total_return_pct", 0) for m in window_metrics]
    drawdowns = [m.get("max_drawdown_pct", 0) for m in window_metrics]
    n_trades_list = [m.get("n_trades", 0) for m in window_metrics]
    win_rates = [m.get("win_rate", 0) for m in window_metrics]
    avg_holds = [m.get("avg_hold_days", 0) for m in window_metrics]

    mean_sortino = float(np.mean(sortinos))
    worst_dd = float(np.min(drawdowns))
    safety = compute_safety_score(mean_sortino, worst_dd)

    combined = {
        "mean_sortino": mean_sortino,
        "min_sortino": float(np.min(sortinos)),
        "mean_return_pct": float(np.mean(returns)),
        "max_drawdown_pct": worst_dd,
        "mean_win_rate": float(np.mean(win_rates)),
        "total_n_trades": int(np.sum(n_trades_list)),
        "mean_n_trades": float(np.mean(n_trades_list)),
        "avg_hold_days": float(np.mean(avg_holds)),
        "safety_score": safety,
        "n_windows": len(window_metrics),
    }
    for i, m in enumerate(window_metrics):
        combined[f"w{i}_sortino"] = m.get("sortino", 0)
        combined[f"w{i}_return_pct"] = m.get("total_return_pct", 0)
        combined[f"w{i}_drawdown_pct"] = m.get("max_drawdown_pct", 0)
        combined[f"w{i}_n_trades"] = m.get("n_trades", 0)
    return combined


def _eval_batch_csim_window(batch_fn, all_bars, configs, start, end):
    results = batch_fn(all_bars, configs, start_date=start, end_date=end)
    out = []
    for r in results:
        if not r or (r.get("total_trades", 0) == 0 and r.get("total_return", 0) == 0):
            out.append(None)
        else:
            r["n_trades"] = r.get("total_trades", 0)
            r["avg_hold_days"] = 0.0
            out.append(r)
    return out


def eval_batch_multi_window_csim(batch_fn, all_bars, configs, windows):
    all_window_results = []
    for start, end in windows:
        wr = _eval_batch_csim_window(batch_fn, all_bars, configs, start, end)
        all_window_results.append(wr)

    combined = []
    for ci in range(len(configs)):
        wms = []
        failed = False
        for wi in range(len(windows)):
            m = all_window_results[wi][ci]
            if m is None:
                failed = True
                break
            wms.append(m)
        if failed:
            combined.append(None)
        else:
            combined.append(_aggregate_window_metrics(wms))
    return combined


# For multiprocessing fallback
_mp_all_bars = None
_mp_windows = None
_mp_prepared_bars = None


def _init_mp_worker(all_bars, windows):
    global _mp_all_bars, _mp_windows, _mp_prepared_bars
    _mp_all_bars = all_bars
    _mp_windows = windows
    _mp_prepared_bars = prepare_backtest_bars(all_bars)


def _mp_eval_config(args):
    keys, combo, cash, base_config = args
    config = combo_to_config(keys, combo, cash=cash, base_config=base_config)
    failures = []
    result = eval_config_multi_window_python(
        _mp_all_bars,
        config,
        _mp_windows,
        prepared_bars=_mp_prepared_bars,
        report_backtest_failure=failures.append,
    )
    return result, failures[0] if failures else None


def get_per_symbol_pnl(all_bars, config, windows):
    sym_pnl = {}
    for start, end in windows:
        try:
            _, trades, _ = run_worksteal_backtest(
                all_bars, config, start_date=start, end_date=end,
            )
        except Exception:
            continue
        exits = [t for t in trades if t.side in ("sell", "cover")]
        for t in exits:
            sym_pnl[t.symbol] = sym_pnl.get(t.symbol, 0) + t.pnl
    return sym_pnl


def run_sweep(
    all_bars, windows, output_csv,
    max_trials=None,
    cash: float | None = None,
    n_workers=None,
    base_config: WorkStealConfig | None = None,
    return_metadata: bool = False,
):
    batch_fn = _try_load_csim_batch()
    prepared_bars = prepare_backtest_bars(all_bars)
    batch_available = batch_fn is not None
    skipped_backtest_failures = 0
    logged_backtest_failures = 0
    backtest_failure_samples: list[str] = []
    c_batch_runtime_failure_samples: list[str] = []
    c_batch_runtime_failure_count = 0
    c_batch_incompatibility_issues = ""
    c_batch_used_any = False

    def report_backtest_failure(message):
        nonlocal skipped_backtest_failures, logged_backtest_failures
        skipped_backtest_failures += 1
        if len(backtest_failure_samples) < 10:
            backtest_failure_samples.append(message)
        if logged_backtest_failures >= 3:
            return
        logged_backtest_failures += 1
        print(f"WARN: {message}")

    keys, combos = generate_grid(max_trials=max_trials)
    total = len(combos)
    configs_all = [combo_to_config(keys, c, cash=cash, base_config=base_config) for c in combos]

    if batch_fn is not None:
        try:
            assert_csim_compatible_configs(configs_all, context="sweep_expanded")
        except ValueError as exc:
            c_batch_incompatibility_issues = str(exc)
            print(
                "WARN: disabling C batch sweep because generated configs use unsupported "
                f"Python-only features: {exc}"
            )
            batch_fn = None

    use_csim = batch_fn is not None

    print(f"Sweep: {total} configs (grid total: {cartesian_product_size(list(SWEEP_GRID.values()))})")
    print(f"Windows: {len(windows)}")
    for i, (s, e) in enumerate(windows):
        print(f"  W{i}: {s} to {e}")
    print(f"Simulator: {'C batch' if use_csim else 'Python'}")

    t0 = time.time()
    results = []

    if use_csim:
        BATCH_SIZE = 256
        params_all = [dict(zip(keys, c)) for c in combos]

        done = 0
        for bi in range(0, total, BATCH_SIZE):
            batch_configs = configs_all[bi:bi + BATCH_SIZE]
            batch_params = params_all[bi:bi + BATCH_SIZE]
            try:
                c_batch_used_any = True
                batch_results = eval_batch_multi_window_csim(batch_fn, all_bars, batch_configs, windows)
            except Exception as exc:
                c_batch_runtime_failure_count += 1
                sample_config = batch_configs[0]
                message = format_eval_failure(
                    "sweep_expanded",
                    "C batch",
                    sample_config,
                    windows[0][0] if windows else None,
                    windows[-1][1] if windows else None,
                    exc,
                )
                if len(c_batch_runtime_failure_samples) < 5:
                    c_batch_runtime_failure_samples.append(message)
                print(f"WARN: {message}; falling back to Python for remaining configs")
                use_csim = False
                for ci in range(bi, total):
                    config = configs_all[ci]
                    multi = eval_config_multi_window_python(
                        all_bars,
                        config,
                        windows,
                        prepared_bars=prepared_bars,
                        report_backtest_failure=report_backtest_failure,
                    )
                    if multi is None:
                        continue
                    params = dict(zip(keys, combos[ci]))
                    row = {**params, **multi}
                    results.append(row)
                break

            for pi, multi in enumerate(batch_results):
                if multi is None:
                    continue
                row = {**batch_params[pi], **multi}
                results.append(row)

            done += len(batch_configs)
            elapsed = time.time() - t0
            rate = done / elapsed if elapsed > 0 else 0
            print(f"  [{done}/{total} {rate:.1f}/s] {len(results)} valid", flush=True)
    else:
        n_workers = n_workers or max(1, cpu_count() - 1)
        if n_workers > 1 and total > 10:
            print(f"  Using {n_workers} workers")
            args_list = [(keys, c, cash, base_config) for c in combos]
            chunksize = max(1, total // (n_workers * 4))
            with Pool(n_workers, initializer=_init_mp_worker, initargs=(all_bars, windows)) as pool:
                iterator = pool.imap(_mp_eval_config, args_list, chunksize=chunksize)
                if tqdm:
                    iterator = tqdm(iterator, total=total, desc="Sweep")
                for ci, item in enumerate(iterator):
                    multi, failure_message = item
                    if failure_message:
                        report_backtest_failure(failure_message)
                    if multi is None:
                        continue
                    params = dict(zip(keys, combos[ci]))
                    row = {**params, **multi}
                    results.append(row)
        else:
            iterator = range(total)
            if tqdm:
                iterator = tqdm(iterator, desc="Sweep")
            for ci in iterator:
                config = configs_all[ci]
                multi = eval_config_multi_window_python(
                    all_bars,
                    config,
                    windows,
                    prepared_bars=prepared_bars,
                    report_backtest_failure=report_backtest_failure,
                )
                if multi is None:
                    continue
                params = dict(zip(keys, combos[ci]))
                row = {**params, **multi}
                results.append(row)

                if (ci + 1) % 50 == 0 and not tqdm:
                    elapsed = time.time() - t0
                    rate = (ci + 1) / elapsed if elapsed > 0 else 0
                    print(f"  [{ci+1}/{total} {rate:.1f}/s] {len(results)} valid", flush=True)

    elapsed = time.time() - t0

    if skipped_backtest_failures:
        print(
            "WARN: skipped "
            f"{skipped_backtest_failures} failed Python window evaluations during expanded sweep"
        )
        suppressed = skipped_backtest_failures - logged_backtest_failures
        if suppressed > 0:
            print(f"WARN: suppressed {suppressed} additional failure details")

    if not results:
        metadata = {
            "skipped_backtest_failure_count": skipped_backtest_failures,
            "backtest_failure_samples": backtest_failure_samples,
            "suppressed_backtest_failure_count": max(skipped_backtest_failures - len(backtest_failure_samples), 0),
            "c_batch_available": batch_available,
            "c_batch_used": c_batch_used_any,
            "c_batch_incompatibility_detected": bool(c_batch_incompatibility_issues),
            "c_batch_incompatibility_issues": c_batch_incompatibility_issues or None,
            "c_batch_runtime_fallback_count": c_batch_runtime_failure_count,
            "c_batch_runtime_fallback_samples": c_batch_runtime_failure_samples,
        }
        if return_metadata:
            return results, metadata
        return results

    df = pd.DataFrame(results)
    df = df.sort_values("safety_score", ascending=False)
    df.to_csv(output_csv, index=False)
    print(f"\nSaved {len(results)} results to {output_csv}")
    print(f"Total time: {elapsed:.1f}s ({len(results)/max(elapsed,0.001):.1f} configs/s)")

    print(f"\nTop 20 by safety_score:")
    for rank, (_, r) in enumerate(df.head(20).iterrows(), 1):
        print(f"  #{rank:2d} safety={r['safety_score']:7.2f} "
              f"sort={r['mean_sortino']:6.2f} ret={r['mean_return_pct']:7.2f}% "
              f"dd={r['max_drawdown_pct']:7.2f}% wr={r['mean_win_rate']:5.1f}% "
              f"tr={r['total_n_trades']:4.0f} | "
              f"dip={r['dip_pct']:.0%} tp={r['profit_target_pct']:.0%} "
              f"sl={r['stop_loss_pct']:.0%} trail={r['trailing_stop_pct']:.0%} "
              f"pos={r['max_positions']:.0f} lev={r['max_leverage']:.0f}x")

    # Per-symbol PnL for top 20
    if not use_csim:
        print(f"\nPer-symbol PnL breakdown (top 20):")
        for rank, (_, r) in enumerate(df.head(20).iterrows(), 1):
            combo = tuple(r[k] for k in keys)
            config = combo_to_config(keys, combo, cash=cash, base_config=base_config)
            sym_pnl = get_per_symbol_pnl(all_bars, config, windows)
            if sym_pnl:
                top_syms = sorted(sym_pnl.items(), key=lambda x: x[1], reverse=True)[:5]
                bot_syms = sorted(sym_pnl.items(), key=lambda x: x[1])[:3]
                top_str = " ".join(f"{s}:${p:.0f}" for s, p in top_syms)
                bot_str = " ".join(f"{s}:${p:.0f}" for s, p in bot_syms)
                print(f"  #{rank:2d} best=[{top_str}] worst=[{bot_str}]")

    metadata = {
        "skipped_backtest_failure_count": skipped_backtest_failures,
        "backtest_failure_samples": backtest_failure_samples,
        "suppressed_backtest_failure_count": max(skipped_backtest_failures - len(backtest_failure_samples), 0),
        "c_batch_available": batch_available,
        "c_batch_used": c_batch_used_any,
        "c_batch_incompatibility_detected": bool(c_batch_incompatibility_issues),
        "c_batch_incompatibility_issues": c_batch_incompatibility_issues or None,
        "c_batch_runtime_fallback_count": c_batch_runtime_failure_count,
        "c_batch_runtime_fallback_samples": c_batch_runtime_failure_samples,
    }
    if return_metadata:
        return results, metadata
    return results


def main(argv: list[str] | None = None):
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(description="Expanded hyperparameter sweep for work-stealing strategy")
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
    parser.add_argument("--windows", "--n-windows", dest="windows", type=int, default=3)
    parser.add_argument("--max-trials", type=int, default=None)
    parser.add_argument("--cash", type=float, default=10000.0)
    parser.add_argument("--output", default=None)
    add_config_file_arg(parser)
    add_print_config_arg(parser)
    add_explain_config_arg(parser)
    add_preview_run_arg(parser)
    add_summary_json_arg(parser, defaults_to_sidecar=True)
    parser.add_argument("--workers", type=int, default=None)
    args = parser.parse_args(raw_argv)

    if args.output is None:
        tag = datetime.now().strftime("%Y%m%d")
        args.output = f"binance_worksteal/sweep_expanded_{tag}.csv"

    config_output_rc = maybe_handle_worksteal_config_output(
        args=args,
        build_config=lambda: build_expanded_sweep_base_config(args, raw_argv),
        base_config=build_expanded_sweep_cli_default_config(args),
        config_file=args.config_file,
        raw_argv=raw_argv,
        flag_to_field=EXPANDED_SWEEP_CONFIG_FLAG_TO_FIELD,
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
                tool="sweep_expanded",
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
                tool="sweep_expanded",
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
                tool="sweep_expanded",
                data_dir=args.data_dir,
                symbol_source=symbol_source,
                symbols=symbols,
                config_file=args.config_file,
                extra={"output_csv": args.output},
            )

        try:
            base_config = build_expanded_sweep_base_config(args, raw_argv)
        except (FileNotFoundError, OSError, ValueError) as exc:
            config_error = build_cli_error(exc)
            print(config_error["error"])
            error_extra: dict[str, object] = {"output_csv": args.output}
            if args.preview_run:
                error_extra["preview_only"] = True
            return 1, build_cli_error_summary(
                tool="sweep_expanded",
                error=config_error["error"],
                error_type=config_error["error_type"],
                data_dir=args.data_dir,
                config_file=args.config_file,
                extra=error_extra,
            )
        if args.preview_run:
            date_mode = "fixed_range" if explicit_window is not None else "rolling_windows"
            total_grid_size = cartesian_product_size(list(SWEEP_GRID.values()))
            extra = {
                "date_mode": date_mode,
                "start_date": explicit_window[0] if explicit_window is not None else None,
                "end_date": explicit_window[1] if explicit_window is not None else None,
                "days": None if explicit_window is not None else args.days,
                "requested_window_count": 1 if explicit_window is not None else args.windows,
                "max_trials_requested": args.max_trials,
                "total_grid_size": total_grid_size,
                "worker_count": args.workers,
                "output_csv": args.output,
                "summary_json": summary_path,
                "require_full_universe": args.require_full_universe,
            }
            print_run_preview(
                tool="sweep_expanded",
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
                            ("max_trials", args.max_trials),
                            ("grid_size", total_grid_size),
                            ("workers", args.workers),
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
                tool="sweep_expanded",
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
            strict_retry_command=build_strict_retry_command(module="binance_worksteal.sweep_expanded", argv=raw_argv),
        )
        if load_failure is not None:
            return 1, build_empty_sweep_run_summary(
                tool="sweep_expanded",
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
                    "max_trials_requested": args.max_trials,
                    "worker_count": args.workers,
                    "error": load_failure["error"],
                    "load_failure": load_failure,
                    "require_full_universe": bool(args.require_full_universe),
                },
            )
        if not all_bars:
            return 1, build_empty_sweep_run_summary(
                tool="sweep_expanded",
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
                    "max_trials_requested": args.max_trials,
                    "worker_count": args.workers,
                    "error": "ERROR: No data",
                    "require_full_universe": bool(args.require_full_universe),
                },
            )

        missing_data_error = require_full_universe_or_print_error(
            require_full_universe=args.require_full_universe,
            load_summary=load_summary,
        )
        if missing_data_error is not None:
            return 1, build_empty_sweep_run_summary(
                tool="sweep_expanded",
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
                    "max_trials_requested": args.max_trials,
                    "worker_count": args.workers,
                    "error": missing_data_error,
                    "require_full_universe": bool(args.require_full_universe),
                },
            )

        if explicit_window is not None:
            windows = [explicit_window]
        else:
            windows = build_windows(all_bars, window_days=args.days, n_windows=args.windows)
            if not windows:
                error = "ERROR: Not enough data for rolling windows"
                print(error)
                return 1, build_empty_sweep_run_summary(
                    tool="sweep_expanded",
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
                        "max_trials_requested": args.max_trials,
                        "requested_window_count": args.windows,
                        "window_days": args.days,
                        "worker_count": args.workers,
                        "error": error,
                        "require_full_universe": bool(args.require_full_universe),
                    },
                )
            if len(windows) < args.windows:
                print(
                    f"WARN: only {len(windows)}/{args.windows} rolling windows of {args.days} days "
                    "fit within loaded data coverage"
                )
        data_coverage = print_window_span_coverage_summary(all_bars, windows)

        sweep_output = run_sweep(
            all_bars, windows, args.output,
            max_trials=args.max_trials,
            n_workers=args.workers,
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
                "c_batch_available": False,
                "c_batch_used": False,
                "c_batch_incompatibility_detected": False,
                "c_batch_incompatibility_issues": None,
                "c_batch_runtime_fallback_count": 0,
                "c_batch_runtime_fallback_samples": [],
            }
        ranked = sorted(results, key=lambda row: row.get("safety_score", float("-inf")), reverse=True)
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
                eval_windows=args.windows if explicit_window is None else None,
                eval_start_date=explicit_window[0] if explicit_window is not None else None,
                eval_end_date=explicit_window[1] if explicit_window is not None else None,
            )
        else:
            print("ERROR: No valid expanded sweep results")
            recommendation = empty_sweep_recommendation()
        summary_payload = build_sweep_run_summary(
            tool="sweep_expanded",
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
                "max_trials_requested": args.max_trials,
                "worker_count": args.workers,
                "skipped_backtest_failure_count": sweep_metadata["skipped_backtest_failure_count"],
                "backtest_failure_samples": sweep_metadata["backtest_failure_samples"],
                "suppressed_backtest_failure_count": sweep_metadata["suppressed_backtest_failure_count"],
                "c_batch_available": sweep_metadata["c_batch_available"],
                "c_batch_used": sweep_metadata["c_batch_used"],
                "c_batch_incompatibility_detected": sweep_metadata["c_batch_incompatibility_detected"],
                "c_batch_incompatibility_issues": sweep_metadata["c_batch_incompatibility_issues"],
                "c_batch_runtime_fallback_count": sweep_metadata["c_batch_runtime_fallback_count"],
                "c_batch_runtime_fallback_samples": sweep_metadata["c_batch_runtime_fallback_samples"],
                "require_full_universe": bool(args.require_full_universe),
            },
        )
        announce_sweep_artifacts(
            recommendation=recommendation,
            output_csv=args.output,
            summary_json_file=summary_path,
            module="binance_worksteal.sweep_expanded",
            argv=raw_argv,
            include_output_csv=bool(ranked),
            warnings=summary_payload.get("warnings"),
        )
        return (0 if results else 1), summary_payload

    return run_with_optional_summary(
        summary_path,
        run,
        module="binance_worksteal.sweep_expanded",
        argv=raw_argv,
        announce_summary_write_on_error=args.summary_json is not None,
        announce_summary_write_on_success=False,
        announce_artifact_manifest_on_success=bool(
            args.summary_json is not None and (args.list_symbols or args.preview_run)
        ),
    )


if __name__ == "__main__":
    sys.exit(main())
