#!/usr/bin/env python3
"""Backtest the work-stealing dip-buying strategy on daily crypto data."""
from __future__ import annotations

import argparse
import sys
from dataclasses import asdict
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from binance_worksteal.cli import (
    build_cli_error,
    add_date_range_args,
    add_require_full_universe_arg,
    load_bars_with_summary,
    build_strict_retry_command,
    add_symbol_selection_args,
    print_data_coverage_summary,
    print_resolved_symbols,
    require_full_universe_or_print_error,
    resolve_cli_symbols_with_error,
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
from binance_worksteal.strategy import (
    WorkStealConfig,
    get_entry_trades,
    get_exit_trades,
    load_daily_bars,
    print_results,
    run_worksteal_backtest,
)


ORIGINAL_30_UNIVERSE = [
    "BTCUSD", "ETHUSD", "SOLUSD", "DOGEUSD", "AVAXUSD", "LINKUSD",
    "AAVEUSD", "LTCUSD", "XRPUSD", "DOTUSD", "UNIUSD", "NEARUSD",
    "APTUSD", "ICPUSD", "SHIBUSD", "ADAUSD", "FILUSD", "ARBUSD",
    "OPUSD", "INJUSD", "SUIUSD", "TIAUSD", "SEIUSD", "ATOMUSD",
    "ALGOUSD", "BCHUSD", "BNBUSD", "TRXUSD", "PEPEUSD", "MATICUSD",
]

EXPANDED_UNIVERSE = [
    "HBARUSD", "VETUSD", "RENDERUSD", "FETUSD", "GRTUSD",
    "SANDUSD", "MANAUSD", "AXSUSD", "CRVUSD", "COMPUSD",
    "MKRUSD", "SNXUSD", "ENJUSD", "1INCHUSD", "SUSHIUSD",
    "YFIUSD", "BATUSD", "ZRXUSD", "THETAUSD", "FTMUSD",
    "RUNEUSD", "KAVAUSD", "EGLDUSD", "CHZUSD", "GALAUSD",
    "APEUSD", "LDOUSD", "GMXUSD", "PENDLEUSD", "WLDUSD",
    "JUPUSD", "WUSD", "ENAUSD", "STXUSD", "FLOKIUSD",
    "TONUSD", "KASUSD", "ONDOUSD", "JASMYUSD", "CFXUSD",
]

FULL_UNIVERSE = ORIGINAL_30_UNIVERSE + EXPANDED_UNIVERSE

BACKTEST_CONFIG_FLAG_TO_FIELD = {
    "--dip-pct": "dip_pct",
    "--proximity-pct": "proximity_pct",
    "--profit-target": "profit_target_pct",
    "--stop-loss": "stop_loss_pct",
    "--max-positions": "max_positions",
    "--max-hold-days": "max_hold_days",
    "--lookback": "lookback_days",
    "--ref-method": ("ref_price_method", "ref_method"),
    "--fee": "maker_fee",
    "--cash": ("initial_cash", "cash"),
    "--trailing-stop": "trailing_stop_pct",
    "--cooldown": ("reentry_cooldown_days", "cooldown"),
    "--max-leverage": "max_leverage",
    "--enable-shorts": "enable_shorts",
    "--short-pump-pct": "short_pump_pct",
    "--base-asset": ("base_asset_symbol", "base_asset"),
    "--base-asset-sma-period": "base_asset_sma_filter_period",
    "--base-asset-momentum-period": "base_asset_momentum_period",
    "--base-asset-min-momentum": "base_asset_min_momentum",
    "--base-asset-rebalance-min-cash": "base_asset_rebalance_min_cash",
    "--sma-filter": "sma_filter_period",
    "--sma-check-method": "sma_check_method",
    "--adaptive-dip": "adaptive_dip",
    "--realistic-fill": "realistic_fill",
    "--daily-checkpoint-only": "daily_checkpoint_only",
}


def _build_backtest_failure(
    *,
    stage: str,
    engine: str,
    config: WorkStealConfig,
    start_date: str | None,
    end_date: str | None,
    exc: Exception,
) -> dict[str, object]:
    message = format_eval_failure(
        f"backtest {stage}",
        engine,
        config,
        start_date,
        end_date,
        exc,
    )
    return {
        "stage": stage,
        "engine": engine,
        "start_date": start_date,
        "end_date": end_date,
        "error_type": exc.__class__.__name__,
        "error": str(exc),
        "message": message,
    }


def build_backtest_cli_default_config(args: argparse.Namespace) -> WorkStealConfig:
    return WorkStealConfig(
        dip_pct=args.dip_pct,
        proximity_pct=args.proximity_pct,
        profit_target_pct=args.profit_target,
        stop_loss_pct=args.stop_loss,
        max_positions=args.max_positions,
        max_hold_days=args.max_hold_days,
        lookback_days=args.lookback,
        ref_price_method=args.ref_method,
        maker_fee=args.fee,
        initial_cash=args.cash,
        trailing_stop_pct=args.trailing_stop,
        reentry_cooldown_days=args.cooldown,
        max_leverage=args.max_leverage,
        enable_shorts=bool(args.enable_shorts),
        short_pump_pct=args.short_pump_pct,
        base_asset_symbol=args.base_asset,
        base_asset_sma_filter_period=args.base_asset_sma_period,
        base_asset_momentum_period=args.base_asset_momentum_period,
        base_asset_min_momentum=args.base_asset_min_momentum,
        base_asset_rebalance_min_cash=args.base_asset_rebalance_min_cash,
        sma_filter_period=args.sma_filter,
        sma_check_method=args.sma_check_method,
        adaptive_dip=args.adaptive_dip,
        realistic_fill=args.realistic_fill,
        daily_checkpoint_only=args.daily_checkpoint_only,
    )


def build_backtest_config(args: argparse.Namespace, raw_argv: list[str]) -> WorkStealConfig:
    return build_worksteal_config_from_args(
        base_config=build_backtest_cli_default_config(args),
        config_file=args.config_file,
        args=args,
        raw_argv=raw_argv,
        flag_to_field=BACKTEST_CONFIG_FLAG_TO_FIELD,
    )


def main(argv: list[str] | None = None):
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(description="Backtest the Binance worksteal strategy on daily bars.")
    parser.add_argument("--data-dir", default="trainingdata/train")
    add_symbol_selection_args(parser)
    add_require_full_universe_arg(parser)
    add_date_range_args(
        parser,
        start_dest="start_date",
        end_dest="end_date",
        include_days=True,
        days_default=30,
        days_help="Backtest last N days if no start/end",
    )
    parser.add_argument("--dip-pct", type=float, default=0.10)
    parser.add_argument("--proximity-pct", type=float, default=0.03)
    parser.add_argument("--profit-target", type=float, default=0.05)
    parser.add_argument("--stop-loss", type=float, default=0.08)
    parser.add_argument("--max-positions", type=int, default=5)
    parser.add_argument("--max-hold-days", type=int, default=14)
    parser.add_argument("--lookback", type=int, default=20)
    parser.add_argument("--ref-method", choices=["high", "sma", "close"], default="high")
    parser.add_argument("--fee", type=float, default=0.001)
    parser.add_argument("--cash", type=float, default=10000.0)
    parser.add_argument("--trailing-stop", type=float, default=0.0)
    parser.add_argument("--cooldown", type=int, default=1)
    parser.add_argument("--max-leverage", type=float, default=1.0)
    parser.add_argument("--enable-shorts", action="store_true")
    parser.add_argument("--short-pump-pct", type=float, default=0.10)
    parser.add_argument("--base-asset", default="", help="Optional idle base asset, e.g. ETHUSD.")
    parser.add_argument("--base-asset-sma-period", type=int, default=0)
    parser.add_argument("--base-asset-momentum-period", type=int, default=0)
    parser.add_argument("--base-asset-min-momentum", type=float, default=0.0)
    parser.add_argument("--base-asset-rebalance-min-cash", type=float, default=1.0)
    parser.add_argument("--sma-filter", type=int, default=20)
    parser.add_argument("--sma-check-method", choices=["current", "pre_dip", "none"], default="pre_dip")
    parser.add_argument("--adaptive-dip", action="store_true")
    parser.add_argument(
        "--realistic-fill",
        action="store_true",
        help="Require the bar to actually touch the limit entry price before filling.",
    )
    parser.add_argument(
        "--daily-checkpoint-only",
        action="store_true",
        help="Decide at bar close, then allow fills on the next bar only.",
    )
    add_config_file_arg(parser)
    add_print_config_arg(parser)
    add_explain_config_arg(parser)
    add_preview_run_arg(parser)
    add_summary_json_arg(parser)
    args = parser.parse_args(raw_argv)

    config_output_rc = maybe_handle_worksteal_config_output(
        args=args,
        build_config=lambda: build_backtest_config(args, raw_argv),
        base_config=build_backtest_cli_default_config(args),
        config_file=args.config_file,
        raw_argv=raw_argv,
        flag_to_field=BACKTEST_CONFIG_FLAG_TO_FIELD,
    )
    if config_output_rc is not None:
        return config_output_rc

    def run():
        validated_range, date_error = validate_date_range_with_error(
            start_date=args.start_date,
            end_date=args.end_date,
        )
        if date_error is not None:
            print(date_error["error"])
            error_extra: dict[str, object] = {}
            if args.list_symbols:
                error_extra["list_symbols_only"] = True
            if args.preview_run:
                error_extra["preview_only"] = True
            return 1, build_cli_error_summary(
                tool="backtest",
                error=date_error["error"],
                error_type=date_error["error_type"],
                data_dir=args.data_dir,
                config_file=args.config_file,
                extra=error_extra or None,
            )
        args.start_date, args.end_date = validated_range

        resolved, symbol_error = resolve_cli_symbols_with_error(
            symbols_arg=args.symbols,
            universe_file=args.universe_file,
            default_symbols=FULL_UNIVERSE,
        )
        if symbol_error is not None:
            print(symbol_error["error"])
            error_extra: dict[str, object] = {}
            if args.list_symbols:
                error_extra["list_symbols_only"] = True
            if args.preview_run:
                error_extra["preview_only"] = True
            return 1, build_cli_error_summary(
                tool="backtest",
                error=symbol_error["error"],
                error_type=symbol_error["error_type"],
                data_dir=args.data_dir,
                config_file=args.config_file,
                extra=error_extra or None,
            )
        symbols, symbol_source = resolved
        if args.list_symbols:
            print_resolved_symbols(symbols, symbol_source)
            return 0, build_symbol_listing_summary(
                tool="backtest",
                data_dir=args.data_dir,
                symbol_source=symbol_source,
                symbols=symbols,
                config_file=args.config_file,
            )

        try:
            config = build_backtest_config(args, raw_argv)
        except (FileNotFoundError, OSError, ValueError) as exc:
            config_error = build_cli_error(exc)
            print(config_error["error"])
            error_extra: dict[str, object] = {}
            if args.preview_run:
                error_extra["preview_only"] = True
            return 1, build_cli_error_summary(
                tool="backtest",
                error=config_error["error"],
                error_type=config_error["error_type"],
                data_dir=args.data_dir,
                config_file=args.config_file,
                extra=error_extra or None,
            )
        if args.preview_run:
            date_mode = "fixed_range" if args.start_date and args.end_date else "auto_last_n_days"
            extra = {
                "date_mode": date_mode,
                "start_date": args.start_date,
                "end_date": args.end_date,
                "days": None if date_mode == "fixed_range" else args.days,
                "summary_json": args.summary_json,
                "require_full_universe": args.require_full_universe,
            }
            print_run_preview(
                tool="backtest",
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
                            ("start_date", args.start_date),
                            ("end_date", args.end_date),
                            ("days", extra["days"]),
                            ("require_full_universe", args.require_full_universe),
                            ("summary_json", args.summary_json),
                        ),
                    ),
                    (
                        "Config",
                        (
                            ("config_file", args.config_file),
                            ("base_asset", config.base_asset_symbol or "cash"),
                            ("realistic_fill", config.realistic_fill),
                            ("daily_checkpoint_only", config.daily_checkpoint_only),
                        ),
                    ),
                ],
            )
            return 0, build_preview_run_summary(
                tool="backtest",
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
        all_bars, load_summary, load_failure = load_bars_with_summary(
            data_dir=args.data_dir,
            requested_symbols=symbols,
            load_bars=load_daily_bars,
            loading_message=f"Loading data for {len(symbols)} symbols from {args.data_dir}",
            no_data_message="ERROR: No data loaded",
            return_summary_on_empty=True,
            return_failure_on_error=True,
            strict_retry_command=build_strict_retry_command(module="binance_worksteal.backtest", argv=raw_argv),
        )
        if load_failure is not None:
            return 1, {
                **build_symbol_run_summary(
                    tool="backtest",
                    data_dir=args.data_dir,
                    symbol_source=symbol_source,
                    symbols=symbols,
                    load_summary=load_summary,
                    config_file=args.config_file,
                    config=asdict(config),
                    require_full_universe=args.require_full_universe,
                ),
                "start_date": args.start_date,
                "end_date": args.end_date,
                "error": load_failure["error"],
                "load_failure": load_failure,
            }
        if not all_bars:
            return 1, {
                **build_symbol_run_summary(
                    tool="backtest",
                    data_dir=args.data_dir,
                    symbol_source=symbol_source,
                    symbols=symbols,
                    load_summary=load_summary,
                    config_file=args.config_file,
                    config=asdict(config),
                    require_full_universe=args.require_full_universe,
                ),
                "start_date": args.start_date,
                "end_date": args.end_date,
                "error": "ERROR: No data loaded",
            }

        missing_data_error = require_full_universe_or_print_error(
            require_full_universe=args.require_full_universe,
            load_summary=load_summary,
        )
        if missing_data_error is not None:
            return 1, {
                **build_symbol_run_summary(
                    tool="backtest",
                    data_dir=args.data_dir,
                    symbol_source=symbol_source,
                    symbols=symbols,
                    load_summary=load_summary,
                    config_file=args.config_file,
                    config=asdict(config),
                    require_full_universe=args.require_full_universe,
                ),
                "start_date": args.start_date,
                "end_date": args.end_date,
                "error": missing_data_error,
            }

        # Auto-compute date range if not specified
        if not args.start_date and not args.end_date:
            latest = max(df["timestamp"].max() for df in all_bars.values())
            args.end_date = str(latest.date())
            args.start_date = str((latest - pd.Timedelta(days=args.days)).date())
            print(f"Auto date range: {args.start_date} to {args.end_date}")
        data_coverage = print_data_coverage_summary(
            all_bars,
            start_date=args.start_date,
            end_date=args.end_date,
        )

        def build_summary_payload(
            *,
            metrics: dict | None,
            trades: list | None,
            backtest_failure: dict[str, object] | None,
        ) -> dict:
            entries = get_entry_trades(trades or [])
            exits = get_exit_trades(trades or [])
            per_symbol_pnl: dict[str, float] = {}
            exit_reasons: dict[str, int] = {}
            for trade in exits:
                per_symbol_pnl[trade.symbol] = per_symbol_pnl.get(trade.symbol, 0.0) + float(trade.pnl)
                exit_reasons[trade.reason] = exit_reasons.get(trade.reason, 0) + 1
            payload = {
                **build_symbol_run_summary(
                    tool="backtest",
                    data_dir=args.data_dir,
                    symbol_source=symbol_source,
                    symbols=symbols,
                    load_summary=load_summary,
                    data_coverage=data_coverage,
                    config_file=args.config_file,
                    config=asdict(config),
                    require_full_universe=args.require_full_universe,
                ),
                "start_date": args.start_date,
                "end_date": args.end_date,
                "metrics": metrics,
                "trade_counts": {
                    "entries": len(entries),
                    "exits": len(exits),
                    "long_entries": sum(1 for t in entries if t.direction == "long"),
                    "short_entries": sum(1 for t in entries if t.direction == "short"),
                },
                "per_symbol_pnl": per_symbol_pnl,
                "exit_reasons": exit_reasons,
            }
            if backtest_failure is not None:
                payload["backtest_failure"] = backtest_failure
            return payload

        print(f"\nConfig: dip={config.dip_pct:.0%} prox={config.proximity_pct:.1%} "
              f"tp={config.profit_target_pct:.0%} sl={config.stop_loss_pct:.0%} "
              f"maxpos={config.max_positions} maxhold={config.max_hold_days}d "
              f"lev={config.max_leverage:.1f}x "
              f"base={config.base_asset_symbol or 'cash'} "
              f"realistic_fill={'on' if config.realistic_fill else 'off'} "
              f"daily_checkpoint_only={'on' if config.daily_checkpoint_only else 'off'}")

        try:
            equity_df, trades, metrics = run_worksteal_backtest(
                all_bars, config,
                start_date=args.start_date,
                end_date=args.end_date,
            )
        except Exception as exc:
            failure = _build_backtest_failure(
                stage="run_worksteal_backtest",
                engine="Python backtest",
                config=config,
                start_date=args.start_date,
                end_date=args.end_date,
                exc=exc,
            )
            print(f"ERROR: {failure['message']}")
            if not args.summary_json:
                return 1, None
            return 1, build_summary_payload(
                metrics=None,
                trades=None,
                backtest_failure=failure,
            )

        try:
            print_results(equity_df, trades, metrics)
        except Exception as exc:
            failure = _build_backtest_failure(
                stage="print_results",
                engine="reporting",
                config=config,
                start_date=args.start_date,
                end_date=args.end_date,
                exc=exc,
            )
            print(f"ERROR: {failure['message']}")
            if not args.summary_json:
                return 1, None
            return 1, build_summary_payload(
                metrics=metrics,
                trades=trades,
                backtest_failure=failure,
            )
        if not args.summary_json:
            return 0, None

        return 0, build_summary_payload(
            metrics=metrics,
            trades=trades,
            backtest_failure=None,
        )

    return run_with_optional_summary(
        args.summary_json,
        run,
        module="binance_worksteal.backtest",
        argv=raw_argv,
        announce_artifact_manifest_on_success=True,
    )


if __name__ == "__main__":
    sys.exit(main())
