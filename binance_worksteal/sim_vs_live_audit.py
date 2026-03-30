#!/usr/bin/env python3
"""Sim-vs-production parity audit for binance_worksteal.

Runs the backtest under production-realistic constraints and reports
where candidates are blocked, helping diagnose why sim trades but
production does not.
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

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
    _build_daily_market_context,
    _base_asset_should_hold,
    _initialize_backtest_cursors,
    _normalize_base_asset_symbol,
    _prepare_backtest_symbol_data,
    WorkStealConfig,
    build_entry_candidates,
    compute_breadth_ratio,
    load_daily_bars,
    resolve_entry_regime,
    run_worksteal_backtest,
)
from binance_worksteal.backtest import FULL_UNIVERSE


AUDIT_CONFIG_FLAG_TO_FIELD = {
    "--dip-pct": "dip_pct",
    "--proximity-pct": "proximity_pct",
    "--profit-target": "profit_target_pct",
    "--stop-loss": "stop_loss_pct",
    "--max-positions": "max_positions",
    "--max-hold-days": "max_hold_days",
    "--lookback": "lookback_days",
    "--sma-filter": "sma_filter_period",
    "--trailing-stop": "trailing_stop_pct",
    "--fee": "maker_fee",
}


def _build_audit_failure(
    *,
    stage: str,
    engine: str,
    config: WorkStealConfig,
    start_date: str | None,
    end_date: str | None,
    exc: Exception,
) -> dict[str, object]:
    message = format_eval_failure(
        f"sim_vs_live_audit {stage}",
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

def audit_entries(
    all_bars: dict[str, pd.DataFrame],
    config: WorkStealConfig,
    start_date: str | None = None,
    end_date: str | None = None,
) -> pd.DataFrame:
    prepared, all_dates, first_date_ns = _prepare_backtest_symbol_data(
        all_bars,
        start_date=start_date,
        end_date=end_date,
    )
    next_indices = _initialize_backtest_cursors(prepared, first_date_ns)
    base_symbol = _normalize_base_asset_symbol(config)

    rows = []
    for date in all_dates:
        current_bars, history = _build_daily_market_context(prepared, next_indices, date)
        if not current_bars:
            continue

        hold_base_asset = _base_asset_should_hold(
            base_symbol=base_symbol,
            current_bars=current_bars,
            history=history,
            config=config,
        )
        market_breadth, _, _ = compute_breadth_ratio(current_bars, history)
        entry_regime = resolve_entry_regime(
            current_bars=current_bars,
            history=history,
            config=config,
        )
        entry_config = entry_regime.config
        market_breadth_blocks = entry_regime.market_breadth_skip
        risk_off_blocks = entry_regime.risk_off

        diagnostics = []
        build_entry_candidates(
            current_bars=current_bars,
            history=history,
            positions={},
            last_exit={},
            date=date,
            config=entry_config,
            base_symbol=base_symbol,
            diagnostics=diagnostics,
        )
        diagnostics_by_symbol = {diag.symbol: diag for diag in diagnostics}

        for sym, bar in current_bars.items():
            diag = diagnostics_by_symbol.get(sym)
            close = float(bar["close"])
            low_bar = float(bar["low"])
            ref_high = float(getattr(diag, "ref_high", 0.0) or 0.0)
            buy_target = float(getattr(diag, "buy_target", 0.0) or 0.0)
            strict_fill = buy_target > 0.0 and low_bar <= buy_target
            local_candidate = bool(getattr(diag, "is_candidate", False))
            is_candidate = local_candidate and not market_breadth_blocks and not risk_off_blocks

            filter_reason = str(getattr(diag, "filter_reason", "") or "")
            if is_candidate:
                filter_reason = ""
            elif local_candidate and risk_off_blocks:
                filter_reason = "risk_off"
            elif local_candidate and market_breadth_blocks:
                filter_reason = "market_breadth"

            dist_pct = float(getattr(diag, "dist_pct", 0.0) or 0.0)
            sma_value = float(getattr(diag, "sma_value", 0.0) or 0.0)
            sma_pass = bool(getattr(diag, "sma_pass", True))
            proximity_bps = dist_pct * 10000.0

            rows.append({
                "date": date,
                "symbol": sym,
                "close": close,
                "low": low_bar,
                "ref_high": ref_high,
                "buy_target": buy_target,
                "sma_value": sma_value,
                "sma_blocks": not sma_pass,
                "proximity_bps": proximity_bps,
                "proximity_blocks": filter_reason.startswith("proximity("),
                "strict_fill_possible": strict_fill,
                "momentum_blocks": filter_reason.startswith("momentum("),
                "market_breadth": float(market_breadth),
                "market_breadth_blocks": bool(market_breadth_blocks),
                "risk_off_blocks": bool(risk_off_blocks),
                "hold_base_asset": bool(hold_base_asset),
                "base_asset_symbol": base_symbol or "",
                "local_candidate": bool(local_candidate),
                "filter_reason": filter_reason,
                "is_candidate": is_candidate,
                "would_fill_realistic": is_candidate and strict_fill,
            })

    return pd.DataFrame(rows)


def build_audit_summary(audit_df: pd.DataFrame) -> dict:
    total = len(audit_df)
    if total == 0:
        return {
            "total_symbol_day_evaluations": 0,
            "blocked_by_sma": 0,
            "blocked_by_proximity": 0,
            "blocked_by_momentum": 0,
            "blocked_by_market_breadth": 0,
            "blocked_by_risk_off": 0,
            "blocked_by_base_asset": 0,
            "candidate_count": 0,
            "strict_fill_count": 0,
            "strict_fill_rate_pct": 0.0,
            "per_symbol_candidates": [],
            "active_dates": [],
        }

    normalized = audit_df.copy()
    default_columns = {
        "sma_blocks": False,
        "proximity_blocks": False,
        "momentum_blocks": False,
        "market_breadth_blocks": False,
        "risk_off_blocks": False,
        "filter_reason": "",
        "is_candidate": False,
        "would_fill_realistic": False,
        "proximity_bps": 0.0,
    }
    for column, default in default_columns.items():
        if column not in normalized:
            normalized[column] = default

    filter_reason = normalized["filter_reason"].fillna("").astype(str)
    no_reason = filter_reason == ""
    n_sma_blocked = int(
        (filter_reason.str.startswith("sma_filter(") | (no_reason & normalized["sma_blocks"].astype(bool))).sum()
    ) if total else 0
    n_prox_blocked = int(
        (filter_reason.str.startswith("proximity(") | (no_reason & normalized["proximity_blocks"].astype(bool))).sum()
    ) if total else 0
    n_mom_blocked = int(
        (filter_reason.str.startswith("momentum(") | (no_reason & normalized["momentum_blocks"].astype(bool))).sum()
    ) if total else 0
    n_breadth_blocked = int(
        ((filter_reason == "market_breadth") | (no_reason & normalized["market_breadth_blocks"].astype(bool))).sum()
    ) if total else 0
    n_risk_off_blocked = int(
        ((filter_reason == "risk_off") | (no_reason & normalized["risk_off_blocks"].astype(bool))).sum()
    ) if total else 0
    n_base_asset_blocked = int((filter_reason == "base_asset").sum()) if total else 0
    n_candidates = int(normalized["is_candidate"].sum()) if total else 0
    n_strict_fill = int(normalized["would_fill_realistic"].sum()) if total else 0

    per_symbol_candidates = []
    cand_df = normalized[normalized["is_candidate"]]
    if not cand_df.empty:
        sym_stats = cand_df.groupby("symbol").agg(
            n_cand=("is_candidate", "sum"),
            n_fill=("would_fill_realistic", "sum"),
            avg_prox=("proximity_bps", "mean"),
        ).sort_values("n_cand", ascending=False)
        for sym, row in sym_stats.iterrows():
            per_symbol_candidates.append(
                {
                    "symbol": sym,
                    "candidate_count": int(row["n_cand"]),
                    "strict_fill_count": int(row["n_fill"]),
                    "fill_rate_pct": float(row["n_fill"] / max(row["n_cand"], 1) * 100.0),
                    "avg_proximity_bps": float(row["avg_prox"]),
                }
            )

    active_dates = []
    if total:
        date_df = normalized.groupby("date").agg(
            total=("symbol", "count"),
            sma_blocked=("sma_blocks", "sum"),
            prox_blocked=("proximity_blocks", "sum"),
            breadth_blocked=("market_breadth_blocks", "max"),
            risk_off_blocked=("risk_off_blocks", "max"),
            candidates=("is_candidate", "sum"),
            fills=("would_fill_realistic", "sum"),
        )
        for dt, row in date_df[date_df["candidates"] > 0].iterrows():
            active_dates.append(
                {
                    "date": dt,
                    "symbol_count": int(row["total"]),
                    "sma_blocked": int(row["sma_blocked"]),
                    "proximity_blocked": int(row["prox_blocked"]),
                    "market_breadth_blocked": bool(row["breadth_blocked"]),
                    "risk_off_blocked": bool(row["risk_off_blocked"]),
                    "candidate_count": int(row["candidates"]),
                    "fill_count": int(row["fills"]),
                }
            )

    return {
        "total_symbol_day_evaluations": total,
        "blocked_by_sma": n_sma_blocked,
        "blocked_by_proximity": n_prox_blocked,
        "blocked_by_momentum": n_mom_blocked,
        "blocked_by_market_breadth": n_breadth_blocked,
        "blocked_by_risk_off": n_risk_off_blocked,
        "blocked_by_base_asset": n_base_asset_blocked,
        "candidate_count": n_candidates,
        "strict_fill_count": n_strict_fill,
        "strict_fill_rate_pct": float(n_strict_fill / max(n_candidates, 1) * 100.0) if n_candidates else 0.0,
        "per_symbol_candidates": per_symbol_candidates,
        "active_dates": active_dates,
    }


def print_audit_summary(audit_df: pd.DataFrame, config: WorkStealConfig):
    summary = build_audit_summary(audit_df)
    print(f"\n{'='*70}")
    print("SIM-vs-LIVE PARITY AUDIT")
    print(f"{'='*70}")

    total = summary["total_symbol_day_evaluations"]
    n_sma_blocked = summary["blocked_by_sma"]
    n_prox_blocked = summary["blocked_by_proximity"]
    n_mom_blocked = summary["blocked_by_momentum"]
    n_breadth_blocked = summary["blocked_by_market_breadth"]
    n_risk_off_blocked = summary["blocked_by_risk_off"]
    n_base_asset_blocked = summary["blocked_by_base_asset"]
    n_candidates = summary["candidate_count"]
    n_strict_fill = summary["strict_fill_count"]

    print(f"\nTotal symbol-day evaluations: {total}")
    print(f"  Blocked by SMA filter:     {n_sma_blocked:>6d} ({n_sma_blocked/max(total,1)*100:.1f}%)")
    print(f"  Blocked by proximity:      {n_prox_blocked:>6d} ({n_prox_blocked/max(total,1)*100:.1f}%)")
    print(f"  Blocked by momentum:       {n_mom_blocked:>6d} ({n_mom_blocked/max(total,1)*100:.1f}%)")
    print(f"  Blocked by breadth:        {n_breadth_blocked:>6d} ({n_breadth_blocked/max(total,1)*100:.1f}%)")
    print(f"  Blocked by risk-off:       {n_risk_off_blocked:>6d} ({n_risk_off_blocked/max(total,1)*100:.1f}%)")
    print(f"  Blocked as base asset:     {n_base_asset_blocked:>6d} ({n_base_asset_blocked/max(total,1)*100:.1f}%)")
    print(f"  Pass all filters:          {n_candidates:>6d} ({n_candidates/max(total,1)*100:.1f}%)")
    print(f"  Would fill (strict):       {n_strict_fill:>6d} ({n_strict_fill/max(total,1)*100:.1f}%)")

    if n_candidates > 0:
        fill_rate = n_strict_fill / n_candidates * 100
        print(f"\n  Fill rate (strict/candidate): {fill_rate:.1f}%")

    if summary["per_symbol_candidates"]:
        print(f"\n{'='*70}")
        print("PER-SYMBOL CANDIDATE BREAKDOWN")
        print(f"{'='*70}")
        print(f"{'Symbol':<12s} {'Candidates':>10s} {'StrictFill':>10s} {'FillRate':>8s} {'AvgProxBps':>10s}")
        print("-" * 52)
        for row in summary["per_symbol_candidates"]:
            print(
                f"{row['symbol']:<12s} {row['candidate_count']:>10d} {row['strict_fill_count']:>10d} "
                f"{row['fill_rate_pct']:>7.1f}% {row['avg_proximity_bps']:>10.1f}"
            )

    if summary["active_dates"]:
        print(f"\n{'='*70}")
        print(f"DATES WITH CANDIDATES ({len(summary['active_dates'])} days)")
        print(f"{'='*70}")
        print(
            f"{'Date':<12s} {'Syms':>5s} {'SMAblk':>6s} {'ProxBlk':>7s} "
            f"{'Breadth':>7s} {'RiskOff':>7s} {'Cands':>5s} {'Fills':>5s}"
        )
        print("-" * 66)
        for row in summary["active_dates"]:
            ds = str(row["date"])[:10]
            print(
                f"{ds:<12s} {row['symbol_count']:>5d} {row['sma_blocked']:>6d} "
                f"{row['proximity_blocked']:>7d} "
                f"{str(bool(row['market_breadth_blocked'])):>7s} "
                f"{str(bool(row['risk_off_blocked'])):>7s} "
                f"{row['candidate_count']:>5d} {row['fill_count']:>5d}"
            )


def build_comparison_summary(
    all_bars: dict[str, pd.DataFrame],
    config: WorkStealConfig,
    start_date: str | None,
    end_date: str | None,
):
    config_default = replace(config, realistic_fill=False, daily_checkpoint_only=False)
    config_realistic = replace(config, realistic_fill=True, daily_checkpoint_only=True)

    _eq_d, _trades_d, met_d = run_worksteal_backtest(all_bars, config_default, start_date, end_date)
    _eq_r, _trades_r, met_r = run_worksteal_backtest(all_bars, config_realistic, start_date, end_date)

    return {
        "default": {
            "total_return_pct": float(met_d.get("total_return_pct", 0.0)),
            "sortino": float(met_d.get("sortino", 0.0)),
            "max_drawdown_pct": float(met_d.get("max_drawdown_pct", 0.0)),
            "win_rate": float(met_d.get("win_rate", 0.0)),
            "entries_executed": int(met_d.get("entries_executed", 0)),
            "candidates_generated": int(met_d.get("candidates_generated", 0)),
            "candidates_visible": int(met_d.get("candidates_visible", 0)),
            "fill_rate": float(met_d.get("fill_rate", 0.0)),
            "visible_fill_rate": float(met_d.get("visible_fill_rate", 0.0)),
        },
        "realistic": {
            "total_return_pct": float(met_r.get("total_return_pct", 0.0)),
            "sortino": float(met_r.get("sortino", 0.0)),
            "max_drawdown_pct": float(met_r.get("max_drawdown_pct", 0.0)),
            "win_rate": float(met_r.get("win_rate", 0.0)),
            "entries_executed": int(met_r.get("entries_executed", 0)),
            "candidates_generated": int(met_r.get("candidates_generated", 0)),
            "candidates_visible": int(met_r.get("candidates_visible", 0)),
            "fill_rate": float(met_r.get("fill_rate", 0.0)),
            "visible_fill_rate": float(met_r.get("visible_fill_rate", 0.0)),
        },
    }


def run_comparison(
    all_bars: dict[str, pd.DataFrame],
    config: WorkStealConfig,
    start_date: str | None,
    end_date: str | None,
    comparison: dict | None = None,
):
    print(f"\n{'='*70}")
    print("BACKTEST COMPARISON: default vs realistic fill")
    print(f"{'='*70}")

    if comparison is None:
        comparison = build_comparison_summary(all_bars, config, start_date, end_date)
    default = comparison["default"]
    realistic = comparison["realistic"]
    print(f"\n{'Metric':<25s} {'Default':>12s} {'Realistic':>12s}")
    print("-" * 51)
    for key in ["total_return_pct", "sortino", "max_drawdown_pct", "win_rate"]:
        vd = default.get(key, 0)
        vr = realistic.get(key, 0)
        print(f"{key:<25s} {vd:>12.2f} {vr:>12.2f}")
    print(f"{'entries_executed':<25s} {default.get('entries_executed', 0):>12d} {realistic.get('entries_executed', 0):>12d}")
    print(f"{'candidates_generated':<25s} {default.get('candidates_generated',0):>12d} {realistic.get('candidates_generated',0):>12d}")
    print(f"{'candidates_visible':<25s} {default.get('candidates_visible',0):>12d} {realistic.get('candidates_visible',0):>12d}")
    print(f"{'fill_rate':<25s} {default.get('fill_rate',0):>12.1%} {realistic.get('fill_rate',0):>12.1%}")
    print(f"{'visible_fill_rate':<25s} {default.get('visible_fill_rate',0):>12.1%} {realistic.get('visible_fill_rate',0):>12.1%}")


def build_audit_cli_default_config(args: argparse.Namespace) -> WorkStealConfig:
    return WorkStealConfig(
        dip_pct=args.dip_pct,
        proximity_pct=args.proximity_pct,
        profit_target_pct=args.profit_target,
        stop_loss_pct=args.stop_loss,
        max_positions=args.max_positions,
        max_hold_days=args.max_hold_days,
        lookback_days=args.lookback,
        sma_filter_period=args.sma_filter,
        trailing_stop_pct=args.trailing_stop,
        maker_fee=args.fee,
        ref_price_method="high",
    )


def build_audit_config(
    args: argparse.Namespace,
    raw_argv: list[str],
) -> WorkStealConfig:
    return build_worksteal_config_from_args(
        base_config=build_audit_cli_default_config(args),
        config_file=args.config_file,
        args=args,
        raw_argv=raw_argv,
        flag_to_field=AUDIT_CONFIG_FLAG_TO_FIELD,
    )


def main(argv: list[str] | None = None):
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(description="Sim-vs-live parity audit")
    parser.add_argument("--data-dir", default="trainingdata/train")
    add_symbol_selection_args(parser)
    add_require_full_universe_arg(parser)
    add_date_range_args(
        parser,
        start_dest="start",
        end_dest="end",
        include_days=True,
        days_default=30,
    )
    parser.add_argument("--dip-pct", type=float, default=0.20)
    parser.add_argument("--proximity-pct", type=float, default=0.02)
    parser.add_argument("--profit-target", type=float, default=0.15)
    parser.add_argument("--stop-loss", type=float, default=0.10)
    parser.add_argument("--max-positions", type=int, default=5)
    parser.add_argument("--max-hold-days", type=int, default=14)
    parser.add_argument("--lookback", type=int, default=20)
    parser.add_argument("--sma-filter", type=int, default=20)
    parser.add_argument("--trailing-stop", type=float, default=0.03)
    parser.add_argument("--fee", type=float, default=0.001)
    add_config_file_arg(parser)
    add_print_config_arg(parser)
    add_explain_config_arg(parser)
    add_preview_run_arg(parser)
    add_summary_json_arg(parser)
    args = parser.parse_args(raw_argv)

    config_output_rc = maybe_handle_worksteal_config_output(
        args=args,
        build_config=lambda: build_audit_config(args, raw_argv),
        base_config=build_audit_cli_default_config(args),
        config_file=args.config_file,
        raw_argv=raw_argv,
        flag_to_field=AUDIT_CONFIG_FLAG_TO_FIELD,
    )
    if config_output_rc is not None:
        return config_output_rc

    def run():
        validated_range, date_error = validate_date_range_with_error(
            start_date=args.start,
            end_date=args.end,
        )
        if date_error is not None:
            print(date_error["error"])
            error_extra: dict[str, object] = {}
            if args.list_symbols:
                error_extra["list_symbols_only"] = True
            if args.preview_run:
                error_extra["preview_only"] = True
            return 1, build_cli_error_summary(
                tool="sim_vs_live_audit",
                error=date_error["error"],
                error_type=date_error["error_type"],
                data_dir=args.data_dir,
                config_file=args.config_file,
                extra=error_extra or None,
            )
        args.start, args.end = validated_range

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
                tool="sim_vs_live_audit",
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
                tool="sim_vs_live_audit",
                data_dir=args.data_dir,
                symbol_source=symbol_source,
                symbols=symbols,
                config_file=args.config_file,
            )

        try:
            config = build_audit_config(args, raw_argv)
        except (FileNotFoundError, OSError, ValueError) as exc:
            config_error = build_cli_error(exc)
            print(config_error["error"])
            error_extra: dict[str, object] = {}
            if args.preview_run:
                error_extra["preview_only"] = True
            return 1, build_cli_error_summary(
                tool="sim_vs_live_audit",
                error=config_error["error"],
                error_type=config_error["error_type"],
                data_dir=args.data_dir,
                config_file=args.config_file,
                extra=error_extra or None,
            )
        if args.preview_run:
            date_mode = "fixed_range" if args.start and args.end else "auto_last_n_days"
            extra = {
                "date_mode": date_mode,
                "start_date": args.start,
                "end_date": args.end,
                "days": None if date_mode == "fixed_range" else args.days,
                "summary_json": args.summary_json,
                "require_full_universe": args.require_full_universe,
            }
            print_run_preview(
                tool="sim_vs_live_audit",
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
                            ("start_date", args.start),
                            ("end_date", args.end),
                            ("days", extra["days"]),
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
                tool="sim_vs_live_audit",
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
            strict_retry_command=build_strict_retry_command(module="binance_worksteal.sim_vs_live_audit", argv=raw_argv),
        )
        if load_failure is not None:
            failure = {
                "stage": "load_bars",
                "engine": "data_load",
                "error_type": load_failure["error_type"],
                "error": load_failure["error"],
                "message": load_failure["error"],
                "start_date": args.start,
                "end_date": args.end,
            }
            return 1, {
                **build_symbol_run_summary(
                    tool="sim_vs_live_audit",
                    data_dir=args.data_dir,
                    symbol_source=symbol_source,
                    symbols=symbols,
                    load_summary=load_summary,
                    data_coverage=None,
                    config_file=args.config_file,
                    config=config,
                    require_full_universe=args.require_full_universe,
                ),
                "start_date": args.start,
                "end_date": args.end,
                "audit_summary": None,
                "comparison": None,
                "audit_failure": failure,
            }
        if not all_bars:
            failure = {
                "stage": "load_bars",
                "engine": "data_load",
                "error_type": "NoDataLoaded",
                "error": "ERROR: No data loaded",
                "message": "ERROR: No data loaded",
                "start_date": args.start,
                "end_date": args.end,
            }
            return 1, {
                **build_symbol_run_summary(
                    tool="sim_vs_live_audit",
                    data_dir=args.data_dir,
                    symbol_source=symbol_source,
                    symbols=symbols,
                    load_summary=load_summary,
                    data_coverage=None,
                    config_file=args.config_file,
                    config=config,
                    require_full_universe=args.require_full_universe,
                ),
                "start_date": args.start,
                "end_date": args.end,
                "audit_summary": None,
                "comparison": None,
                "audit_failure": failure,
            }

        missing_data_error = require_full_universe_or_print_error(
            require_full_universe=args.require_full_universe,
            load_summary=load_summary,
        )
        if missing_data_error is not None:
            failure = {
                "stage": "load_bars",
                "engine": "data_load",
                "error_type": "PartialUniverse",
                "error": missing_data_error,
                "message": missing_data_error,
                "start_date": args.start,
                "end_date": args.end,
            }
            return 1, {
                **build_symbol_run_summary(
                    tool="sim_vs_live_audit",
                    data_dir=args.data_dir,
                    symbol_source=symbol_source,
                    symbols=symbols,
                    load_summary=load_summary,
                    data_coverage=None,
                    config_file=args.config_file,
                    config=config,
                    require_full_universe=args.require_full_universe,
                ),
                "start_date": args.start,
                "end_date": args.end,
                "audit_summary": None,
                "comparison": None,
                "audit_failure": failure,
            }

        if not args.start and not args.end:
            latest = max(df["timestamp"].max() for df in all_bars.values())
            args.end = str(latest.date())
            args.start = str((latest - pd.Timedelta(days=args.days)).date())
            print(f"Auto date range: {args.start} to {args.end}")
        data_coverage = print_data_coverage_summary(
            all_bars,
            start_date=args.start,
            end_date=args.end,
        )

        def build_summary_payload(
            *,
            audit_summary: dict | None,
            comparison: dict | None,
            audit_failure: dict[str, object] | None,
        ) -> dict:
            return {
                **build_symbol_run_summary(
                    tool="sim_vs_live_audit",
                    data_dir=args.data_dir,
                    symbol_source=symbol_source,
                    symbols=symbols,
                    load_summary=load_summary,
                    data_coverage=data_coverage,
                    config_file=args.config_file,
                    config=config,
                    require_full_universe=args.require_full_universe,
                ),
                "start_date": args.start,
                "end_date": args.end,
                "audit_summary": audit_summary,
                "comparison": comparison,
                "audit_failure": audit_failure,
            }

        try:
            audit_df = audit_entries(
                all_bars,
                config,
                start_date=args.start,
                end_date=args.end,
            )
        except Exception as exc:
            failure = _build_audit_failure(
                stage="audit_entries",
                engine="python",
                config=config,
                start_date=args.start,
                end_date=args.end,
                exc=exc,
            )
            print(f"ERROR: {failure['message']}")
            if not args.summary_json:
                return 1, None
            return 1, build_summary_payload(
                audit_summary=None,
                comparison=None,
                audit_failure=failure,
            )

        audit_summary = build_audit_summary(audit_df)

        try:
            print_audit_summary(audit_df, config)
        except Exception as exc:
            failure = _build_audit_failure(
                stage="print_audit_summary",
                engine="reporting",
                config=config,
                start_date=args.start,
                end_date=args.end,
                exc=exc,
            )
            print(f"ERROR: {failure['message']}")
            if not args.summary_json:
                return 1, None
            return 1, build_summary_payload(
                audit_summary=audit_summary,
                comparison=None,
                audit_failure=failure,
            )

        try:
            comparison = build_comparison_summary(all_bars, config, args.start, args.end)
        except Exception as exc:
            failure = _build_audit_failure(
                stage="build_comparison_summary",
                engine="python_backtest",
                config=config,
                start_date=args.start,
                end_date=args.end,
                exc=exc,
            )
            print(f"ERROR: {failure['message']}")
            if not args.summary_json:
                return 1, None
            return 1, build_summary_payload(
                audit_summary=audit_summary,
                comparison=None,
                audit_failure=failure,
            )

        try:
            run_comparison(all_bars, config, args.start, args.end, comparison=comparison)
        except Exception as exc:
            failure = _build_audit_failure(
                stage="run_comparison",
                engine="reporting",
                config=config,
                start_date=args.start,
                end_date=args.end,
                exc=exc,
            )
            print(f"ERROR: {failure['message']}")
            if not args.summary_json:
                return 1, None
            return 1, build_summary_payload(
                audit_summary=audit_summary,
                comparison=comparison,
                audit_failure=failure,
            )

        if not args.summary_json:
            return 0, None

        return 0, build_summary_payload(
            audit_summary=audit_summary,
            comparison=comparison,
            audit_failure=None,
        )

    return run_with_optional_summary(
        args.summary_json,
        run,
        module="binance_worksteal.sim_vs_live_audit",
        argv=raw_argv,
        announce_artifact_manifest_on_success=True,
    )


if __name__ == "__main__":
    sys.exit(main() or 0)
