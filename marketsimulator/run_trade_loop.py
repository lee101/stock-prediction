from __future__ import annotations

import argparse
import csv
import importlib
import json
import os
import sys
from datetime import UTC, datetime
from pathlib import Path
from subprocess import CalledProcessError, check_output
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent

if __package__ in (None, ""):
    repo_root = Path(__file__).resolve().parent.parent
    repo_str = str(repo_root)
    if repo_str not in sys.path:
        sys.path.insert(0, repo_str)
    from marketsimulator.logging_utils import logger
    from marketsimulator.telemetry import (
        build_symbol_performance_table,
        build_portfolio_stack_series,
        build_price_history_table,
        build_trade_events_table,
        compute_breakdowns,
        compute_equity_timeseries,
        compute_fee_breakdown,
        compute_risk_timeseries,
        summarize_daily_analysis,
    )
    from wandboard import WandBoardLogger
else:  # pragma: no cover
    from .logging_utils import logger
    from .telemetry import (
        build_symbol_performance_table,
        build_portfolio_stack_series,
        build_price_history_table,
        build_trade_events_table,
        compute_breakdowns,
        compute_equity_timeseries,
        compute_fee_breakdown,
        compute_risk_timeseries,
        summarize_daily_analysis,
    )
    from wandboard import WandBoardLogger


ENV_MOCK_ANALYTICS = "MARKETSIM_USE_MOCK_ANALYTICS"


def parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    argv_list = list(argv) if argv is not None else None
    if argv_list is None:
        argv_list = sys.argv[1:]

    if "--stub-config" in argv_list:
        stub_path = Path("analysis") / "stub_hit.txt"
        try:
            stub_path.parent.mkdir(parents=True, exist_ok=True)
            stub_path.write_text("1", encoding="utf-8")
        except OSError:
            print(f"warning: unable to record stub hit at {stub_path}", file=sys.stderr)

    parser = argparse.ArgumentParser(
        description="Simulate trade_stock_e2e with a mocked Alpaca stack."
    )
    parser.add_argument(
        "--symbols",
        nargs="+",
        default=["AAPL", "MSFT", "NVDA"],
        help="Symbols to simulate.",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=30,
        help="Number of simulation iterations to run (treated as trading days).",
    )
    parser.add_argument(
        "--step-size",
        type=int,
        default=1,
        help="Data rows to advance between iterations.",
    )
    parser.add_argument(
        "--initial-cash",
        type=float,
        default=100_000.0,
        help="Starting cash balance.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=4,
        help="Number of picks to keep each iteration.",
    )
    parser.add_argument(
        "--kronos-only",
        action="store_true",
        help="Force Kronos forecasting pipeline even if another model is selected.",
    )
    parser.add_argument(
        "--flatten-end",
        action="store_true",
        help="Liquidate all open positions at the end of the run for realised PnL metrics.",
    )
    parser.add_argument(
        "--sharpe-cutoff",
        type=float,
        default=None,
        help="Override walk-forward Sharpe threshold used for gating (default 0.30).",
    )
    parser.add_argument(
        "--kronos-sharpe-cutoff",
        type=float,
        default=None,
        help="Override Sharpe threshold applied under --kronos-only (default -0.25).",
    )
    parser.add_argument(
        "--metrics-json",
        type=Path,
        default=None,
        help="Optional path to write summary metrics as JSON.",
    )
    parser.add_argument(
        "--metrics-csv",
        type=Path,
        default=None,
        help="Optional path to append summary metrics as CSV (header auto-added if file missing).",
    )
    parser.add_argument(
        "--trades-csv",
        type=Path,
        default=None,
        help="Optional path to append per-trade timeline as CSV.",
    )
    parser.add_argument(
        "--trades-summary-json",
        type=Path,
        default=None,
        help="Optional path to write aggregated trade statistics as JSON.",
    )
    parser.add_argument(
        "--real-analytics",
        dest="real_analytics",
        action="store_true",
        help="Use the full forecasting/backtest stack instead of simulator mocks.",
    )
    parser.add_argument(
        "--mock-analytics",
        dest="real_analytics",
        action="store_false",
        help="Force lightweight simulator analytics (skips heavy forecasting models).",
    )
    parser.set_defaults(real_analytics=True)
    parser.add_argument(
        "--compact-logs",
        action="store_true",
        help="Reduce console log noise by using compact formatting.",
    )
    parser.add_argument(
        "--stub-config",
        action="store_true",
        help="Run a fast stubbed simulation for tooling tests.",
    )
    return parser.parse_args(argv_list)


def run_stub(args: argparse.Namespace) -> dict[str, float]:
    metrics = {
        "return": 0.0,
        "sharpe": 0.0,
        "pnl": 0.0,
        "balance": getattr(args, "initial_cash", 0.0),
        "steps": getattr(args, "steps", 0),
        "symbols": getattr(args, "symbols", []),
    }
    print("Stub simulator executed")
    print("Stub metrics:", json.dumps(metrics, sort_keys=True))
    return metrics


def _set_logger_level(name: str, level: int) -> None:
    import logging

    log = logging.getLogger(name)
    log.setLevel(level)
    for handler in log.handlers:
        handler.setLevel(level)


def _configure_compact_logging_pre(enabled: bool) -> None:
    if not enabled:
        return
    os.environ.setdefault("COMPACT_TRADING_LOGS", "1")
    from loguru import logger as loguru_logger

    loguru_logger.remove()
    loguru_logger.add(
        sys.stdout,
        level=os.getenv("SIM_LOGURU_LEVEL", "WARNING"),
        format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}",
    )


def _configure_compact_logging_post(enabled: bool) -> None:
    if not enabled:
        return
    import logging

    levels: dict[str, int] = {
        "backtest_test3_inline": logging.WARNING,
        "data_curate_daily": logging.WARNING,
        "sizing_utils": logging.WARNING,
    }
    for name, level in levels.items():
        _set_logger_level(name, level)


def _real_stack_available() -> bool:
    try:
        from marketsimulator import backtest_test3_inline as sim_backtest
        from marketsimulator import predict_stock_forecasting_proxy as sim_forecasting

        real_backtest = getattr(sim_backtest, "_REAL_BACKTEST_MODULE", None)
        if real_backtest is None:
            raise ImportError("real backtest module not loaded")
        real_forecast = getattr(sim_forecasting, "_real_module", None)
        if real_forecast is None:
            raise ImportError("real forecasting module not loaded")
        return True
    except Exception as exc:  # pragma: no cover - environment dependent
        import traceback
        tb = traceback.format_exc()
        logger.warning(f"[sim] Real analytics stack unavailable ({exc}). Falling back to simulator mocks.\n{tb}")
        return False


def _load_runner():
    if __package__ in (None, ""):
        module = importlib.import_module("marketsimulator.runner")
    else:  # pragma: no cover
        module = importlib.import_module(".runner", package=__package__)
    return module


def _compute_step_metrics(report) -> dict[str, float]:
    closes = [snap.equity for snap in report.daily_snapshots if snap.phase == "close"]
    if not closes:
        closes = [report.initial_cash, report.final_equity]
    else:
        closes = [report.initial_cash] + closes

    series = np.asarray(closes, dtype=np.float64)
    if series.size < 2:
        step_returns = np.array([], dtype=np.float64)
    else:
        prev = series[:-1]
        with np.errstate(divide="ignore", invalid="ignore"):
            step_returns = np.where(prev != 0.0, (series[1:] - prev) / prev, 0.0)

    mean_return = float(step_returns.mean()) if step_returns.size else 0.0
    std_return = float(step_returns.std(ddof=1)) if step_returns.size > 1 else 0.0
    if std_return > 0.0:
        sharpe = mean_return / std_return * np.sqrt(252.0)
    else:
        sharpe = 0.0

    return {
        "return": float(report.total_return_pct),
        "sharpe": float(sharpe),
        "pnl": float(report.total_return),
        "cash": float(report.final_cash),
        "balance": float(report.final_equity),
        "max_drawdown": float(report.max_drawdown),
        "max_drawdown_pct": float(report.max_drawdown_pct),
        "fees_paid": float(report.fees_paid),
    }


def _flatten_entry_limits(entry_snapshot: Dict[str, Dict]) -> Dict[str, float]:
    flattened: Dict[str, float] = {}
    per_symbol = entry_snapshot.get("per_symbol", {})
    for symbol, info in sorted(per_symbol.items()):
        symbol_key = symbol.lower()
        entries = info.get("entries")
        entry_limit = info.get("entry_limit")
        if entries is not None:
            flattened[f"entry_entries_{symbol_key}"] = float(entries)
        if entry_limit is not None:
            flattened[f"entry_limit_{symbol_key}"] = float(entry_limit)
        try:
            if entries is not None and entry_limit:
                utilization = float(entries) / float(entry_limit)
                flattened[f"entry_util_{symbol_key}"] = utilization
        except ZeroDivisionError:
            pass
    return flattened


def _format_metric_line(key: str, value: float) -> str:
    if key in {"return", "sharpe", "max_drawdown_pct"}:
        return f"{key}={value:.6f}"
    return f"{key}={value:.2f}"


def _restore_env(key: str, previous: Optional[str]) -> None:
    if previous is None:
        os.environ.pop(key, None)
    else:
        os.environ[key] = previous


def _export_metrics(metrics: dict, json_path: Optional[Path], csv_path: Optional[Path]) -> None:
    if json_path:
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")
    if not csv_path:
        return
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not csv_path.exists()
    fieldnames = list(metrics.keys())
    row: Dict[str, object] = {}
    for key, value in metrics.items():
        if isinstance(value, (dict, list)):
            row[key] = json.dumps(value, sort_keys=True)
        else:
            row[key] = value
    with csv_path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def _export_trades(trades, csv_path: Optional[Path]) -> None:
    if not csv_path or not trades:
        return
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not csv_path.exists()
    fieldnames = [
        "index",
        "timestamp",
        "symbol",
        "side",
        "qty",
        "price",
        "notional",
        "fee",
        "cash_delta",
        "slip_bps",
    ]
    with csv_path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        for idx, trade in enumerate(trades):
            writer.writerow(
                {
                    "index": idx,
                    "timestamp": trade.timestamp.isoformat(),
                    "symbol": trade.symbol,
                    "side": trade.side,
                    "qty": f"{trade.qty:.6f}",
                    "price": f"{trade.price:.4f}",
                    "notional": f"{trade.notional:.2f}",
                    "fee": f"{trade.fee:.4f}",
                    "cash_delta": f"{trade.cash_delta:.2f}",
                    "slip_bps": f"{trade.slip_bps:.4f}",
                }
            )


def _summarize_trades(trades) -> Dict[str, Dict[str, float]]:
    summary: Dict[str, Dict[str, float]] = {}
    positions: Dict[str, float] = {}
    entry_time: Dict[str, datetime] = {}
    cycle_durations: Dict[str, List[float]] = {}
    cumulative_pnl = 0.0
    worst_pnl = 0.0

    for trade in trades:
        symbol = trade.symbol
        stats = summary.setdefault(
            symbol,
            {
                "trades": 0,
                "buy_trades": 0,
                "sell_trades": 0,
                "gross_notional": 0.0,
                "fees": 0.0,
                "cash_delta": 0.0,
                "total_qty": 0.0,
                "slip_bps_sum": 0.0,
            },
        )
        stats["trades"] += 1
        is_buy = trade.side.lower() == "buy"
        if is_buy:
            stats["buy_trades"] += 1
        else:
            stats["sell_trades"] += 1
        stats["gross_notional"] += float(trade.notional)
        stats["fees"] += float(trade.fee)
        stats["cash_delta"] += float(trade.cash_delta)
        stats["total_qty"] += float(trade.qty)
        stats["slip_bps_sum"] += abs(float(trade.slip_bps))

        qty_signed = float(trade.qty) if is_buy else -float(trade.qty)
        prev_pos = positions.get(symbol, 0.0)
        if abs(prev_pos) < 1e-9:
            entry_time[symbol] = trade.timestamp
        positions[symbol] = prev_pos + qty_signed
        if abs(positions[symbol]) < 1e-9:
            start = entry_time.pop(symbol, None)
            if start:
                duration = (trade.timestamp - start).total_seconds()
                cycle_durations.setdefault(symbol, []).append(duration)
        cumulative_pnl += float(trade.cash_delta)
        worst_pnl = min(worst_pnl, cumulative_pnl)

    for symbol, stats in summary.items():
        durations = cycle_durations.get(symbol, [])
        if durations:
            stats["average_holding_seconds"] = sum(durations) / len(durations)
            stats["holding_cycles"] = len(durations)
        else:
            stats["average_holding_seconds"] = 0.0
            stats["holding_cycles"] = 0
        stats["average_slip_bps"] = (
            stats["slip_bps_sum"] / stats["trades"] if stats["trades"] else 0.0
        )

    summary["__overall__"] = {
        "trades": sum(stats["trades"] for stats in summary.values()),
        "buy_trades": sum(stats["buy_trades"] for stats in summary.values()),
        "sell_trades": sum(stats["sell_trades"] for stats in summary.values()),
        "gross_notional": sum(stats["gross_notional"] for stats in summary.values()),
        "fees": sum(stats["fees"] for stats in summary.values()),
        "cash_delta": sum(stats["cash_delta"] for stats in summary.values()),
        "total_qty": sum(stats["total_qty"] for stats in summary.values()),
        "slip_bps_sum": sum(stats["slip_bps_sum"] for stats in summary.values()),
        "average_holding_seconds": (
            sum(
                stats["average_holding_seconds"] * stats["holding_cycles"]
                for stats in summary.values()
                if stats["holding_cycles"] > 0
            )
            / max(1, sum(stats["holding_cycles"] for stats in summary.values()))
        ),
        "average_slip_bps": (
            sum(stats["slip_bps_sum"] for stats in summary.values())
            / max(1, sum(stats["trades"] for stats in summary.values()))
        ),
        "holding_cycles": sum(stats["holding_cycles"] for stats in summary.values()),
        "worst_cumulative_cash": worst_pnl,
    }
    return summary


def _export_trade_summary(summary: Dict[str, Dict[str, float]], json_path: Optional[Path]) -> None:
    if not json_path or not summary:
        return
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")


def _collect_run_metadata() -> Dict[str, object]:
    metadata: Dict[str, object] = {}
    try:
        metadata["git_sha"] = check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip()
    except (CalledProcessError, FileNotFoundError):
        metadata["git_sha"] = "unknown"
    else:
        try:
            status = check_output(["git", "status", "--porcelain"], cwd=REPO_ROOT, text=True)
            metadata["git_dirty"] = bool(status.strip())
        except (CalledProcessError, FileNotFoundError):
            metadata["git_dirty"] = None
    metadata["python_version"] = sys.version.split()[0]
    metadata["data_root"] = os.getenv("MARKETSIM_DATA_ROOT", "")
    metadata["wandb_project"] = os.getenv("WANDB_PROJECT", "")
    metadata["wandb_entity"] = os.getenv("WANDB_ENTITY", "")
    return metadata


def _make_portfolio_stack_figure(report) -> Optional["plt.Figure"]:
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except Exception:
        return None

    closes = [snap for snap in report.daily_snapshots if snap.phase == "close"]
    if not closes:
        closes = list(report.daily_snapshots)
    closes = sorted(closes, key=lambda snap: snap.timestamp)
    if not closes:
        return None
    value_map: Dict[str, List[float]] = {}
    for snap in closes:
        for symbol, details in (snap.positions_detail or {}).items():
            value_map.setdefault(symbol, []).append(abs(float(details.get("market_value", 0.0))))
        missing_symbols = set(value_map.keys()) - set((snap.positions_detail or {}).keys())
        for symbol in missing_symbols:
            value_map[symbol].append(0.0)
        for symbol in set((snap.positions_detail or {}).keys()) - set(value_map.keys()):
            prior_len = len(next(iter(value_map.values()))) if value_map else 0
            value_map[symbol] = [0.0] * prior_len + [abs(float(snap.positions_detail[symbol].get("market_value", 0.0)))]
    if not value_map:
        return None

    lengths = {len(series) for series in value_map.values()}
    max_len = max(lengths)
    for series in value_map.values():
        if len(series) < max_len:
            series.extend([0.0] * (max_len - len(series)))

    symbol_order = sorted(value_map.keys(), key=lambda sym: value_map[sym][-1], reverse=True)
    top_symbols = symbol_order[:8]
    other_symbols = symbol_order[8:]

    times = [snap.timestamp for snap in closes]
    stack_series: List[List[float]] = [value_map[sym] for sym in top_symbols]
    labels = list(top_symbols)
    if other_symbols:
        aggregated = [0.0] * max_len
        for idx in range(max_len):
            aggregated[idx] = sum(value_map[sym][idx] for sym in other_symbols)
        stack_series.append(aggregated)
        labels.append("OTHER")

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.stackplot(times, stack_series, labels=labels)
    ax.set_title("Portfolio Allocation (Abs Market Value)")
    ax.set_xlabel("Timestamp")
    ax.set_ylabel("Absolute Market Value ($)")
    ax.legend(loc="upper left", fontsize="small", ncols=2)
    ax.grid(alpha=0.3)
    fig.autofmt_xdate()
    return fig


def _parse_timestamp(value: Any) -> Optional[datetime]:
    if isinstance(value, datetime):
        return value
    if isinstance(value, str):
        try:
            return datetime.fromisoformat(value)
        except ValueError:
            return None
    return None


def _make_trade_price_figure(report, symbol: str) -> Optional["plt.Figure"]:
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except Exception:
        return None

    history = report.price_history.get(symbol)
    if not history:
        return None
    times: List[datetime] = []
    closes: List[float] = []
    for entry in history:
        ts = _parse_timestamp(entry.get("timestamp"))
        close = entry.get("close")
        if ts is None or close is None:
            continue
        try:
            close_val = float(close)
        except (TypeError, ValueError):
            continue
        times.append(ts)
        closes.append(close_val)
    if not times or not closes:
        return None

    trades = [trade for trade in report.trade_executions if trade.symbol == symbol]
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(times, closes, label="Close", color="tab:blue")

    if trades:
        buy_times = [trade.timestamp for trade in trades if trade.side.lower() == "buy"]
        buy_prices = [float(trade.price) for trade in trades if trade.side.lower() == "buy"]
        sell_times = [trade.timestamp for trade in trades if trade.side.lower() == "sell"]
        sell_prices = [float(trade.price) for trade in trades if trade.side.lower() == "sell"]
        if buy_times:
            ax.scatter(buy_times, buy_prices, marker="^", color="tab:green", label="Buy", zorder=3)
        if sell_times:
            ax.scatter(sell_times, sell_prices, marker="v", color="tab:red", label="Sell", zorder=3)

    meta = report.symbol_metadata.get(symbol, {})
    strategy = meta.get("strategy", "unknown")
    trade_mode = meta.get("trade_mode", "unknown")
    ax.set_title(f"{symbol} Close Price with Trades (strategy={strategy}, mode={trade_mode})")
    ax.set_xlabel("Timestamp")
    ax.set_ylabel("Price ($)")
    ax.legend(loc="best", fontsize="small")
    ax.grid(alpha=0.3)
    fig.autofmt_xdate()
    return fig


def _log_to_wandboard(
    args: argparse.Namespace,
    report,
    metrics_payload: Dict[str, object],
    trade_summary: Dict[str, Dict[str, float]],
    *,
    risk_series: Optional[List[Dict[str, Any]]] = None,
    analysis_summary: Optional[Dict[str, Any]] = None,
) -> None:
    tags = ["marketsimulator", "simulation"]
    tags.append("real-analytics" if args.real_analytics else "mock-analytics")
    if args.kronos_only:
        tags.append("kronos-only")

    config = {
        "symbols": list(args.symbols),
        "steps": int(args.steps),
        "step_size": int(args.step_size),
        "top_k": int(args.top_k),
        "initial_cash": float(args.initial_cash),
        "flatten_end": bool(args.flatten_end),
        "real_analytics": bool(args.real_analytics),
        "kronos_only": bool(args.kronos_only),
        "sharpe_cutoff": args.sharpe_cutoff,
        "kronos_sharpe_cutoff": args.kronos_sharpe_cutoff,
    }
    config.update({key: value for key, value in _collect_run_metadata().items() if value not in ("", None)})

    try:
        wand_logger = WandBoardLogger(
            run_name=f"marketsim_{datetime.now(UTC).strftime('%Y%m%d_%H%M%S')}",
            tags=tags,
            config=config,
            notes="Automated market simulator run.",
            log_metrics=True,
        )
    except Exception as exc:  # pragma: no cover - defensive logging
        logger.warning(f"[sim] Unable to initialise WandBoardLogger: {exc}")
        return

    with wand_logger as metrics_logger:
        metrics_logger.log(metrics_payload, step=0)

        equity_curve = compute_equity_timeseries(report)
        for entry in equity_curve:
            step = int(entry["day_index"]) + 1
            metrics_logger.log(
                {
                    "equity/close": float(entry["equity"]),
                    "cash/close": float(entry["cash"]),
                    "returns/daily": float(entry["daily_return"]),
                    "returns/cumulative": float(entry["cumulative_return"]),
                    "positions/open_count": float(entry["open_positions"]),
                    "positions/gross_qty": float(entry["gross_position_qty"]),
                },
                step=step,
            )

        final_step = int(equity_curve[-1]["day_index"]) + 1 if equity_curve else 0

        risk_series = risk_series or compute_risk_timeseries(report)
        risk_columns = [
            "timestamp",
            "day_index",
            "phase",
            "equity",
            "cash",
            "gross_exposure",
            "net_exposure",
            "leverage",
            "drawdown",
            "drawdown_pct",
        ]
        risk_rows: List[Sequence[Any]] = []
        for idx, entry in enumerate(risk_series, start=1):
            metrics_logger.log(
                {
                    "risk/gross_exposure": float(entry["gross_exposure"]),
                    "risk/net_exposure": float(entry["net_exposure"]),
                    "risk/leverage": float(entry["leverage"]),
                    "risk/drawdown": float(entry["drawdown"]),
                    "risk/drawdown_pct": float(entry["drawdown_pct"]),
                    "risk/cash": float(entry["cash"]),
                    "risk/equity": float(entry["equity"]),
                },
                step=len(equity_curve) + idx,
            )
            timestamp = entry["timestamp"]
            ts_value = timestamp.isoformat() if hasattr(timestamp, "isoformat") else timestamp
            risk_rows.append(
                [
                    ts_value,
                    entry["day_index"],
                    entry["phase"],
                    float(entry["equity"]),
                    float(entry["cash"]),
                    float(entry["gross_exposure"]),
                    float(entry["net_exposure"]),
                    float(entry["leverage"]),
                    float(entry["drawdown"]),
                    float(entry["drawdown_pct"]),
                ]
            )
        if risk_rows:
            metrics_logger.log_table("risk/snapshots", risk_columns, risk_rows, step=len(equity_curve) + len(risk_rows))
        final_step = max(final_step, len(equity_curve) + len(risk_rows))

        fee_metrics = compute_fee_breakdown(report)
        metrics_logger.log(fee_metrics, step=final_step)

        breakdowns = compute_breakdowns(report)

        asset_metrics: Dict[str, float] = {}
        for asset_key, stats in breakdowns.get("asset", {}).items():
            asset_metrics[f"asset/{asset_key}/realised_pnl"] = float(stats["realised_pnl"])
            asset_metrics[f"asset/{asset_key}/total_value"] = float(stats["total_value"])
            asset_metrics[f"asset/{asset_key}/trades"] = float(stats["trades"])
        if asset_metrics:
            metrics_logger.log(asset_metrics, step=final_step)

        mode_metrics: Dict[str, float] = {}
        for mode_key, stats in breakdowns.get("trade_mode", {}).items():
            mode_metrics[f"trade_mode/{mode_key}/realised_pnl"] = float(stats["realised_pnl"])
            mode_metrics[f"trade_mode/{mode_key}/cash_flow"] = float(stats["cash_flow"])
        if mode_metrics:
            metrics_logger.log(mode_metrics, step=final_step)

        strategy_metrics: Dict[str, float] = {}
        for strategy_key, stats in breakdowns.get("strategy", {}).items():
            strategy_metrics[f"strategy/{strategy_key}/realised_pnl"] = float(stats["realised_pnl"])
            strategy_metrics[f"strategy/{strategy_key}/unrealized_pl"] = float(stats["unrealized_pl"])
        if strategy_metrics:
            metrics_logger.log(strategy_metrics, step=final_step)

        summary_metrics: Dict[str, float] = {}
        equity_pnl_key = "asset/equity/realised_pnl"
        crypto_pnl_key = "asset/crypto/realised_pnl"
        normal_mode_key = "trade_mode/normal/realised_pnl"
        if equity_pnl_key in asset_metrics:
            summary_metrics["summary/non_crypto_realised_pnl"] = asset_metrics[equity_pnl_key]
        if crypto_pnl_key in asset_metrics:
            summary_metrics["summary/crypto_realised_pnl"] = asset_metrics[crypto_pnl_key]
        if normal_mode_key in mode_metrics:
            summary_metrics["summary/strategy_only_realised_pnl"] = mode_metrics[normal_mode_key]
        summary_metrics["summary/trading_fees"] = float(report.trading_fees_paid)
        summary_metrics["summary/financing_cost"] = float(report.financing_cost_paid)
        if summary_metrics:
            metrics_logger.log(summary_metrics, step=final_step)

        overall = trade_summary.get("__overall__")
        if overall:
            overall_payload = {}
            for key, value in overall.items():
                if isinstance(value, (int, float)):
                    overall_payload[f"trades/overall/{key}"] = float(value)
            if overall_payload:
                metrics_logger.log(overall_payload, step=final_step)

        columns, rows = build_symbol_performance_table(report)
        if rows:
            metrics_logger.log_table("symbol_performance", columns, rows, step=final_step)

        stack_columns, stack_rows = build_portfolio_stack_series(report)
        if stack_rows:
            metrics_logger.log_table("portfolio/stack", stack_columns, stack_rows, step=final_step)

        trade_columns, trade_rows = build_trade_events_table(report)
        if trade_rows:
            metrics_logger.log_table("trades/events", trade_columns, trade_rows, step=final_step)

        price_columns, price_rows = build_price_history_table(report)
        if price_rows:
            metrics_logger.log_table("prices/history_preview", price_columns, price_rows[:2000], step=final_step)

        analysis_summary = analysis_summary or summarize_daily_analysis(report)
        if analysis_summary:
            analysis_metrics: Dict[str, float] = {}
            for key, value in analysis_summary.items():
                if isinstance(value, (int, float)):
                    analysis_metrics[f"analysis/{key}"] = float(value)
            for strategy, count in analysis_summary.get("strategy_counts", {}).items():
                analysis_metrics[f"analysis/strategy/{strategy}"] = float(count)
            for mode, count in analysis_summary.get("trade_mode_counts", {}).items():
                analysis_metrics[f"analysis/trade_mode/{mode}"] = float(count)
            if analysis_metrics:
                metrics_logger.log(analysis_metrics, step=final_step)

        summary_text = report.render_summary()
        metrics_logger.log_text("simulation/summary", summary_text, step=final_step)

        stack_fig = _make_portfolio_stack_figure(report)
        if stack_fig is not None:
            metrics_logger.log_figure("figures/portfolio_stack", stack_fig, step=final_step)
            try:
                import matplotlib.pyplot as plt  # type: ignore
            except Exception:
                pass
            else:
                plt.close(stack_fig)

        symbols_with_trades = sorted({trade.symbol for trade in report.trade_executions})
        for symbol in symbols_with_trades[:5]:
            fig = _make_trade_price_figure(report, symbol)
            if fig is None:
                continue
            metrics_logger.log_figure(f"figures/trades/{symbol}", fig, step=final_step)
            try:
                import matplotlib.pyplot as plt  # type: ignore
            except Exception:
                pass
            else:
                plt.close(fig)


def main(argv: Optional[Iterable[str]] = None) -> int:
    args = parse_args(argv)
    if getattr(args, "stub_config", False):
        stub_return = 0.0125
        stub_sharpe = 1.0500
        stub_pnl = args.initial_cash * stub_return
        stub_cash = args.initial_cash + stub_pnl
        summary = {
            "mode": "stub",
            "return": stub_return,
            "sharpe": stub_sharpe,
            "pnl": stub_pnl,
            "cash": stub_cash,
            "steps": args.steps,
            "step_size": args.step_size,
            "symbols": args.symbols,
            "top_k": args.top_k,
            "initial_cash": args.initial_cash,
            "kronos_only": args.kronos_only,
            "compact_logs": args.compact_logs,
            "real_analytics": args.real_analytics,
        }
        summary_json = json.dumps(summary, sort_keys=True)
        for metric_line in (
            f"return={stub_return:.6f}",
            f"sharpe={stub_sharpe:.6f}",
            f"pnl={stub_pnl:.2f}",
            f"cash={stub_cash:.2f}",
            f"balance={stub_cash:.2f}",
        ):
            print(metric_line)
        print(f"stub-summary={summary_json}")
        return 0
    _configure_compact_logging_pre(args.compact_logs)

    use_mock = not args.real_analytics
    if not use_mock and not _real_stack_available():
        use_mock = True

    previous_mock_setting = os.environ.get(ENV_MOCK_ANALYTICS)
    os.environ[ENV_MOCK_ANALYTICS] = "1" if use_mock else "0"
    override_keys = {
        "MARKETSIM_SHARPE_CUTOFF": args.sharpe_cutoff,
        "MARKETSIM_KRONOS_SHARPE_CUTOFF": args.kronos_sharpe_cutoff,
    }
    previous_overrides = {key: os.environ.get(key) for key in override_keys}
    for key, value in override_keys.items():
        if value is None:
            continue
        os.environ[key] = str(value)

    runner_module = _load_runner()
    simulate_strategy = getattr(runner_module, "simulate_strategy")
    lookahead_value = str(max(1, args.step_size))
    previous_lookahead = os.environ.get("MARKETSIM_FORECAST_LOOKAHEAD")
    os.environ["MARKETSIM_FORECAST_LOOKAHEAD"] = lookahead_value

    try:
        report = simulate_strategy(
            symbols=args.symbols,
            days=max(1, args.steps),
            step_size=max(1, args.step_size),
            initial_cash=args.initial_cash,
            top_k=max(1, args.top_k),
            force_kronos=args.kronos_only,
            flatten_end=args.flatten_end,
        )
    except Exception:
        logger.exception("[sim] Simulation run failed.")
        return 1
    finally:
        _restore_env("MARKETSIM_FORECAST_LOOKAHEAD", previous_lookahead)
        for key, prev in previous_overrides.items():
            _restore_env(key, prev)
        _restore_env(ENV_MOCK_ANALYTICS, previous_mock_setting)
        _configure_compact_logging_post(args.compact_logs)

    if report is None:  # pragma: no cover - defensive
        logger.error("[sim] Simulation did not produce a report.")
        return 1

    print(report.render_summary())
    metrics = _compute_step_metrics(report)

    entry_snapshot: Dict[str, Dict] = {}
    try:
        trade_module = importlib.import_module("trade_stock_e2e")
        if hasattr(trade_module, "get_entry_counter_snapshot"):
            entry_snapshot = trade_module.get_entry_counter_snapshot()
    except Exception as exc:  # pragma: no cover - best-effort telemetry
        logger.warning(f"[sim] Unable to capture entry counter snapshot: {exc}")

    metrics_payload = {
        "mode": "simulation",
        "steps": int(args.steps),
        "step_size": int(args.step_size),
        "top_k": int(args.top_k),
        "symbols": list(args.symbols),
        "initial_cash": float(args.initial_cash),
        "kronos_only": bool(args.kronos_only),
        "real_analytics": bool(args.real_analytics),
        "sharpe_cutoff": args.sharpe_cutoff,
        "kronos_sharpe_cutoff": args.kronos_sharpe_cutoff,
        "flatten_end": bool(args.flatten_end),
        **metrics,
    }
    metrics_payload["trading_fees_paid"] = float(report.trading_fees_paid)
    metrics_payload["financing_cost_paid"] = float(report.financing_cost_paid)

    risk_series = compute_risk_timeseries(report)
    if risk_series:
        metrics_payload["max_leverage"] = max(float(entry["leverage"]) for entry in risk_series)
        metrics_payload["final_gross_exposure"] = float(risk_series[-1]["gross_exposure"])
        metrics_payload["final_net_exposure"] = float(risk_series[-1]["net_exposure"])
        metrics_payload["final_drawdown_pct"] = float(risk_series[-1]["drawdown_pct"])

    analysis_summary = summarize_daily_analysis(report)
    if analysis_summary:
        metrics_payload["analysis_summary"] = analysis_summary

    if entry_snapshot:
        metrics_payload["entry_limits"] = entry_snapshot
        metrics_payload.update(_flatten_entry_limits(entry_snapshot))
    _export_metrics(metrics_payload, args.metrics_json, args.metrics_csv)
    _export_trades(report.trade_executions, args.trades_csv)
    trade_summary = _summarize_trades(report.trade_executions)
    _export_trade_summary(trade_summary, args.trades_summary_json)
    _log_to_wandboard(
        args,
        report,
        metrics_payload,
        trade_summary,
        risk_series=risk_series,
        analysis_summary=analysis_summary,
    )
    print("sim-summary=" + json.dumps(metrics_payload, sort_keys=True))
    ordered_keys = [
        "return",
        "sharpe",
        "pnl",
        "cash",
        "balance",
        "max_drawdown",
        "max_drawdown_pct",
        "fees_paid",
    ]
    for key in ordered_keys:
        print(_format_metric_line(key, metrics[key]))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
