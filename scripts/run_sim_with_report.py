#!/usr/bin/env python3
"""Run marketsimulator trade loop and emit a post-run trade summary.

Example:
    python scripts/run_sim_with_report.py -- TRADE_STATE_SUFFIX=sim \
        python marketsimulator/run_trade_loop.py --symbols AAPL MSFT \
        --steps 20 --step-size 1 --initial-cash 100000 --kronos-only --flatten-end

Any arguments after ``--`` are forwarded to the child process exactly as provided.
The script automatically injects metrics/trade export flags and prints a concise
report using ``scripts/analyze_trades_csv.py`` once the run completes.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Dict, List

from trade_limit_utils import apply_fee_slip_defaults, parse_trade_limit_map

REPO_ROOT = Path(__file__).resolve().parent.parent
RUN_LOG_DIR = REPO_ROOT / "marketsimulator" / "run_logs"
ANALYZER = REPO_ROOT / "scripts" / "analyze_trades_csv.py"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Wrapper that runs marketsimulator/run_trade_loop.py and prints a trade summary."
    )
    parser.add_argument(
        "run_args",
        nargs=argparse.REMAINDER,
        help="Command to execute (prepend with '--' to separate from wrapper arguments).",
    )
    parser.add_argument(
        "--prefix",
        default=datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S"),
        help="Prefix for generated metric/trade files (default: UTC timestamp).",
    )
    parser.add_argument(
        "--skip-summary",
        action="store_true",
        help="Skip running the analyzer after the simulation completes.",
    )
    parser.add_argument(
        "--max-fee-bps",
        type=float,
        default=None,
        help="Alert if any symbol fees exceed this basis-point ratio of gross notional.",
    )
    parser.add_argument(
        "--max-avg-slip",
        type=float,
        default=None,
        help="Alert if any symbol average absolute slip (bps) exceeds this threshold.",
    )
    parser.add_argument(
        "--max-drawdown-pct",
        type=float,
        default=None,
        help="Alert if reported max drawdown percentage exceeds this threshold (0-100).",
    )
    parser.add_argument(
        "--min-final-pnl",
        type=float,
        default=None,
        help="Alert if final PnL (USD) is below this amount.",
    )
    parser.add_argument(
        "--max-worst-cash",
        type=float,
        default=None,
        help="Alert if worst cumulative cash delta falls below this negative USD amount.",
    )
    parser.add_argument(
        "--min-symbol-pnl",
        type=float,
        default=None,
        help="Alert if any symbol net cash delta (PnL) falls below this USD threshold.",
    )
    parser.add_argument(
        "--max-trades",
        type=float,
        default=None,
        help="Alert if any symbol executes more trades than this limit.",
    )
    parser.add_argument(
        "--max-trades-map",
        type=str,
        default=None,
        help="Comma-separated overrides for per-symbol trade limits (e.g., 'NVDA:6,MSFT:20' "
        "or entries with strategy tags like 'NVDA@ci_guard:6').",
    )
    parser.add_argument(
        "--fail-on-alert",
        action="store_true",
        help="Exit with non-zero status if any alert threshold is breached.",
    )
    return parser.parse_args()


def ensure_analyzer_available() -> None:
    if not ANALYZER.exists():
        raise FileNotFoundError(f"Analyzer script not found: {ANALYZER}")


def build_output_paths(prefix: str) -> dict[str, Path]:
    RUN_LOG_DIR.mkdir(parents=True, exist_ok=True)
    return {
        "metrics_json": RUN_LOG_DIR / f"{prefix}_metrics.json",
        "metrics_csv": RUN_LOG_DIR / f"{prefix}_metrics.csv",
        "trades_csv": RUN_LOG_DIR / f"{prefix}_trades.csv",
        "trades_summary_json": RUN_LOG_DIR / f"{prefix}_trades_summary.json",
    }


def run_simulation(cmd: list[str]) -> int:
    completed = subprocess.run(cmd)
    return completed.returncode


def run_analyzer(trades_csv: Path, trades_summary: Path) -> None:
    if not trades_csv.exists():
        print(f"[warn] trades CSV not found at {trades_csv}; skipping summary.", file=sys.stderr)
        return
    summary_cmd = [sys.executable, str(ANALYZER), str(trades_csv)]
    if trades_summary.exists():
        summary_cmd.extend(["--per-cycle"])
    print("\n=== Post-run trade summary ===")
    subprocess.run(summary_cmd, check=False)


def main() -> int:
    args = parse_args()
    if not args.run_args:
        print("No command provided. Supply run arguments after '--'.", file=sys.stderr)
        return 1
    run_cmd = args.run_args
    if run_cmd and run_cmd[0] == "--":
        run_cmd = run_cmd[1:]
    if not run_cmd:
        print("No command provided after '--'.", file=sys.stderr)
        return 1
    ensure_analyzer_available()
    paths = build_output_paths(args.prefix)

    # Inject export flags into the child process.
    injected_flags = [
        "--metrics-json",
        str(paths["metrics_json"]),
        "--metrics-csv",
        str(paths["metrics_csv"]),
        "--trades-csv",
        str(paths["trades_csv"]),
        "--trades-summary-json",
        str(paths["trades_summary_json"]),
    ]

    run_cmd = run_cmd + injected_flags
    print(f"[info] Running command: {' '.join(run_cmd)}")
    rc = run_simulation(run_cmd)
    if rc != 0:
        print(f"[error] Simulation failed with exit code {rc}.", file=sys.stderr)
        return rc

    if not args.skip_summary:
        run_analyzer(paths["trades_csv"], paths["trades_summary_json"])
    else:
        print("[info] Summary skipped (--skip-summary).")
    max_trades_overrides = parse_trade_limit_map(args.max_trades_map, verbose=False)
    max_fee_bps, max_avg_slip = apply_fee_slip_defaults(args.max_fee_bps, args.max_avg_slip)
    alerts_triggered = check_alerts(
        paths["trades_summary_json"],
        max_fee_bps=max_fee_bps,
        max_avg_slip=max_avg_slip,
        metrics_json=paths["metrics_json"],
        max_drawdown_pct=args.max_drawdown_pct,
        min_final_pnl=args.min_final_pnl,
        max_worst_cash=args.max_worst_cash,
        min_symbol_pnl=args.min_symbol_pnl,
        max_trades=args.max_trades,
        max_trades_map=max_trades_overrides,
    )
    if alerts_triggered:
        print("[warn] Alert thresholds exceeded:")
        for line in alerts_triggered:
            print(f"  - {line}")
        if args.fail_on_alert:
            return 2
    print(f"[info] Outputs written with prefix '{args.prefix}' to {RUN_LOG_DIR}")
    return 0


def check_alerts(
    summary_path: Path,
    *,
    max_fee_bps: float | None,
    max_avg_slip: float | None,
    metrics_json: Path,
    max_drawdown_pct: float | None,
    min_final_pnl: float | None,
    max_worst_cash: float | None,
    min_symbol_pnl: float | None,
    max_trades: float | None,
    max_trades_map: Dict[str, float],
) -> List[str]:
    alerts: List[str] = []
    summary = {}
    if summary_path.exists():
        try:
            with summary_path.open("r", encoding="utf-8") as handle:
                summary = json.load(handle)
        except Exception as exc:
            alerts.append(f"Unable to parse trade summary ({exc}).")

    metrics = {}
    if metrics_json.exists():
        try:
            with metrics_json.open("r", encoding="utf-8") as handle:
                metrics = json.load(handle)
        except Exception as exc:
            alerts.append(f"Unable to parse metrics JSON ({exc}).")

    for symbol, stats in summary.items():
        if symbol == "__overall__":
            continue
        gross = float(stats.get("gross_notional", 0.0))
        fees = float(stats.get("fees", 0.0))
        avg_slip = float(stats.get("average_slip_bps", 0.0))
        cash_delta = float(stats.get("cash_delta", 0.0))
        trades = float(stats.get("trades", 0.0))
        if max_fee_bps and gross > 0:
            fee_bps = (fees / gross) * 1e4
            if fee_bps > max_fee_bps:
                alerts.append(
                    f"{symbol}: fee ratio {fee_bps:.2f} bps exceeds threshold {max_fee_bps:.2f} bps"
                )
        if max_avg_slip and avg_slip > max_avg_slip:
            alerts.append(
                f"{symbol}: average slip {avg_slip:.2f} bps exceeds threshold {max_avg_slip:.2f} bps"
            )
        if min_symbol_pnl is not None and cash_delta < min_symbol_pnl:
            alerts.append(
                f"{symbol}: net PnL ${cash_delta:,.2f} below threshold ${min_symbol_pnl:,.2f}"
            )
        trade_limit = max_trades_map.get(symbol)
        if trade_limit is None:
            trade_limit = max_trades
        if trade_limit is not None and trades > trade_limit:
            alerts.append(
                f"{symbol}: trade count {trades:.0f} exceeds limit {trade_limit:.0f}"
            )
    if metrics:
        if max_drawdown_pct is not None:
            drawdown_pct = float(metrics.get("max_drawdown_pct", 0.0)) * 100.0
            if drawdown_pct > max_drawdown_pct:
                alerts.append(
                    f"Max drawdown {drawdown_pct:.2f}% exceeds threshold {max_drawdown_pct:.2f}%"
                )
        if min_final_pnl is not None:
            final_pnl = float(metrics.get("pnl", 0.0))
            if final_pnl < min_final_pnl:
                alerts.append(
                    f"Final PnL ${final_pnl:,.2f} below threshold ${min_final_pnl:,.2f}"
                )
        if max_worst_cash is not None:
            worst_cash = float(summary.get("__overall__", {}).get("worst_cumulative_cash", 0.0))
            if worst_cash < max_worst_cash:
                alerts.append(
                    f"Worst cumulative cash ${worst_cash:,.2f} below threshold ${max_worst_cash:,.2f}"
                )
    return alerts


if __name__ == "__main__":
    sys.exit(main())
