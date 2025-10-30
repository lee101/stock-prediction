#!/usr/bin/env python3
"""
Evaluate how widening the portfolio (N concurrent positions) impacts expected
profitability using the marketsimulator stack. The script first runs the full
Toto+Kronos analytics; if those produce no actionable picks, it falls back to a
Kronos-only pass so we can still quantify the diversification effect.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List

import importlib
import matplotlib.pyplot as plt
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from marketsimulator import activate_simulation


RUN_LOG_DIR = REPO_ROOT / "marketsimulator" / "run_logs"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze N-position spreads using marketsimulator analytics."
    )
    parser.add_argument(
        "--symbols",
        nargs="+",
        default=[
            "AAPL",
            "MSFT",
            "NVDA",
            "AMZN",
            "GOOG",
            "META",
            "TSLA",
            "AMD",
            "NFLX",
            "AVGO",
        ],
        help="Universe of symbols to analyze.",
    )
    parser.add_argument(
        "--min-n",
        type=int,
        default=1,
        help="Minimum portfolio width to evaluate.",
    )
    parser.add_argument(
        "--max-n",
        type=int,
        default=10,
        help="Maximum portfolio width to evaluate.",
    )
    parser.add_argument(
        "--initial-cash",
        type=float,
        default=100_000.0,
        help="Initial cash balance for the synthetic account.",
    )
    parser.add_argument(
        "--backtest-sims",
        type=int,
        default=12,
        help="Number of Toto/Kronos backtest simulations per symbol.",
    )
    parser.add_argument(
        "--toto-num-samples",
        type=int,
        default=384,
        help="Baseline Toto sample count per inference window.",
    )
    parser.add_argument(
        "--toto-batch-size",
        type=int,
        default=64,
        help="Baseline Toto samples-per-batch setting.",
    )
    parser.add_argument(
        "--kronos-sample-count",
        type=int,
        default=64,
        help="Samples per Kronos inference request.",
    )
    parser.add_argument(
        "--min-strategy-return-map",
        type=str,
        default="AAPL:-0.03,MSFT:-0.03,NVDA:-0.03,AMZN:-0.02,GOOG:-0.02,META:-0.02,TSLA:-0.02,AMD:-0.02,NFLX:-0.015,AVGO:-0.02",
        help="Overrides for MARKETSIM_SYMBOL_MIN_STRATEGY_RETURN_MAP.",
    )
    parser.add_argument(
        "--min-predicted-move-map",
        type=str,
        default="AAPL:0.005,MSFT:0.005,NVDA:0.01,AMZN:0.006,GOOG:0.006,META:0.006,TSLA:0.01,AMD:0.008,NFLX:0.008,AVGO:0.008",
        help="Overrides for MARKETSIM_SYMBOL_MIN_PREDICTED_MOVE_MAP.",
    )
    parser.add_argument(
        "--output-prefix",
        type=str,
        default=None,
        help="Prefix for generated artefacts (defaults to UTC timestamp).",
    )
    return parser.parse_args()


def _configure_base_environment(args: argparse.Namespace) -> None:
    os.environ.setdefault("WANDB_MODE", "offline")
    os.environ["MARKETSIM_ALLOW_MOCK_ANALYTICS"] = "0"
    os.environ["MARKETSIM_USE_MOCK_ANALYTICS"] = "0"
    os.environ["MARKETSIM_FORCE_MARKET_OPEN"] = "1"
    os.environ["MARKETSIM_DISABLE_GATES"] = "1"
    os.environ["MARKETSIM_RELAX_SPREAD"] = "1"
    os.environ["MARKETSIM_BACKTEST_SIMULATIONS"] = str(max(1, args.backtest_sims))
    os.environ["MARKETSIM_KRONOS_SAMPLE_COUNT"] = str(max(1, args.kronos_sample_count))
    os.environ["TOTO_NUM_SAMPLES"] = str(max(8, args.toto_num_samples))
    os.environ["TOTO_SAMPLES_PER_BATCH"] = str(max(4, args.toto_batch_size))
    os.environ["MARKETSIM_TOTO_MIN_NUM_SAMPLES"] = str(max(8, args.toto_num_samples // 2))
    os.environ["MARKETSIM_TOTO_MIN_SAMPLES_PER_BATCH"] = str(max(4, args.toto_batch_size // 2))
    os.environ["MARKETSIM_TOTO_MAX_NUM_SAMPLES"] = str(max(args.toto_num_samples, args.toto_num_samples * 2))
    os.environ["MARKETSIM_TOTO_MAX_SAMPLES_PER_BATCH"] = str(max(args.toto_batch_size, args.toto_batch_size * 2))
    os.environ.setdefault("COMPILED_MODELS_DIR", str(REPO_ROOT / "compiled_models"))
    if args.min_strategy_return_map:
        os.environ["MARKETSIM_SYMBOL_MIN_STRATEGY_RETURN_MAP"] = args.min_strategy_return_map
    if args.min_predicted_move_map:
        os.environ["MARKETSIM_SYMBOL_MIN_PREDICTED_MOVE_MAP"] = args.min_predicted_move_map


@contextmanager
def _force_kronos(enabled: bool):
    key = "MARKETSIM_FORCE_KRONOS"
    previous = os.environ.get(key)
    if enabled:
        os.environ[key] = "1"
    else:
        os.environ.pop(key, None)
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = previous


def _collect_analysis(args: argparse.Namespace, *, force_kronos: bool) -> pd.DataFrame:
    with _force_kronos(force_kronos):
        with activate_simulation(symbols=args.symbols, initial_cash=args.initial_cash, force_kronos=force_kronos) as controller:
            trade_module = importlib.import_module("trade_stock_e2e")
            predict_module = importlib.import_module("predict_stock_forecasting")
            alpaca_module = importlib.import_module("alpaca_wrapper")
            if hasattr(trade_module, "reset_symbol_entry_counters"):
                trade_module.reset_symbol_entry_counters(run_id="spread-analysis")
            predict_module.make_predictions(
                input_data_path="_simulator",
                pred_name="SPREAD-ANALYSIS",
                retrain=False,
                alpaca_wrapper=alpaca_module,
                symbols=args.symbols,
            )
            analyzed: Dict[str, Dict] = trade_module.analyze_symbols(args.symbols)

    rows: List[Dict] = []
    for symbol, payload in analyzed.items():
        row: Dict[str, float | str] = {"symbol": symbol}
        for key, value in payload.items():
            if isinstance(value, (int, float)):
                row[key] = float(value)
        rows.append(row)
    df = pd.DataFrame(rows)
    if not df.empty:
        df.sort_values("composite_score", ascending=False, inplace=True)
        df.reset_index(drop=True, inplace=True)
    return df


def _sum_column(frame: pd.DataFrame, column: str) -> float:
    if column not in frame.columns:
        return float("nan")
    return float(frame[column].sum())


def _compute_spread_curve(df: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    metrics: List[Dict[str, float]] = []
    for n in range(args.min_n, args.max_n + 1):
        subset = df.head(n)
        payload = {
            "positions": n,
            "avg_return_pct": _sum_column(subset, "avg_return") * 100.0,
            "simple_return_pct": _sum_column(subset, "simple_return") * 100.0,
            "entry_takeprofit_return_pct": _sum_column(subset, "entry_takeprofit_return") * 100.0,
            "maxdiff_return_pct": _sum_column(subset, "maxdiff_return") * 100.0,
            "composite_score": _sum_column(subset, "composite_score"),
            "aggregate_predicted_move": _sum_column(subset, "predicted_movement"),
        }
        metrics.append(payload)
    return pd.DataFrame(metrics)


def _build_prefix(custom_prefix: str | None) -> str:
    if custom_prefix:
        return custom_prefix
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    return f"position_spread_{timestamp}"


def _save_outputs(
    curve_df: pd.DataFrame,
    symbol_df: pd.DataFrame,
    prefix: str,
) -> tuple[Path, Path, Path]:
    RUN_LOG_DIR.mkdir(parents=True, exist_ok=True)
    curve_csv = RUN_LOG_DIR / f"{prefix}_curve.csv"
    symbol_csv = RUN_LOG_DIR / f"{prefix}_symbols.csv"
    chart_path = RUN_LOG_DIR / f"{prefix}.png"

    curve_df.to_csv(curve_csv, index=False)
    symbol_df.to_csv(symbol_csv, index=False)

    plt.figure(figsize=(9, 4.5))
    plt.plot(curve_df["positions"], curve_df["avg_return_pct"], marker="o", label="Avg return (%)")
    if "maxdiff_return_pct" in curve_df.columns:
        plt.plot(
            curve_df["positions"],
            curve_df["maxdiff_return_pct"],
            marker="x",
            linestyle="--",
            label="MaxDiff return (%)",
        )
    plt.xlabel("Concurrent Positions (N)")
    plt.ylabel("Aggregated Return (%)")
    plt.title("Impact of Portfolio Width on Simulated Return")
    plt.grid(True, linestyle=":", alpha=0.4)
    plt.legend()
    plt.tight_layout()
    plt.savefig(chart_path)
    plt.close()

    return curve_csv, symbol_csv, chart_path


def main() -> int:
    args = parse_args()
    if args.min_n <= 0 or args.max_n <= 0:
        raise SystemExit("Both --min-n and --max-n must be positive integers.")
    if args.min_n > args.max_n:
        raise SystemExit("--min-n cannot exceed --max-n.")

    _configure_base_environment(args)
    prefix = _build_prefix(args.output_prefix)

    print("[info] Running Toto + Kronos analytics ...", flush=True)
    toto_df = _collect_analysis(args, force_kronos=False)
    if toto_df.empty:
        print("[warn] Toto pipeline produced no tradable candidates under the current gating.", flush=True)
    else:
        toto_path = RUN_LOG_DIR / f"{prefix}_toto_symbols.csv"
        toto_df.to_csv(toto_path, index=False)
        print(f"[info] Stored Toto analytics snapshot at {toto_path}", flush=True)

    print("[info] Running Kronos-only analytics ...", flush=True)
    kronos_df = _collect_analysis(args, force_kronos=True)
    if kronos_df.empty:
        raise SystemExit("Kronos analytics returned no candidates; unable to compute spread curve.")

    curve_df = _compute_spread_curve(kronos_df, args)
    curve_csv, symbols_csv, chart_path = _save_outputs(curve_df, kronos_df, prefix)

    print(f"\n[info] Aggregated curve written to {curve_csv}")
    print(f"[info] Symbol-level metrics saved to {symbols_csv}")
    print(f"[info] Chart saved to {chart_path}\n")
    print("[info] Summary (Kronos-only spread curve):")
    print(
        curve_df.to_string(
            index=False,
            formatters={
                "avg_return_pct": "{:.4f}".format,
                "simple_return_pct": "{:.4f}".format,
                "entry_takeprofit_return_pct": "{:.4f}".format,
                "maxdiff_return_pct": "{:.4f}".format,
                "composite_score": "{:.4f}".format,
                "aggregate_predicted_move": "{:.4f}".format,
            },
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

