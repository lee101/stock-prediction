#!/usr/bin/env python3
"""
XGBoost hourly open-to-close (bar-by-bar) strategy.

Reads MKTD v2 binary hourly data (pufferlib_market/data/stocks*_hourly*.bin),
trains XGBoost on historical bars to predict next-hour open-to-close direction,
then backtests on held-out recent data.

Strategy
--------
Each trading hour H:
  1. Score all available symbols with XGBStockModel on hourly features
  2. Pick top-N by score (min confidence threshold)
  3. Simulate: buy at H's open fill, sell at H's close fill
  4. Apply explicit fee/fill-buffer costs, optional leverage

Data
----
Uses pufferlib_market/data/*_hourly_*_v2_*.bin  (MKTD v2 format, 10-15 symbols).
Falls back to daily CSVs treated as 1-bar-per-day if no binary data available.

Usage
-----
  python -m xgbnew.run_hourly \\
      --mktd-file pufferlib_market/data/stocks13_hourly_forecast_mktd_v2_20260214.bin \\
      --top-n 2 --leverage 1.0 --output-dir analysis/xgbnew_hourly
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import pandas as pd


REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from xgbnew.backtest import (  # noqa: E402,I001
    PRODUCTION_STOCK_FEE_RATE,
    STOCK_HOURS_PER_YEAR,
    BacktestConfig,
    BacktestResult,
    _compute_result,
    print_summary,
    simulate_hourly as _shared_simulate_hourly,
)
from xgbnew.artifacts import write_json_atomic  # noqa: E402
from xgbnew.cli_realism import validate_nonnegative_realism_args  # noqa: E402
from xgbnew.features import HOURLY_FEATURE_COLS, build_features_for_symbol_hourly  # noqa: E402
from xgbnew.mktd_reader import read_mktd_hourly  # noqa: E402
from xgbnew.model import XGBStockModel  # noqa: E402


logger = logging.getLogger(__name__)

# Hourly backtest (different cadence from daily)

def _simulate_hourly(
    feat_dfs: dict[str, pd.DataFrame],
    model: XGBStockModel,
    config: BacktestConfig,
    train_end_ts: pd.Timestamp,
) -> BacktestResult:
    """Run shared hourly backtest on test data (bars after train_end_ts)."""
    parts = []
    for sym, df in feat_dfs.items():
        d = df[df["timestamp"] > train_end_ts].copy()
        d["symbol"] = sym
        parts.append(d)

    if not parts:
        logger.warning("No test bars found after %s", train_end_ts)
        return _compute_result([], config)

    merged = pd.concat(parts, ignore_index=True).sort_values("timestamp")
    return _shared_simulate_hourly(
        merged,
        model,
        config,
        bars_per_year=STOCK_HOURS_PER_YEAR,
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mktd-file", type=Path,
                   default=REPO / "pufferlib_market/data/stocks13_hourly_forecast_mktd_v2_20260214.bin")
    p.add_argument("--train-fraction", type=float, default=0.75,
                   help="Fraction of total bars to use for training (default 0.75)")
    p.add_argument("--top-n",      type=int,   default=2)
    p.add_argument("--leverage",   type=float, default=1.0)
    p.add_argument("--xgb-weight", type=float, default=1.0,
                   help="XGB weight (1.0=pure XGB; no Chronos2 for hourly)")
    p.add_argument("--commission-bps", type=float, default=0.0)
    p.add_argument(
        "--fee-rate",
        type=float,
        default=PRODUCTION_STOCK_FEE_RATE,
        help="Per-side fee fraction (default 0.001 = 10 bps production stress)",
    )
    p.add_argument("--fill-buffer-bps", type=float, default=5.0)
    p.add_argument("--min-score",  type=float, default=0.52,
                   help="Minimum P(up) to trade (default 0.52)")
    p.add_argument("--initial-cash", type=float, default=10_000.0)
    p.add_argument("--n-estimators", type=int, default=200)
    p.add_argument("--max-depth",    type=int, default=4)
    p.add_argument(
        "--device",
        default="cuda",
        help="XGBoost device (default 'cuda' when available). Pass 'cpu' to force CPU.",
    )
    p.add_argument("--output-dir", type=Path,
                   default=REPO / "analysis/xgbnew_hourly")
    p.add_argument("--verbose", "-v", action="store_true")
    return p.parse_args(argv)


def _validate_realism_args(args: argparse.Namespace) -> list[str]:
    return validate_nonnegative_realism_args(args)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)s %(message)s",
    )

    validation_failures = _validate_realism_args(args)
    if validation_failures:
        for failure in validation_failures:
            print(f"ERROR: {failure}", file=sys.stderr)
        return 2

    # Load hourly MKTD data
    if not args.mktd_file.exists():
        print(f"ERROR: MKTD file not found: {args.mktd_file}", file=sys.stderr)
        print("Available files:", flush=True)
        for f in sorted((REPO / "pufferlib_market/data").glob("*.bin"))[:5]:
            print(f"  {f}", flush=True)
        return 1

    print(f"[xgb-hourly] Reading {args.mktd_file.name} ...", flush=True)
    t0 = time.perf_counter()
    raw_data = read_mktd_hourly(args.mktd_file, market_hours_only=True)
    print(f"  {len(raw_data)} symbols loaded in {time.perf_counter()-t0:.1f}s", flush=True)
    for sym, df in raw_data.items():
        print(f"    {sym}: {len(df)} bars  "
              f"{df['timestamp'].iloc[0].date()} -> {df['timestamp'].iloc[-1].date()}", flush=True)

    if not raw_data:
        print("ERROR: No symbols loaded.", file=sys.stderr)
        return 1

    # Build hourly features
    print("[xgb-hourly] Building features...", flush=True)
    feat_dfs: dict[str, pd.DataFrame] = {}
    for sym, df in raw_data.items():
        feat = build_features_for_symbol_hourly(df, symbol=sym)
        feat = feat.dropna(subset=HOURLY_FEATURE_COLS[:5])
        if len(feat) < 200:
            continue
        feat_dfs[sym] = feat

    print(f"  {len(feat_dfs)} symbols with sufficient features", flush=True)

    if not feat_dfs:
        print("ERROR: No symbols with sufficient data.", file=sys.stderr)
        return 1

    # Train/test split (temporal)
    all_ts = sorted({
        ts for df in feat_dfs.values()
        for ts in df["timestamp"]
        if pd.notna(ts)
    })
    split_idx = int(len(all_ts) * args.train_fraction)
    train_end_ts = all_ts[split_idx] if split_idx < len(all_ts) else all_ts[-1]
    print(f"  Train -> {train_end_ts.date()}  |  Test -> after {train_end_ts.date()}", flush=True)

    # Build combined train DataFrame
    train_parts = []
    for _sym, df in feat_dfs.items():
        tr = df[df["timestamp"] <= train_end_ts].copy()
        if len(tr) > 50:
            train_parts.append(tr)

    if not train_parts:
        print("ERROR: No training data.", file=sys.stderr)
        return 1

    train_df = pd.concat(train_parts, ignore_index=True)
    print(f"  Training rows: {len(train_df):,}", flush=True)

    # Train XGBoost
    print("[xgb-hourly] Training XGBStockModel...", flush=True)
    model = XGBStockModel(
        device=args.device,
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        learning_rate=0.05,
    )
    model.fit(train_df, HOURLY_FEATURE_COLS, verbose=args.verbose)

    # Feature importances
    print("\n  Top-10 hourly feature importances:", flush=True)
    for feat, imp in model.feature_importances().head(10).items():
        print(f"    {feat:<25} {imp:.4f}")

    # Backtest
    configs = [
        BacktestConfig(top_n=args.top_n, leverage=1.0,
                       xgb_weight=1.0, commission_bps=args.commission_bps,
                       fee_rate=args.fee_rate, fill_buffer_bps=args.fill_buffer_bps,
                       min_score=args.min_score, initial_cash=args.initial_cash),
    ]
    if args.leverage > 1.0:
        configs.append(BacktestConfig(top_n=args.top_n, leverage=args.leverage,
                                      xgb_weight=1.0, commission_bps=args.commission_bps,
                                      fee_rate=args.fee_rate,
                                      fill_buffer_bps=args.fill_buffer_bps,
                                      min_score=args.min_score, initial_cash=args.initial_cash))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    results_summary = []

    for cfg in configs:
        label = f"hourly_top{cfg.top_n}_lev{cfg.leverage:.1f}_minsc{cfg.min_score:.2f}"
        result = _simulate_hourly(feat_dfs, model, cfg, train_end_ts)
        print_summary(result, label=label)
        results_summary.append({
            "label": label,
            "top_n": cfg.top_n,
            "leverage": cfg.leverage,
            "min_score": cfg.min_score,
            "fee_rate": cfg.fee_rate,
            "fill_buffer_bps": cfg.fill_buffer_bps,
            "commission_bps": cfg.commission_bps,
            "total_return_pct": result.total_return_pct,
            "monthly_return_pct": result.monthly_return_pct,
            "sharpe": result.sharpe_ratio,
            "sortino": result.sortino_ratio,
            "max_dd_pct": result.max_drawdown_pct,
            "win_rate_pct": result.win_rate_pct,
            "dir_acc_pct": result.directional_accuracy_pct,
            "total_trades": result.total_trades,
        })

    ts_str = time.strftime("%Y%m%d_%H%M%S")
    summary_path = args.output_dir / f"hourly_summary_{ts_str}.json"
    write_json_atomic(summary_path, {"results": results_summary})
    print(f"\n  Summary -> {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
