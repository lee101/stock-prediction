#!/usr/bin/env python3
"""Hourly analog of ``xgbnew.eval_multiwindow``.

Reads hourly OHLCV CSVs from ``trainingdatahourly/{stocks,crypto}/``, trains
XGBoost on bars before ``--train-end``, and walks rolling OOS windows with
realistic per-bar execution costs. Reports monthly/annualised PnL,
sortino, max-DD and negative-window count — same metrics as the daily
driver so results are apples-to-apples (modulo the time horizon).

Annualisation is universe-aware:
    stocks-only:  252 × 6.5  = 1,638 bars/yr   (monthly ≈ 136.5 bars)
    crypto-only:  365 × 24   = 8,760 bars/yr   (monthly ≈ 730 bars)

``--universe both`` builds one combined dataset but still reports
universe-segmented metrics so you can see which side is paying PnL.

Usage::

    python -m xgbnew.eval_hourly_multiwindow \\
        --data-root trainingdatahourly \\
        --universe stocks \\
        --train-end 2025-09-30 \\
        --val-end 2025-12-31 \\
        --test-end 2026-04-10 \\
        --window-bars 137 --stride-bars 30 --top-n 1 \\
        --n-estimators 400 --max-depth 5 --learning-rate 0.03
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from xgbnew.backtest import (
    BacktestConfig,
    CRYPTO_HOURS_PER_MONTH,
    CRYPTO_HOURS_PER_YEAR,
    STOCK_HOURS_PER_MONTH,
    STOCK_HOURS_PER_YEAR,
    simulate_hourly,
)
from xgbnew.dataset import build_hourly_dataset
from xgbnew.features import HOURLY_FEATURE_COLS
from xgbnew.model import XGBStockModel


logger = logging.getLogger(__name__)


def _bars_per_year(universe: str) -> float:
    u = universe.lower()
    if u == "crypto":
        return CRYPTO_HOURS_PER_YEAR
    if u == "both":
        # Use stock hours as the canonical annualisation; caller segments.
        return STOCK_HOURS_PER_YEAR
    return STOCK_HOURS_PER_YEAR


def _bars_per_month(universe: str) -> float:
    u = universe.lower()
    if u == "crypto":
        return CRYPTO_HOURS_PER_MONTH
    return STOCK_HOURS_PER_MONTH


def _monthly_return_from_total(total_ret_pct: float, n_bars: int, bars_per_month: float) -> float:
    if n_bars <= 0:
        return 0.0
    try:
        return ((1.0 + total_ret_pct / 100.0) ** (bars_per_month / n_bars) - 1.0) * 100.0
    except Exception:
        return 0.0


def _build_hourly_windows(
    all_ts: list[pd.Timestamp],
    *,
    window_bars: int,
    stride_bars: int,
) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    """Return a list of (start_ts, end_ts) pairs over a sorted list of bars."""
    if window_bars <= 0 or stride_bars <= 0 or len(all_ts) < window_bars:
        return []
    out: list[tuple[pd.Timestamp, pd.Timestamp]] = []
    idx = 0
    while idx + window_bars <= len(all_ts):
        span = all_ts[idx : idx + window_bars]
        out.append((span[0], span[-1]))
        idx += stride_bars
    return out


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--data-root", type=Path, default=REPO / "trainingdatahourly")
    p.add_argument(
        "--universe",
        choices=["stocks", "crypto", "both"],
        default="stocks",
        help="Which CSV subtree to load.",
    )
    p.add_argument(
        "--symbols",
        default=None,
        help="Comma-separated symbol override (otherwise load whole universe).",
    )
    p.add_argument("--train-start", default="", help="Optional ISO timestamp floor for train.")
    p.add_argument("--train-end", default="2025-09-30")
    p.add_argument("--val-end", default="2025-12-31")
    p.add_argument("--test-end", default="2026-04-10")

    p.add_argument("--top-n", type=int, default=1)
    p.add_argument("--leverage", type=float, default=1.0)
    p.add_argument("--xgb-weight", type=float, default=1.0)
    p.add_argument("--commission-bps", type=float, default=0.0)
    p.add_argument("--fill-buffer-bps", type=float, default=5.0)
    p.add_argument(
        "--fee-rate",
        type=float,
        default=None,
        help="Per-side fee fraction. Default: per-symbol (stocks ~5bps, crypto 8-15bps).",
    )
    p.add_argument("--min-score", type=float, default=0.0)
    p.add_argument(
        "--min-dollar-vol",
        type=float,
        default=5e5,
        help="Per-bar dollar volume floor (hourly, default 500k).",
    )
    p.add_argument(
        "--max-spread-bps",
        type=float,
        default=30.0,
        help="Skip stocks whose hourly volume-based spread exceeds this (default 30bps).",
    )

    p.add_argument("--n-estimators", type=int, default=400)
    p.add_argument("--max-depth", type=int, default=5)
    p.add_argument("--learning-rate", type=float, default=0.03)
    p.add_argument("--random-state", type=int, default=42)

    p.add_argument(
        "--window-bars",
        type=int,
        default=137,
        help="OOS window size in bars (stocks: 137≈21td; crypto: 730=1mo).",
    )
    p.add_argument("--stride-bars", type=int, default=30)
    p.add_argument("--min-bars", type=int, default=400)

    p.add_argument("--output-dir", type=Path, default=REPO / "analysis/xgbnew_hourly")
    p.add_argument("--verbose", "-v", action="store_true")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)s %(message)s",
    )

    if not args.data_root.exists():
        print(f"ERROR: data root not found: {args.data_root}", file=sys.stderr)
        return 1

    # Parse dates / symbols
    train_end_ts = pd.Timestamp(args.train_end, tz="UTC")
    val_end_ts = pd.Timestamp(args.val_end, tz="UTC")
    test_end_ts = pd.Timestamp(args.test_end, tz="UTC")
    train_start_ts = pd.Timestamp(args.train_start, tz="UTC") if args.train_start else None

    symbols = None
    if args.symbols:
        symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]

    print(
        f"[xgb-hourly-mw] data_root={args.data_root} universe={args.universe} "
        f"train_end={args.train_end} val_end={args.val_end} test_end={args.test_end}",
        flush=True,
    )
    t0 = time.perf_counter()
    train_df, val_df, test_df, kind_map = build_hourly_dataset(
        args.data_root,
        symbols,
        train_start=train_start_ts,
        train_end=train_end_ts,
        val_end=val_end_ts,
        test_end=test_end_ts,
        universe=args.universe,
        min_bars=int(args.min_bars),
        min_dollar_vol=float(args.min_dollar_vol),
    )
    print(
        f"[xgb-hourly-mw] dataset in {time.perf_counter()-t0:.1f}s | "
        f"train={len(train_df):,} val={len(val_df):,} test={len(test_df):,} | "
        f"{len(kind_map)} symbols "
        f"(stocks={sum(1 for k in kind_map.values() if k=='stocks')}, "
        f"crypto={sum(1 for k in kind_map.values() if k=='crypto')})",
        flush=True,
    )

    if len(train_df) < 2_000:
        print("ERROR: too few training rows", file=sys.stderr)
        return 1
    if len(test_df) < 200:
        print("ERROR: too few OOS rows", file=sys.stderr)
        return 1

    # Train
    print("[xgb-hourly-mw] training XGB...", flush=True)
    t0 = time.perf_counter()
    model = XGBStockModel(
        n_estimators=int(args.n_estimators),
        max_depth=int(args.max_depth),
        learning_rate=float(args.learning_rate),
        random_state=int(args.random_state),
    )
    model.fit(train_df, HOURLY_FEATURE_COLS, val_df=val_df, verbose=args.verbose)
    print(f"[xgb-hourly-mw] trained in {time.perf_counter()-t0:.1f}s", flush=True)

    # Feature importances
    imps = model.feature_importances().head(10)
    print("\n  Top-10 hourly feature importances:", flush=True)
    for feat, imp in imps.items():
        print(f"    {feat:<22} {imp:.4f}")

    # OOS backtest — walk windows across the test slice
    cfg = BacktestConfig(
        top_n=int(args.top_n),
        leverage=float(args.leverage),
        xgb_weight=float(args.xgb_weight),
        commission_bps=float(args.commission_bps),
        fill_buffer_bps=float(args.fill_buffer_bps),
        fee_rate=float(args.fee_rate) if args.fee_rate is not None else None,
        min_score=float(args.min_score),
        min_dollar_vol=float(args.min_dollar_vol),
        max_spread_bps=float(args.max_spread_bps),
    )

    bpy = _bars_per_year(args.universe)
    bpm = _bars_per_month(args.universe)

    # Precompute scores once over the full test slice
    oos_prob = model.predict_scores(test_df)

    all_ts = sorted(test_df["timestamp"].unique())
    windows = _build_hourly_windows(
        [pd.Timestamp(t) for t in all_ts],
        window_bars=int(args.window_bars),
        stride_bars=int(args.stride_bars),
    )
    if not windows:
        print("ERROR: no OOS windows could be built", file=sys.stderr)
        return 1

    print(f"[xgb-hourly-mw] running {len(windows)} window(s) "
          f"({int(args.window_bars)} bars each)...", flush=True)

    window_results: list[dict] = []
    for w_start, w_end in windows:
        mask = (test_df["timestamp"] >= w_start) & (test_df["timestamp"] <= w_end)
        w_df = test_df[mask]
        if len(w_df) < 5:
            continue
        w_scores = oos_prob.loc[w_df.index]

        res = simulate_hourly(
            w_df, model, cfg,
            bars_per_year=bpy, bars_per_month=bpm,
            kind_map=kind_map, precomputed_scores=w_scores,
        )
        n_bars = len(res.day_results)
        monthly = _monthly_return_from_total(res.total_return_pct, n_bars, bpm)
        window_results.append({
            "w_start": str(w_start),
            "w_end": str(w_end),
            "n_bars": n_bars,
            "total_return_pct": res.total_return_pct,
            "monthly_return_pct": monthly,
            "annualized_return_pct": res.annualized_return_pct,
            "sharpe": res.sharpe_ratio,
            "sortino": res.sortino_ratio,
            "max_dd_pct": res.max_drawdown_pct,
            "win_rate_pct": res.win_rate_pct,
            "dir_acc_pct": res.directional_accuracy_pct,
            "total_trades": res.total_trades,
            "avg_spread_bps": res.avg_spread_bps,
            "avg_fee_bps": res.avg_fee_bps,
        })

    if not window_results:
        print("ERROR: no window produced results", file=sys.stderr)
        return 1

    monthly = np.array([r["monthly_return_pct"] for r in window_results], dtype=np.float64)
    annual = np.array([r["annualized_return_pct"] for r in window_results], dtype=np.float64)
    sortinos = np.array([r["sortino"] for r in window_results], dtype=np.float64)
    dd = np.array([r["max_dd_pct"] for r in window_results], dtype=np.float64)
    n_neg = int(np.sum(monthly < 0.0))

    print(f"\n{'='*78}")
    print(f"  Hourly XGB multi-window OOS  (universe={args.universe})")
    print(f"  bars_per_year={bpy:.1f}  bars_per_month={bpm:.1f}")
    print(f"  top_n={cfg.top_n} lev={cfg.leverage:.2f} fee_rate={cfg.fee_rate}")
    print(f"{'='*78}")
    print(f"  Windows              : {len(window_results)} (n_bars each ~ {int(args.window_bars)})")
    print(f"  Median monthly%      : {float(np.median(monthly)):+.2f}%")
    print(f"  P10 monthly%         : {float(np.percentile(monthly, 10)):+.2f}%")
    print(f"  P90 monthly%         : {float(np.percentile(monthly, 90)):+.2f}%")
    print(f"  Median annualised%   : {float(np.median(annual)):+.2f}%")
    print(f"  Median sortino       : {float(np.median(sortinos)):.2f}")
    print(f"  Median max DD%       : {float(np.median(dd)):.2f}%")
    print(f"  Worst max DD%        : {float(np.max(dd)):.2f}%")
    print(f"  Neg windows          : {n_neg}/{len(window_results)}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    out = {
        "universe": args.universe,
        "train_end": args.train_end,
        "val_end": args.val_end,
        "test_end": args.test_end,
        "bars_per_year": bpy,
        "bars_per_month": bpm,
        "config": {
            "top_n": int(args.top_n),
            "leverage": float(args.leverage),
            "xgb_weight": float(args.xgb_weight),
            "commission_bps": float(args.commission_bps),
            "fill_buffer_bps": float(args.fill_buffer_bps),
            "fee_rate": float(args.fee_rate) if args.fee_rate is not None else None,
            "n_estimators": int(args.n_estimators),
            "max_depth": int(args.max_depth),
            "learning_rate": float(args.learning_rate),
            "random_state": int(args.random_state),
            "min_dollar_vol": float(args.min_dollar_vol),
            "max_spread_bps": float(args.max_spread_bps),
            "window_bars": int(args.window_bars),
            "stride_bars": int(args.stride_bars),
        },
        "median_monthly_pct": float(np.median(monthly)),
        "p10_monthly_pct": float(np.percentile(monthly, 10)),
        "median_sortino": float(np.median(sortinos)),
        "n_neg_monthly": n_neg,
        "n_windows": len(window_results),
        "n_symbols": len(kind_map),
        "kind_counts": {
            "stocks": sum(1 for k in kind_map.values() if k == "stocks"),
            "crypto": sum(1 for k in kind_map.values() if k == "crypto"),
        },
        "windows": window_results,
    }
    out_path = args.output_dir / f"hourly_multiwindow_{args.universe}_{ts}.json"
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\n  Results → {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
