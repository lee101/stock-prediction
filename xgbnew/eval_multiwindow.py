#!/usr/bin/env python3
"""Multi-window regime robustness evaluation for XGBoost daily strategy.

Trains once on 2021-01-01 → 2023-12-31, then tests on rolling 50-day windows
from 2024-01-02 onwards with 21-day stride.  This covers:
  - 2024: post-rate-hike recovery, rotation, election rally
  - 2025: mixed/choppy, DeepSeek correction, some bull
  - 2026: AI/semiconductor bull run (where original backtest ran)

Aggregates median, p10, neg/N, sortino across all windows — same yardstick
as the screened32 RL ensemble (though via xgbnew's own backtest, not marketsim).

Usage:
    python -m xgbnew.eval_multiwindow \\
        --symbols-file symbol_lists/stocks_wide_1000_v1.txt \\
        --data-root trainingdata \\
        --window-days 50 --stride-days 21 \\
        --top-n 2 --output-dir analysis/xgbnew_multiwindow
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from xgbnew.dataset import build_daily_dataset
from xgbnew.features import DAILY_FEATURE_COLS
from xgbnew.model import XGBStockModel
from xgbnew.backtest import BacktestConfig, simulate

logger = logging.getLogger(__name__)

TRADING_DAYS_PER_MONTH = 21.0


def _monthly_return(total_ret: float, n_days: int) -> float:
    try:
        return (1.0 + total_ret / 100.0) ** (TRADING_DAYS_PER_MONTH / n_days) - 1.0
    except Exception:
        return 0.0


def _trading_days_between(df_all: pd.DataFrame, start: date, end: date) -> list[date]:
    """Return sorted trading days in [start, end] present in df_all."""
    mask = (df_all["date"] >= start) & (df_all["date"] <= end)
    return sorted(df_all.loc[mask, "date"].unique())


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--symbols-file", type=Path,
                   default=REPO / "symbol_lists/stocks_wide_1000_v1.txt")
    p.add_argument("--data-root", type=Path, default=REPO / "trainingdata")
    p.add_argument("--train-start", default="2021-01-01")
    p.add_argument("--train-end",   default="2023-12-31")
    p.add_argument("--oos-start",   default="2024-01-02",
                   help="Start of out-of-sample period (first test window start)")
    p.add_argument("--oos-end",     default="",
                   help="End of OOS period (default: latest available data)")
    p.add_argument("--window-days", type=int, default=50,
                   help="Test window length in trading days (default 50)")
    p.add_argument("--stride-days", type=int, default=21,
                   help="Window stride in trading days (default 21)")
    p.add_argument("--top-n",       type=int, default=2)
    p.add_argument("--commission-bps", type=float, default=10.0)
    p.add_argument("--min-dollar-vol", type=float, default=5e6)
    p.add_argument("--n-estimators", type=int, default=300)
    p.add_argument("--max-depth",    type=int, default=4)
    p.add_argument("--output-dir", type=Path,
                   default=REPO / "analysis/xgbnew_multiwindow")
    p.add_argument("--model-save-path", type=Path, default=None,
                   help="Save trained model here for live_trader.py to load")
    p.add_argument("--verbose", "-v", action="store_true")
    return p.parse_args(argv)


def _load_symbols(path: Path) -> list[str]:
    syms = []
    for line in path.read_text(encoding="utf-8").splitlines():
        s = line.strip().split("#", 1)[0].strip().upper()
        if s:
            syms.append(s)
    return syms


def main(argv=None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)s %(message)s",
    )

    symbols = _load_symbols(args.symbols_file)
    print(f"[xgb-eval] {len(symbols)} symbols | train {args.train_start}–{args.train_end}", flush=True)

    oos_start = date.fromisoformat(args.oos_start)

    # ── Build full dataset: train + all OOS data in one pass ─────────────────
    oos_end_str = args.oos_end or date.today().isoformat()
    oos_end = date.fromisoformat(oos_end_str)

    print(f"[xgb-eval] Building features ({args.train_start} → {oos_end_str})...", flush=True)
    t0 = time.perf_counter()
    train_df, _, oos_df = build_daily_dataset(
        data_root=args.data_root,
        symbols=symbols,
        train_start=date.fromisoformat(args.train_start),
        train_end=date.fromisoformat(args.train_end),
        val_start=oos_start,
        val_end=oos_end,
        test_start=oos_start,
        test_end=oos_end,
        chronos_cache=None,
        min_dollar_vol=args.min_dollar_vol,
    )
    print(f"  done in {time.perf_counter()-t0:.1f}s | train={len(train_df):,} oos={len(oos_df):,}", flush=True)

    if len(train_df) < 1000:
        print("ERROR: Too few training rows.", file=sys.stderr)
        return 1
    if len(oos_df) < 100:
        print("ERROR: No OOS data found.", file=sys.stderr)
        return 1

    # ── Train once on 2021–2023 ───────────────────────────────────────────────
    print("[xgb-eval] Training XGBStockModel on 2021–2023...", flush=True)
    t1 = time.perf_counter()
    model = XGBStockModel(n_estimators=args.n_estimators, max_depth=args.max_depth)
    model.fit(train_df, DAILY_FEATURE_COLS, verbose=args.verbose)
    print(f"  trained in {time.perf_counter()-t1:.1f}s", flush=True)

    if args.model_save_path:
        args.model_save_path.parent.mkdir(parents=True, exist_ok=True)
        model.save(args.model_save_path)
        print(f"  model saved → {args.model_save_path}", flush=True)

    # ── Build rolling windows ─────────────────────────────────────────────────
    all_trading_days = sorted(oos_df["date"].unique())
    if not all_trading_days:
        print("ERROR: No trading days in OOS data.", file=sys.stderr)
        return 1

    windows = []
    idx = 0
    while idx + args.window_days <= len(all_trading_days):
        window_days_list = all_trading_days[idx: idx + args.window_days]
        windows.append((window_days_list[0], window_days_list[-1]))
        idx += args.stride_days

    print(f"[xgb-eval] {len(windows)} windows of {args.window_days}d, stride {args.stride_days}d", flush=True)
    print(f"  OOS: {all_trading_days[0]} → {all_trading_days[-1]}", flush=True)

    # ── Run each window ────────────────────────────────────────────────────────
    cfg = BacktestConfig(
        top_n=args.top_n, leverage=1.0, xgb_weight=1.0,
        commission_bps=args.commission_bps, min_dollar_vol=args.min_dollar_vol,
    )

    window_results = []
    for w_start, w_end in windows:
        w_df = oos_df[(oos_df["date"] >= w_start) & (oos_df["date"] <= w_end)]
        if len(w_df) < 5:
            continue
        result = simulate(w_df, model, cfg)
        n_days = len(result.day_results)
        monthly = _monthly_return(result.total_return_pct, max(n_days, 1)) * 100.0
        window_results.append({
            "w_start": str(w_start),
            "w_end": str(w_end),
            "n_trading_days": n_days,
            "total_return_pct": result.total_return_pct,
            "monthly_return_pct": monthly,
            "sharpe": result.sharpe_ratio,
            "sortino": result.sortino_ratio,
            "max_dd_pct": result.max_drawdown_pct,
            "win_rate_pct": result.win_rate_pct,
            "dir_acc_pct": result.directional_accuracy_pct,
            "total_trades": result.total_trades,
        })

    if not window_results:
        print("ERROR: No window results.", file=sys.stderr)
        return 1

    # ── Aggregate ─────────────────────────────────────────────────────────────
    monthly_rets = np.array([r["monthly_return_pct"] for r in window_results])
    total_rets   = np.array([r["total_return_pct"]   for r in window_results])
    sortinos     = np.array([r["sortino"]             for r in window_results])
    n_neg = int(np.sum(monthly_rets < 0))
    n_win = len(window_results)

    print(f"\n{'='*72}")
    print(f"  XGBoost Multi-Window Eval  (top-{args.top_n}, 1x lev, {args.window_days}d windows)")
    print(f"  Train: {args.train_start} → {args.train_end}   OOS: {args.oos_start} → {oos_end_str}")
    print(f"{'='*72}")
    print(f"  Windows            : {n_win}")
    print(f"  Neg windows (mon%) : {n_neg}/{n_win}  ({100*n_neg/n_win:.0f}%)")
    print(f"  Median monthly%    : {np.median(monthly_rets):+.2f}%")
    print(f"  P10 monthly%       : {np.percentile(monthly_rets, 10):+.2f}%")
    print(f"  P90 monthly%       : {np.percentile(monthly_rets, 90):+.2f}%")
    print(f"  Mean monthly%      : {np.mean(monthly_rets):+.2f}%")
    print(f"  Median sortino     : {np.median(sortinos):.2f}")
    print(f"  Median total ret%  : {np.median(total_rets):+.2f}%")
    print(f"{'='*72}")

    # Per-window table
    print(f"\n  {'Window':>23}  {'Total%':>8}  {'Monthly%':>9}  {'Sharpe':>7}  {'DirAcc%':>8}  {'DD%':>6}")
    print("  " + "-"*74)
    for r in window_results:
        flag = " ← NEG" if r["monthly_return_pct"] < 0 else ""
        print(f"  {r['w_start']} → {r['w_end']}  "
              f"{r['total_return_pct']:>+8.1f}  "
              f"{r['monthly_return_pct']:>+9.1f}  "
              f"{r['sharpe']:>7.2f}  "
              f"{r['dir_acc_pct']:>8.1f}  "
              f"{r['max_dd_pct']:>6.1f}{flag}")

    # ── Save ──────────────────────────────────────────────────────────────────
    args.output_dir.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    out = {
        "train_start": args.train_start,
        "train_end": args.train_end,
        "oos_start": args.oos_start,
        "oos_end": oos_end_str,
        "window_days": args.window_days,
        "stride_days": args.stride_days,
        "top_n": args.top_n,
        "n_windows": n_win,
        "n_neg_monthly": n_neg,
        "median_monthly_pct": float(np.median(monthly_rets)),
        "p10_monthly_pct": float(np.percentile(monthly_rets, 10)),
        "p90_monthly_pct": float(np.percentile(monthly_rets, 90)),
        "median_sortino": float(np.median(sortinos)),
        "windows": window_results,
    }
    out_path = args.output_dir / f"multiwindow_{ts}.json"
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\n  Results → {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
