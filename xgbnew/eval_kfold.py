#!/usr/bin/env python3
"""XGB daily champion k-fold CV (time-series).

Splits the OOS period into K contiguous folds. For each fold, trains on
*everything outside that fold* (both pre-fold and post-fold data from the
combined train+OOS span) and evaluates only on the held-out fold. Reports
per-fold monthly%, sortino, neg-window count, and aggregate stats across
all folds.

Note on time-series leakage: features use past-lagged info only (verified
in xgbnew/features.py); including post-fold data in training is a
contamination concern for the strict "no look-ahead" principle, so by
default we use **expanding-window time-series CV** (anchored back at
train_start, train up to fold_start, test fold, move forward). Flip
``--cv-mode k_contiguous`` to use the leave-one-fold-out variant.

Usage::

    python -m xgbnew.eval_kfold \\
        --symbols-file symbol_lists/stocks_wide_1000_v1.txt \\
        --data-root trainingdata \\
        --train-start 2020-01-01 \\
        --cv-start 2024-01-01 --cv-end 2026-04-10 \\
        --n-folds 5 --window-days 30 --stride-days 14 \\
        --top-n 1 --leverage 1.0 \\
        --n-estimators 400 --max-depth 5 --learning-rate 0.03 \\
        --device cpu
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

from xgbnew.backtest import BacktestConfig, simulate
from xgbnew.dataset import build_daily_dataset, load_chronos_cache
from xgbnew.features import DAILY_FEATURE_COLS
from xgbnew.model import XGBStockModel

logger = logging.getLogger(__name__)


def _build_windows(days: list, window_days: int, stride_days: int) -> list[tuple]:
    if len(days) < window_days:
        return []
    out = []
    i = 0
    while i + window_days <= len(days):
        span = days[i : i + window_days]
        out.append((span[0], span[-1]))
        i += stride_days
    return out


def _monthly_return(total_pct: float, n_days: int) -> float:
    if n_days <= 0:
        return 0.0
    return (1.0 + total_pct / 100.0) ** (21.0 / n_days) - 1.0


def _load_symbols(path: Path) -> list[str]:
    syms: list[str] = []
    for line in path.read_text().splitlines():
        s = line.strip().upper()
        if s and not s.startswith("#"):
            syms.append(s)
    seen = set()
    return [s for s in syms if not (s in seen or seen.add(s))]


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--symbols-file", type=Path, required=True)
    p.add_argument("--data-root", type=Path, default=REPO / "trainingdata")
    p.add_argument("--chronos-cache", type=Path,
                   default=REPO / "analysis/top2_backtest/forecast_cache")
    p.add_argument("--train-start", default="2020-01-01")
    p.add_argument("--cv-start", default="2024-01-01",
                   help="Start of the CV block (first fold begins here).")
    p.add_argument("--cv-end", default="2026-04-10",
                   help="End of the CV block.")
    p.add_argument("--n-folds", type=int, default=5)
    p.add_argument(
        "--cv-mode", default="expanding",
        choices=["expanding", "k_contiguous"],
        help="expanding = anchor back at --train-start, grow train set fold by fold. "
             "k_contiguous = hold out each fold, train on everything else (may leak).",
    )

    p.add_argument("--window-days", type=int, default=30)
    p.add_argument("--stride-days", type=int, default=14)

    p.add_argument("--top-n", type=int, default=1)
    p.add_argument("--leverage", type=float, default=1.0)
    p.add_argument("--xgb-weight", type=float, default=1.0)
    p.add_argument("--commission-bps", type=float, default=0.0)
    p.add_argument("--fill-buffer-bps", type=float, default=5.0)
    p.add_argument("--fee-rate", type=float, default=0.0000278)
    p.add_argument("--min-dollar-vol", type=float, default=5_000_000.0)

    p.add_argument("--n-estimators", type=int, default=400)
    p.add_argument("--max-depth", type=int, default=5)
    p.add_argument("--learning-rate", type=float, default=0.03)
    p.add_argument("--random-state", type=int, default=42)

    p.add_argument("--allocation-mode", default="equal",
                   choices=["equal", "softmax", "score_norm"])
    p.add_argument("--allocation-temp", type=float, default=1.0)

    p.add_argument("--device", default="cpu",
                   help="CPU is the deployable path; CUDA for quick directional check.")
    p.add_argument("--output-dir", type=Path,
                   default=REPO / "analysis/xgbnew_kfold")
    p.add_argument("--verbose", "-v", action="store_true")
    return p.parse_args(argv)


def _split_folds(cv_start: date, cv_end: date, n_folds: int) -> list[tuple[date, date]]:
    span_days = (cv_end - cv_start).days
    if span_days < n_folds * 14:
        raise ValueError(f"CV span ({span_days}d) too short for {n_folds} folds")
    fold_len = span_days // n_folds
    folds = []
    for i in range(n_folds):
        fs = cv_start + timedelta(days=i * fold_len)
        fe = cv_end if i == n_folds - 1 else cv_start + timedelta(days=(i + 1) * fold_len - 1)
        folds.append((fs, fe))
    return folds


def main(argv=None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING,
                        format="%(levelname)s %(message)s")

    symbols = _load_symbols(args.symbols_file)
    train_start = date.fromisoformat(args.train_start)
    cv_start = date.fromisoformat(args.cv_start)
    cv_end = date.fromisoformat(args.cv_end)

    chronos_cache = {}
    if args.chronos_cache.exists():
        chronos_cache = load_chronos_cache(args.chronos_cache)

    folds = _split_folds(cv_start, cv_end, int(args.n_folds))
    print(f"[xgb-kfold] {len(symbols)} symbols | train_start={train_start} cv_start={cv_start} cv_end={cv_end} "
          f"| n_folds={len(folds)} mode={args.cv_mode} device={args.device}", flush=True)
    for i, (fs, fe) in enumerate(folds):
        print(f"  fold {i+1}: {fs} → {fe}", flush=True)

    # Load full span: train_start → cv_end (or the max of that and cv_end)
    full_start = train_start
    full_end = cv_end
    t0 = time.perf_counter()
    train_df_full, _, _ = build_daily_dataset(
        data_root=args.data_root,
        symbols=symbols,
        train_start=full_start, train_end=full_end,
        val_start=full_end, val_end=full_end,
        test_start=full_end, test_end=full_end,
        chronos_cache=chronos_cache if chronos_cache else None,
        min_dollar_vol=args.min_dollar_vol,
        fast_features=False,
    )
    print(f"[xgb-kfold] master dataset in {time.perf_counter()-t0:.1f}s | rows={len(train_df_full):,}",
          flush=True)

    # The master df has 'date' column. We'll slice it for train/fold.
    master = train_df_full.copy()
    if "date" not in master.columns:
        print("ERROR: dataset missing 'date' column", file=sys.stderr)
        return 1

    fold_results = []
    for f_idx, (fs, fe) in enumerate(folds, start=1):
        print(f"\n[xgb-kfold] === Fold {f_idx}/{len(folds)}  hold-out {fs} → {fe} ===", flush=True)
        fold_mask = (master["date"] >= fs) & (master["date"] <= fe)
        fold_df = master[fold_mask]
        if args.cv_mode == "expanding":
            train_mask = master["date"] < fs
        else:
            train_mask = ~fold_mask
        train_df = master[train_mask]

        if len(train_df) < 1000:
            print(f"  WARN: fold {f_idx} has only {len(train_df)} train rows; skipping")
            continue
        if len(fold_df) < 50:
            print(f"  WARN: fold {f_idx} has only {len(fold_df)} eval rows; skipping")
            continue

        print(f"  train_rows={len(train_df):,}  fold_rows={len(fold_df):,}", flush=True)

        t_fit = time.perf_counter()
        model = XGBStockModel(
            device=args.device,
            n_estimators=int(args.n_estimators),
            max_depth=int(args.max_depth),
            learning_rate=float(args.learning_rate),
            random_state=int(args.random_state),
        )
        model.fit(train_df, DAILY_FEATURE_COLS, verbose=args.verbose)
        print(f"  fit in {time.perf_counter()-t_fit:.1f}s", flush=True)

        oos_prob = model.predict_scores(fold_df)

        cfg = BacktestConfig(
            top_n=int(args.top_n),
            leverage=float(args.leverage),
            xgb_weight=float(args.xgb_weight),
            commission_bps=float(args.commission_bps),
            fill_buffer_bps=float(args.fill_buffer_bps),
            fee_rate=float(args.fee_rate),
            min_dollar_vol=float(args.min_dollar_vol),
            allocation_mode=str(args.allocation_mode),
            allocation_temp=float(args.allocation_temp),
        )

        all_days = sorted(fold_df["date"].unique())
        windows = _build_windows(all_days, int(args.window_days), int(args.stride_days))
        if not windows:
            print(f"  no windows; skipping fold {f_idx}")
            continue

        w_rows = []
        for w_start, w_end in windows:
            w_df = fold_df[(fold_df["date"] >= w_start) & (fold_df["date"] <= w_end)]
            if len(w_df) < 5:
                continue
            w_scores = oos_prob.loc[w_df.index]
            res = simulate(w_df, model, cfg, precomputed_scores=w_scores)
            n_days = len(res.day_results)
            monthly = _monthly_return(res.total_return_pct, max(n_days, 1)) * 100.0
            w_rows.append({
                "w_start": str(w_start), "w_end": str(w_end),
                "n_trading_days": n_days,
                "monthly_return_pct": monthly,
                "sortino": res.sortino_ratio,
                "max_dd_pct": res.max_drawdown_pct,
            })

        if not w_rows:
            continue
        m = np.array([r["monthly_return_pct"] for r in w_rows])
        s = np.array([r["sortino"] for r in w_rows])
        dd = np.array([r["max_dd_pct"] for r in w_rows])
        nn = int(np.sum(m < 0))
        fold_rec = {
            "fold_idx": f_idx,
            "fold_start": str(fs), "fold_end": str(fe),
            "n_windows": len(w_rows),
            "median_monthly_pct": float(np.median(m)),
            "p10_monthly_pct": float(np.percentile(m, 10)),
            "mean_monthly_pct": float(np.mean(m)),
            "median_sortino": float(np.median(s)),
            "median_dd_pct": float(np.median(dd)),
            "worst_dd_pct": float(np.max(dd)),
            "n_neg_monthly": nn,
            "windows": w_rows,
        }
        fold_results.append(fold_rec)
        print(f"  fold {f_idx}: med={fold_rec['median_monthly_pct']:+.2f}%  "
              f"p10={fold_rec['p10_monthly_pct']:+.2f}%  "
              f"sortino={fold_rec['median_sortino']:.2f}  "
              f"worst_dd={fold_rec['worst_dd_pct']:.2f}%  neg={nn}/{len(w_rows)}",
              flush=True)

    if not fold_results:
        print("ERROR: no folds produced results", file=sys.stderr)
        return 1

    med = np.array([r["median_monthly_pct"] for r in fold_results])
    p10 = np.array([r["p10_monthly_pct"] for r in fold_results])
    sort = np.array([r["median_sortino"] for r in fold_results])
    neg = np.array([r["n_neg_monthly"] / r["n_windows"] for r in fold_results])

    print(f"\n{'='*78}")
    print(f"  XGB K-FOLD CV  (mode={args.cv_mode})")
    print(f"{'='*78}")
    print(f"  Folds:                       {len(fold_results)}")
    print(f"  Per-fold median monthly%:    min {med.min():+.2f}  mean {med.mean():+.2f}  max {med.max():+.2f}")
    print(f"  Per-fold p10 monthly%:       min {p10.min():+.2f}  mean {p10.mean():+.2f}  max {p10.max():+.2f}")
    print(f"  Per-fold median sortino:     min {sort.min():.2f}  mean {sort.mean():.2f}  max {sort.max():.2f}")
    print(f"  Per-fold neg-window frac:    min {neg.min():.3f}  mean {neg.mean():.3f}  max {neg.max():.3f}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    out = {
        "cv_mode": args.cv_mode,
        "n_folds": len(fold_results),
        "summary": {
            "median_monthly_pct_min": float(med.min()),
            "median_monthly_pct_mean": float(med.mean()),
            "median_monthly_pct_max": float(med.max()),
            "p10_monthly_pct_min": float(p10.min()),
            "p10_monthly_pct_mean": float(p10.mean()),
            "neg_window_frac_max": float(neg.max()),
        },
        "config": {
            "n_estimators": int(args.n_estimators),
            "max_depth": int(args.max_depth),
            "learning_rate": float(args.learning_rate),
            "top_n": int(args.top_n),
            "leverage": float(args.leverage),
            "fee_rate": float(args.fee_rate),
            "fill_buffer_bps": float(args.fill_buffer_bps),
            "min_dollar_vol": float(args.min_dollar_vol),
            "train_start": str(train_start),
            "cv_start": str(cv_start),
            "cv_end": str(cv_end),
            "device": args.device,
            "random_state": int(args.random_state),
        },
        "folds": fold_results,
    }
    out_path = args.output_dir / f"kfold_{ts}.json"
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\n  Results → {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
