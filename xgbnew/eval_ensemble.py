#!/usr/bin/env python3
"""XGB multi-seed ensemble daily eval.

Trains N XGBoost models with identical hyperparams but different random_state,
averages `predict_proba` across the N models to form ensemble scores, then runs
the same windowed OOS backtest as ``xgbnew.eval_multiwindow``. This tests
whether seed-ensembling smooths sortino / lifts p10 without sacrificing the
champion's median (analogous to how our RL ensemble trades member fragility
for robustness).

Usage::

    python -m xgbnew.eval_ensemble \\
        --symbols-file symbol_lists/stocks_wide_1000_v1.txt \\
        --data-root trainingdata \\
        --train-start 2020-01-01 --train-end 2024-12-31 \\
        --oos-start 2025-01-02 --oos-end 2026-04-10 \\
        --window-days 30 --stride-days 14 \\
        --top-n 1 --leverage 1.0 \\
        --n-estimators 400 --max-depth 5 --learning-rate 0.03 \\
        --seeds 0,7,42,73,197 \\
        --device cuda
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from datetime import date
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
    out: list[str] = []
    for line in path.read_text().splitlines():
        s = line.strip().upper()
        if s and not s.startswith("#"):
            out.append(s)
    # de-dup preserving order
    seen = set()
    return [s for s in out if not (s in seen or seen.add(s))]


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--symbols-file", type=Path, required=True)
    p.add_argument("--data-root", type=Path, default=REPO / "trainingdata")
    p.add_argument("--chronos-cache", type=Path,
                   default=REPO / "analysis/top2_backtest/forecast_cache")
    p.add_argument("--train-start", default="2020-01-01")
    p.add_argument("--train-end", default="2024-12-31")
    p.add_argument("--oos-start", default="2025-01-02")
    p.add_argument("--oos-end", default="")
    p.add_argument("--window-days", type=int, default=30)
    p.add_argument("--stride-days", type=int, default=14)

    p.add_argument("--top-n", type=int, default=1)
    p.add_argument("--leverage", type=float, default=1.0)
    p.add_argument("--xgb-weight", type=float, default=1.0)
    p.add_argument("--commission-bps", type=float, default=0.0)
    p.add_argument("--fill-buffer-bps", type=float, default=5.0)
    p.add_argument("--fee-rate", type=float, default=0.0000278,
                   help="Per-side fee fraction (stocks default ≈ 2.78bps).")
    p.add_argument("--min-dollar-vol", type=float, default=5_000_000.0)

    p.add_argument("--n-estimators", type=int, default=400)
    p.add_argument("--max-depth", type=int, default=5)
    p.add_argument("--learning-rate", type=float, default=0.03)
    p.add_argument("--seeds", default="0,7,42,73,197",
                   help="Comma-separated seed list — one model per seed; proba averaged.")
    p.add_argument("--blend-mode", default="mean", choices=["mean", "rank_mean", "median"],
                   help="How to combine per-seed predict_proba: mean=arithmetic mean "
                        "(probability average), rank_mean=per-day percentile-rank mean, "
                        "median=per-row median across seeds.")

    p.add_argument("--allocation-mode", default="equal",
                   choices=["equal", "softmax", "score_norm"])
    p.add_argument("--allocation-temp", type=float, default=1.0)
    p.add_argument("--hold-through", action="store_true",
                   help="Carry positions when today's pick set == yesterday's "
                        "(skip sell-close + buy-open round-trip).")
    p.add_argument("--min-score", type=float, default=0.0,
                   help="Conviction gate: drop picks with ensemble score < min_score. "
                        "NOTE: ensemble blending shrinks the score distribution, so "
                        "the usable knee differs from single-seed. Sweep per ensemble.")

    p.add_argument("--device", default="cuda",
                   help="'cuda' or 'cpu'. Bonferroni-style validations should use 'cpu' to match prod.")
    p.add_argument("--output-dir", type=Path,
                   default=REPO / "analysis/xgbnew_ensemble")
    p.add_argument("--verbose", "-v", action="store_true")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING,
                        format="%(levelname)s %(message)s")

    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    if len(seeds) < 2:
        print("ERROR: need at least 2 seeds for ensemble", file=sys.stderr)
        return 1

    symbols = _load_symbols(args.symbols_file)
    oos_start = date.fromisoformat(args.oos_start)
    oos_end = date.fromisoformat(args.oos_end or date.today().isoformat())

    chronos_cache = {}
    if args.chronos_cache.exists():
        chronos_cache = load_chronos_cache(args.chronos_cache)

    print(f"[xgb-ens] {len(symbols)} symbols | train {args.train_start}–{args.train_end} | "
          f"OOS {args.oos_start}–{oos_end} | windows={args.window_days}d stride={args.stride_days}d | "
          f"seeds={seeds} blend={args.blend_mode} device={args.device}", flush=True)

    t0 = time.perf_counter()
    train_df, _, oos_df = build_daily_dataset(
        data_root=args.data_root,
        symbols=symbols,
        train_start=date.fromisoformat(args.train_start),
        train_end=date.fromisoformat(args.train_end),
        val_start=oos_start, val_end=oos_end,
        test_start=oos_start, test_end=oos_end,
        chronos_cache=chronos_cache if chronos_cache else None,
        min_dollar_vol=args.min_dollar_vol,
        fast_features=False,
    )
    print(f"[xgb-ens] dataset built in {time.perf_counter()-t0:.1f}s | "
          f"train={len(train_df):,} oos={len(oos_df):,}", flush=True)

    if len(train_df) < 1000 or len(oos_df) < 100:
        print("ERROR: not enough data", file=sys.stderr)
        return 1

    # Train each seed, collect OOS probas
    oos_probs: list[pd.Series] = []
    for i, seed in enumerate(seeds, start=1):
        print(f"[xgb-ens] train model {i}/{len(seeds)} (seed={seed})", flush=True)
        t_fit = time.perf_counter()
        m = XGBStockModel(
            device=args.device,
            n_estimators=int(args.n_estimators),
            max_depth=int(args.max_depth),
            learning_rate=float(args.learning_rate),
            random_state=int(seed),
        )
        m.fit(train_df, DAILY_FEATURE_COLS, verbose=args.verbose)
        print(f"[xgb-ens]   fit in {time.perf_counter()-t_fit:.1f}s", flush=True)
        oos_probs.append(m.predict_scores(oos_df))
        del m

    # Blend across seeds
    mat = np.stack([p.values for p in oos_probs], axis=0)  # [n_seeds, n_rows]
    if args.blend_mode == "mean":
        blended = mat.mean(axis=0)
    elif args.blend_mode == "median":
        blended = np.median(mat, axis=0)
    elif args.blend_mode == "rank_mean":
        # Per-day per-seed pct-rank then mean.
        ranks = np.empty_like(mat)
        dates = oos_df["date"].values
        # Cache date → row indices
        unique_dates, inv = np.unique(dates, return_inverse=True)
        for d_i in range(len(unique_dates)):
            rows = np.where(inv == d_i)[0]
            for s_i in range(mat.shape[0]):
                x = mat[s_i, rows]
                order = np.argsort(np.argsort(x))
                ranks[s_i, rows] = order / max(len(rows) - 1, 1)
        blended = ranks.mean(axis=0)
    else:
        blended = mat.mean(axis=0)

    ensemble_scores = pd.Series(blended, index=oos_df.index, name="ensemble_score")

    # Backtest
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
        hold_through=bool(args.hold_through),
        min_score=float(args.min_score),
    )

    # Dummy model just to satisfy simulate() signature — backtest uses precomputed_scores
    dummy = XGBStockModel(device="cpu", n_estimators=1, max_depth=1, learning_rate=0.1)
    dummy.feature_cols = DAILY_FEATURE_COLS
    dummy._col_medians = np.zeros(len(DAILY_FEATURE_COLS), dtype=np.float32)
    dummy._fitted = True

    all_days = sorted(oos_df["date"].unique())
    windows = _build_windows(all_days, int(args.window_days), int(args.stride_days))
    if not windows:
        print("ERROR: no windows", file=sys.stderr)
        return 1

    window_rows: list[dict] = []
    for w_start, w_end in windows:
        w_df = oos_df[(oos_df["date"] >= w_start) & (oos_df["date"] <= w_end)]
        if len(w_df) < 5:
            continue
        w_scores = ensemble_scores.loc[w_df.index]
        res = simulate(w_df, dummy, cfg, precomputed_scores=w_scores)
        n_days = len(res.day_results)
        monthly = _monthly_return(res.total_return_pct, max(n_days, 1)) * 100.0
        window_rows.append({
            "w_start": str(w_start), "w_end": str(w_end),
            "n_trading_days": n_days,
            "total_return_pct": res.total_return_pct,
            "monthly_return_pct": monthly,
            "sortino": res.sortino_ratio,
            "sharpe": res.sharpe_ratio,
            "max_dd_pct": res.max_drawdown_pct,
            "win_rate_pct": res.win_rate_pct,
            "dir_acc_pct": res.directional_accuracy_pct,
            "total_trades": res.total_trades,
        })

    if not window_rows:
        print("ERROR: no window results", file=sys.stderr)
        return 1

    monthly = np.array([r["monthly_return_pct"] for r in window_rows])
    sortinos = np.array([r["sortino"] for r in window_rows])
    dd = np.array([r["max_dd_pct"] for r in window_rows])
    n_neg = int(np.sum(monthly < 0))

    print(f"\n{'='*78}")
    print(f"  XGB ENSEMBLE  seeds={seeds}  blend={args.blend_mode}  top_n={cfg.top_n}  lev={cfg.leverage}")
    print(f"{'='*78}")
    print(f"  Windows              : {len(window_rows)}")
    print(f"  Median monthly%      : {float(np.median(monthly)):+.2f}%")
    print(f"  P10 monthly%         : {float(np.percentile(monthly, 10)):+.2f}%")
    print(f"  P90 monthly%         : {float(np.percentile(monthly, 90)):+.2f}%")
    print(f"  Mean monthly%        : {float(np.mean(monthly)):+.2f}%")
    print(f"  Median sortino       : {float(np.median(sortinos)):.2f}")
    print(f"  Median max DD%       : {float(np.median(dd)):.2f}%")
    print(f"  Worst max DD%        : {float(np.max(dd)):.2f}%")
    print(f"  Neg windows          : {n_neg}/{len(window_rows)}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    out = {
        "seeds": seeds,
        "blend_mode": args.blend_mode,
        "config": {
            "n_estimators": int(args.n_estimators),
            "max_depth": int(args.max_depth),
            "learning_rate": float(args.learning_rate),
            "top_n": int(args.top_n),
            "leverage": float(args.leverage),
            "fee_rate": float(args.fee_rate),
            "fill_buffer_bps": float(args.fill_buffer_bps),
            "commission_bps": float(args.commission_bps),
            "min_dollar_vol": float(args.min_dollar_vol),
            "allocation_mode": str(args.allocation_mode),
            "allocation_temp": float(args.allocation_temp),
            "train_start": args.train_start,
            "train_end": args.train_end,
            "oos_start": args.oos_start,
            "oos_end": str(oos_end),
            "window_days": int(args.window_days),
            "stride_days": int(args.stride_days),
            "device": args.device,
        },
        "median_monthly_pct": float(np.median(monthly)),
        "p10_monthly_pct": float(np.percentile(monthly, 10)),
        "median_sortino": float(np.median(sortinos)),
        "n_neg_monthly": n_neg,
        "n_windows": len(window_rows),
        "windows": window_rows,
    }
    out_path = args.output_dir / f"ensemble_{ts}.json"
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"  Results → {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
