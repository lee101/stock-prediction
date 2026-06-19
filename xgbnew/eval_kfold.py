#!/usr/bin/env python3
"""XGB daily champion k-fold CV.

Splits the OOS period into K contiguous folds. For each fold, trains on
*everything outside that fold* (both pre-fold and post-fold data from the
combined train+OOS span) and evaluates only on the held-out fold. Reports
per-fold monthly%, sortino, neg-window count, and aggregate stats across
all folds.

Note on time-series leakage: features use past-lagged info only (verified
in xgbnew/features.py); including post-fold data in training is a
contamination concern for the strict "no look-ahead" principle. The default
``k_contiguous`` mode is actual leave-one-contiguous-fold-out validation.
Flip ``--cv-mode expanding`` to use anchored time-series validation.

Usage::

    python -m xgbnew.eval_kfold \\
        --symbols-file symbol_lists/stocks_wide_1000_v1.txt \\
        --data-root trainingdata \\
        --train-start 2020-01-01 \\
        --cv-start 2024-01-01 --cv-end 2026-04-10 \\
        --n-folds 5 --cv-mode k_contiguous --window-days 30 --stride-days 14 \\
        --top-n 1 --leverage 1.0 \\
        --n-estimators 400 --max-depth 5 --learning-rate 0.03 \\
        --device cpu
"""

from __future__ import annotations

import argparse
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

from xgbnew.artifacts import write_json_atomic  # noqa: E402
from xgbnew.backtest import PRODUCTION_STOCK_FEE_RATE, BacktestConfig, simulate  # noqa: E402
from xgbnew.cli_realism import validate_nonnegative_realism_args  # noqa: E402
from xgbnew.dataset import build_daily_dataset, load_chronos_cache  # noqa: E402
from xgbnew.features import (  # noqa: E402
    CHRONOS_FEATURE_COLS,
    DAILY_DISPERSION_FEATURE_COLS,
    DAILY_FEATURE_COLS,
    DAILY_RANK_FEATURE_COLS,
)
from xgbnew.model import XGBStockModel  # noqa: E402
from xgbnew.model_xgb_ranker import XGBRankerStockModel  # noqa: E402


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
        "--cv-mode", default="k_contiguous",
        choices=["expanding", "k_contiguous"],
        help="expanding = anchor back at --train-start, grow train set fold by fold. "
             "k_contiguous = true leave-one-contiguous-fold-out CV.",
    )
    p.add_argument(
        "--embargo-days",
        type=int,
        default=0,
        help="Calendar-day gap to drop around each held-out fold before fitting.",
    )

    p.add_argument("--window-days", type=int, default=30)
    p.add_argument("--stride-days", type=int, default=14)

    p.add_argument("--top-n", type=int, default=1)
    p.add_argument("--leverage", type=float, default=1.0)
    p.add_argument("--xgb-weight", type=float, default=1.0)
    p.add_argument("--min-score", type=float, default=0.0)
    p.add_argument("--hold-through", action="store_true")
    p.add_argument("--commission-bps", type=float, default=0.0)
    p.add_argument("--fill-buffer-bps", type=float, default=5.0)
    p.add_argument(
        "--fee-rate",
        type=float,
        default=PRODUCTION_STOCK_FEE_RATE,
        help="Per-side fee fraction. Default 0.001 = 10 bps production-realism stress fee.",
    )
    p.add_argument("--min-dollar-vol", type=float, default=5_000_000.0)

    p.add_argument("--n-estimators", type=int, default=400)
    p.add_argument("--max-depth", type=int, default=5)
    p.add_argument("--learning-rate", type=float, default=0.03)
    p.add_argument("--random-state", type=int, default=42)
    p.add_argument("--model-family", choices=["xgb", "xgb_rank"], default="xgb")
    p.add_argument("--ranker-deciles", type=int, default=10)
    p.add_argument(
        "--ranker-sample-weight-mode",
        choices=["none", "abs_target", "abs_target_sqrt"],
        default="none",
    )
    p.add_argument("--ranker-sample-weight-clip", type=float, default=0.05)
    p.add_argument("--include-ranks", action="store_true")
    p.add_argument("--include-dispersion", action="store_true")
    p.add_argument(
        "--learn-chronos-features",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Fit XGBoost directly on cached Chronos2 features when the cache exists.",
    )

    p.add_argument("--allocation-mode", default="equal",
                   choices=["equal", "softmax", "score_norm"])
    p.add_argument("--allocation-temp", type=float, default=1.0)

    p.add_argument("--device", default="cpu",
                   help="CPU is the deployable path; CUDA for quick directional check.")
    p.add_argument("--output-dir", type=Path,
                   default=REPO / "analysis/xgbnew_kfold")
    p.add_argument("--verbose", "-v", action="store_true")
    return p.parse_args(argv)


def _validate_realism_args(args: argparse.Namespace) -> list[str]:
    return validate_nonnegative_realism_args(args)


def _split_folds(cv_start: date, cv_end: date, n_folds: int) -> list[tuple[date, date]]:
    if n_folds <= 0:
        raise ValueError("n_folds must be positive")
    if cv_end < cv_start:
        raise ValueError("cv_end must be >= cv_start")
    total_days = (cv_end - cv_start).days + 1
    if total_days < n_folds:
        raise ValueError(f"CV span ({total_days}d) too short for {n_folds} folds")
    base = total_days // n_folds
    remainder = total_days % n_folds
    folds: list[tuple[date, date]] = []
    cursor = cv_start
    for i in range(n_folds):
        fold_days = base + (1 if i < remainder else 0)
        fs = cursor
        fe = fs + timedelta(days=fold_days - 1)
        folds.append((fs, fe))
        cursor = fe + timedelta(days=1)
    return folds


def _train_mask_for_fold(
    dates: pd.Series,
    *,
    fold_start: date,
    fold_end: date,
    cv_mode: str,
    embargo_days: int,
) -> pd.Series:
    if embargo_days < 0:
        raise ValueError("embargo_days must be nonnegative")
    left_cut = fold_start - timedelta(days=embargo_days)
    right_cut = fold_end + timedelta(days=embargo_days)
    if cv_mode == "expanding":
        return dates < left_cut
    if cv_mode == "k_contiguous":
        return (dates < left_cut) | (dates > right_cut)
    raise ValueError(f"unknown cv_mode: {cv_mode}")


def _feature_columns(args: argparse.Namespace, chronos_cache_loaded: bool) -> list[str]:
    feature_cols = list(DAILY_FEATURE_COLS)
    if args.include_ranks:
        feature_cols += list(DAILY_RANK_FEATURE_COLS)
    if args.include_dispersion:
        feature_cols += list(DAILY_DISPERSION_FEATURE_COLS)
    if bool(args.learn_chronos_features) and chronos_cache_loaded:
        feature_cols += list(CHRONOS_FEATURE_COLS)
    return feature_cols


def _build_model(args: argparse.Namespace):
    if args.model_family == "xgb_rank":
        return XGBRankerStockModel(
            device=args.device,
            n_estimators=int(args.n_estimators),
            max_depth=int(args.max_depth),
            learning_rate=float(args.learning_rate),
            random_state=int(args.random_state),
            n_deciles=int(args.ranker_deciles),
            sample_weight_mode=str(args.ranker_sample_weight_mode),
            sample_weight_clip=float(args.ranker_sample_weight_clip),
        )
    return XGBStockModel(
        device=args.device,
        n_estimators=int(args.n_estimators),
        max_depth=int(args.max_depth),
        learning_rate=float(args.learning_rate),
        random_state=int(args.random_state),
    )


def _chronos_coverage(df: pd.DataFrame) -> dict[str, float | int]:
    if df.empty or "chronos_available" not in df.columns:
        return {"rows": int(len(df)), "available_rows": 0, "available_frac": 0.0}
    available = int((df["chronos_available"].astype(float) > 0.0).sum())
    rows = int(len(df))
    return {
        "rows": rows,
        "available_rows": available,
        "available_frac": float(available / rows) if rows else 0.0,
    }


def main(argv=None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING,
                        format="%(levelname)s %(message)s")

    validation_failures = _validate_realism_args(args)
    if validation_failures:
        for failure in validation_failures:
            print(f"ERROR: {failure}", file=sys.stderr)
        return 2
    if int(args.embargo_days) < 0:
        print("ERROR: --embargo-days must be nonnegative", file=sys.stderr)
        return 2

    symbols = _load_symbols(args.symbols_file)
    train_start = date.fromisoformat(args.train_start)
    cv_start = date.fromisoformat(args.cv_start)
    cv_end = date.fromisoformat(args.cv_end)

    chronos_cache = {}
    if args.chronos_cache.exists():
        chronos_cache = load_chronos_cache(args.chronos_cache)

    folds = _split_folds(cv_start, cv_end, int(args.n_folds))
    print(f"[xgb-kfold] {len(symbols)} symbols | train_start={train_start} cv_start={cv_start} cv_end={cv_end} "
          f"| n_folds={len(folds)} mode={args.cv_mode} embargo_days={args.embargo_days} "
          f"family={args.model_family} device={args.device}", flush=True)
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
        include_cross_sectional_ranks=bool(args.include_ranks),
        include_cross_sectional_dispersion=bool(args.include_dispersion),
        attach_chronos_to_train=bool(args.learn_chronos_features),
    )
    print(f"[xgb-kfold] master dataset in {time.perf_counter()-t0:.1f}s | rows={len(train_df_full):,}",
          flush=True)

    # The master df has 'date' column. We'll slice it for train/fold.
    master = train_df_full.copy()
    if "date" not in master.columns:
        print("ERROR: dataset missing 'date' column", file=sys.stderr)
        return 1
    feature_cols = _feature_columns(args, bool(chronos_cache))
    missing_features = sorted(set(feature_cols) - set(master.columns))
    if missing_features:
        print(f"ERROR: dataset missing feature columns: {missing_features}", file=sys.stderr)
        return 1
    coverage = _chronos_coverage(master)
    print(
        f"[xgb-kfold] feature_cols={len(feature_cols)} "
        f"chronos_learn={bool(args.learn_chronos_features and chronos_cache)} "
        f"chronos_available={coverage['available_rows']}/{coverage['rows']} "
        f"({coverage['available_frac']:.3f})",
        flush=True,
    )

    fold_results = []
    for f_idx, (fs, fe) in enumerate(folds, start=1):
        print(f"\n[xgb-kfold] === Fold {f_idx}/{len(folds)}  hold-out {fs} → {fe} ===", flush=True)
        fold_mask = (master["date"] >= fs) & (master["date"] <= fe)
        fold_df = master[fold_mask]
        train_mask = _train_mask_for_fold(
            master["date"],
            fold_start=fs,
            fold_end=fe,
            cv_mode=str(args.cv_mode),
            embargo_days=int(args.embargo_days),
        )
        train_df = master[train_mask]

        if len(train_df) < 1000:
            print(f"  WARN: fold {f_idx} has only {len(train_df)} train rows; skipping")
            continue
        if len(fold_df) < 50:
            print(f"  WARN: fold {f_idx} has only {len(fold_df)} eval rows; skipping")
            continue

        print(f"  train_rows={len(train_df):,}  fold_rows={len(fold_df):,}", flush=True)

        t_fit = time.perf_counter()
        model = _build_model(args)
        model.fit(train_df, feature_cols, verbose=args.verbose)
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
            min_score=float(args.min_score),
            hold_through=bool(args.hold_through),
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
            n_days = len(pd.unique(w_df["date"]))
            n_active_days = len(res.day_results)
            monthly = _monthly_return(res.total_return_pct, max(n_days, 1)) * 100.0
            w_rows.append({
                "w_start": str(w_start), "w_end": str(w_end),
                "n_trading_days": n_days,
                "n_active_days": n_active_days,
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
            "train_date_min": str(train_df["date"].min()),
            "train_date_max": str(train_df["date"].max()),
            "train_rows": int(len(train_df)),
            "fold_rows": int(len(fold_df)),
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
            "model_family": str(args.model_family),
            "n_estimators": int(args.n_estimators),
            "max_depth": int(args.max_depth),
            "learning_rate": float(args.learning_rate),
            "top_n": int(args.top_n),
            "leverage": float(args.leverage),
            "min_score": float(args.min_score),
            "hold_through": bool(args.hold_through),
            "fee_rate": float(args.fee_rate),
            "fill_buffer_bps": float(args.fill_buffer_bps),
            "min_dollar_vol": float(args.min_dollar_vol),
            "train_start": str(train_start),
            "cv_start": str(cv_start),
            "cv_end": str(cv_end),
            "cv_mode": str(args.cv_mode),
            "embargo_days": int(args.embargo_days),
            "device": args.device,
            "random_state": int(args.random_state),
            "learn_chronos_features": bool(args.learn_chronos_features),
            "include_ranks": bool(args.include_ranks),
            "include_dispersion": bool(args.include_dispersion),
            "feature_cols": feature_cols,
            "chronos_cache": str(args.chronos_cache),
            "chronos_coverage": coverage,
        },
        "folds": fold_results,
    }
    out_path = args.output_dir / f"kfold_{ts}.json"
    write_json_atomic(out_path, out)
    print(f"\n  Results → {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
