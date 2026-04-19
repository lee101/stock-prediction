#!/usr/bin/env python3
"""Multi-window backtest of pre-saved XGB pickles (no training).

Usage::

    # Single model (overfit by design when OOS overlaps train):
    python -m xgbnew.eval_pretrained \\
        --model-path analysis/xgbnew_daily/live_model_alltrain.pkl \\
        --symbols-file symbol_lists/stocks_wide_1000_v1.txt \\
        --oos-start 2025-01-02 --oos-end 2026-04-10 \\
        --window-days 30 --stride-days 14 --top-n 1 --leverage 1.0

    # Ensemble (predict_proba averaged across models):
    python -m xgbnew.eval_pretrained \\
        --model-paths analysis/xgbnew_daily/alltrain_ensemble_gpu/alltrain_seed0.pkl,\\
analysis/xgbnew_daily/alltrain_ensemble_gpu/alltrain_seed7.pkl,\\
analysis/xgbnew_daily/alltrain_ensemble_gpu/alltrain_seed42.pkl,\\
analysis/xgbnew_daily/alltrain_ensemble_gpu/alltrain_seed73.pkl,\\
analysis/xgbnew_daily/alltrain_ensemble_gpu/alltrain_seed197.pkl \\
        --symbols-file symbol_lists/stocks_wide_1000_v1.txt \\
        --oos-start 2025-01-02 --oos-end 2026-04-10 \\
        --blend-mode mean --top-n 1 --leverage 1.0

⚠ If the OOS window overlaps training (e.g. alltrain model), numbers are IN-SAMPLE.
  Use for sanity check only — do not cite as OOS. Honest OOS requires train_end <
  oos_start.
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

TRADING_DAYS_PER_MONTH = 21.0


def _load_symbols(path: Path) -> list[str]:
    syms = []
    for line in path.read_text(encoding="utf-8").splitlines():
        s = line.strip().split("#", 1)[0].strip().upper()
        if s:
            syms.append(s)
    seen = set()
    return [s for s in syms if not (s in seen or seen.add(s))]


def _monthly_return(total_ret: float, n_days: int) -> float:
    try:
        return (1.0 + total_ret / 100.0) ** (TRADING_DAYS_PER_MONTH / n_days) - 1.0
    except Exception:
        return 0.0


def _build_windows(days, *, window_days: int, stride_days: int):
    out = []
    i = 0
    while i + window_days <= len(days):
        span = days[i:i + window_days]
        out.append((span[0], span[-1]))
        i += stride_days
    return out


def _load_pretrained(path: Path) -> XGBStockModel:
    if not path.exists():
        raise FileNotFoundError(f"model pickle not found: {path}")
    return XGBStockModel.load(path)


def _blend(probs: list[pd.Series], mode: str, oos_df: pd.DataFrame) -> pd.Series:
    mat = np.stack([p.to_numpy(dtype=np.float64, copy=False) for p in probs], axis=0)
    if mode == "mean":
        blended = mat.mean(axis=0)
    elif mode == "median":
        blended = np.median(mat, axis=0)
    elif mode == "rank_mean":
        dates = oos_df["date"].to_numpy()
        uniq, inv = np.unique(dates, return_inverse=True)
        ranks = np.empty_like(mat)
        for d_i in range(len(uniq)):
            rows = np.where(inv == d_i)[0]
            for s_i in range(mat.shape[0]):
                x = mat[s_i, rows]
                order = np.argsort(np.argsort(x))
                ranks[s_i, rows] = order / max(len(rows) - 1, 1)
        blended = ranks.mean(axis=0)
    else:
        raise ValueError(f"unknown blend-mode: {mode!r}")
    return pd.Series(blended, index=probs[0].index, name="blended_score")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--model-path", type=Path,
                   help="Single pre-saved pickle — skip ensemble blend.")
    g.add_argument("--model-paths", type=str,
                   help="Comma-separated pickle paths for ensemble blend.")

    p.add_argument("--symbols-file", type=Path, required=True)
    p.add_argument("--data-root", type=Path, default=REPO / "trainingdata")
    p.add_argument("--chronos-cache", type=Path,
                   default=REPO / "analysis/top2_backtest/forecast_cache")
    p.add_argument("--oos-start", default="2025-01-02")
    p.add_argument("--oos-end", default="")
    p.add_argument("--window-days", type=int, default=30)
    p.add_argument("--stride-days", type=int, default=14)

    p.add_argument("--top-n", type=int, default=1)
    p.add_argument("--leverage", type=float, default=1.0)
    p.add_argument("--xgb-weight", type=float, default=1.0)
    p.add_argument("--fee-rate", type=float, default=0.0000278)
    p.add_argument("--commission-bps", type=float, default=0.0)
    p.add_argument("--fill-buffer-bps", type=float, default=5.0)
    p.add_argument("--min-dollar-vol", type=float, default=5_000_000.0)

    p.add_argument("--allocation-mode", default="equal",
                   choices=["equal", "softmax", "score_norm"])
    p.add_argument("--allocation-temp", type=float, default=1.0)
    p.add_argument("--min-score", type=float, default=0.0,
                   help="Skip trade if top-1 blended predict_proba < min_score. "
                        "0.0 (default) = no filter. Values ~0.55-0.70 gate on conviction.")

    p.add_argument("--blend-mode", default="mean",
                   choices=["mean", "median", "rank_mean"])
    p.add_argument("--output-path", type=Path,
                   default=REPO / "analysis/xgbnew_daily/eval_pretrained_latest.json")
    p.add_argument("--log-picks", action="store_true",
                   help="Include per-window picks (symbol, score) in output JSON.")
    p.add_argument("--verbose", "-v", action="store_true")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING,
                        format="%(levelname)s %(message)s")

    if args.model_path is not None:
        model_paths = [args.model_path]
    else:
        model_paths = [Path(p.strip()) for p in args.model_paths.split(",") if p.strip()]

    models: list[XGBStockModel] = []
    for mp in model_paths:
        t = time.perf_counter()
        m = _load_pretrained(mp)
        models.append(m)
        print(f"[xgb-eval-pre] loaded {mp.name} in {time.perf_counter()-t:.2f}s", flush=True)

    symbols = _load_symbols(args.symbols_file)
    oos_start = date.fromisoformat(args.oos_start)
    oos_end = date.fromisoformat(args.oos_end) if args.oos_end else date.today()

    chronos_cache = {}
    if args.chronos_cache.exists():
        chronos_cache = load_chronos_cache(args.chronos_cache)

    t0 = time.perf_counter()
    # We only need the OOS dataset; pass degenerate train range so the dataset
    # builder doesn't do extra work.
    _, _, oos_df = build_daily_dataset(
        data_root=args.data_root,
        symbols=symbols,
        train_start=oos_start, train_end=oos_start,
        val_start=oos_start, val_end=oos_start,
        test_start=oos_start, test_end=oos_end,
        chronos_cache=chronos_cache if chronos_cache else None,
        min_dollar_vol=args.min_dollar_vol,
        fast_features=False,
    )
    print(f"[xgb-eval-pre] OOS dataset built in {time.perf_counter()-t0:.1f}s | "
          f"rows={len(oos_df):,}  symbols={oos_df['symbol'].nunique()}", flush=True)

    if len(oos_df) < 100:
        print("ERROR: too few OOS rows", file=sys.stderr)
        return 1

    # Predict with each model
    probs: list[pd.Series] = []
    for i, m in enumerate(models, start=1):
        t = time.perf_counter()
        p_scores = m.predict_scores(oos_df)
        probs.append(p_scores)
        print(f"[xgb-eval-pre] predict {i}/{len(models)} in "
              f"{time.perf_counter()-t:.2f}s", flush=True)

    blended_scores = _blend(probs, args.blend_mode, oos_df) if len(probs) > 1 else probs[0]

    # Run the multi-window backtest
    all_days = sorted(oos_df["date"].unique())
    windows = _build_windows(all_days, window_days=int(args.window_days),
                             stride_days=int(args.stride_days))
    if not windows:
        print("ERROR: no windows built", file=sys.stderr)
        return 1

    backtest_cfg = BacktestConfig(
        top_n=int(args.top_n),
        leverage=float(args.leverage),
        xgb_weight=float(args.xgb_weight),
        commission_bps=float(args.commission_bps),
        min_dollar_vol=float(args.min_dollar_vol),
        fee_rate=float(args.fee_rate),
        fill_buffer_bps=float(args.fill_buffer_bps),
        allocation_mode=str(args.allocation_mode),
        allocation_temp=float(args.allocation_temp),
        min_score=float(args.min_score),
    )

    window_results = []
    for w_start, w_end in windows:
        w_df = oos_df[(oos_df["date"] >= w_start) & (oos_df["date"] <= w_end)]
        if len(w_df) < 5:
            continue
        w_scores = blended_scores.loc[w_df.index]
        res = simulate(w_df, models[0], backtest_cfg, precomputed_scores=w_scores)
        n_days = len(res.day_results)
        monthly = _monthly_return(res.total_return_pct, max(n_days, 1)) * 100.0
        wr = {
            "w_start": str(w_start), "w_end": str(w_end),
            "n_trading_days": n_days,
            "total_return_pct": res.total_return_pct,
            "monthly_return_pct": monthly,
            "sortino": res.sortino_ratio,
            "max_dd_pct": res.max_drawdown_pct,
        }
        if args.log_picks:
            wr["picks"] = [
                {"day": str(dr.day),
                 "trades": [{"sym": t.symbol, "score": float(t.score),
                             "net_ret_pct": float(t.net_return_pct)}
                            for t in dr.trades]}
                for dr in res.day_results
            ]
        window_results.append(wr)

    if not window_results:
        print("ERROR: no complete windows", file=sys.stderr)
        return 1

    monthlies = np.array([r["monthly_return_pct"] for r in window_results], dtype=np.float64)
    sortinos = np.array([r["sortino"] for r in window_results], dtype=np.float64)
    dds = np.array([r["max_dd_pct"] for r in window_results], dtype=np.float64)
    summary = {
        "n_windows": len(window_results),
        "median_monthly_pct": float(np.median(monthlies)),
        "p10_monthly_pct": float(np.percentile(monthlies, 10)),
        "mean_monthly_pct": float(np.mean(monthlies)),
        "median_sortino": float(np.median(sortinos)),
        "worst_dd_pct": float(np.max(dds)),
        "n_neg": int(np.sum(monthlies < 0.0)),
    }

    print("\n=== SUMMARY ===")
    print(f"  models:   {len(models)} {'(ensemble)' if len(models) > 1 else '(single)'}")
    print(f"  windows:  {summary['n_windows']}")
    print(f"  median monthly%: {summary['median_monthly_pct']:+.2f}")
    print(f"  p10    monthly%: {summary['p10_monthly_pct']:+.2f}")
    print(f"  mean   monthly%: {summary['mean_monthly_pct']:+.2f}")
    print(f"  median sortino:  {summary['median_sortino']:.2f}")
    print(f"  worst DD%:       {summary['worst_dd_pct']:.2f}")
    print(f"  neg windows:     {summary['n_neg']}/{summary['n_windows']}")

    out = {
        "models": [str(mp) for mp in model_paths],
        "oos_start": str(oos_start),
        "oos_end": str(oos_end),
        "window_days": int(args.window_days),
        "stride_days": int(args.stride_days),
        "top_n": int(args.top_n),
        "leverage": float(args.leverage),
        "fee_rate": float(args.fee_rate),
        "fill_buffer_bps": float(args.fill_buffer_bps),
        "blend_mode": str(args.blend_mode),
        "summary": summary,
        "windows": window_results,
    }
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    args.output_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\n[xgb-eval-pre] results -> {args.output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
