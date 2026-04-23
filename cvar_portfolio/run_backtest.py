"""CLI: run a rolling CVaR backtest on our real universe.

Example
-------
    python -m cvar_portfolio.run_backtest \
        --symbols symbol_lists/stocks_wide_1000_v1.txt \
        --data-root trainingdata \
        --start 2020-01-01 --end 2026-04-18 \
        --fit-window 252 --hold-days 21 \
        --w-max 0.05 --cardinality 20 \
        --api cuopt_python --kde-device GPU \
        --out analysis/cvar_portfolio/run1

Writes:
    <out>/weights.parquet      one row per rebalance
    <out>/daily_returns.csv    realised portfolio log returns per trading day
    <out>/summary.json         sortino / ann_return / DD / neg_frac / solve_time
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from cvar_portfolio.backtest import run_backtest
from cvar_portfolio.data import load_price_panel, read_symbol_list


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbols", type=Path, required=True)
    ap.add_argument("--data-root", type=Path, default=Path("trainingdata"))
    ap.add_argument("--start", default="2020-01-01")
    ap.add_argument("--end", default=None)
    ap.add_argument("--max-symbols", type=int, default=None,
                    help="cap universe to first N tickers after liquidity filter")
    ap.add_argument("--min-avg-dol-vol", type=float, default=5e6)
    ap.add_argument("--fit-window", type=int, default=252)
    ap.add_argument("--hold-days", type=int, default=21)
    ap.add_argument("--num-scen", type=int, default=2500)
    ap.add_argument("--fit-type", default="kde", choices=["kde", "gaussian", "historical"])
    ap.add_argument("--kde-device", default="CPU", choices=["CPU", "GPU"])
    ap.add_argument("--kde-bandwidth", type=float, default=0.01)
    ap.add_argument("--risk-aversion", type=float, default=1.0)
    ap.add_argument("--confidence", type=float, default=0.95)
    ap.add_argument("--w-max", type=float, default=0.05)
    ap.add_argument("--w-min", type=float, default=0.0)
    ap.add_argument("--c-min", type=float, default=0.0)
    ap.add_argument("--c-max", type=float, default=1.0)
    ap.add_argument("--l-tar", type=float, default=1.0)
    ap.add_argument("--cardinality", type=int, default=None)
    ap.add_argument("--api", default="cvxpy", choices=["cvxpy", "cuopt_python", "pytorch_kelly"])
    ap.add_argument("--kelly-lr", type=float, default=0.01)
    ap.add_argument("--kelly-steps", type=int, default=1500)
    ap.add_argument("--kelly-l2-reg", type=float, default=0.0)
    ap.add_argument("--kelly-turnover-penalty", type=float, default=0.0)
    ap.add_argument("--kelly-warm-start", action="store_true")
    ap.add_argument("--kelly-device", default=None,
                    help="Torch device for api=pytorch_kelly, e.g. cuda or cpu.")
    ap.add_argument("--rng-seed", type=int, default=0)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--verbose", action="store_true")

    # Alpha-aware injection: pipe XGB ensemble scores into `mean_override`.
    ap.add_argument("--xgb-ensemble-dir", type=Path, default=None,
                    help="If set, score every OOS day/symbol with this ensemble "
                         "and pass μ = k·(p − base) to the LP as mean_override.")
    ap.add_argument("--xgb-train-start", default="2020-01-01")
    ap.add_argument("--xgb-train-end", default="2025-06-30")
    ap.add_argument("--xgb-k", type=float, default=0.01,
                    help="Scale factor mapping ensemble-score to daily log return.")
    ap.add_argument("--xgb-mode", default="center", choices=["center", "demean", "rank"])
    ap.add_argument("--xgb-panel-cache", type=Path, default=None,
                    help="Parquet cache path for panel scores (skip expensive rebuild).")
    ap.add_argument("--xgb-top-k", type=int, default=0,
                    help="If >0, restrict LP universe at each rebalance to the "
                         "top-K XGB-scored symbols. Separate from --xgb-k/mode: "
                         "pre-filter vs mu-tilt can be combined or used alone.")
    ap.add_argument("--xgb-min-score", type=float, default=0.0,
                    help="Drop symbols with ensemble_score < this threshold from "
                         "the pre-filtered top-K set (e.g. 0.50 = only long-side).")
    ap.add_argument("--no-alpha-mu", action="store_true",
                    help="Use XGB only as universe pre-filter (top-K), skip "
                         "mean_override injection. Pair with --xgb-top-k.")

    args = ap.parse_args()

    syms = read_symbol_list(args.symbols)
    if args.max_symbols:
        syms = syms[: args.max_symbols]
    print(f"Loading {len(syms)} symbols from {args.data_root}…")
    prices = load_price_panel(
        syms,
        args.data_root,
        start=args.start,
        end=args.end,
        min_avg_dollar_vol=args.min_avg_dol_vol,
    )
    print(f"Loaded panel: {prices.shape[0]} days × {prices.shape[1]} tickers  [{prices.index[0].date()}..{prices.index[-1].date()}]")
    if prices.shape[1] == 0:
        raise SystemExit("No symbols passed liquidity/history filter.")

    alpha_fn = None
    universe_fn = None
    if args.xgb_ensemble_dir is not None:
        from datetime import date as _date
        from cvar_portfolio.xgb_alpha import (
            build_xgb_panel_scores, make_alpha_fn, make_universe_fn,
        )
        panel_tickers = prices.columns.tolist()

        cache_path = args.xgb_panel_cache
        if cache_path is not None and cache_path.exists():
            print(f"[run_backtest] Loading cached XGB panel scores → {cache_path}")
            panel_scores = pd.read_parquet(cache_path)
        else:
            panel_scores = build_xgb_panel_scores(
                symbols=panel_tickers,
                data_root=args.data_root,
                oos_start=_date.fromisoformat(args.start),
                oos_end=_date.fromisoformat(args.end) if args.end else prices.index[-1].date(),
                ensemble_dir=args.xgb_ensemble_dir,
                train_start=_date.fromisoformat(args.xgb_train_start),
                train_end=_date.fromisoformat(args.xgb_train_end),
                min_dollar_vol=args.min_avg_dol_vol,
                fast_features=True,
            )
            if cache_path is not None:
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                panel_scores.to_parquet(cache_path)

        if not args.no_alpha_mu:
            alpha_fn = make_alpha_fn(panel_scores, k=args.xgb_k, mode=args.xgb_mode)
            print(f"[run_backtest] mu-tilt mode: k={args.xgb_k} mode={args.xgb_mode}")
        if args.xgb_top_k and args.xgb_top_k > 0:
            universe_fn = make_universe_fn(
                panel_scores, top_k=args.xgb_top_k, min_score=args.xgb_min_score,
            )
            print(f"[run_backtest] universe pre-filter: top_k={args.xgb_top_k} "
                  f"min_score={args.xgb_min_score}")

    result = run_backtest(
        prices,
        fit_window=args.fit_window,
        hold_days=args.hold_days,
        num_scen=args.num_scen,
        fit_type=args.fit_type,
        kde_device=args.kde_device,
        kde_bandwidth=args.kde_bandwidth,
        risk_aversion=args.risk_aversion,
        confidence=args.confidence,
        w_max=args.w_max,
        w_min=args.w_min,
        c_min=args.c_min,
        c_max=args.c_max,
        L_tar=args.l_tar,
        cardinality=args.cardinality,
        api=args.api,
        alpha_fn=alpha_fn,
        universe_fn=universe_fn,
        kelly_lr=args.kelly_lr,
        kelly_steps=args.kelly_steps,
        kelly_l2_reg=args.kelly_l2_reg,
        kelly_turnover_penalty=args.kelly_turnover_penalty,
        kelly_warm_start=args.kelly_warm_start,
        kelly_device=args.kelly_device,
        rng_seed=args.rng_seed,
        verbose=args.verbose,
    )

    args.out.mkdir(parents=True, exist_ok=True)
    result.weights_history.to_parquet(args.out / "weights.parquet")
    result.portfolio_returns.to_csv(args.out / "daily_returns.csv", header=["log_ret"])
    (args.out / "summary.json").write_text(json.dumps(result.summary, indent=2, default=float))
    print("\n=== Summary ===")
    for k, v in result.summary.items():
        print(f"  {k:24s} {v}")


if __name__ == "__main__":
    main()
