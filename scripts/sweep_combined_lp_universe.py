"""Combined-LP sweep: universe = top-K_mom ∪ top-K_xgb in a SINGLE CVaR LP.

Replaces the post-hoc blend (run two solo LPs, blend daily returns) with one
LP over the union universe. Hypothesis: the LP can exploit cross-asset
covariance the post-hoc blend cannot, should be ≥ blend champion (good +6.37,
ann +88%, maxDD −20.65%).

Grid (compact, ~10 cells × ~15 min cpu = ~2.5h):
  K_mom × K_xgb × mom_lb × hd × ats × pts × wmom_pseudo (rank-tilt scale)

The LP gets the union as its universe; the alpha vector is the XGB-derived
μ tilt within that universe. Symbols only in mom-top get μ=0; symbols in xgb-top
get center-mode tilt (μ_i = k * (p - 0.5)). This makes the LP "neutral" on the
mom side (rank-only) and tilted on the xgb side, mirroring the relative
strengths of each signal in the post-hoc blend.
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from cvar_portfolio.backtest import run_backtest
from cvar_portfolio.data import read_symbol_list
from cvar_portfolio.sweep_wide_momentum import load_active_panel, make_momentum_topk
from cvar_portfolio.xgb_alpha import make_alpha_fn, make_universe_fn


def make_combined_universe_fn(
    mom_fn: Callable[[pd.Timestamp, list[str]], list[str]],
    xgb_fn: Callable[[pd.Timestamp, list[str]], list[str]],
) -> Callable[[pd.Timestamp, list[str]], list[str]]:
    def universe_fn(asof, tickers: list[str]) -> list[str]:
        a = mom_fn(asof, tickers)
        b = xgb_fn(asof, tickers)
        if not a and not b:
            return []
        seen = set(a)
        merged = list(a)
        for t in b:
            if t not in seen:
                merged.append(t)
                seen.add(t)
        return merged

    return universe_fn


def _run_cell(
    prices: pd.DataFrame,
    panel_scores: pd.DataFrame,
    *,
    k_mom: int,
    k_xgb: int,
    mom_lookback: int,
    min_score: float,
    alpha_k: float,
    w_max: float,
    L_tar: float,
    risk_aversion: float,
    confidence: float,
    c_min: float,
    num_scen: int,
    fit_type: str,
    api: str,
    fee_bps: float,
    slip_bps: float,
    hold_days: int,
    per_asset_stop_loss_pct: float,
    portfolio_stop_loss_pct: float,
    per_asset_trailing_stop_pct: float,
    portfolio_trailing_stop_pct: float,
    out_dir: Path,
) -> dict:
    cell_tag = (
        f"km{k_mom:03d}_kx{k_xgb:03d}_mom{mom_lookback:02d}_"
        f"ms{int(round(min_score*100)):03d}_ak{alpha_k:.3f}_"
        f"wmax{w_max:.2f}_Ltar{L_tar:.2f}_ra{risk_aversion:.2f}_conf{confidence:.3f}_"
        f"cmin{c_min:+.2f}_hd{hold_days:02d}_"
        f"fee{int(fee_bps):02d}_slip{int(slip_bps):02d}_"
        f"asl{int(per_asset_stop_loss_pct):02d}_psl{int(portfolio_stop_loss_pct):02d}_"
        f"ats{int(per_asset_trailing_stop_pct):02d}_pts{int(portfolio_trailing_stop_pct):02d}"
    )
    cell_out = out_dir / cell_tag
    cell_out.mkdir(parents=True, exist_ok=True)
    summary_path = cell_out / "summary.json"
    if summary_path.exists():
        return {**json.loads(summary_path.read_text()), "cell": cell_tag, "cached": True}

    mom_fn = make_momentum_topk(prices, top_k=k_mom, lookback=mom_lookback)
    xgb_universe_fn = make_universe_fn(panel_scores, top_k=k_xgb, min_score=min_score)
    combined_fn = make_combined_universe_fn(mom_fn, xgb_universe_fn)
    alpha_fn = make_alpha_fn(panel_scores, k=alpha_k, mode="center") if alpha_k > 0 else None

    res = run_backtest(
        prices,
        fit_window=252,
        hold_days=int(hold_days),
        num_scen=num_scen,
        fit_type=fit_type,
        w_max=w_max,
        L_tar=L_tar,
        c_min=c_min,
        risk_aversion=risk_aversion,
        confidence=confidence,
        cardinality=None,
        api=api,
        fee_bps=fee_bps,
        slip_bps=slip_bps,
        per_asset_stop_loss_pct=per_asset_stop_loss_pct,
        portfolio_stop_loss_pct=portfolio_stop_loss_pct,
        per_asset_trailing_stop_pct=per_asset_trailing_stop_pct,
        portfolio_trailing_stop_pct=portfolio_trailing_stop_pct,
        universe_fn=combined_fn,
        alpha_fn=alpha_fn,
        rng_seed=11,
    )
    res.weights_history.to_parquet(cell_out / "weights.parquet")
    res.portfolio_returns.to_csv(cell_out / "daily_returns.csv", header=["log_ret"])
    summary_path.write_text(json.dumps(res.summary, indent=2, default=float))
    return {**res.summary, "cell": cell_tag, "cached": False}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbols", type=Path, required=True)
    ap.add_argument("--data-root", type=Path, default=Path("trainingdata"))
    ap.add_argument("--start", default="2023-06-01")
    ap.add_argument("--end", default="2026-04-18")
    ap.add_argument("--max-symbols", type=int, default=1000)
    ap.add_argument("--min-avg-dol-vol", type=float, default=1e6)
    ap.add_argument("--num-scen", type=int, default=1500)
    ap.add_argument("--fit-type", default="gaussian", choices=["gaussian", "kde", "historical"])
    ap.add_argument("--api", default="cvxpy", choices=["cvxpy", "cuopt_python", "pytorch_kelly"])
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--xgb-score-cache", type=Path, required=True)
    ap.add_argument("--k-mom-grid", nargs="+", type=int, default=[25])
    ap.add_argument("--k-xgb-grid", nargs="+", type=int, default=[15])
    ap.add_argument("--mom-lookback-grid", nargs="+", type=int, default=[20])
    ap.add_argument("--min-score-grid", nargs="+", type=float, default=[0.0])
    ap.add_argument("--alpha-k-grid", nargs="+", type=float, default=[0.0, 0.005])
    ap.add_argument("--w-max-grid", nargs="+", type=float, default=[0.15])
    ap.add_argument("--ltar-grid", nargs="+", type=float, default=[1.0])
    ap.add_argument("--risk-aversion-grid", nargs="+", type=float, default=[1.0])
    ap.add_argument("--confidence-grid", nargs="+", type=float, default=[0.95])
    ap.add_argument("--fee-bps", type=float, default=10.0)
    ap.add_argument("--slip-bps", type=float, default=5.0)
    ap.add_argument("--hold-days-grid", nargs="+", type=int, default=[5])
    ap.add_argument("--per-asset-stop-loss-grid", nargs="+", type=float, default=[15.0])
    ap.add_argument("--portfolio-stop-loss-grid", nargs="+", type=float, default=[0.0])
    ap.add_argument("--per-asset-trailing-stop-grid", nargs="+", type=float, default=[12.0])
    ap.add_argument("--portfolio-trailing-stop-grid", nargs="+", type=float, default=[0.0])
    args = ap.parse_args()

    syms = read_symbol_list(args.symbols)
    if args.max_symbols:
        syms = syms[: args.max_symbols]
    print(f"Loading {len(syms)} symbols from {args.data_root}…", flush=True)
    prices = load_active_panel(
        syms, args.data_root,
        start=args.start, end=args.end,
        min_avg_dollar_vol=args.min_avg_dol_vol,
    )
    print(f"Panel: {prices.shape[0]} days × {prices.shape[1]} tickers  "
          f"[{prices.index[0].date()}..{prices.index[-1].date()}]", flush=True)

    print(f"Loading XGB score cache: {args.xgb_score_cache}", flush=True)
    panel_scores = pd.read_parquet(args.xgb_score_cache)
    print(f"Scores: {len(panel_scores):,} rows, {panel_scores.symbol.nunique()} syms, "
          f"{panel_scores.date.nunique()} days [{panel_scores.date.min().date()}..{panel_scores.date.max().date()}]",
          flush=True)

    args.out.mkdir(parents=True, exist_ok=True)
    rows = []
    cells = list(itertools.product(
        args.k_mom_grid, args.k_xgb_grid, args.mom_lookback_grid,
        args.min_score_grid, args.alpha_k_grid,
        args.w_max_grid, args.ltar_grid, args.risk_aversion_grid, args.confidence_grid,
        args.hold_days_grid,
        args.per_asset_stop_loss_grid, args.portfolio_stop_loss_grid,
        args.per_asset_trailing_stop_grid, args.portfolio_trailing_stop_grid,
    ))
    for i, (km, kx, mom, ms, ak, wmax, ltar, ra, conf, hd, asl, psl, ats, pts) in enumerate(cells, 1):
        cmin = 0.0
        print(f"[{i}/{len(cells)}] km={km} kx={kx} mom_lb={mom} ms={ms:.2f} ak={ak:.3f} "
              f"wmax={wmax:.2f} Ltar={ltar:.2f} ra={ra:.2f} hd={hd} ats={ats:.0f} pts={pts:.0f}",
              flush=True)
        row = _run_cell(
            prices, panel_scores,
            k_mom=km, k_xgb=kx, mom_lookback=mom, min_score=ms, alpha_k=ak,
            w_max=wmax, L_tar=ltar, risk_aversion=ra, confidence=conf, c_min=cmin,
            num_scen=args.num_scen, fit_type=args.fit_type, api=args.api,
            fee_bps=args.fee_bps, slip_bps=args.slip_bps,
            hold_days=hd,
            per_asset_stop_loss_pct=asl, portfolio_stop_loss_pct=psl,
            per_asset_trailing_stop_pct=ats, portfolio_trailing_stop_pct=pts,
            out_dir=args.out,
        )
        rows.append({
            "cell": row["cell"], "k_mom": km, "k_xgb": kx, "mom_lb": mom,
            "min_score": ms, "alpha_k": ak,
            "wmax": wmax, "Ltar": ltar, "ra": ra, "conf": conf,
            "hold_days": hd, "ats": ats, "pts": pts,
            "med_mo": row.get("median_monthly_return_pct", 0.0),
            "p10_mo": row.get("p10_monthly_return_pct", 0.0),
            "w5": row.get("worst_5d_drawdown_pct", 0.0),
            "w21": row.get("worst_21d_drawdown_pct", 0.0),
            "w63": row.get("worst_63d_drawdown_pct", 0.0),
            "frac21": row.get("frac_21d_dd_gt_25pct", 0.0),
            "maxdd": row["max_drawdown_pct"],
            "sortino": row["sortino"],
            "ann": row["ann_return_pct"],
            "good": row.get("goodness_score", 0.0),
            "cached": row.get("cached", False),
        })
        print(f"    good={row.get('goodness_score',0):+7.2f}  "
              f"med/mo={row.get('median_monthly_return_pct',0):+6.2f}% "
              f"p10/mo={row.get('p10_monthly_return_pct',0):+6.2f}% "
              f"w21dDD={row.get('worst_21d_drawdown_pct',0):+6.2f}% "
              f"maxDD={row['max_drawdown_pct']:+6.1f}% sortino={row['sortino']:+.2f} "
              f"ann={row['ann_return_pct']:+7.1f}%",
              flush=True)
    df = pd.DataFrame(rows)
    df = df.sort_values("good", ascending=False)
    csv_path = args.out / "_results.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nWrote {csv_path}", flush=True)
    print(df.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
