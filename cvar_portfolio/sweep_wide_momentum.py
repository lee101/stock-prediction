"""Wider-universe CVaR sweep with momentum-top-K universe_fn.

The 100-sym panel hits a physical oracle ceiling at +5.6%/mo
(see memory/project_cvar_panel_physical_ceiling_2026_04_24.md). The
user's +25-30%/mo target requires a wider universe and shorter-horizon
top-K concentration: 807-sym × hd=1 × top-3 oracle reaches +30%/mo.

This sweep:
  - Loads a wide panel (default 1000 syms, min_avg_dollar_vol=1e6).
  - Picks per-rebalance universe = top-K by past-N-day log-return
    (crude momentum signal — cheap, no lookahead).
  - Solves LP on just those K tickers, so solve cost stays small.
  - Sweeps hd × K × L_tar × w_max × trailing-stop params.

Outputs:
    <out>/<cell>/summary.json
    <out>/_results.csv
"""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from cvar_portfolio.backtest import run_backtest
from cvar_portfolio.data import load_price_panel, read_symbol_list


def make_momentum_topk(
    prices: pd.DataFrame, *, top_k: int, lookback: int
) -> Callable[[pd.Timestamp, list[str]], list[str]]:
    """Return `universe_fn(asof, tickers) -> list[str]` selecting top-K
    tickers by their past `lookback`-day log-return ending strictly before
    `asof` (no lookahead).
    """
    log_prices = np.log(prices)

    def universe_fn(asof, tickers: list[str]) -> list[str]:
        idx = prices.index
        # Find the last date strictly < asof
        pos = idx.searchsorted(asof, side="left")
        if pos < lookback + 1:
            return []
        start_pos = pos - lookback - 1
        end_pos = pos - 1
        start_vals = log_prices.iloc[start_pos]
        end_vals = log_prices.iloc[end_pos]
        mom = (end_vals - start_vals).dropna()
        mom = mom[mom.index.isin(tickers)]
        if mom.empty:
            return []
        return mom.sort_values(ascending=False).head(int(top_k)).index.tolist()

    return universe_fn


def _run_cell(
    prices: pd.DataFrame,
    *,
    w_max: float,
    L_tar: float,
    risk_aversion: float,
    confidence: float,
    num_scen: int,
    fit_type: str,
    api: str,
    out_dir: Path,
    c_min: float,
    fee_bps: float,
    slip_bps: float,
    hold_days: int,
    per_asset_stop_loss_pct: float,
    portfolio_stop_loss_pct: float,
    per_asset_trailing_stop_pct: float,
    portfolio_trailing_stop_pct: float,
    top_k: int,
    momentum_lookback: int,
) -> dict:
    stop_tag = ""
    if per_asset_stop_loss_pct > 0 or portfolio_stop_loss_pct > 0:
        stop_tag = f"_asl{int(per_asset_stop_loss_pct):02d}_psl{int(portfolio_stop_loss_pct):02d}"
    ts_tag = ""
    if per_asset_trailing_stop_pct > 0 or portfolio_trailing_stop_pct > 0:
        ts_tag = f"_ats{int(per_asset_trailing_stop_pct):02d}_pts{int(portfolio_trailing_stop_pct):02d}"
    cell_tag = (
        f"k{int(top_k):03d}_mom{int(momentum_lookback):02d}_"
        f"wmax{w_max:.2f}_Ltar{L_tar:.2f}_ra{risk_aversion:.2f}_"
        f"conf{confidence:.3f}_cmin{c_min:+.2f}_hd{int(hold_days):02d}_"
        f"fee{int(fee_bps):02d}_slip{int(slip_bps):02d}{stop_tag}{ts_tag}"
    )
    cell_out = out_dir / cell_tag
    cell_out.mkdir(parents=True, exist_ok=True)
    summary_path = cell_out / "summary.json"
    if summary_path.exists():
        return {**json.loads(summary_path.read_text()), "cell": cell_tag, "cached": True}

    universe_fn = make_momentum_topk(prices, top_k=top_k, lookback=momentum_lookback)
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
        universe_fn=universe_fn,
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
    ap.add_argument("--out", type=Path, default=Path("analysis/cvar_portfolio/sweep_wide_momentum"))
    ap.add_argument("--w-max-grid", nargs="+", type=float, default=[0.15, 0.25])
    ap.add_argument("--ltar-grid", nargs="+", type=float, default=[1.0, 2.0])
    ap.add_argument("--risk-aversion-grid", nargs="+", type=float, default=[1.0])
    ap.add_argument("--confidence-grid", nargs="+", type=float, default=[0.95])
    ap.add_argument("--auto-lever", action="store_true")
    ap.add_argument("--fee-bps", type=float, default=0.0)
    ap.add_argument("--slip-bps", type=float, default=0.0)
    ap.add_argument("--hold-days-grid", nargs="+", type=int, default=[1, 3, 5])
    ap.add_argument("--per-asset-stop-loss-grid", nargs="+", type=float, default=[0.0])
    ap.add_argument("--portfolio-stop-loss-grid", nargs="+", type=float, default=[0.0])
    ap.add_argument("--per-asset-trailing-stop-grid", nargs="+", type=float, default=[0.0])
    ap.add_argument("--portfolio-trailing-stop-grid", nargs="+", type=float, default=[0.0])
    ap.add_argument("--topk-grid", nargs="+", type=int, default=[10, 20, 30])
    ap.add_argument("--momentum-lookback-grid", nargs="+", type=int, default=[20])
    ap.add_argument("--sort-by", default="goodness_score",
                    choices=["goodness_score", "median_monthly_return_pct",
                             "sortino", "ann_return_pct"])
    args = ap.parse_args()

    syms = read_symbol_list(args.symbols)
    if args.max_symbols:
        syms = syms[: args.max_symbols]
    print(f"Loading {len(syms)} symbols from {args.data_root}…")
    prices = load_price_panel(
        syms, args.data_root, start=args.start, end=args.end,
        min_avg_dollar_vol=args.min_avg_dol_vol,
    )
    print(f"Panel: {prices.shape[0]} days × {prices.shape[1]} tickers  "
          f"[{prices.index[0].date()}..{prices.index[-1].date()}]")

    args.out.mkdir(parents=True, exist_ok=True)
    rows = []
    cells = list(itertools.product(
        args.topk_grid, args.momentum_lookback_grid,
        args.w_max_grid, args.ltar_grid, args.risk_aversion_grid, args.confidence_grid,
        args.hold_days_grid, args.per_asset_stop_loss_grid, args.portfolio_stop_loss_grid,
        args.per_asset_trailing_stop_grid, args.portfolio_trailing_stop_grid,
    ))
    for i, (k, mom, wmax, ltar, ra, conf, hd, asl, psl, ats, pts) in enumerate(cells, 1):
        cmin = (1.0 - float(ltar)) if args.auto_lever else 0.0
        print(f"[{i}/{len(cells)}] k={k} mom={mom} wmax={wmax:.2f} Ltar={ltar:.2f} "
              f"ra={ra:.2f} conf={conf:.3f} cmin={cmin:+.2f} hd={hd} "
              f"asl={asl:.0f}%% psl={psl:.0f}%% ats={ats:.0f}%% pts={pts:.0f}%%",
              flush=True)
        row = _run_cell(
            prices, w_max=wmax, L_tar=ltar, risk_aversion=ra, confidence=conf,
            num_scen=args.num_scen, fit_type=args.fit_type, api=args.api, out_dir=args.out,
            c_min=cmin, fee_bps=args.fee_bps, slip_bps=args.slip_bps,
            hold_days=hd, per_asset_stop_loss_pct=asl, portfolio_stop_loss_pct=psl,
            per_asset_trailing_stop_pct=ats, portfolio_trailing_stop_pct=pts,
            top_k=k, momentum_lookback=mom,
        )
        rows.append({
            "cell": row["cell"],
            "top_k": k, "momentum_lookback": mom,
            "wmax": wmax, "Ltar": ltar, "ra": ra, "conf": conf, "cmin": cmin,
            "hold_days": hd, "per_asset_stop_loss_pct": asl, "portfolio_stop_loss_pct": psl,
            "per_asset_trailing_stop_pct": ats, "portfolio_trailing_stop_pct": pts,
            "ann_return_pct": row["ann_return_pct"],
            "sortino": row["sortino"],
            "ann_vol_pct": row["ann_vol_pct"],
            "max_drawdown_pct": row["max_drawdown_pct"],
            "neg_day_frac": row["neg_day_frac"],
            "mean_solve_s": row["mean_solve_s"],
            "mean_turnover": row.get("mean_turnover", 0.0),
            "total_fee_cost_pct": row.get("total_fee_cost_pct", 0.0),
            "median_monthly_return_pct": row.get("median_monthly_return_pct", 0.0),
            "p10_monthly_return_pct": row.get("p10_monthly_return_pct", 0.0),
            "worst_monthly_drawdown_pct": row.get("worst_monthly_drawdown_pct", 0.0),
            "neg_months": row.get("neg_months", 0),
            "worst_5d_drawdown_pct": row.get("worst_5d_drawdown_pct", 0.0),
            "worst_21d_drawdown_pct": row.get("worst_21d_drawdown_pct", 0.0),
            "worst_63d_drawdown_pct": row.get("worst_63d_drawdown_pct", 0.0),
            "frac_21d_dd_gt_25pct": row.get("frac_21d_dd_gt_25pct", 0.0),
            "goodness_score": row.get("goodness_score", 0.0),
            "cached": row.get("cached", False),
        })
        print(f"    good={row.get('goodness_score', 0):+7.2f}  "
              f"med/mo={row.get('median_monthly_return_pct', 0):+6.2f}% "
              f"p10/mo={row.get('p10_monthly_return_pct', 0):+6.2f}% "
              f"w21dDD={row.get('worst_21d_drawdown_pct', 0):+6.2f}% "
              f"frac21>25={row.get('frac_21d_dd_gt_25pct', 0)*100:4.1f}%  "
              f"ann={row['ann_return_pct']:+7.1f}% DD={row['max_drawdown_pct']:+6.1f}% "
              f"sortino={row['sortino']:+.2f} "
              f"turn={row.get('mean_turnover', 0):.2f} fees={row.get('total_fee_cost_pct', 0):.2f}%",
              flush=True)
    df = pd.DataFrame(rows)
    df = df.sort_values(
        [args.sort_by, "worst_21d_drawdown_pct", "sortino"],
        ascending=[False, False, False],
    )
    csv_path = args.out / "_results.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nWrote {csv_path}")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
