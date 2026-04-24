"""Sweep vanilla CVaR hyperparameters on the 100-sym 2024-06 → 2026-04 panel.

Loads the same cached price panel used for the alpha/prefilter refutations
and reruns `run_backtest` over (w_max × L_tar × risk_aversion × confidence)
cells. No XGB hook — this isolates the optimiser's own operating point.

Outputs:
    analysis/cvar_portfolio/sweep_vanilla_hparam/<cell>/summary.json
    analysis/cvar_portfolio/sweep_vanilla_hparam/_results.csv
"""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import pandas as pd

from cvar_portfolio.backtest import run_backtest
from cvar_portfolio.data import load_price_panel, read_symbol_list


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
    c_min: float = 0.0,
    fee_bps: float = 0.0,
    slip_bps: float = 0.0,
    hold_days: int = 21,
    per_asset_stop_loss_pct: float = 0.0,
    portfolio_stop_loss_pct: float = 0.0,
    per_asset_trailing_stop_pct: float = 0.0,
    portfolio_trailing_stop_pct: float = 0.0,
) -> dict:
    fee_tag = f"_fee{int(fee_bps):02d}_slip{int(slip_bps):02d}" if (fee_bps or slip_bps) else ""
    hold_tag = f"_hd{int(hold_days):02d}"
    stop_tag = ""
    if per_asset_stop_loss_pct > 0 or portfolio_stop_loss_pct > 0:
        stop_tag = f"_asl{int(per_asset_stop_loss_pct):02d}_psl{int(portfolio_stop_loss_pct):02d}"
    ts_tag = ""
    if per_asset_trailing_stop_pct > 0 or portfolio_trailing_stop_pct > 0:
        ts_tag = f"_ats{int(per_asset_trailing_stop_pct):02d}_pts{int(portfolio_trailing_stop_pct):02d}"
    cell_tag = (
        f"wmax{w_max:.2f}_Ltar{L_tar:.2f}_ra{risk_aversion:.2f}_"
        f"conf{confidence:.3f}_cmin{c_min:+.2f}{hold_tag}{fee_tag}{stop_tag}{ts_tag}"
    )
    cell_out = out_dir / cell_tag
    cell_out.mkdir(parents=True, exist_ok=True)
    summary_path = cell_out / "summary.json"
    if summary_path.exists():
        return {**json.loads(summary_path.read_text()), "cell": cell_tag, "cached": True}
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
    ap.add_argument("--max-symbols", type=int, default=100)
    ap.add_argument("--min-avg-dol-vol", type=float, default=5e6)
    ap.add_argument("--num-scen", type=int, default=2500)
    ap.add_argument("--fit-type", default="gaussian", choices=["gaussian", "kde", "historical"])
    ap.add_argument("--api", default="cvxpy", choices=["cvxpy", "cuopt_python", "pytorch_kelly"])
    ap.add_argument("--out", type=Path, default=Path("analysis/cvar_portfolio/sweep_vanilla_hparam"))
    ap.add_argument("--w-max-grid", nargs="+", type=float, default=[0.05, 0.10, 0.15, 0.20])
    ap.add_argument("--ltar-grid", nargs="+", type=float, default=[1.0, 1.5, 2.0])
    ap.add_argument("--risk-aversion-grid", nargs="+", type=float, default=[1.0])
    ap.add_argument("--confidence-grid", nargs="+", type=float, default=[0.95])
    ap.add_argument("--auto-lever", action="store_true",
                    help="Set c_min=1-L_tar so cash can go negative; lets the "
                         "LP actually reach sum(|w|)=L_tar. Default keeps c_min=0.")
    ap.add_argument("--fee-bps", type=float, default=0.0,
                    help="One-way transaction fee in bps per unit of turnover.")
    ap.add_argument("--slip-bps", type=float, default=0.0,
                    help="One-way slippage in bps per unit of turnover.")
    ap.add_argument("--hold-days-grid", nargs="+", type=int, default=[21],
                    help="Rebalance cadence grid (trading days between rebalances). "
                         "Shorter = more responsive but more fees.")
    ap.add_argument("--per-asset-stop-loss-grid", nargs="+", type=float, default=[0.0],
                    help="Per-asset stop-loss threshold in %% (0 = disabled). "
                         "An asset whose cumulative log-return since buy drops "
                         "below −N%% is liquidated mid-window.")
    ap.add_argument("--portfolio-stop-loss-grid", nargs="+", type=float, default=[0.0],
                    help="Portfolio-wide stop-loss threshold in %% (0 = disabled). "
                         "If the portfolio cum log-return since rebalance drops "
                         "below −N%%, the whole book liquidates.")
    ap.add_argument("--per-asset-trailing-stop-grid", nargs="+", type=float, default=[0.0],
                    help="Per-asset trailing stop in %% (0 = disabled). "
                         "Asset liquidated if cum log-return drops >N%% from "
                         "its running peak since entry.")
    ap.add_argument("--portfolio-trailing-stop-grid", nargs="+", type=float, default=[0.0],
                    help="Portfolio-wide trailing stop in %% (0 = disabled). "
                         "Book fully liquidated for remainder of window if "
                         "portfolio cum log-return drops >N%% from rebalance peak.")
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
        args.w_max_grid, args.ltar_grid, args.risk_aversion_grid, args.confidence_grid,
        args.hold_days_grid, args.per_asset_stop_loss_grid, args.portfolio_stop_loss_grid,
        args.per_asset_trailing_stop_grid, args.portfolio_trailing_stop_grid,
    ))
    for i, (wmax, ltar, ra, conf, hd, asl, psl, ats, pts) in enumerate(cells, 1):
        # If auto-lever is set, let cash go negative so LP can actually
        # reach sum(|w|)=L_tar; otherwise cash≥0 caps sum(w)≤1 regardless
        # of L_tar (budget constraint sum(w)+c=1 binds).
        cmin = (1.0 - float(ltar)) if args.auto_lever else 0.0
        print(f"[{i}/{len(cells)}] wmax={wmax:.2f} Ltar={ltar:.2f} ra={ra:.2f} "
              f"conf={conf:.3f} cmin={cmin:+.2f} hd={hd} asl={asl:.0f}%% psl={psl:.0f}%% "
              f"ats={ats:.0f}%% pts={pts:.0f}%%", flush=True)
        row = _run_cell(
            prices, w_max=wmax, L_tar=ltar, risk_aversion=ra, confidence=conf,
            num_scen=args.num_scen, fit_type=args.fit_type, api=args.api, out_dir=args.out,
            c_min=cmin, fee_bps=args.fee_bps, slip_bps=args.slip_bps,
            hold_days=hd, per_asset_stop_loss_pct=asl, portfolio_stop_loss_pct=psl,
            per_asset_trailing_stop_pct=ats, portfolio_trailing_stop_pct=pts,
        )
        rows.append({
            "cell": row["cell"],
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
              f"w5dDD={row.get('worst_5d_drawdown_pct', 0):+6.2f}% "
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
