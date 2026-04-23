"""Compare pytorch_kelly vs LP champion on the 100-sym 2024-06 → 2026-04 panel.

Runs `run_backtest(api="pytorch_kelly")` over a small grid at the same
cells as the LP aggressive sweep so we can see whether differentiable
Kelly + CVaR matches or exceeds the LP's +10503%/yr champion.

Outputs: analysis/cvar_portfolio/bench_pytorch_kelly/<cell>/summary.json
         analysis/cvar_portfolio/bench_pytorch_kelly/_results.csv
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
    cvar_penalty: float,
    confidence: float,
    num_scen: int,
    fit_type: str,
    out_dir: Path,
    fee_bps: float,
    slip_bps: float,
    kelly_lr: float,
    kelly_steps: int,
    kelly_l2_reg: float,
    kelly_turnover_penalty: float,
    kelly_warm_start: bool,
    kelly_device: str | None,
) -> dict:
    fee_tag = f"_fee{int(fee_bps):02d}_slip{int(slip_bps):02d}" if (fee_bps or slip_bps) else ""
    cell_tag = (
        f"kelly_wmax{w_max:.2f}_Ltar{L_tar:.2f}_pen{cvar_penalty:.3f}_"
        f"conf{confidence:.3f}{fee_tag}"
    )
    cell_out = out_dir / cell_tag
    cell_out.mkdir(parents=True, exist_ok=True)
    summary_path = cell_out / "summary.json"
    if summary_path.exists():
        return {**json.loads(summary_path.read_text()), "cell": cell_tag, "cached": True}
    res = run_backtest(
        prices,
        fit_window=252,
        hold_days=21,
        num_scen=num_scen,
        fit_type=fit_type,
        w_max=w_max,
        L_tar=L_tar,
        risk_aversion=cvar_penalty,  # maps to cvar_penalty inside kelly wrapper
        confidence=confidence,
        cardinality=None,
        api="pytorch_kelly",
        fee_bps=fee_bps,
        slip_bps=slip_bps,
        kelly_lr=kelly_lr,
        kelly_steps=kelly_steps,
        kelly_l2_reg=kelly_l2_reg,
        kelly_turnover_penalty=kelly_turnover_penalty,
        kelly_warm_start=kelly_warm_start,
        kelly_device=kelly_device,
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
    ap.add_argument("--out", type=Path, default=Path("analysis/cvar_portfolio/bench_pytorch_kelly"))
    ap.add_argument("--w-max-grid", nargs="+", type=float, default=[0.10, 0.25])
    ap.add_argument("--ltar-grid", nargs="+", type=float, default=[1.0, 2.0, 4.0, 6.0])
    ap.add_argument("--penalty-grid", nargs="+", type=float, default=[0.01, 0.1])
    ap.add_argument("--confidence-grid", nargs="+", type=float, default=[0.95])
    ap.add_argument("--fee-bps", type=float, default=0.0)
    ap.add_argument("--slip-bps", type=float, default=0.0)
    ap.add_argument("--kelly-lr", type=float, default=0.01)
    ap.add_argument("--kelly-steps", type=int, default=1500)
    ap.add_argument("--kelly-l2-reg", type=float, default=0.0)
    ap.add_argument("--kelly-turnover-penalty", type=float, default=0.0)
    ap.add_argument("--kelly-warm-start", action="store_true")
    ap.add_argument("--kelly-device", default=None)
    args = ap.parse_args()

    syms = read_symbol_list(args.symbols)
    if args.max_symbols:
        syms = syms[: args.max_symbols]
    print(f"Loading {len(syms)} symbols from {args.data_root}…")
    prices = load_price_panel(
        syms, args.data_root, start=args.start, end=args.end,
        min_avg_dollar_vol=args.min_avg_dol_vol,
    )
    print(f"Panel: {prices.shape[0]} days × {prices.shape[1]} tickers")

    args.out.mkdir(parents=True, exist_ok=True)
    rows = []
    cells = list(itertools.product(
        args.w_max_grid, args.ltar_grid, args.penalty_grid, args.confidence_grid,
    ))
    for i, (wmax, ltar, pen, conf) in enumerate(cells, 1):
        print(f"[{i}/{len(cells)}] kelly wmax={wmax:.2f} Ltar={ltar:.2f} pen={pen:.3f} "
              f"conf={conf:.3f} fee={args.fee_bps:.0f}bps+{args.slip_bps:.0f}bps", flush=True)
        row = _run_cell(
            prices, w_max=wmax, L_tar=ltar, cvar_penalty=pen, confidence=conf,
            num_scen=args.num_scen, fit_type=args.fit_type, out_dir=args.out,
            fee_bps=args.fee_bps, slip_bps=args.slip_bps,
            kelly_lr=args.kelly_lr,
            kelly_steps=args.kelly_steps,
            kelly_l2_reg=args.kelly_l2_reg,
            kelly_turnover_penalty=args.kelly_turnover_penalty,
            kelly_warm_start=args.kelly_warm_start,
            kelly_device=args.kelly_device,
        )
        rows.append({
            "cell": row["cell"],
            "wmax": wmax, "Ltar": ltar, "pen": pen, "conf": conf,
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
            "cached": row.get("cached", False),
        })
        print(f"    med/mo={row.get('median_monthly_return_pct', 0):+6.2f}% "
              f"p10/mo={row.get('p10_monthly_return_pct', 0):+6.2f}% "
              f"worstMDD={row.get('worst_monthly_drawdown_pct', 0):+6.2f}% "
              f"ann={row['ann_return_pct']:+8.2f}% sortino={row['sortino']:+.2f} "
              f"DD={row['max_drawdown_pct']:+7.2f}% neg={row['neg_day_frac']*100:4.1f}% "
              f"turnover={row.get('mean_turnover', 0):.2f} fees={row.get('total_fee_cost_pct', 0):.2f}%",
              flush=True)
    df = pd.DataFrame(rows).sort_values(
        ["median_monthly_return_pct", "worst_monthly_drawdown_pct", "sortino"],
        ascending=[False, False, False],
    )
    csv_path = args.out / "_results.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nWrote {csv_path}")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
