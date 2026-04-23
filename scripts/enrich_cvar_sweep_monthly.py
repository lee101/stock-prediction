"""Recompute monthly stats post-hoc for a CVaR sweep directory.

When backtest.py's monthly-stats helper lands mid-sweep, cached summaries
miss the monthly columns. This reads each cell's daily_returns.csv,
recomputes median/p10/worst/neg monthly and monthly DD, and emits an
enriched `_results_monthly.csv` next to the sweep's `_results.csv`.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def _max_drawdown(cum_log_ret: np.ndarray) -> float:
    equity = np.exp(cum_log_ret)
    peak = np.maximum.accumulate(equity)
    dd = (equity - peak) / peak
    return float(dd.min())


def _monthly_stats(port_rets: pd.Series) -> dict:
    if len(port_rets) == 0:
        return {"n_months": 0, "median_monthly_return_pct": 0.0,
                "p10_monthly_return_pct": 0.0, "worst_monthly_return_pct": 0.0,
                "neg_months": 0, "worst_monthly_drawdown_pct": 0.0}
    idx = port_rets.index
    if getattr(idx, "tz", None) is not None:
        idx = idx.tz_localize(None)
    port_rets = pd.Series(port_rets.values, index=idx)
    g = port_rets.groupby(idx.to_period("M"))
    monthly_ret = g.sum().map(lambda x: (np.exp(float(x)) - 1.0) * 100.0)
    monthly_dd = g.apply(lambda s: _max_drawdown(s.cumsum().values) * 100.0)
    return {
        "n_months": int(len(monthly_ret)),
        "median_monthly_return_pct": float(np.median(monthly_ret.values)),
        "p10_monthly_return_pct": float(np.quantile(monthly_ret.values, 0.10)),
        "worst_monthly_return_pct": float(np.min(monthly_ret.values)),
        "neg_months": int((monthly_ret < 0.0).sum()),
        "worst_monthly_drawdown_pct": float(np.min(monthly_dd.values)),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep-dir", type=Path, required=True)
    args = ap.parse_args()

    base_csv = args.sweep_dir / "_results.csv"
    df = pd.read_csv(base_csv)

    rows = []
    for _, row in df.iterrows():
        cell = row["cell"]
        daily_path = args.sweep_dir / cell / "daily_returns.csv"
        if not daily_path.exists():
            continue
        daily = pd.read_csv(daily_path, index_col=0, parse_dates=True)
        series = daily.iloc[:, 0]
        m = _monthly_stats(series)
        rows.append({**row.to_dict(), **m,
                     "median_monthly_pct_check": float((np.exp(row["ann_return_pct"] / 100.0 + 1e-12)) ** (1 / 12) - 1)})

    out = pd.DataFrame(rows)
    out = out.sort_values(
        ["median_monthly_return_pct", "worst_monthly_drawdown_pct", "sortino"],
        ascending=[False, False, False],
    )
    out_path = args.sweep_dir / "_results_monthly.csv"
    out.to_csv(out_path, index=False)
    cols = ["cell", "ann_return_pct", "sortino", "max_drawdown_pct",
            "median_monthly_return_pct", "p10_monthly_return_pct",
            "worst_monthly_return_pct", "worst_monthly_drawdown_pct",
            "neg_months", "n_months", "mean_turnover" if "mean_turnover" in out else "ann_vol_pct"]
    cols = [c for c in cols if c in out.columns]
    print(f"Wrote {out_path}")
    print(out[cols].to_string(index=False))


if __name__ == "__main__":
    main()
