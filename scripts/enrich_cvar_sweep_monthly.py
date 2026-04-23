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


def _rolling_drawdown_stats(port_rets: pd.Series) -> dict:
    if len(port_rets) == 0:
        return {"worst_5d_drawdown_pct": 0.0, "worst_21d_drawdown_pct": 0.0,
                "worst_63d_drawdown_pct": 0.0, "frac_5d_dd_gt_10pct": 0.0,
                "frac_21d_dd_gt_25pct": 0.0, "frac_63d_dd_gt_40pct": 0.0}
    equity = np.exp(port_rets.cumsum().values)
    out = {}
    for win_days, label, thresh in [(5, "5d", 10.0), (21, "21d", 25.0), (63, "63d", 40.0)]:
        if len(equity) < win_days:
            out[f"worst_{label}_drawdown_pct"] = 0.0
            out[f"frac_{label}_dd_gt_{int(thresh)}pct"] = 0.0
            continue
        roll_peak = pd.Series(equity).rolling(win_days, min_periods=1).max().values
        dd = (equity - roll_peak) / roll_peak * 100.0
        window_min = pd.Series(dd).rolling(win_days, min_periods=1).min().values
        out[f"worst_{label}_drawdown_pct"] = float(np.min(window_min))
        out[f"frac_{label}_dd_gt_{int(thresh)}pct"] = float((window_min < -thresh).mean())
    return out


def _goodness_score(summary: dict) -> float:
    med = float(summary.get("median_monthly_return_pct", 0.0))
    w5 = abs(float(summary.get("worst_5d_drawdown_pct", 0.0)))
    w21 = abs(float(summary.get("worst_21d_drawdown_pct", 0.0)))
    w63 = abs(float(summary.get("worst_63d_drawdown_pct", 0.0)))
    frac21 = float(summary.get("frac_21d_dd_gt_25pct", 0.0))
    penalty = (2.0 * max(0.0, w21 - 25.0) + 1.5 * max(0.0, w5 - 10.0)
               + 0.5 * max(0.0, w63 - 40.0) + 5.0 * frac21 * 100.0)
    return float(med - penalty)


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
    if base_csv.exists():
        df = pd.read_csv(base_csv)
        base_rows = df.to_dict(orient="records")
    else:
        # Fallback: derive rows from each cell's summary.json (partial sweep).
        base_rows = []
        for cell_dir in sorted(p for p in args.sweep_dir.iterdir() if p.is_dir()):
            sp = cell_dir / "summary.json"
            if sp.exists():
                s = json.loads(sp.read_text())
                base_rows.append({"cell": cell_dir.name, **s})

    rows = []
    for row in base_rows:
        cell = row["cell"]
        daily_path = args.sweep_dir / cell / "daily_returns.csv"
        if not daily_path.exists():
            continue
        daily = pd.read_csv(daily_path, index_col=0, parse_dates=True)
        series = daily.iloc[:, 0]
        m = _monthly_stats(series)
        r = _rolling_drawdown_stats(series)
        enriched = {**row, **m, **r}
        enriched["goodness_score"] = _goodness_score(enriched)
        ann = row.get("ann_return_pct", 0.0)
        enriched["median_monthly_pct_check"] = float((np.exp(ann / 100.0 + 1e-12)) ** (1 / 12) - 1)
        rows.append(enriched)

    out = pd.DataFrame(rows)
    out = out.sort_values(
        ["goodness_score", "worst_21d_drawdown_pct", "sortino"],
        ascending=[False, False, False],
    )
    out_path = args.sweep_dir / "_results_monthly.csv"
    out.to_csv(out_path, index=False)
    cols = ["cell", "goodness_score", "median_monthly_return_pct",
            "p10_monthly_return_pct", "worst_21d_drawdown_pct",
            "worst_5d_drawdown_pct", "worst_63d_drawdown_pct",
            "frac_21d_dd_gt_25pct", "ann_return_pct", "sortino",
            "max_drawdown_pct", "mean_turnover"]
    cols = [c for c in cols if c in out.columns]
    print(f"Wrote {out_path}")
    print(out[cols].to_string(index=False))


if __name__ == "__main__":
    main()
