"""Portfolio trailing stop sweep on the 34mo blend champion equity curve.

Goal: cap blend maxDD (currently −29.65%) to ≤25% while preserving goodness
(currently +5.59 = first positive composite goodness across CVaR sweeps).

Approach:
  1. Load top-K blend cells from `blend_grid_mom_xgb_34mo.csv` (already
     contains pre-computed metrics for 1920 cells).
  2. For each cell, load the underlying mom + xgb daily log-return series.
  3. Compute log-blend: r_t = log(wM*exp(mom_t) + wX*exp(xgb_t)).
  4. Apply a global trailing stop on the blended equity curve: if peak-to-
     current drawdown exceeds `pts%`, force returns to 0 (cash) for the
     next `cooldown_days` trading days, then re-enter and reset peak.
  5. Recompute monthly + rolling-DD metrics + goodness. Compare to baseline.

Picks: best-goodness cell × (pts, cooldown) combo with maxDD ≤ 25%.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from cvar_portfolio.backtest import (  # noqa: E402
    _goodness_score,
    _max_drawdown,
    _monthly_stats,
    _rolling_drawdown_stats,
    _sortino,
)


MOM_BASE = REPO / "analysis/cvar_portfolio/sweep_wide_momentum_phase3"
XGB_BASE = REPO / "analysis/cvar_portfolio/sweep_wide_xgb_34mo"
BLEND_GRID = REPO / "analysis/cvar_portfolio/blend_grid_mom_xgb_34mo.csv"
OUT_CSV = REPO / "analysis/cvar_portfolio/blend_portfolio_trailing_stop_34mo.csv"


def _load_returns(base: Path, cell_name: str) -> pd.Series:
    df = pd.read_csv(base / cell_name / "daily_returns.csv")
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True).dt.tz_localize(None)
    return df.set_index("timestamp")["log_ret"]


def _blend_log_returns(mom: pd.Series, xgb: pd.Series, wmom: float) -> pd.Series:
    aligned = pd.concat([mom.rename("mom"), xgb.rename("xgb")], axis=1).dropna()
    blended_lin = wmom * np.exp(aligned["mom"].values) + (1 - wmom) * np.exp(aligned["xgb"].values)
    return pd.Series(np.log(blended_lin), index=aligned.index)


def _apply_portfolio_trailing_stop(
    log_rets: pd.Series, pts_pct: float, cooldown_days: int
) -> pd.Series:
    """Global peak-to-current trailing stop on equity curve.

    When equity drops `pts_pct%` below its running peak, set returns to 0
    for `cooldown_days` trading days. After cooldown, re-enter and reset
    peak to current equity.
    """
    if pts_pct <= 0:
        return log_rets.copy()
    out = log_rets.copy().values
    pts = pts_pct / 100.0
    cum = 0.0
    peak_cum = 0.0
    cooldown = 0
    for i in range(len(out)):
        if cooldown > 0:
            out[i] = 0.0
            cooldown -= 1
            if cooldown == 0:
                # reset peak to current cum (which is unchanged across cash days)
                peak_cum = cum
            continue
        cum += out[i]
        peak_cum = max(peak_cum, cum)
        # equity drawdown = exp(cum) / exp(peak_cum) - 1 = exp(cum - peak_cum) - 1
        dd = float(np.exp(cum - peak_cum) - 1.0)
        if dd <= -pts:
            # pull out today: realize the drop, then go to cash for cooldown
            cooldown = max(0, cooldown_days)
            # after-stop: peak resets at cooldown end (or immediately if cooldown=0)
            if cooldown == 0:
                peak_cum = cum
    return pd.Series(out, index=log_rets.index)


def _summarize(port_rets: pd.Series) -> dict:
    cum = port_rets.cumsum()
    monthly = _monthly_stats(port_rets)
    summary = {
        "n_days": int(len(port_rets)),
        "ann_return_pct": float((np.exp(port_rets.mean() * 252) - 1) * 100),
        "sortino": _sortino(port_rets.values),
        "max_drawdown_pct": _max_drawdown(cum.values) * 100,
    }
    summary.update(monthly)
    summary.update(_rolling_drawdown_stats(port_rets))
    summary["goodness_score"] = _goodness_score(summary)
    return summary


def main() -> None:
    grid = pd.read_csv(BLEND_GRID)
    # Top blend cells by goodness — apply trailing stops to each, look for
    # best (pts × cooldown) that holds maxDD ≤ 25 with highest goodness.
    top_cells = grid.sort_values("good", ascending=False).head(15).reset_index(drop=True)
    print(f"Sweeping {len(top_cells)} top blend cells × pts × cooldown grid")

    pts_grid = [0, 10, 12, 15, 18, 20, 22, 25, 30]
    cooldown_grid = [0, 3, 5, 10, 21]

    rows = []
    for _, cell in top_cells.iterrows():
        mom_ret = _load_returns(MOM_BASE, cell["mom"])
        xgb_ret = _load_returns(XGB_BASE, cell["xgb"])
        blended = _blend_log_returns(mom_ret, xgb_ret, float(cell["wmom"]))
        for pts in pts_grid:
            for cd in cooldown_grid:
                if pts == 0 and cd != 0:
                    continue  # no stop = no cooldown sweep
                stopped = _apply_portfolio_trailing_stop(blended, pts, cd)
                summary = _summarize(stopped)
                rows.append({
                    "mom": cell["mom"],
                    "xgb": cell["xgb"],
                    "wmom": float(cell["wmom"]),
                    "pts": pts,
                    "cooldown": cd,
                    "med": summary["median_monthly_return_pct"],
                    "p10": summary["p10_monthly_return_pct"],
                    "w5": summary["worst_5d_drawdown_pct"],
                    "w21": summary["worst_21d_drawdown_pct"],
                    "w63": summary["worst_63d_drawdown_pct"],
                    "frac21": summary["frac_21d_dd_gt_25pct"],
                    "maxdd": summary["max_drawdown_pct"],
                    "sortino": summary["sortino"],
                    "ann": summary["ann_return_pct"],
                    "n_months": summary["n_months"],
                    "neg_months": summary["neg_months"],
                    "good": summary["goodness_score"],
                })

    df = pd.DataFrame(rows)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_CSV, index=False)
    print(f"\nWrote {len(df)} rows → {OUT_CSV}")

    # Filter to maxDD-compliant + report top.
    compliant = df[df["maxdd"].abs() <= 25].sort_values("good", ascending=False).head(15)
    print("\nTop 15 maxDD ≤ 25% cells:")
    print(compliant[["wmom", "pts", "cooldown", "med", "p10", "w21", "maxdd", "sortino", "ann", "good"]]
          .to_string(index=False, float_format=lambda x: f"{x:+7.2f}"))

    print("\nBaseline (pts=0):")
    base = df[df["pts"] == 0].sort_values("good", ascending=False).head(5)
    print(base[["wmom", "med", "p10", "w21", "maxdd", "sortino", "ann", "good"]]
          .to_string(index=False, float_format=lambda x: f"{x:+7.2f}"))


if __name__ == "__main__":
    main()
