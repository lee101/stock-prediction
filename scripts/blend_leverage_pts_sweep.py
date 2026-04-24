"""Leverage × portfolio trailing stop sweep on blend champion.

Approach: post-hoc leverage on the blended daily returns. If the underlying
strategy is scaled by L (allocation knob in live trader), then:
    leveraged_lin_ret = L * (exp(r) - 1)
    leveraged_log_ret = log(1 + L * (exp(r) - 1))
                      = log(1 - L + L * exp(r))   (no NaN if 1-L+L*exp(r)>0)

For L>1, leveraged_log_ret blows up if r << 0 (1+L(e^r−1) → 0 → log → −inf).
We cap at log(0.01) = −4.6 (99% loss day) to keep numerics clean — that's
basically a margin call event in real trading.

Trailing stop fires on the LEVERAGED equity curve (correct model: the
broker would liquidate when the leveraged position drops too far).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts.blend_portfolio_trailing_stop_sweep import (  # noqa: E402
    _apply_portfolio_trailing_stop,
    _blend_log_returns,
    _load_returns,
    _summarize,
    MOM_BASE,
    XGB_BASE,
)


OUT_CSV = REPO / "analysis/cvar_portfolio/blend_leverage_pts_sweep_34mo.csv"


def _leverage_log_ret(log_ret: pd.Series, L: float) -> pd.Series:
    if L == 1.0:
        return log_ret.copy()
    lin = np.exp(log_ret.values) - 1.0
    lev_lin = L * lin
    # Floor at -0.99 to avoid log of zero/negative (margin call).
    lev_lin = np.clip(lev_lin, -0.99, None)
    lev_log = np.log1p(lev_lin)
    return pd.Series(lev_log, index=log_ret.index)


def main() -> None:
    # Top blend cells by 34mo baseline goodness (from blend_grid_mom_xgb_34mo.csv).
    top_cells = [
        # (mom_cell, xgb_cell, wmom, label)
        (
            "k025_mom20_wmax0.15_Ltar1.00_ra1.00_conf0.950_cmin+0.00_hd05_fee10_slip05_asl15_psl00_ats12_pts00",
            "k015_xgb_wmax0.15_Ltar1.00_ra1.00_conf0.950_cmin+0.00_hd03_fee10_slip05_asl15_psl00_ats10_pts00",
            0.70,
            "champion_w70",
        ),
        (
            "k025_mom20_wmax0.15_Ltar1.00_ra1.00_conf0.950_cmin+0.00_hd05_fee10_slip05_asl15_psl00_ats12_pts00",
            "k015_xgb_wmax0.15_Ltar1.00_ra1.00_conf0.950_cmin+0.00_hd03_fee10_slip05_asl15_psl00_ats10_pts00",
            0.60,
            "champion_w60",
        ),
        (
            "k025_mom20_wmax0.15_Ltar1.00_ra1.00_conf0.950_cmin+0.00_hd05_fee10_slip05_asl15_psl00_ats12_pts00",
            "k015_xgb_wmax0.15_Ltar1.00_ra1.00_conf0.950_cmin+0.00_hd03_fee10_slip05_asl15_psl00_ats10_pts00",
            0.50,
            "champion_w50",
        ),
        (
            "k025_mom20_wmax0.15_Ltar1.00_ra1.00_conf0.950_cmin+0.00_hd05_fee10_slip05_asl15_psl00_ats12_pts00",
            "k015_xgb_wmax0.15_Ltar1.00_ra1.00_conf0.950_cmin+0.00_hd03_fee10_slip05_asl15_psl00_ats10_pts00",
            0.80,
            "champion_w80",
        ),
    ]

    L_grid = [1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 3.0]
    pts_grid = [0, 8, 10, 12, 15, 18, 20, 25]
    cooldown_grid = [10, 21]

    rows = []
    for mom_cell, xgb_cell, wmom, label in top_cells:
        mom_ret = _load_returns(MOM_BASE, mom_cell)
        xgb_ret = _load_returns(XGB_BASE, xgb_cell)
        blend = _blend_log_returns(mom_ret, xgb_ret, wmom)
        for L in L_grid:
            lev = _leverage_log_ret(blend, L)
            for pts in pts_grid:
                for cd in cooldown_grid:
                    if pts == 0 and cd != 10:
                        continue  # no stop = no cooldown sweep
                    stopped = _apply_portfolio_trailing_stop(lev, pts, cd)
                    s = _summarize(stopped)
                    rows.append({
                        "label": label,
                        "wmom": wmom,
                        "L": L,
                        "pts": pts,
                        "cooldown": cd,
                        "med": s["median_monthly_return_pct"],
                        "p10": s["p10_monthly_return_pct"],
                        "w5": s["worst_5d_drawdown_pct"],
                        "w21": s["worst_21d_drawdown_pct"],
                        "w63": s["worst_63d_drawdown_pct"],
                        "frac21": s["frac_21d_dd_gt_25pct"],
                        "maxdd": s["max_drawdown_pct"],
                        "sortino": s["sortino"],
                        "ann": s["ann_return_pct"],
                        "n_months": s["n_months"],
                        "neg_months": s["neg_months"],
                        "good": s["goodness_score"],
                    })

    df = pd.DataFrame(rows)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_CSV, index=False)
    print(f"\nWrote {len(df)} rows → {OUT_CSV}")

    print("\n=== TOP 20 BY GOODNESS, ALL CELLS ===")
    top = df.sort_values("good", ascending=False).head(20)
    print(top[["label","wmom","L","pts","cooldown","med","p10","w21","maxdd","sortino","ann","good"]]
          .to_string(index=False, float_format=lambda x: f"{x:+7.2f}"))

    print("\n=== TOP 20 BY GOODNESS, maxDD ≤ 25 ===")
    compliant = df[df["maxdd"].abs() <= 25].sort_values("good", ascending=False).head(20)
    print(compliant[["label","wmom","L","pts","cooldown","med","p10","w21","maxdd","sortino","ann","good"]]
          .to_string(index=False, float_format=lambda x: f"{x:+7.2f}"))

    print("\n=== TOP 10 BY MEDIAN, maxDD ≤ 25 (push toward 25-30%/mo target) ===")
    compliant_by_med = df[df["maxdd"].abs() <= 25].sort_values("med", ascending=False).head(10)
    print(compliant_by_med[["label","wmom","L","pts","cooldown","med","p10","w21","maxdd","sortino","ann","good"]]
          .to_string(index=False, float_format=lambda x: f"{x:+7.2f}"))


if __name__ == "__main__":
    main()
