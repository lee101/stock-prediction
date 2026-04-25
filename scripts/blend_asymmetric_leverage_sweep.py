"""Asymmetric per-sleeve leverage on the 34mo blend champion.

Hypothesis: XGB solo sortino (+4.08) is 5× MOM solo (+0.76) → leveraging the
high-sortino sleeve more aggressively at the same effective leverage should
preserve median while improving sortino. Confirmed: produces the first
Pareto improvement over the symmetric L=1.25 champion.

Two phases:
  1. Coarse sweep: wmom ∈ {0.40-0.60} × L_mom ∈ {1.00-1.25} ×
     L_xgb ∈ {1.00-1.75} × pts ∈ {0,10,12,14,16,18,20} × cd ∈ {10,21}.
     Constraint L_xgb ≥ L_mom.
  2. Refine sweep: wmom ∈ {0.45-0.60} × L_mom ∈ {1.05-1.20} ×
     L_xgb ∈ {1.30-2.00} × pts ∈ {6-16} × cd ∈ {15,21,30}.

Smoothness champion (refine): wmom=0.55 L_mom=1.10 L_xgb=1.30 pts=11 cd=21
  = sortino +2.81, maxDD −16.21, ann +84.75, good +6.10.
Max-ann champion (refine): wmom=0.55 L_mom=1.20 L_xgb=1.40 pts=12 cd=21
  = ann +93.89, sortino +2.78, maxDD −17.50, good +6.01.
"""
from __future__ import annotations

import sys
from pathlib import Path

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
from scripts.blend_leverage_pts_sweep import _leverage_log_ret  # noqa: E402


COARSE_OUT = REPO / "analysis/cvar_portfolio/blend_asymmetric_leverage_34mo.csv"
REFINE_OUT = REPO / "analysis/cvar_portfolio/blend_asymmetric_leverage_refine_34mo.csv"

MOM_CELL = "k025_mom20_wmax0.15_Ltar1.00_ra1.00_conf0.950_cmin+0.00_hd05_fee10_slip05_asl15_psl00_ats12_pts00"
XGB_CELL = "k015_xgb_wmax0.15_Ltar1.00_ra1.00_conf0.950_cmin+0.00_hd03_fee10_slip05_asl15_psl00_ats10_pts00"


def _row(*, wmom, L_mom, L_xgb, pts, cd, mom, xgb):
    mom_lev = _leverage_log_ret(mom, L_mom)
    xgb_lev = _leverage_log_ret(xgb, L_xgb)
    blend = _blend_log_returns(mom_lev, xgb_lev, wmom)
    stopped = _apply_portfolio_trailing_stop(blend, pts, cd)
    s = _summarize(stopped)
    return {
        "wmom": wmom, "L_mom": L_mom, "L_xgb": L_xgb,
        "pts": pts, "cd": cd,
        "med": s["median_monthly_return_pct"],
        "p10": s["p10_monthly_return_pct"],
        "w5": s["worst_5d_drawdown_pct"],
        "w21": s["worst_21d_drawdown_pct"],
        "maxdd": s["max_drawdown_pct"],
        "sortino": s["sortino"],
        "ann": s["ann_return_pct"],
        "good": s["goodness_score"],
    }


def coarse(mom: pd.Series, xgb: pd.Series) -> pd.DataFrame:
    rows = []
    for wmom in [0.40, 0.45, 0.50, 0.55, 0.60]:
        for L_mom in [1.0, 1.10, 1.20, 1.25]:
            for L_xgb in [1.0, 1.10, 1.20, 1.30, 1.40, 1.50, 1.60, 1.75]:
                if L_xgb < L_mom:
                    continue
                for pts in [0, 10, 12, 14, 16, 18, 20]:
                    for cd in [10, 21]:
                        if pts == 0 and cd != 10:
                            continue
                        rows.append(_row(
                            wmom=wmom, L_mom=L_mom, L_xgb=L_xgb,
                            pts=pts, cd=cd, mom=mom, xgb=xgb,
                        ))
    return pd.DataFrame(rows)


def refine(mom: pd.Series, xgb: pd.Series) -> pd.DataFrame:
    rows = []
    for wmom in [0.45, 0.50, 0.55, 0.60]:
        for L_mom in [1.05, 1.10, 1.15, 1.20]:
            for L_xgb in [1.30, 1.40, 1.50, 1.60, 1.75, 2.00]:
                for pts in [6, 8, 10, 11, 12, 13, 14, 16]:
                    for cd in [15, 21, 30]:
                        rows.append(_row(
                            wmom=wmom, L_mom=L_mom, L_xgb=L_xgb,
                            pts=pts, cd=cd, mom=mom, xgb=xgb,
                        ))
    return pd.DataFrame(rows)


def _report(df: pd.DataFrame, label: str) -> None:
    print(f"\n=== {label}: TOP 10 by sortino, maxDD <= 25, good >= 5 ===")
    cols = ["wmom","L_mom","L_xgb","pts","cd","med","p10","w5","w21","maxdd","sortino","ann","good"]
    sup = df[(df["maxdd"].abs() <= 25) & (df["good"] >= 5.0)]
    print(sup.sort_values("sortino", ascending=False).head(10)[cols]
          .to_string(index=False, float_format=lambda x: f"{x:+7.2f}"))

    print(f"\n=== {label}: TOP 10 by goodness, maxDD <= 25 ===")
    print(df[df["maxdd"].abs() <= 25].sort_values("good", ascending=False).head(10)[cols]
          .to_string(index=False, float_format=lambda x: f"{x:+7.2f}"))


def main() -> None:
    mom = _load_returns(MOM_BASE, MOM_CELL)
    xgb = _load_returns(XGB_BASE, XGB_CELL)
    print(f"MOM: {len(mom)} days; XGB: {len(xgb)} days")

    COARSE_OUT.parent.mkdir(parents=True, exist_ok=True)
    coarse_df = coarse(mom, xgb)
    coarse_df.to_csv(COARSE_OUT, index=False)
    print(f"Wrote {len(coarse_df)} coarse cells → {COARSE_OUT}")
    _report(coarse_df, "COARSE")

    refine_df = refine(mom, xgb)
    refine_df.to_csv(REFINE_OUT, index=False)
    print(f"\nWrote {len(refine_df)} refine cells → {REFINE_OUT}")
    _report(refine_df, "REFINE")

    print("\nReference symmetric champion: wmom=0.55 L=1.25 pts=12 cd=21")
    print("    = good +6.37, ann +88.26, maxDD -20.65, sortino +2.66")


if __name__ == "__main__":
    main()
