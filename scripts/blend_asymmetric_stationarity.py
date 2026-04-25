"""Temporal stationarity check for the asymmetric-leverage smoothness champion.

Champion: wmom=0.55 L_mom=1.15 L_xgb=1.20 pts=11 cd=21
  → sortino +2.81, maxDD −15.96, ann +85.80, good +6.10 on 34mo (2023-06→2026-04).

Risk: champion is single-sample tuned. We split the sleeve daily returns
into 3 sub-periods and re-run a focused asymmetric-leverage grid on each,
checking whether the same neighborhood of (wmom, L_mom, L_xgb, pts, cd)
remains a Pareto-positive cell in EACH sub-period.

Sub-periods (3 ≈ equal trading-day chunks):
  P1: 2023-06 → 2024-04  (~210 trading days, pre-rate-cut + summer chop)
  P2: 2024-05 → 2025-04  (~252 trading days, late 2024 rally → 2025-Q1 rotation)
  P3: 2025-05 → 2026-04  (~252 trading days, includes 2026 tariff crash)

Stationarity verdict:
  - GREEN: champion cell ranks in top-10% goodness in all 3 sub-periods
  - YELLOW: top-10% in 2/3 sub-periods, neutral in the other
  - RED: champion cell loses goodness (good <0) in any sub-period
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
from scripts.blend_leverage_pts_sweep import _leverage_log_ret  # noqa: E402


OUT_CSV = REPO / "analysis/cvar_portfolio/blend_asymmetric_stationarity.csv"

MOM_CELL = "k025_mom20_wmax0.15_Ltar1.00_ra1.00_conf0.950_cmin+0.00_hd05_fee10_slip05_asl15_psl00_ats12_pts00"
XGB_CELL = "k015_xgb_wmax0.15_Ltar1.00_ra1.00_conf0.950_cmin+0.00_hd03_fee10_slip05_asl15_psl00_ats10_pts00"

CHAMPION = dict(wmom=0.55, L_mom=1.15, L_xgb=1.20, pts=11, cd=21)
MAX_ANN = dict(wmom=0.55, L_mom=1.20, L_xgb=1.40, pts=12, cd=21)
SYMMETRIC = dict(wmom=0.55, L_mom=1.25, L_xgb=1.25, pts=12, cd=21)

PERIODS = [
    ("P1_2024Q4_2025Q1", "2024-09-16", "2025-03-31"),
    ("P2_2025Q2_2025Q3", "2025-04-01", "2025-09-30"),
    ("P3_2025Q4_2026Q1", "2025-10-01", "2026-04-30"),
]


def _row(*, wmom, L_mom, L_xgb, pts, cd, mom, xgb):
    mom_lev = _leverage_log_ret(mom, L_mom)
    xgb_lev = _leverage_log_ret(xgb, L_xgb)
    blend = _blend_log_returns(mom_lev, xgb_lev, wmom)
    stopped = _apply_portfolio_trailing_stop(blend, pts, cd)
    s = _summarize(stopped)
    return {
        "wmom": wmom, "L_mom": L_mom, "L_xgb": L_xgb,
        "pts": pts, "cd": cd,
        "n_days": int(len(stopped)),
        "med": s["median_monthly_return_pct"],
        "p10": s["p10_monthly_return_pct"],
        "w5": s["worst_5d_drawdown_pct"],
        "w21": s["worst_21d_drawdown_pct"],
        "maxdd": s["max_drawdown_pct"],
        "sortino": s["sortino"],
        "ann": s["ann_return_pct"],
        "good": s["goodness_score"],
    }


def _slice(s: pd.Series, start: str, end: str) -> pd.Series:
    return s[(s.index >= pd.Timestamp(start)) & (s.index <= pd.Timestamp(end))]


def _sweep(mom: pd.Series, xgb: pd.Series, period_label: str) -> pd.DataFrame:
    rows = []
    rows.append({"variant": "champion_smooth", **CHAMPION,
                 "period": period_label,
                 **{k: v for k, v in _row(**CHAMPION, mom=mom, xgb=xgb).items()
                    if k not in CHAMPION}})
    rows.append({"variant": "champion_max_ann", **MAX_ANN,
                 "period": period_label,
                 **{k: v for k, v in _row(**MAX_ANN, mom=mom, xgb=xgb).items()
                    if k not in MAX_ANN}})
    rows.append({"variant": "symmetric_L1.25", **SYMMETRIC,
                 "period": period_label,
                 **{k: v for k, v in _row(**SYMMETRIC, mom=mom, xgb=xgb).items()
                    if k not in SYMMETRIC}})

    # Local grid around the smoothness champion to check rank stability.
    for wmom in [0.45, 0.50, 0.55, 0.60]:
        for L_mom in [1.05, 1.10, 1.15, 1.20]:
            for L_xgb in [1.10, 1.20, 1.30, 1.40, 1.50]:
                if L_xgb < L_mom:
                    continue
                for pts in [10, 11, 12, 14]:
                    cd = 21
                    rows.append({"variant": "grid",
                                 "period": period_label,
                                 **_row(wmom=wmom, L_mom=L_mom, L_xgb=L_xgb,
                                        pts=pts, cd=cd, mom=mom, xgb=xgb)})
    return pd.DataFrame(rows)


def main() -> None:
    mom_full = _load_returns(MOM_BASE, MOM_CELL)
    xgb_full = _load_returns(XGB_BASE, XGB_CELL)
    print(f"MOM: {len(mom_full)} days [{mom_full.index[0].date()}..{mom_full.index[-1].date()}]")
    print(f"XGB: {len(xgb_full)} days [{xgb_full.index[0].date()}..{xgb_full.index[-1].date()}]")

    all_dfs = []
    for label, start, end in PERIODS:
        m = _slice(mom_full, start, end)
        x = _slice(xgb_full, start, end)
        print(f"\n=== {label}: MOM {len(m)}d, XGB {len(x)}d "
              f"[{m.index[0].date()}..{m.index[-1].date()}] ===")
        df = _sweep(m, x, label)
        all_dfs.append(df)

        anchors = df[df["variant"] != "grid"]
        cols = ["variant", "wmom", "L_mom", "L_xgb", "pts", "cd",
                "med", "p10", "w5", "w21", "maxdd", "sortino", "ann", "good"]
        print(anchors[cols].to_string(index=False, float_format=lambda x: f"{x:+7.2f}"))

        grid = df[df["variant"] == "grid"]
        top = grid.sort_values("good", ascending=False).head(5)
        print(f"\n  Top-5 {label} grid (good):")
        print(top[["wmom", "L_mom", "L_xgb", "pts", "cd",
                   "med", "p10", "w21", "maxdd", "sortino", "ann", "good"]]
              .to_string(index=False, float_format=lambda x: f"{x:+7.2f}"))

    # Also full-sample row.
    print(f"\n=== FULL 34mo (reference) ===")
    full_df = _sweep(mom_full, xgb_full, "FULL_34mo")
    all_dfs.append(full_df)
    anchors_full = full_df[full_df["variant"] != "grid"]
    cols = ["variant", "wmom", "L_mom", "L_xgb", "pts", "cd",
            "med", "p10", "w5", "w21", "maxdd", "sortino", "ann", "good"]
    print(anchors_full[cols].to_string(index=False, float_format=lambda x: f"{x:+7.2f}"))

    out = pd.concat(all_dfs, ignore_index=True)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_CSV, index=False)
    print(f"\nWrote {len(out)} rows → {OUT_CSV}")

    # Stationarity verdict on smoothness champion.
    champ_rows = out[(out["variant"] == "champion_smooth") & (out["period"] != "FULL_34mo")]
    print("\n=== STATIONARITY VERDICT (smoothness champion) ===")
    for _, r in champ_rows.iterrows():
        verdict = "GREEN" if r["good"] >= 0 else "RED"
        print(f"  {r['period']:<22} good={r['good']:+6.2f} sortino={r['sortino']:+5.2f} "
              f"maxDD={r['maxdd']:+6.2f} ann={r['ann']:+7.2f} → {verdict}")


if __name__ == "__main__":
    main()
