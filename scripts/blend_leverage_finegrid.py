"""Fine-grid leverage × pts × cooldown sweep around the L=1.25 winner.

Refines `blend_leverage_pts_sweep.py` in a tighter neighborhood:
  wmom ∈ {0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70}
  L    ∈ {1.0, 1.10, 1.15, 1.20, 1.25, 1.30, 1.35, 1.40}
  pts  ∈ {8, 9, 10, 11, 12, 14, 16, 18, 20, 22}
  cd   ∈ {10, 15, 21}
= 1680 cells.
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


OUT_CSV = REPO / "analysis/cvar_portfolio/blend_leverage_finegrid_34mo.csv"


def main() -> None:
    mom_cell = "k025_mom20_wmax0.15_Ltar1.00_ra1.00_conf0.950_cmin+0.00_hd05_fee10_slip05_asl15_psl00_ats12_pts00"
    xgb_cell = "k015_xgb_wmax0.15_Ltar1.00_ra1.00_conf0.950_cmin+0.00_hd03_fee10_slip05_asl15_psl00_ats10_pts00"
    mom = _load_returns(MOM_BASE, mom_cell)
    xgb = _load_returns(XGB_BASE, xgb_cell)

    rows = []
    for wmom in [0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]:
        blend = _blend_log_returns(mom, xgb, wmom)
        for L in [1.0, 1.10, 1.15, 1.20, 1.25, 1.30, 1.35, 1.40]:
            lev = _leverage_log_ret(blend, L)
            for pts in [8, 9, 10, 11, 12, 14, 16, 18, 20, 22]:
                for cd in [10, 15, 21]:
                    stopped = _apply_portfolio_trailing_stop(lev, pts, cd)
                    s = _summarize(stopped)
                    rows.append({
                        "wmom": wmom, "L": L, "pts": pts, "cd": cd,
                        "med": s["median_monthly_return_pct"],
                        "p10": s["p10_monthly_return_pct"],
                        "w5": s["worst_5d_drawdown_pct"],
                        "w21": s["worst_21d_drawdown_pct"],
                        "w63": s["worst_63d_drawdown_pct"],
                        "maxdd": s["max_drawdown_pct"],
                        "sortino": s["sortino"],
                        "ann": s["ann_return_pct"],
                        "good": s["goodness_score"],
                    })
    df = pd.DataFrame(rows)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_CSV, index=False)
    print(f"Wrote {len(df)} rows → {OUT_CSV}")

    print("\nTop 10 by goodness, maxDD ≤ 25:")
    print(df[df["maxdd"].abs() <= 25].sort_values("good", ascending=False).head(10)
          .to_string(index=False, float_format=lambda x: f"{x:+7.2f}"))


if __name__ == "__main__":
    main()
