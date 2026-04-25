"""Volatility-targeting on the asymmetric-leverage smoothness champion.

Hypothesis: instead of regime-classifying via trailing momentum (refuted),
scale daily leverage by target_vol / realized_vol of the blend itself.
When realized vol is low (calm regime), the multiplier is >1 → leverage
up. When realized vol spikes (chop / crash), multiplier <1 → de-leverage
automatically. Symmetric in direction; doesn't need to predict the regime.

Apply ON TOP of the smoothness champion's L_mom=1.15 / L_xgb=1.20 base.
Compare goodness vs no-vol-target across all 3 sub-periods.

Lever:
  σ_t   = sqrt(252) * std(blend[t-N..t-1])     (lagged 1 day, no lookahead)
  m_t   = clip(τ / σ_t, m_min, m_max)
  out_t = log1p(m_t * (exp(blend_t) - 1))      (re-applied via log-leverage)

Grid:
  N (window) ∈ {21, 42, 63}
  τ (target ann vol) ∈ {0.15, 0.20, 0.25, 0.30, 0.35}
  m_min ∈ {0.5, 0.75}
  m_max ∈ {1.25, 1.5, 1.75, 2.0}
  → 3 × 5 × 2 × 4 = 120 cells, plus baseline.
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


OUT_CSV = REPO / "analysis/cvar_portfolio/blend_vol_target_34mo.csv"

MOM_CELL = "k025_mom20_wmax0.15_Ltar1.00_ra1.00_conf0.950_cmin+0.00_hd05_fee10_slip05_asl15_psl00_ats12_pts00"
XGB_CELL = "k015_xgb_wmax0.15_Ltar1.00_ra1.00_conf0.950_cmin+0.00_hd03_fee10_slip05_asl15_psl00_ats10_pts00"

WMOM = 0.55
L_MOM_BASE = 1.15
L_XGB_BASE = 1.20
PTS = 11
CD = 21

PERIODS = [
    ("P1", "2024-09-16", "2025-03-31"),
    ("P2", "2025-04-01", "2025-09-30"),
    ("P3", "2025-10-01", "2026-04-30"),
]


def _apply_vol_target(blend: pd.Series, N: int, tau: float, mmin: float, mmax: float) -> pd.Series:
    sigma = blend.rolling(N, min_periods=N).std().shift(1) * np.sqrt(252)
    mult = (tau / sigma).clip(lower=mmin, upper=mmax).fillna(1.0)
    lin = np.expm1(blend.values)
    lev_lin = np.clip(mult.values * lin, -0.99, None)
    return pd.Series(np.log1p(lev_lin), index=blend.index)


def _slice(s: pd.Series, start: str, end: str) -> pd.Series:
    return s[(s.index >= pd.Timestamp(start)) & (s.index <= pd.Timestamp(end))]


def _eval(blend: pd.Series, N, tau, mmin, mmax) -> dict:
    if N is None:  # baseline = no vol target
        scaled = blend.copy()
    else:
        scaled = _apply_vol_target(blend, N, tau, mmin, mmax)
    stopped = _apply_portfolio_trailing_stop(scaled, PTS, CD)
    s = _summarize(stopped)
    out = {
        "N": N if N is not None else 0,
        "tau": tau, "mmin": mmin, "mmax": mmax,
        "med": s["median_monthly_return_pct"],
        "p10": s["p10_monthly_return_pct"],
        "w5": s["worst_5d_drawdown_pct"],
        "w21": s["worst_21d_drawdown_pct"],
        "maxdd": s["max_drawdown_pct"],
        "sortino": s["sortino"],
        "ann": s["ann_return_pct"],
        "good": s["goodness_score"],
    }
    for label, start, end in PERIODS:
        sub = _slice(stopped, start, end)
        if len(sub) < 20:
            continue
        s_sub = _summarize(sub)
        out[f"{label}_med"] = s_sub["median_monthly_return_pct"]
        out[f"{label}_sortino"] = s_sub["sortino"]
        out[f"{label}_ann"] = s_sub["ann_return_pct"]
        out[f"{label}_good"] = s_sub["goodness_score"]
    return out


def main() -> None:
    mom = _load_returns(MOM_BASE, MOM_CELL)
    xgb = _load_returns(XGB_BASE, XGB_CELL)
    print(f"MOM/XGB days: {len(mom)} [{mom.index[0].date()}..{mom.index[-1].date()}]")

    mom_lev = _leverage_log_ret(mom, L_MOM_BASE)
    xgb_lev = _leverage_log_ret(xgb, L_XGB_BASE)
    blend = _blend_log_returns(mom_lev, xgb_lev, WMOM)

    blend_realized_vol = blend.std() * np.sqrt(252)
    print(f"Blend realized ann vol (full sample): {blend_realized_vol*100:.2f}%")

    rows = []
    rows.append({"variant": "baseline", **_eval(blend, None, 0.0, 0.0, 0.0)})

    for N in [21, 42, 63]:
        for tau in [0.15, 0.20, 0.25, 0.30, 0.35]:
            for mmin in [0.5, 0.75]:
                for mmax in [1.25, 1.5, 1.75, 2.0]:
                    rows.append({"variant": "voltgt", **_eval(blend, N, tau, mmin, mmax)})

    df = pd.DataFrame(rows)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_CSV, index=False)
    print(f"\nWrote {len(df)} cells → {OUT_CSV}")

    base = df[df["variant"] == "baseline"].iloc[0]
    print("\n=== BASELINE (no vol target) ===")
    print(f"  good={base['good']:+5.2f} med={base['med']:+5.2f} "
          f"sortino={base['sortino']:+5.2f} maxDD={base['maxdd']:+5.2f} "
          f"ann={base['ann']:+7.2f}")
    print(f"  P1 good={base.get('P1_good',np.nan):+5.2f} ann={base.get('P1_ann',np.nan):+7.2f} | "
          f"P2 good={base.get('P2_good',np.nan):+5.2f} ann={base.get('P2_ann',np.nan):+7.2f} | "
          f"P3 good={base.get('P3_good',np.nan):+5.2f} ann={base.get('P3_ann',np.nan):+7.2f}")

    grid = df[df["variant"] == "voltgt"].copy()
    grid["delta_good"] = grid["good"] - base["good"]
    grid["P1_delta"] = grid["P1_good"] - base["P1_good"]

    cols = ["N","tau","mmin","mmax",
            "med","p10","w21","maxdd","sortino","ann","good","delta_good",
            "P1_good","P1_delta","P2_good","P3_good"]

    print("\n=== TOP 10 by goodness ===")
    print(grid.sort_values("good", ascending=False).head(10)[cols]
          .to_string(index=False, float_format=lambda x: f"{x:+7.2f}"))

    print("\n=== TOP 10 by ann% (DD-compliant maxdd≥-25) ===")
    compliant = grid[grid["maxdd"].abs() <= 25]
    print(compliant.sort_values("ann", ascending=False).head(10)[cols]
          .to_string(index=False, float_format=lambda x: f"{x:+7.2f}"))

    print("\n=== TOP 10 by sortino ===")
    print(grid.sort_values("sortino", ascending=False).head(10)[cols]
          .to_string(index=False, float_format=lambda x: f"{x:+7.2f}"))

    pareto = grid[(grid["good"] >= base["good"]) & (grid["P1_good"] >= base["P1_good"])
                  & (grid["P3_good"] >= base["P3_good"]) & (grid["maxdd"].abs() <= 25)]
    print(f"\n=== PARETO: matches/improves baseline good AND P1 AND P3, DD-compliant (n={len(pareto)}) ===")
    if len(pareto):
        print(pareto.sort_values("good", ascending=False).head(15)[cols]
              .to_string(index=False, float_format=lambda x: f"{x:+7.2f}"))


if __name__ == "__main__":
    main()
