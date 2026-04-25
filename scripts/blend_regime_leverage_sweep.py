"""Regime-aware leverage scaler on the asymmetric-leverage smoothness champion.

Hypothesis: P1 (2024Q4-2025Q1) had ann +9.65% / sortino +0.60 in the
smoothness champion, vs P2 +276 / +6.12. A simple regime detector that
scales leverage DOWN during P1-like flat regimes (trailing 21d
momentum near 0) and UP during P2-like rallies could lift the
flat-month median without hurting bull-regime returns.

Detector: trailing-N-day SUM of blend log-returns (lagged by 1 day to
avoid lookahead). Bins:
  bull   (trailing_sum >= τ_hi):   L = L_bull
  bear   (trailing_sum <= τ_lo):   L = L_bear (often <1.0 to de-leverage)
  neutral:                         L = L_mid

Apply per-day scalar to the asymmetric blend log return BEFORE the
global trailing stop. Compare goodness vs the 1-leverage-everywhere
smoothness champion across all 3 sub-periods and the full window.

Grid: (lookback, τ_lo, τ_hi, L_bear, L_mid, L_bull, pts, cd) ≈ 200 cells.
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


OUT_CSV = REPO / "analysis/cvar_portfolio/blend_regime_leverage_34mo.csv"

MOM_CELL = "k025_mom20_wmax0.15_Ltar1.00_ra1.00_conf0.950_cmin+0.00_hd05_fee10_slip05_asl15_psl00_ats12_pts00"
XGB_CELL = "k015_xgb_wmax0.15_Ltar1.00_ra1.00_conf0.950_cmin+0.00_hd03_fee10_slip05_asl15_psl00_ats10_pts00"

WMOM = 0.55
L_MOM_BASE = 1.15
L_XGB_BASE = 1.20
PTS = 11
CD = 21


def _apply_regime_leverage(
    blend: pd.Series, lookback: int, tau_lo: float, tau_hi: float,
    L_bear: float, L_mid: float, L_bull: float,
) -> pd.Series:
    """Scale daily log-return by per-day L (bear/mid/bull) using lagged
    trailing-sum signal. L=L_bear/L_mid/L_bull applied via lev_log = log1p(L*(exp(r)-1))."""
    sig = blend.rolling(lookback, min_periods=lookback).sum().shift(1)
    L = pd.Series(L_mid, index=blend.index, dtype=float)
    L[sig <= tau_lo] = L_bear
    L[sig >= tau_hi] = L_bull
    L = L.fillna(L_mid)
    lin = np.expm1(blend.values)
    lev_lin = np.clip(L.values * lin, -0.99, None)
    out = np.log1p(lev_lin)
    return pd.Series(out, index=blend.index)


def _slice(s: pd.Series, start: str, end: str) -> pd.Series:
    return s[(s.index >= pd.Timestamp(start)) & (s.index <= pd.Timestamp(end))]


PERIODS = [
    ("P1", "2024-09-16", "2025-03-31"),
    ("P2", "2025-04-01", "2025-09-30"),
    ("P3", "2025-10-01", "2026-04-30"),
]


def _eval(blend: pd.Series, lookback, tlo, thi, lb, lm, lh, pts, cd) -> dict:
    regime = _apply_regime_leverage(blend, lookback, tlo, thi, lb, lm, lh)
    stopped = _apply_portfolio_trailing_stop(regime, pts, cd)
    s = _summarize(stopped)
    out = {
        "lookback": lookback, "tlo": tlo, "thi": thi,
        "Lb": lb, "Lm": lm, "Lh": lh,
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

    rows = []
    # Baseline = no regime scaling (Lb=Lm=Lh=1.0 effectively).
    rows.append({"variant": "baseline_smooth_champ",
                 **_eval(blend, 21, 0.0, 0.0, 1.0, 1.0, 1.0, PTS, CD)})

    for lookback in [10, 21, 42]:
        for tlo in [-0.02, -0.01, 0.0]:
            for thi in [0.01, 0.02, 0.04]:
                if thi <= tlo:
                    continue
                for Lb in [0.50, 0.75, 1.00]:
                    for Lh in [1.00, 1.25, 1.50]:
                        # Lm = 1.0 (no change in neutral regime)
                        rows.append({"variant": "regime",
                                     **_eval(blend, lookback, tlo, thi,
                                             Lb, 1.0, Lh, PTS, CD)})

    df = pd.DataFrame(rows)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_CSV, index=False)
    print(f"\nWrote {len(df)} cells → {OUT_CSV}")

    base = df[df["variant"] == "baseline_smooth_champ"].iloc[0]
    print("\n=== BASELINE (no regime scaling) ===")
    print(f"  good={base['good']:+5.2f} med={base['med']:+5.2f} "
          f"sortino={base['sortino']:+5.2f} maxDD={base['maxdd']:+5.2f} "
          f"ann={base['ann']:+7.2f}")
    print(f"  P1 good={base.get('P1_good',np.nan):+5.2f} ann={base.get('P1_ann',np.nan):+7.2f} | "
          f"P2 good={base.get('P2_good',np.nan):+5.2f} ann={base.get('P2_ann',np.nan):+7.2f} | "
          f"P3 good={base.get('P3_good',np.nan):+5.2f} ann={base.get('P3_ann',np.nan):+7.2f}")

    grid = df[df["variant"] == "regime"].copy()
    grid["delta_good"] = grid["good"] - base["good"]
    grid["P1_delta"] = grid["P1_good"] - base["P1_good"]

    cols = ["lookback","tlo","thi","Lb","Lh",
            "med","p10","w21","maxdd","sortino","ann","good","delta_good",
            "P1_good","P1_delta","P2_good","P3_good"]

    print("\n=== TOP 10 by goodness (regime cells) ===")
    print(grid.sort_values("good", ascending=False).head(10)[cols]
          .to_string(index=False, float_format=lambda x: f"{x:+7.2f}"))

    print("\n=== TOP 10 by P1 lift (regime cells) ===")
    print(grid.sort_values("P1_delta", ascending=False).head(10)[cols]
          .to_string(index=False, float_format=lambda x: f"{x:+7.2f}"))

    pareto = grid[(grid["good"] >= base["good"]) & (grid["P1_good"] > base["P1_good"])]
    print(f"\n=== PARETO: improves both full-sample goodness AND P1 (n={len(pareto)}) ===")
    if len(pareto):
        print(pareto.sort_values("good", ascending=False).head(10)[cols]
              .to_string(index=False, float_format=lambda x: f"{x:+7.2f}"))


if __name__ == "__main__":
    main()
