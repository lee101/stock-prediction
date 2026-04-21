"""Diagnostic: does cross-sectional dispersion predict regime direction?

Four inference-side levers have been eliminated on the 2025-07→2026-04
true-OOS (band-pass vol, SPY-vta, per-pick inv-vol, momentum-rank
filters). The previous `--invert-scores` experiment showed that the
ensemble still has rank-order info but its DIRECTION is inverted in
the tariff-crash regime.

SPY-based regime signals (20d realized vol, 200d MA) don't fire because
SPY itself stayed calm. This script tests CROSS-SECTIONAL regime
signals instead:

  * cs_iqr_ret5  = per-day IQR of ret_5d across the universe
  * cs_iqr_ret20 = per-day IQR of ret_20d across the universe
  * cs_std_ret5  = per-day std of ret_5d
  * cs_skew_ret5 = per-day skewness of ret_5d
  * breadth_dn   = % of universe with ret_5d < −2%
  * breadth_up   = % with ret_5d > +2%

For each signal, stratify top-1/day picks by its quartile and measure
mean target_oc per bucket. If one quartile shows strongly positive
target_oc and the complement shows strongly negative, a "trade only
when CS-signal is in bucket X" gate is a candidate regime mask.

Usage:
    python -m scripts.xgb_diagnose_cs_dispersion_regime
"""
from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from xgbnew.dataset import build_daily_dataset  # noqa: E402
from xgbnew.model import XGBStockModel  # noqa: E402


def _load_symbols(path: Path) -> list[str]:
    out: list[str] = []
    with open(path) as f:
        for ln in f:
            s = ln.strip()
            if s and not s.startswith("#"):
                out.append(s)
    return out


def main() -> int:
    sym_file = REPO / "symbol_lists/stocks_wide_1000_v1.txt"
    models_dir = REPO / "analysis/xgbnew_daily/oos2025h1_ensemble_gpu_fresh"
    model_paths = sorted(models_dir.glob("alltrain_seed*.pkl"))
    if not model_paths:
        print(f"ERR: no ensemble pkl at {models_dir}")
        return 1

    symbols = _load_symbols(sym_file)
    print(f"[cs-diag] loading {len(model_paths)} ensemble members, "
          f"{len(symbols)} symbols")

    train_df, _, oos_df = build_daily_dataset(
        data_root=REPO / "trainingdata",
        symbols=symbols,
        train_start=date(2020, 1, 1), train_end=date(2025, 6, 30),
        val_start=date(2025, 7, 1), val_end=date(2026, 4, 20),
        test_start=date(2025, 7, 1), test_end=date(2026, 4, 20),
        min_dollar_vol=50_000_000,
        fast_features=False,
    )
    print(f"[cs-diag] oos rows: {len(oos_df):,}")

    # Compute per-day cross-sectional regime signals from the FULL pool
    # (before any filtering — these are regime features, not pick features).
    daily_cs = oos_df.groupby("date").agg(
        cs_iqr_ret5=("ret_5d",  lambda s: float(s.quantile(0.75) - s.quantile(0.25))),
        cs_iqr_ret20=("ret_20d", lambda s: float(s.quantile(0.75) - s.quantile(0.25))),
        cs_std_ret5=("ret_5d",  "std"),
        cs_skew_ret5=("ret_5d", "skew"),
        breadth_dn=("ret_5d",  lambda s: float((s < -0.02).mean())),
        breadth_up=("ret_5d",  lambda s: float((s >  0.02).mean())),
        cs_mean_ret5=("ret_5d", "mean"),
    ).reset_index()

    print("\n[cs-diag] regime signal distributions across "
          f"{len(daily_cs)} OOS days:")
    for col in ["cs_iqr_ret5", "cs_iqr_ret20", "cs_std_ret5",
                "cs_skew_ret5", "breadth_dn", "breadth_up", "cs_mean_ret5"]:
        s = daily_cs[col]
        print(f"  {col:>18}  mean={s.mean():+.4f}  std={s.std():.4f}  "
              f"q10={s.quantile(0.10):+.4f}  q50={s.median():+.4f}  "
              f"q90={s.quantile(0.90):+.4f}  max={s.max():+.4f}")

    # Load ensemble + blend.
    models = [XGBStockModel.load(p) for p in model_paths]
    mat = np.stack([m.predict_scores(oos_df).values for m in models], axis=0)
    blended = mat.mean(axis=0)
    oos_df = oos_df.assign(score=blended).copy()

    # Apply live inference gate (50M dolvol + vol>=0.10).
    mask = (
        (np.exp(oos_df["dolvol_20d_log"]) - 1.0 >= 50_000_000)
        & (oos_df["vol_20d"] >= 0.10)
    )
    pool = oos_df[mask].copy()
    print(f"\n[cs-diag] pool after 50M+vol0.10: {len(pool):,} rows")

    # Take top-1/day.
    pool_sorted = pool.sort_values(["date", "score"], ascending=[True, False])
    top1 = pool_sorted.groupby("date", group_keys=False).head(1).copy()
    top1 = top1.merge(daily_cs, on="date", how="left")
    print(f"[cs-diag] top-1/day picks: {len(top1):,}")

    print(f"\n[top1 overall]  mean target_oc = {top1['target_oc'].mean():+.4%}"
          f"  hit = {(top1['target_oc'] > 0).mean():.1%}")

    # For each regime signal, stratify picks by its quartile.
    signals = ["cs_iqr_ret5", "cs_iqr_ret20", "cs_std_ret5",
               "cs_skew_ret5", "breadth_dn", "breadth_up", "cs_mean_ret5"]
    print("\n[stratified top-1/day mean target_oc by daily-regime-signal "
          "quartile]")
    print(f"  {'signal':>16}  {'Q1 (low)':>12} {'Q2':>12} "
          f"{'Q3':>12} {'Q4 (high)':>12}  {'Q4-Q1':>10}")
    best_spreads = []
    for sig in signals:
        q = pd.qcut(top1[sig], 4, duplicates="drop")
        stats = top1.groupby(q, observed=True)["target_oc"].agg(["mean", "count"])
        means = stats["mean"].tolist()
        while len(means) < 4:
            means.append(np.nan)
        spread = means[3] - means[0] if not any(np.isnan(m) for m in means) else np.nan
        line = f"  {sig:>16}"
        for m in means[:4]:
            line += f"  {m:+.4%}" if not np.isnan(m) else "         NaN"
        line += f"  {spread:+.4%}" if not np.isnan(spread) else "       NaN"
        print(line)
        if not np.isnan(spread):
            best_spreads.append((sig, spread, means[0], means[3]))

    # ── Counterfactual: if we gate ON the most-separating signal,
    # what does top-1/day mean target_oc look like over the "trade"
    # days vs "skip" days?
    print("\n[counterfactual — trade only when signal is in specified bucket]")
    print(f"  {'filter':>50}  {'n_days':>7}  {'mean_pnl':>10}  "
          f"{'hit%':>6}  {'cum_return':>10}")

    def _simulate_gate(mask_rows: pd.Series, label: str, n_days: int):
        if int(mask_rows.sum()) == 0:
            print(f"  {label:>50}  {0:7d}  {'   N/A  ':>10}  "
                  f"{'   N/A':>6}  {'    N/A':>10}")
            return
        sub = top1[mask_rows]
        mean = sub["target_oc"].mean()
        hit = (sub["target_oc"] > 0).mean()
        # Naive cum-return across trade days (geometric).
        cum = float((1.0 + sub["target_oc"]).prod() - 1.0)
        print(f"  {label:>50}  {len(sub):7d}  {mean:+10.4%}  {hit:6.1%}  "
              f"{cum:+10.2%}")

    total_days = len(top1)
    _simulate_gate(pd.Series(True, index=top1.index),
                   "ALL DAYS (baseline)", total_days)
    for sig in signals:
        q25 = top1[sig].quantile(0.25)
        q50 = top1[sig].quantile(0.50)
        q75 = top1[sig].quantile(0.75)
        _simulate_gate(top1[sig] <= q25, f"{sig} <= Q25 ({q25:+.3f})",
                       total_days)
        _simulate_gate(top1[sig] >= q75, f"{sig} >= Q75 ({q75:+.3f})",
                       total_days)
        # For skew, also test the middle-half (avoid both tails).
        if sig == "cs_skew_ret5":
            _simulate_gate((top1[sig] > q25) & (top1[sig] < q75),
                           f"{sig} IQR middle (avoid extreme skew)", total_days)

    # ── Combined gates: AND / OR of the two best single signals.
    print("\n[combined gates — cs_iqr_ret5 AND cs_skew_ret5]")
    print(f"  {'filter':>55}  {'n_days':>7}  {'mean_pnl':>10}  "
          f"{'hit%':>6}  {'cum_return':>10}")
    for iqr_tight, iqr_label in [
        (top1["cs_iqr_ret5"] <= top1["cs_iqr_ret5"].quantile(0.25), "iqr<=Q25"),
        (top1["cs_iqr_ret5"] <= top1["cs_iqr_ret5"].quantile(0.50), "iqr<=Q50"),
    ]:
        for skew_ok, skew_label in [
            (top1["cs_skew_ret5"] >= top1["cs_skew_ret5"].quantile(0.25), "skew>=Q25"),
            (top1["cs_skew_ret5"] >= top1["cs_skew_ret5"].quantile(0.50), "skew>=Q50"),
            (top1["cs_skew_ret5"] >= 0, "skew>=0 (right-skew only)"),
        ]:
            m = iqr_tight & skew_ok
            _simulate_gate(m, f"{iqr_label} AND {skew_label}", total_days)
    # Union too — wider set.
    _simulate_gate(
        (top1["cs_iqr_ret5"] <= top1["cs_iqr_ret5"].quantile(0.50))
        | (top1["cs_skew_ret5"] >= top1["cs_skew_ret5"].quantile(0.50)),
        "iqr<=Q50 OR skew>=Q50 (wider trade set)", total_days,
    )

    # ── CAUSAL rolling-threshold gate — critical for live deploy.
    # On day T, compute the Q25/Q50 threshold using ONLY days
    # [T - rolling_window, T - 1]. No lookahead.
    print("\n[CAUSAL rolling-threshold gate — no lookahead]")
    print(f"  {'filter':>55}  {'n_days':>7}  {'mean_pnl':>10}  "
          f"{'hit%':>6}  {'cum_return':>10}")
    top1_sorted = top1.sort_values("date").reset_index(drop=True).copy()
    for window in [60, 90, 120]:
        # Compute rolling Q25 / Q50 of cs_iqr_ret5 across PRIOR days.
        iqr_vals = top1_sorted["cs_iqr_ret5"].values
        skew_vals = top1_sorted["cs_skew_ret5"].values
        roll_iqr_q25 = np.full_like(iqr_vals, np.nan, dtype=float)
        roll_iqr_q50 = np.full_like(iqr_vals, np.nan, dtype=float)
        roll_skew_q50 = np.full_like(skew_vals, np.nan, dtype=float)
        for i in range(len(iqr_vals)):
            lo = max(0, i - window)
            if i - lo < 30:  # need some history
                continue
            roll_iqr_q25[i] = np.nanquantile(iqr_vals[lo:i], 0.25)
            roll_iqr_q50[i] = np.nanquantile(iqr_vals[lo:i], 0.50)
            roll_skew_q50[i] = np.nanquantile(skew_vals[lo:i], 0.50)

        # Gate: iqr_today <= rolling_Q50 AND skew_today >= rolling_Q50.
        gate_mask = (
            (top1_sorted["cs_iqr_ret5"] <= pd.Series(roll_iqr_q50, index=top1_sorted.index))
            & (top1_sorted["cs_skew_ret5"] >= pd.Series(roll_skew_q50, index=top1_sorted.index))
        ).fillna(False)
        _simulate_gate(gate_mask,
                       f"roll{window}d: iqr<=Q50 AND skew>=Q50", total_days)

        # Tighter: iqr_today <= rolling_Q25 AND skew_today >= rolling_Q50
        gate_mask = (
            (top1_sorted["cs_iqr_ret5"] <= pd.Series(roll_iqr_q25, index=top1_sorted.index))
            & (top1_sorted["cs_skew_ret5"] >= pd.Series(roll_skew_q50, index=top1_sorted.index))
        ).fillna(False)
        _simulate_gate(gate_mask,
                       f"roll{window}d: iqr<=Q25 AND skew>=Q50", total_days)
    # Simple single-signal causal gate (best single signal).
    for window in [60, 90, 120]:
        iqr_vals = top1_sorted["cs_iqr_ret5"].values
        roll_q25 = np.full_like(iqr_vals, np.nan, dtype=float)
        for i in range(len(iqr_vals)):
            lo = max(0, i - window)
            if i - lo < 30:
                continue
            roll_q25[i] = np.nanquantile(iqr_vals[lo:i], 0.25)
        gate_mask = (
            top1_sorted["cs_iqr_ret5"] <= pd.Series(roll_q25, index=top1_sorted.index)
        ).fillna(False)
        _simulate_gate(gate_mask, f"roll{window}d: iqr<=Q25 only", total_days)

    # Fixed absolute thresholds — simplest deploy.
    print("\n[fixed absolute thresholds — simplest deploy]")
    print(f"  {'filter':>55}  {'n_days':>7}  {'mean_pnl':>10}  "
          f"{'hit%':>6}  {'cum_return':>10}")
    for thr in [0.040, 0.042, 0.045, 0.048, 0.050]:
        _simulate_gate(top1["cs_iqr_ret5"] <= thr,
                       f"fixed: cs_iqr_ret5 <= {thr:.3f}", total_days)

    # ── Per-window / monthly breakdown for the best gate.
    print("\n[monthly breakdown — best gate = cs_iqr_ret5 <= Q25]")
    best_mask = top1["cs_iqr_ret5"] <= top1["cs_iqr_ret5"].quantile(0.25)
    gated = top1[best_mask].copy()
    gated["month"] = pd.to_datetime(gated["date"]).dt.to_period("M")
    monthly = gated.groupby("month").agg(
        n=("target_oc", "size"),
        mean=("target_oc", "mean"),
        hit=("target_oc", lambda s: float((s > 0).mean())),
        cum=("target_oc", lambda s: float((1 + s).prod() - 1)),
    )
    print(monthly.to_string())

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
