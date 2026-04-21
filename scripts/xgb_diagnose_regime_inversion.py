"""Diagnostic: on the inverted-regime OOS, which features predict the
top-N picks' realized direction?

The --invert-scores sweep showed:
  regular top-N long = negative median PnL
  inverted (bottom-N) long = positive median PnL

This script stratifies the TOP-1 daily pick by various features and
measures how each feature's quartile co-moves with realized target_oc.
If some feature stratifies strongly (e.g. high `price_vs_52w_high` →
mostly losers), that's a candidate inference-time mask.

Usage:
    python -m scripts.xgb_diagnose_regime_inversion
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
    print(f"[diag] loading {len(model_paths)} ensemble members, {len(symbols)} symbols")

    train_df, _, oos_df = build_daily_dataset(
        data_root=REPO / "trainingdata",
        symbols=symbols,
        train_start=date(2020, 1, 1), train_end=date(2025, 6, 30),
        val_start=date(2025, 7, 1), val_end=date(2026, 4, 20),
        test_start=date(2025, 7, 1), test_end=date(2026, 4, 20),
        min_dollar_vol=50_000_000,
        fast_features=False,
    )
    print(f"[diag] oos rows: {len(oos_df):,}")

    # Blend scores.
    models = [XGBStockModel.load(p) for p in model_paths]
    mat = np.stack([m.predict_scores(oos_df).values for m in models], axis=0)
    blended = mat.mean(axis=0)
    oos_df = oos_df.assign(score=blended).copy()

    # Apply deployed inference-side filter: min_dollar_vol=50M, min_vol_20d=0.10.
    mask = (
        np.exp(oos_df["dolvol_20d_log"]) - 1.0 >= 50_000_000
    ) & (oos_df["vol_20d"] >= 0.10)
    pool = oos_df[mask].copy()
    print(f"[diag] pool after 50M + vol>=0.10 gate: {len(pool):,} rows "
          f"({100*len(pool)/len(oos_df):.1f}%)")

    # For each day, take the TOP-5 picks (matches min_score-then-top-N path).
    pool_sorted = pool.sort_values(["date", "score"], ascending=[True, False])
    top5 = pool_sorted.groupby("date", group_keys=False).head(5).copy()
    print(f"[diag] top-5/day picks: {len(top5):,}")

    # Realized forward return at OPEN→CLOSE (target_oc is already built into dataset).
    top5["hit"] = (top5["target_oc"] > 0).astype(int)
    print("\n[top5 overall]")
    print(f"  mean target_oc: {top5['target_oc'].mean():+.4%}")
    print(f"  hit rate:       {top5['hit'].mean():.1%}")
    print(f"  median:         {top5['target_oc'].median():+.4%}")

    # Stratify by feature quartile. The ones worth checking:
    # - price_vs_52w_high: near-highs = momentum/growth. Crashes in tariff regime.
    # - ret_20d: prior 20d return. Hot names cool off.
    # - vol_20d: higher vol = more tail risk.
    # - rsi_14: overbought → mean reversion.
    # - dolvol_20d_log: liquidity.
    # - atr_14: range.
    strat_cols = [
        "price_vs_52w_high", "ret_20d", "vol_20d", "rsi_14",
        "dolvol_20d_log", "atr_14", "ret_5d", "price_vs_52w_range",
    ]
    print("\n[stratified mean(target_oc) by quartile of feature]")
    print(f"  {'feature':>22} {'Q1 (low)':>12} {'Q2':>12} {'Q3':>12} {'Q4 (high)':>12}  rows_total")
    for col in strat_cols:
        if col not in top5.columns:
            continue
        q = pd.qcut(top5[col], 4, duplicates="drop")
        stats = top5.groupby(q, observed=True)["target_oc"].agg(["mean", "count"])
        means = stats["mean"].tolist()
        # Pad to 4 if qcut merged bins.
        while len(means) < 4:
            means.append(np.nan)
        line = f"  {col:>22}"
        for m in means[:4]:
            line += f"  {m:+.4%}" if not np.isnan(m) else "         NaN"
        line += f"   {int(stats['count'].sum())}"
        print(line)

    # The key signal: if Q4 (high feature value) is STRONGLY negative
    # compared to Q1, then masking Q4 out at inference is a candidate
    # lever. Report the Q4−Q1 spread for each feature, sorted.
    print("\n[spread: Q4 mean − Q1 mean]  (negative spread = Q4 underperforms Q1)")
    spreads = []
    for col in strat_cols:
        if col not in top5.columns:
            continue
        q = pd.qcut(top5[col], 4, duplicates="drop")
        m = top5.groupby(q, observed=True)["target_oc"].mean().tolist()
        if len(m) >= 4:
            spreads.append((col, m[3] - m[0], m[0], m[3]))
    spreads.sort(key=lambda x: x[1])
    for col, s, q1, q4 in spreads:
        print(f"  {col:>22}  spread={s:+.4%}  (Q1={q1:+.4%}, Q4={q4:+.4%})")

    # ── Counterfactual: what does top-1/day look like WITH vs WITHOUT
    # each candidate mask applied to the pick pool, before selection?
    # This is the closest proxy for "deploy this filter and rerun the
    # top_n=1 backtest" without actually running the full sim.
    print("\n[counterfactual top-1/day mean target_oc — per-day thresholds]")
    print(f"  {'filter':>55}  {'n_days':>7}  {'mean_pnl':>10}  {'hit%':>6}")

    def _per_day_topk(df: pd.DataFrame, k: int = 1,
                      ret20_max_pct: float | None = None,
                      ret5_min_pct: float | None = None) -> pd.DataFrame:
        pieces = []
        for _d, grp in df.groupby("date", sort=False):
            if ret20_max_pct is not None:
                cut = grp["ret_20d"].quantile(ret20_max_pct)
                grp = grp[grp["ret_20d"] <= cut]
            if ret5_min_pct is not None:
                cut = grp["ret_5d"].quantile(ret5_min_pct)
                grp = grp[grp["ret_5d"] >= cut]
            if len(grp) == 0:
                continue
            pieces.append(grp.nlargest(k, "score"))
        return pd.concat(pieces) if pieces else pool.iloc[:0]

    for name, kwargs in [
        ("NO FILTER (baseline)", {}),
        ("drop top-25% ret_20d (hot names)",          dict(ret20_max_pct=0.75)),
        ("drop top-50% ret_20d",                      dict(ret20_max_pct=0.50)),
        ("drop bottom-25% ret_5d (weak recent)",      dict(ret5_min_pct=0.25)),
        ("drop bottom-50% ret_5d",                    dict(ret5_min_pct=0.50)),
        ("drop both extremes",                        dict(ret20_max_pct=0.75, ret5_min_pct=0.25)),
        ("aggressive: drop top-50 r20 + bot-50 r5",   dict(ret20_max_pct=0.50, ret5_min_pct=0.50)),
    ]:
        top1 = _per_day_topk(pool, k=1, **kwargs)
        if len(top1) == 0:
            continue
        pnl = float(top1["target_oc"].mean())
        hit = float((top1["target_oc"] > 0).mean())
        n = len(top1)
        print(f"  {name:>55}  {n:7d}  {pnl:+10.4%}  {hit:6.1%}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
