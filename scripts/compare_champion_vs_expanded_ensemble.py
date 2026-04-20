"""Compare post-2026-04-20 champion (846-sym) ensemble vs expanded
(2700-sym) ensemble. Both were retrained on the same fresh data window
(2020-01-01 → 2026-04-20, alltrain / no holdout), so any OOS window
overlaps training — but the *delta* between the two on the deployed
846-symbol universe is still informative.

Scores last 90 days on the deployed universe (stocks_wide_1000_v1.txt)
under deploy-time filters (min_dolvol 50M, min_vol_20d 0.10). Reports:

- Per-ensemble: max-score distribution, pass-rate at ms=0.85.
- Head-to-head: # days where each ensemble's top pick BEAT the other,
  and the simple PnL of each (top-1 equal weight, no leverage, open-
  to-close, no fees).

Usage (from REPO root with .venv active):
    python scripts/compare_champion_vs_expanded_ensemble.py
"""
from __future__ import annotations

import json
import sys
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from xgbnew.dataset import build_daily_dataset  # noqa: E402
from xgbnew.model import XGBStockModel  # noqa: E402


CHAMPION_DIR = REPO / "analysis/xgbnew_daily/alltrain_ensemble_gpu"
EXPANDED_DIR = REPO / "analysis/xgbnew_daily/alltrain_ensemble_gpu_expanded_2700_20260420"
SYMBOL_LIST = REPO / "symbol_lists/stocks_wide_1000_v1.txt"
MIN_DOLVOL = 50_000_000.0
MIN_VOL_20D = 0.10


def _load_symbols() -> list[str]:
    return [s.strip() for s in SYMBOL_LIST.read_text().splitlines() if s.strip()]


def _score(ens_dir: Path, df: pd.DataFrame) -> np.ndarray:
    acc = None
    for seed in (0, 7, 42, 73, 197):
        m = XGBStockModel.load(ens_dir / f"alltrain_seed{seed}.pkl")
        s = m.predict_scores(df).values
        acc = s if acc is None else acc + s
    return acc / 5.0


def main() -> int:
    syms = _load_symbols()
    oos_end = date.today()
    oos_start = oos_end - timedelta(days=120)

    _tr, _va, oos = build_daily_dataset(
        symbols=syms,
        data_root=REPO / "trainingdata",
        train_start=date(2020, 1, 1), train_end=date(2024, 12, 31),
        val_start=date(2025, 1, 2), val_end=date(2025, 2, 1),
        test_start=oos_start, test_end=oos_end,
    )
    print(f"oos rows: {len(oos)}  days: {oos['date'].nunique()}")

    oos = oos.copy()
    oos["s_champ"] = _score(CHAMPION_DIR, oos)
    oos["s_exp"] = _score(EXPANDED_DIR, oos)

    # Apply deploy filters
    mask = (
        (np.exp(oos["dolvol_20d_log"].values) >= MIN_DOLVOL)
        & (oos["vol_20d"].values >= MIN_VOL_20D)
    )
    f = oos[mask].copy()
    print(f"after dep filters: rows={len(f)}  days={f['date'].nunique()}")

    rows = []
    for d, g in f.groupby("date"):
        top_c = g.loc[g["s_champ"].idxmax()]
        top_e = g.loc[g["s_exp"].idxmax()]
        rows.append({
            "date": str(d),
            "champ_sym": top_c["symbol"],   "champ_score": float(top_c["s_champ"]),
            "champ_ret":  float(top_c.get("target_oc", np.nan)),
            "exp_sym":   top_e["symbol"],   "exp_score":   float(top_e["s_exp"]),
            "exp_ret":   float(top_e.get("target_oc", np.nan)),
            "n_candidates": int(len(g)),
        })
    summary = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)

    # target_oc is open-to-close return in decimal — e.g. 0.012 = +1.2%
    # (feature engineering guarantees no lookahead at feature time)
    champ_ret = summary["champ_ret"].dropna()
    exp_ret = summary["exp_ret"].dropna()

    print("\n=== per-day top pick (most recent 15) ===")
    print(summary.tail(15).to_string(index=False))

    print("\n=== max-score distribution ===")
    for name, col in (("champ", "champ_score"), ("exp", "exp_score")):
        v = summary[col].values
        print(f"  {name}: mean={v.mean():.4f} median={np.median(v):.4f} "
              f"p5={np.percentile(v,5):.4f} p95={np.percentile(v,95):.4f} "
              f"max={v.max():.4f}")
        for t in (0.70, 0.75, 0.80, 0.85, 0.90):
            r = float((v >= t).mean() * 100.0)
            print(f"    pass ms={t}: {r:5.1f}%")

    print("\n=== top-1 PnL (open-to-close, per day, no fees, no lev) ===")
    for name, r in (("champ", champ_ret), ("exp", exp_ret)):
        if len(r) == 0:
            print(f"  {name}: empty")
            continue
        cum = (1 + r).prod() - 1
        sharpe = r.mean() / (r.std(ddof=0) + 1e-12) * np.sqrt(252)
        print(f"  {name}: n={len(r)} mean_daily={r.mean()*100:.3f}% "
              f"cum={cum*100:.2f}% sharpe_ann={sharpe:.2f} "
              f"neg_days={int((r<0).sum())}")

    # Head-to-head: which ensemble's pick beat the other that day?
    both = summary.dropna(subset=["champ_ret", "exp_ret"])
    champ_wins = int((both["champ_ret"] > both["exp_ret"]).sum())
    exp_wins = int((both["exp_ret"] > both["champ_ret"]).sum())
    ties = int((both["champ_ret"] == both["exp_ret"]).sum())
    same_pick = int((both["champ_sym"] == both["exp_sym"]).sum())
    print(f"\n=== head-to-head (n={len(both)}) ===")
    print(f"  champ beats exp: {champ_wins}")
    print(f"  exp beats champ: {exp_wins}")
    print(f"  ties:            {ties}")
    print(f"  same-pick days:  {same_pick}")

    out = REPO / "analysis/xgbnew_daily/compare_champ_vs_expanded_20260420.json"
    out.write_text(json.dumps({
        "oos_start": str(oos_start), "oos_end": str(oos_end),
        "champion_dir": str(CHAMPION_DIR),
        "expanded_dir": str(EXPANDED_DIR),
        "n_days": len(summary),
        "champ_mean_daily": float(champ_ret.mean()) if len(champ_ret) else None,
        "exp_mean_daily":   float(exp_ret.mean())   if len(exp_ret)   else None,
        "champ_sharpe_ann": float(champ_ret.mean() / (champ_ret.std(ddof=0)+1e-12) * np.sqrt(252)) if len(champ_ret) > 1 else None,
        "exp_sharpe_ann":   float(exp_ret.mean()   / (exp_ret.std(ddof=0)+1e-12)   * np.sqrt(252)) if len(exp_ret)   > 1 else None,
        "champ_cum": float((1+champ_ret).prod()-1) if len(champ_ret) else None,
        "exp_cum":   float((1+exp_ret).prod()-1)   if len(exp_ret)   else None,
        "champ_wins": champ_wins, "exp_wins": exp_wins, "ties": ties, "same_pick_days": same_pick,
        "per_day": rows,
    }, indent=2))
    print(f"\nreport: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
