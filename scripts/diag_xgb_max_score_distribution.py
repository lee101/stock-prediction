"""Measure the deployed ensemble's max-score distribution over a recent
OOS window to answer: "how often does min_score=0.85 pass?"

Usage (from REPO root with .venv active):
    python scripts/diag_xgb_max_score_distribution.py

Uses the same symbol list + model paths as the deployed bot. Writes a
json report + prints a histogram.
"""
from __future__ import annotations

import json
import sys
from datetime import date, timedelta
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from xgbnew.dataset import build_daily_dataset  # noqa: E402
from xgbnew.model import XGBStockModel  # noqa: E402


DEPLOYED_MODELS: List[Path] = [
    REPO / f"analysis/xgbnew_daily/alltrain_ensemble_gpu/alltrain_seed{s}.pkl"
    for s in (0, 7, 42, 73, 197)
]
SYMBOL_LIST = REPO / "symbol_lists/stocks_wide_1000_v1.txt"
MIN_DOLVOL = 50_000_000.0
MIN_VOL_20D = 0.10
MS_THRESHOLDS = [0.55, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90]


def _load_symbols() -> list[str]:
    return [s.strip() for s in SYMBOL_LIST.read_text().splitlines() if s.strip()]


def _score_ensemble(df: pd.DataFrame) -> np.ndarray:
    """Average P(up) across the 5 deployed seeds."""
    acc = None
    for path in DEPLOYED_MODELS:
        model = XGBStockModel.load(path)
        s = model.predict_scores(df).values
        acc = s if acc is None else acc + s
    return acc / float(len(DEPLOYED_MODELS))


def main() -> int:
    symbols = _load_symbols()
    # Score the last 90 trading days in the expanded OOS window.
    oos_end = date.today()
    oos_start = oos_end - timedelta(days=120)

    _train, _val, oos_df = build_daily_dataset(
        symbols=symbols,
        data_root=REPO / "trainingdata",
        train_start=date(2020, 1, 1), train_end=date(2024, 12, 31),
        val_start=date(2025, 1, 2), val_end=date(2025, 2, 1),
        test_start=oos_start, test_end=oos_end,
    )
    print(f"oos_df rows: {len(oos_df)}, dates: {oos_df['date'].nunique()}")

    oos_df = oos_df.copy()
    oos_df["xgb_score"] = _score_ensemble(oos_df)

    # Deployed liquidity + vol floors on every row.
    mask = (
        (np.exp(oos_df["dolvol_20d_log"].values) >= MIN_DOLVOL)
        & (oos_df["vol_20d"].values >= MIN_VOL_20D)
    )
    filtered = oos_df[mask].copy()

    rows = []
    for d, grp in filtered.groupby("date"):
        max_s = float(grp["xgb_score"].max())
        top_sym = grp.loc[grp["xgb_score"].idxmax(), "symbol"]
        rows.append({"date": str(d), "max_score": max_s, "top_symbol": top_sym,
                     "n_candidates": int(len(grp))})
    summary = pd.DataFrame(rows).sort_values("date")

    print("\n=== Per-day max score (most recent 15) ===")
    print(summary.tail(15).to_string(index=False))

    max_scores = summary["max_score"].values
    print("\n=== Distribution of daily max-score ===")
    print(f"n_days={len(max_scores)}, mean={max_scores.mean():.4f}, "
          f"median={np.median(max_scores):.4f}, max={max_scores.max():.4f}")
    for p in (5, 10, 25, 50, 75, 90, 95, 99):
        print(f"  p{p}: {np.percentile(max_scores, p):.4f}")

    print("\n=== Pass rate at each min_score threshold ===")
    for t in MS_THRESHOLDS:
        n_pass = int((max_scores >= t).sum())
        rate = n_pass / len(max_scores) * 100.0
        print(f"  ms>={t:.2f}: {n_pass:>3d}/{len(max_scores)} days ({rate:5.1f}%)")

    out = REPO / "analysis/xgbnew_daily/diag_max_score_distribution.json"
    out.write_text(json.dumps({
        "oos_start": str(oos_start), "oos_end": str(oos_end),
        "n_days": len(max_scores), "mean": float(max_scores.mean()),
        "percentiles": {f"p{p}": float(np.percentile(max_scores, p)) for p in (5, 10, 25, 50, 75, 90, 95, 99)},
        "pass_rate": {f"ms{t}": float((max_scores >= t).sum() / len(max_scores)) for t in MS_THRESHOLDS},
        "per_day": rows,
    }, indent=2))
    print(f"\nReport: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
