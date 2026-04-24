"""Extend XGB wide-panel scores back to 2023-06-01 for 34mo-matched blends.

Scores pre-2025-07-01 are IN-SAMPLE for the `oos2025h1_ensemble_gpu_fresh`
ensemble (train_end=2025-06-30). This is acceptable as a robustness-check
companion to the OOS-only cache; the 2025-07+ portion is genuine OOS.

Output: `analysis/cvar_portfolio/xgb_panel_scores_wide_oos2025h1_fresh_2023start.parquet`.
"""
from __future__ import annotations

import logging
import sys
from datetime import date
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from cvar_portfolio.data import read_symbol_list  # noqa: E402
from cvar_portfolio.sweep_wide_momentum import load_active_panel  # noqa: E402
from cvar_portfolio.xgb_alpha import build_xgb_panel_scores  # noqa: E402


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    data_root = REPO / "trainingdata"
    ensemble_dir = REPO / "analysis/xgbnew_daily/oos2025h1_ensemble_gpu_fresh"
    cache_path = REPO / "analysis/cvar_portfolio/xgb_panel_scores_wide_oos2025h1_fresh_2023start.parquet"

    syms = read_symbol_list(REPO / "symbol_lists/stocks_wide_1000_v1.txt")[:1000]
    prices = load_active_panel(
        syms, data_root,
        start=pd.Timestamp("2022-06-01"),
        end=pd.Timestamp("2026-04-18"),
        min_avg_dollar_vol=1e6,
    )
    panel_syms = prices.columns.tolist()
    print(f"Panel: {prices.shape[0]} days × {prices.shape[1]} tickers")

    if cache_path.exists():
        print(f"Cache already exists: {cache_path} — skip rebuild.")
        return

    scores = build_xgb_panel_scores(
        symbols=panel_syms,
        data_root=data_root,
        oos_start=date(2023, 6, 1),
        oos_end=date(2026, 4, 18),
        ensemble_dir=ensemble_dir,
        train_start=date(2020, 1, 1),
        train_end=date(2025, 6, 30),
        min_dollar_vol=1e6,
        fast_features=True,
    )
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    scores.to_parquet(cache_path)
    print(f"\nWrote {len(scores):,} score rows for {scores.symbol.nunique()} syms over "
          f"{scores.date.nunique()} days → {cache_path}")


if __name__ == "__main__":
    main()
