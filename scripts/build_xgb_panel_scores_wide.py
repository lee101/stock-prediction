"""Build XGB ensemble scores for the 761-sym wide panel.

Uses the same 15-seed `oos2025h1_ensemble_gpu_fresh` ensemble as the existing
100-sym cache, but scores the full wide-universe panel used by
`sweep_wide_momentum`. Output goes to
`analysis/cvar_portfolio/xgb_panel_scores_wide_oos2025h1_fresh.parquet`.

Target OOS window is 2024-07-01 → 2026-04-18 so we have 252d fit + ~190d
OOS (2025-07+) of XGB-scored days. Anything before 2025-07-01 is technically
in-sample for this ensemble, but the sweep will slice appropriately.
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
    cache_path = REPO / "analysis/cvar_portfolio/xgb_panel_scores_wide_oos2025h1_fresh.parquet"

    syms = read_symbol_list(REPO / "symbol_lists/stocks_wide_1000_v1.txt")[:1000]
    # Panel is discovered the same way as the sweep so the tickers align.
    prices = load_active_panel(
        syms, data_root,
        start=pd.Timestamp("2023-06-01"),
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
        oos_start=date(2024, 7, 1),
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
