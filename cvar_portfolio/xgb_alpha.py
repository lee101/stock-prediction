"""Inject XGB daily-trader ensemble scores as `mean_override` into the CVaR LP.

Loads a fresh N-seed XGB ensemble, scores each (date, symbol) over the OOS
window, and returns an ``alpha_fn`` callable that ``run_backtest`` plugs into
the LP's μ vector. Mapping: ``μ_i = k · (p_i - 0.5)`` centred on the neutral
prob, so an unscored symbol contributes zero tilt and the LP falls back to the
empirical-μ CVaR behaviour.

Usage
-----
    from cvar_portfolio.xgb_alpha import build_xgb_alpha
    alpha_fn = build_xgb_alpha(
        symbols=panel.columns.tolist(),
        data_root=Path("trainingdata"),
        oos_start=date(2025, 7, 1), oos_end=date(2026, 4, 18),
        ensemble_dir=Path("analysis/xgbnew_daily/oos2025h1_ensemble_gpu_fresh"),
        train_start=date(2020, 1, 1), train_end=date(2025, 6, 30),
        k=0.01, mode="center",
    )
    result = run_backtest(panel, alpha_fn=alpha_fn, ...)
"""
from __future__ import annotations

import json
import logging
import sys
from datetime import date
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from xgbnew.dataset import build_daily_dataset  # noqa: E402
from xgbnew.features import DAILY_FEATURE_COLS  # noqa: E402
from xgbnew.model import XGBStockModel  # noqa: E402

logger = logging.getLogger(__name__)


def load_xgb_ensemble(ensemble_dir: Path, force_cpu: bool = True) -> list[XGBStockModel]:
    """Load all XGBStockModel pkls listed in ``alltrain_ensemble.json``.

    ``force_cpu=True`` flips the device metadata to CPU before prediction so the
    XGB booster runs on the host — required when this process also holds a
    cuOpt/cuML CUDA context (driver segfaults otherwise).
    """
    ensemble_dir = Path(ensemble_dir)
    meta_path = ensemble_dir / "alltrain_ensemble.json"
    meta = json.loads(meta_path.read_text())
    models: list[XGBStockModel] = []
    for m in meta["models"]:
        p = Path(m["path"])
        if not p.is_absolute():
            p = REPO / p
        if not p.exists():
            p = ensemble_dir / Path(m["path"]).name
        mdl = XGBStockModel.load(p)
        if force_cpu:
            mdl.device = None
            try:
                mdl.clf.set_params(device="cpu", tree_method="hist")
            except Exception:
                pass
        models.append(mdl)
    logger.info("Loaded %d XGB models from %s (force_cpu=%s)", len(models), ensemble_dir, force_cpu)
    return models


def build_xgb_panel_scores(
    *,
    symbols: list[str],
    data_root: Path,
    oos_start: date,
    oos_end: date,
    ensemble_dir: Path,
    train_start: date = date(2020, 1, 1),
    train_end: date = date(2025, 6, 30),
    min_dollar_vol: float = 5_000_000.0,
    fast_features: bool = True,
) -> pd.DataFrame:
    """Build a long DataFrame of ensemble scores per (date, symbol).

    Columns: ``date`` (pd.Timestamp tz-naive normalised), ``symbol``, ``ensemble_score``.
    ``ensemble_score`` is the arithmetic mean of ``predict_proba`` across seeds.
    Empty rows mean the day/symbol was filtered out by liquidity (or the sym
    simply has no features that day); the caller treats those as neutral.
    """
    train_df, _, oos_df = build_daily_dataset(
        data_root=Path(data_root),
        symbols=list(symbols),
        train_start=train_start, train_end=train_end,
        val_start=oos_start, val_end=oos_end,
        test_start=oos_start, test_end=oos_end,
        chronos_cache=None,
        min_dollar_vol=min_dollar_vol,
        fast_features=bool(fast_features),
    )
    if len(oos_df) == 0:
        raise RuntimeError(
            "XGB feature build produced 0 OOS rows — check symbol list / date range."
        )
    models = load_xgb_ensemble(ensemble_dir)
    probs = np.stack([m.predict_scores(oos_df).values for m in models], axis=0)
    ensemble_score = probs.mean(axis=0).astype(np.float32)
    out = pd.DataFrame({
        "date": pd.to_datetime(oos_df["date"].values).normalize(),
        "symbol": oos_df["symbol"].values,
        "ensemble_score": ensemble_score,
    })
    return out


def _normalize_asof(asof) -> pd.Timestamp:
    ts = pd.Timestamp(asof)
    if ts.tz is not None:
        ts = ts.tz_convert("UTC").tz_localize(None)
    return ts.normalize()


def make_alpha_fn(
    panel_scores: pd.DataFrame,
    *,
    k: float = 0.01,
    mode: str = "center",
) -> Callable[[pd.Timestamp, list[str]], np.ndarray]:
    """Return ``alpha_fn(asof, tickers) -> np.ndarray`` for ``run_backtest``.

    Modes:
      * ``center``   μ_i = k · (p_i - 0.5)               — neutral=0
      * ``demean``   μ_i = k · (p_i - mean_p_day)         — day-centred tilt
      * ``rank``     μ_i = k · (rank_pct_i - 0.5)         — per-day cross-sectional
    """
    if mode not in {"center", "demean", "rank"}:
        raise ValueError(f"Unknown alpha mode: {mode}")
    ps = panel_scores.copy()
    ps["date"] = pd.to_datetime(ps["date"]).dt.tz_localize(None).dt.normalize()
    # For fast lookup at each rebalance date
    by_date = dict(tuple(ps.groupby("date")))

    def alpha_fn(asof, tickers: list[str]) -> np.ndarray:
        day = _normalize_asof(asof)
        # If no scores that exact day, walk backward up to 5 biz-days
        lookup_day = None
        for lag in range(6):
            cand = day - pd.Timedelta(days=lag)
            if cand in by_date:
                lookup_day = cand
                break
        if lookup_day is None:
            return np.zeros(len(tickers), dtype=np.float32)
        d = by_date[lookup_day].set_index("symbol")["ensemble_score"]
        if mode == "demean":
            base = float(d.mean())
        elif mode == "rank":
            # map to per-day percentile rank in [0,1]
            ranks = d.rank(pct=True, method="average")
            d = ranks
            base = 0.5
        else:
            base = 0.5
        vec = np.zeros(len(tickers), dtype=np.float32)
        for j, t in enumerate(tickers):
            if t in d.index:
                vec[j] = float(k) * (float(d[t]) - base)
        return vec

    return alpha_fn


def build_xgb_alpha(
    *,
    symbols: list[str],
    data_root: Path,
    oos_start: date,
    oos_end: date,
    ensemble_dir: Path,
    train_start: date = date(2020, 1, 1),
    train_end: date = date(2025, 6, 30),
    min_dollar_vol: float = 5_000_000.0,
    k: float = 0.01,
    mode: str = "center",
    cache_path: Optional[Path] = None,
    fast_features: bool = True,
) -> Callable[[pd.Timestamp, list[str]], np.ndarray]:
    """Convenience: build panel scores (optionally cache to parquet) + alpha_fn."""
    if cache_path is not None and Path(cache_path).exists():
        logger.info("Loading cached XGB panel scores from %s", cache_path)
        ps = pd.read_parquet(cache_path)
    else:
        ps = build_xgb_panel_scores(
            symbols=symbols,
            data_root=data_root,
            oos_start=oos_start,
            oos_end=oos_end,
            ensemble_dir=ensemble_dir,
            train_start=train_start,
            train_end=train_end,
            min_dollar_vol=min_dollar_vol,
            fast_features=fast_features,
        )
        if cache_path is not None:
            Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
            ps.to_parquet(cache_path)
            logger.info("Cached XGB panel scores → %s (%d rows)", cache_path, len(ps))
    return make_alpha_fn(ps, k=k, mode=mode)


__all__ = [
    "load_xgb_ensemble",
    "build_xgb_panel_scores",
    "make_alpha_fn",
    "build_xgb_alpha",
]
