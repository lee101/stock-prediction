"""Train one XGB binary classifier per forward horizon.

Keep the recipe minimal and aligned with xgbnew so variance from pipeline
differences does not contaminate the 1-day-vs-multi-day comparison. Defaults:
  - Same feature set as xgbnew daily model.
  - ``binary:logistic`` objective on ``target_fwd_{N}d_up``.
  - 600 trees, max_depth=6, lr=0.05 (matches xgbnew champion family).
  - GPU if available (``tree_method=hist, device=cuda``).
  - Multi-seed ensemble (default 5 seeds), prob averaged.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import xgboost as xgb

from xgbnew.features import DAILY_FEATURE_COLS

logger = logging.getLogger(__name__)

# Drop features that are targets or forward-looking (actual_*)
FEATURE_COLS = list(DAILY_FEATURE_COLS)


@dataclass
class TrainConfig:
    n_estimators: int = 600
    max_depth: int = 6
    learning_rate: float = 0.05
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    min_child_weight: float = 1.0
    device: str = "cuda"   # "cpu" fallback if no GPU
    seeds: Sequence[int] = (0, 7, 42, 73, 197)  # same seed-stems as xgbnew
    early_stopping_rounds: int | None = None


def _make_dmatrix(df: pd.DataFrame, *, y: np.ndarray | None = None):
    X = df[FEATURE_COLS].astype(np.float32)
    if y is None:
        return xgb.DMatrix(X)
    return xgb.DMatrix(X, label=y.astype(np.float32))


def train_per_horizon(
    train_df: pd.DataFrame,
    horizons: Sequence[int],
    cfg: TrainConfig = TrainConfig(),
    *,
    out_dir: Path | None = None,
) -> dict[int, list[xgb.Booster]]:
    """Train an ensemble (len(seeds) models) per horizon N. Returns {N: [boosters]}."""
    models: dict[int, list[xgb.Booster]] = {}
    for n in horizons:
        ycol = f"target_fwd_{n}d_up"
        vcol = f"valid_fwd_{n}d"
        if ycol not in train_df.columns or vcol not in train_df.columns:
            raise ValueError(f"train_df missing {ycol} / {vcol}")
        sub = train_df[train_df[vcol] == 1].copy()
        # Drop rows with NaN in any feature col (XGBoost handles NaNs but we
        # still want the label aligned)
        y = sub[ycol].astype(np.float32).values
        X = sub[FEATURE_COLS].astype(np.float32)
        logger.info("horizon=%d: %d rows, pos_rate=%.4f", n, len(y), float(y.mean()) if len(y) else 0.0)
        if len(y) < 1000:
            logger.warning("horizon=%d has only %d rows; skipping", n, len(y))
            continue

        dtrain = xgb.DMatrix(X, label=y)
        boosters: list[xgb.Booster] = []
        for seed in cfg.seeds:
            params = dict(
                objective="binary:logistic",
                eval_metric="logloss",
                max_depth=cfg.max_depth,
                learning_rate=cfg.learning_rate,
                subsample=cfg.subsample,
                colsample_bytree=cfg.colsample_bytree,
                min_child_weight=cfg.min_child_weight,
                tree_method="hist",
                device=cfg.device,
                seed=int(seed),
                verbosity=0,
            )
            bst = xgb.train(params, dtrain, num_boost_round=cfg.n_estimators)
            boosters.append(bst)
        models[n] = boosters

        if out_dir is not None:
            out_dir.mkdir(parents=True, exist_ok=True)
            for i, bst in enumerate(boosters):
                bst.save_model(str(out_dir / f"fwd_{n}d_seed{cfg.seeds[i]}.ubj"))
    return models


def score_per_horizon(
    test_df: pd.DataFrame,
    models: dict[int, list[xgb.Booster]],
    feature_cols: Sequence[str] = FEATURE_COLS,
) -> dict[int, np.ndarray]:
    """Return {N: prob_up_array_aligned_to_test_df.index} — seed-averaged probs."""
    X = test_df[list(feature_cols)].astype(np.float32)
    d = xgb.DMatrix(X)
    out: dict[int, np.ndarray] = {}
    for n, boosters in models.items():
        probs = np.zeros(len(test_df), dtype=np.float64)
        for bst in boosters:
            probs += bst.predict(d) / len(boosters)
        out[n] = probs
    return out
