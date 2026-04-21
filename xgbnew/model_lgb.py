"""LightGBM daily open-to-close direction model.

Same contract as ``XGBStockModel``: ``predict_scores(df) -> pd.Series`` in [0, 1]
indexed like ``df``. Saves to a pickle compatible with
``xgbnew.model_registry.load_any_model``.

Usage::

    from xgbnew.model_lgb import LGBMStockModel
    from xgbnew.features import DAILY_FEATURE_COLS
    m = LGBMStockModel(n_estimators=400, num_leaves=31, learning_rate=0.03)
    m.fit(train_df, DAILY_FEATURE_COLS)
    scores = m.predict_scores(oos_df)
"""
from __future__ import annotations

import logging
import time
from typing import Sequence

import numpy as np
import pandas as pd

from xgbnew.model_base import BaseBinaryDailyModel

logger = logging.getLogger(__name__)


def _check_lgb() -> None:
    try:
        import lightgbm  # noqa: F401
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "lightgbm not installed. Run: uv pip install lightgbm"
        ) from exc


class LGBMStockModel(BaseBinaryDailyModel):
    """LightGBM classifier for daily open-to-close direction."""

    family = "lgb"

    DEFAULT_PARAMS = dict(
        n_estimators=400,
        num_leaves=31,
        max_depth=-1,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.7,
        min_child_samples=20,
        reg_alpha=0.1,
        reg_lambda=1.0,
        objective="binary",
        random_state=42,
        n_jobs=-1,
        verbosity=-1,
    )

    def __init__(self, device: str | None = None, **kwargs) -> None:
        super().__init__(device=device)
        _check_lgb()
        from lightgbm import LGBMClassifier
        params = {**self.DEFAULT_PARAMS, **kwargs}
        # lightgbm's GPU path is different from xgb's; keep it explicit.
        if device and device.startswith("cuda"):
            params["device"] = "gpu"
            params.setdefault("gpu_use_dp", False)
        self.clf = LGBMClassifier(**params)

    def fit(self, train_df: pd.DataFrame,
            feature_cols: Sequence[str],
            val_df: pd.DataFrame | None = None,
            verbose: bool = True) -> "LGBMStockModel":
        self._fit_medians(train_df, feature_cols)

        X_train = train_df[self.feature_cols].values.astype(np.float32)
        y_train = train_df["target_oc_up"].values.astype(np.int32)
        # NaN → median (lightgbm handles NaN natively, but we match xgb behaviour
        # so feature_cols/col_medians round-trip is identical across families)
        nan_mask = np.isnan(X_train)
        if nan_mask.any():
            X_train[nan_mask] = np.take(self._col_medians,
                                        np.where(nan_mask)[1])

        fit_kwargs: dict = {}
        if val_df is not None and len(val_df) > 0:
            X_val = val_df[self.feature_cols].values.astype(np.float32)
            vnm = np.isnan(X_val)
            if vnm.any():
                X_val[vnm] = np.take(self._col_medians, np.where(vnm)[1])
            y_val = val_df["target_oc_up"].values.astype(np.int32)
            fit_kwargs["eval_set"] = [(X_val, y_val)]

        t0 = time.perf_counter()
        logger.info("LGBMStockModel.fit: rows=%d feats=%d",
                    len(X_train), len(self.feature_cols))
        self.clf.fit(X_train, y_train, **fit_kwargs)
        logger.info("LGBM fit in %.1fs", time.perf_counter() - t0)
        self._fitted = True
        return self

    def _predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.clf.predict_proba(X)[:, 1]


__all__ = ["LGBMStockModel"]
