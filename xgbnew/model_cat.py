"""CatBoost daily open-to-close direction model.

Same contract as ``XGBStockModel``: ``predict_scores(df) -> pd.Series`` in [0, 1].
"""
from __future__ import annotations

import logging
import time
from typing import Sequence

import numpy as np
import pandas as pd

from xgbnew.model_base import BaseBinaryDailyModel

logger = logging.getLogger(__name__)


def _check_cat() -> None:
    try:
        import catboost  # noqa: F401
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "catboost not installed. Run: uv pip install catboost"
        ) from exc


class CatBoostStockModel(BaseBinaryDailyModel):
    """CatBoost classifier for daily open-to-close direction."""

    family = "cat"

    DEFAULT_PARAMS = dict(
        iterations=400,
        depth=6,
        learning_rate=0.03,
        l2_leaf_reg=3.0,
        subsample=0.8,
        colsample_bylevel=0.7,
        bootstrap_type="Bernoulli",
        loss_function="Logloss",
        eval_metric="Logloss",
        random_seed=42,
        verbose=False,
        allow_writing_files=False,
    )

    def __init__(self, device: str | None = None, **kwargs) -> None:
        super().__init__(device=device)
        _check_cat()
        from catboost import CatBoostClassifier
        params = {**self.DEFAULT_PARAMS, **kwargs}
        if device and device.startswith("cuda"):
            params["task_type"] = "GPU"
            params["devices"] = device.split(":", 1)[1] if ":" in device else "0"
        self.clf = CatBoostClassifier(**params)

    def fit(self, train_df: pd.DataFrame,
            feature_cols: Sequence[str],
            val_df: pd.DataFrame | None = None,
            verbose: bool = True) -> "CatBoostStockModel":
        self._fit_medians(train_df, feature_cols)

        X_train = train_df[self.feature_cols].values.astype(np.float32)
        y_train = train_df["target_oc_up"].values.astype(np.int32)
        nan_mask = np.isnan(X_train)
        if nan_mask.any():
            X_train[nan_mask] = np.take(self._col_medians,
                                        np.where(nan_mask)[1])

        fit_kwargs: dict = {"verbose": False}
        if val_df is not None and len(val_df) > 0:
            X_val = val_df[self.feature_cols].values.astype(np.float32)
            vnm = np.isnan(X_val)
            if vnm.any():
                X_val[vnm] = np.take(self._col_medians, np.where(vnm)[1])
            y_val = val_df["target_oc_up"].values.astype(np.int32)
            fit_kwargs["eval_set"] = (X_val, y_val)

        t0 = time.perf_counter()
        logger.info("CatBoostStockModel.fit: rows=%d feats=%d",
                    len(X_train), len(self.feature_cols))
        self.clf.fit(X_train, y_train, **fit_kwargs)
        logger.info("CatBoost fit in %.1fs", time.perf_counter() - t0)
        self._fitted = True
        return self

    def _predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.clf.predict_proba(X)[:, 1]


__all__ = ["CatBoostStockModel"]
