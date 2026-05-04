"""Shared base class for non-XGB daily direction models.

All models predict P(open_to_close_return > 0) for each (symbol, date) row
and expose the same contract consumed by ``xgbnew.live_trader`` and
``xgbnew.sweep_ensemble_grid``:

    model.feature_cols : list[str]
    model.predict_scores(df: pd.DataFrame) -> pd.Series  # values in [0, 1]
    model.save(path)
    cls.load(path)

Pickle format (shared across families):
    {"family": "<family>", "clf": <learner>, "feature_cols": [...],
     "col_medians": np.ndarray, "device": str | None,
     "state": dict | None}   # optional extra state for torch models

``XGBStockModel`` in ``xgbnew.model`` predates this abstraction and uses a
superset of this format (no "family" key). ``model_registry.load_any_model``
handles both legacy xgb pickles and new family pickles.
"""
from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from xgbnew.artifacts import write_pickle_atomic


logger = logging.getLogger(__name__)


class BaseBinaryDailyModel:
    """Shared save/load + NaN-handling for non-XGB daily direction models."""

    family: str = ""  # subclasses override ("lgb", "cat", "mlp", ...)

    def __init__(self, device: str | None = None) -> None:
        self.device = device
        self.feature_cols: list[str] = []
        self._col_medians: np.ndarray | None = None
        self._fitted = False
        self.clf = None

    # ── Data prep ────────────────────────────────────────────────────────────

    def _prep_X(self, df: pd.DataFrame) -> np.ndarray:
        if self._col_medians is None:
            raise RuntimeError("Model not fitted (col_medians missing).")
        X = df[self.feature_cols].values.astype(np.float32)
        nan_mask = np.isnan(X)
        if nan_mask.any():
            X[nan_mask] = np.take(self._col_medians, np.where(nan_mask)[1])
        return X

    def _fit_medians(self, train_df: pd.DataFrame,
                     feature_cols: Sequence[str]) -> np.ndarray:
        self.feature_cols = list(feature_cols)
        X_train = train_df[self.feature_cols].values.astype(np.float32)
        col_medians = np.nanmedian(X_train, axis=0)
        col_medians = np.nan_to_num(col_medians, nan=0.0)
        self._col_medians = col_medians
        return col_medians

    # ── Interface subclasses must implement ─────────────────────────────────

    def fit(self, train_df: pd.DataFrame,
            feature_cols: Sequence[str],
            val_df: pd.DataFrame | None = None,
            verbose: bool = True) -> "BaseBinaryDailyModel":
        raise NotImplementedError

    def _predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return P(up) for each row of X as a 1-D float array in [0, 1]."""
        raise NotImplementedError

    # ── Public inference ────────────────────────────────────────────────────

    def predict_scores(self, df: pd.DataFrame) -> pd.Series:
        if not self._fitted:
            raise RuntimeError("Model not fitted yet. Call fit() first.")
        X = self._prep_X(df)
        proba = self._predict_proba(X)
        proba = np.asarray(proba, dtype=np.float64).ravel()
        return pd.Series(proba, index=df.index, name=f"{self.family}_score")

    # ── Save / load ─────────────────────────────────────────────────────────

    def _extra_state(self) -> dict | None:
        """Subclass hook: return additional pickle-safe state (e.g. torch sd)."""
        return None

    def _load_extra_state(self, state: dict | None) -> None:
        """Subclass hook: restore additional state from the pickle."""
        return

    def save(self, path: Path | str) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "family": self.family,
            "clf": self.clf,
            "feature_cols": self.feature_cols,
            "col_medians": self._col_medians,
            "device": self.device,
            "state": self._extra_state(),
        }
        write_pickle_atomic(path, payload)
        logger.info("%s model saved to %s", self.family, path)

    @classmethod
    def load(cls, path: Path | str) -> "BaseBinaryDailyModel":
        path = Path(path)
        with open(path, "rb") as f:
            data = pickle.load(f)
        if data.get("family") != cls.family:
            raise ValueError(
                f"{cls.__name__}.load() got family={data.get('family')!r}, "
                f"expected {cls.family!r}. Use model_registry.load_any_model() "
                f"for cross-family dispatch."
            )
        obj = cls.__new__(cls)
        obj.clf = data["clf"]
        obj.feature_cols = list(data["feature_cols"])
        obj._col_medians = data["col_medians"]
        obj.device = data.get("device")
        obj._fitted = True
        obj._load_extra_state(data.get("state"))
        return obj


__all__ = ["BaseBinaryDailyModel"]
