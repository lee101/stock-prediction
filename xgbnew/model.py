"""XGBoost model for open-to-close directional prediction.

Trains an XGBClassifier to predict P(open-to-close return > 0).
Also exposes a combined score method that blends XGB probability with
a Chronos2 signal rank.

Usage:
    model = XGBStockModel()
    model.fit(train_df, DAILY_FEATURE_COLS)
    scores = model.predict_scores(test_df)  # Series of P(up), indexed like test_df
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def _check_xgb() -> None:
    try:
        import xgboost  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "xgboost not installed. Run: uv pip install xgboost"
        ) from exc


# ── Model class ───────────────────────────────────────────────────────────────

class XGBStockModel:
    """XGBoost classifier for daily open-to-close direction.

    Predicts P(open_to_close_return > 0) for each (symbol, date) row.
    After fitting, use ``predict_scores`` to rank candidates.
    """

    DEFAULT_PARAMS = dict(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.7,
        min_child_weight=20,   # regularise: need many samples per leaf
        gamma=0.1,
        reg_alpha=0.1,
        reg_lambda=1.0,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1,
        use_label_encoder=False,
    )

    def __init__(self, **kwargs) -> None:
        _check_xgb()
        from xgboost import XGBClassifier
        params = {**self.DEFAULT_PARAMS, **kwargs}
        self.clf = XGBClassifier(**params)
        self.feature_cols: list[str] = []
        self._fitted = False

    def fit(
        self,
        train_df: pd.DataFrame,
        feature_cols: Sequence[str],
        val_df: pd.DataFrame | None = None,
        early_stopping_rounds: int = 30,
        verbose: bool = True,
    ) -> "XGBStockModel":
        """Train the model.

        Args:
            train_df: DataFrame with feature columns + 'target_oc_up'.
            feature_cols: Feature column names to use.
            val_df: Optional validation set for early stopping.
            early_stopping_rounds: Stop if val loss doesn't improve.
            verbose: Print training progress.

        Returns:
            self (for chaining)
        """
        self.feature_cols = list(feature_cols)

        X_train = train_df[self.feature_cols].values.astype(np.float32)
        y_train = train_df["target_oc_up"].values.astype(np.int32)

        # Replace NaN with median of each column
        col_medians = np.nanmedian(X_train, axis=0)
        nan_mask = np.isnan(X_train)
        X_train[nan_mask] = np.take(col_medians, np.where(nan_mask)[1])

        eval_set = []
        if val_df is not None and len(val_df) > 0:
            X_val = val_df[self.feature_cols].values.astype(np.float32)
            nan_mask_v = np.isnan(X_val)
            X_val[nan_mask_v] = np.take(col_medians, np.where(nan_mask_v)[1])
            y_val = val_df["target_oc_up"].values.astype(np.int32)
            eval_set = [(X_val, y_val)]

        self._col_medians = col_medians

        fit_kwargs: dict = {}
        if eval_set:
            fit_kwargs["eval_set"] = eval_set
            fit_kwargs["early_stopping_rounds"] = early_stopping_rounds
            fit_kwargs["verbose"] = verbose

        logger.info(
            "Fitting XGBStockModel on %d rows, %d features...",
            len(X_train), len(self.feature_cols),
        )
        self.clf.fit(X_train, y_train, **fit_kwargs)
        self._fitted = True

        # Log feature importances
        imps = self.feature_importances()
        top5 = imps.head(5)
        logger.info("Top-5 features: %s", top5.to_dict())

        # Val accuracy if available
        if val_df is not None and len(val_df) > 0:
            preds = self.predict_scores(val_df)
            pred_labels = (preds > 0.5).astype(int)
            acc = (pred_labels.values == val_df["target_oc_up"].values).mean()
            logger.info("Val directional accuracy: %.2f%%", acc * 100)

        return self

    def predict_scores(self, df: pd.DataFrame) -> pd.Series:
        """Return P(oc_return > 0) for each row.

        Args:
            df: DataFrame with the same feature columns used during fit.

        Returns:
            Series of float in [0, 1], indexed like ``df``.
        """
        if not self._fitted:
            raise RuntimeError("Model not fitted yet. Call fit() first.")

        X = df[self.feature_cols].values.astype(np.float32)
        nan_mask = np.isnan(X)
        X[nan_mask] = np.take(self._col_medians, np.where(nan_mask)[1])

        proba = self.clf.predict_proba(X)[:, 1]
        return pd.Series(proba, index=df.index, name="xgb_score")

    def feature_importances(self) -> pd.Series:
        """Return feature importances sorted descending."""
        imps = self.clf.feature_importances_
        return pd.Series(imps, index=self.feature_cols).sort_values(ascending=False)

    def save(self, path: Path) -> None:
        """Save model to disk."""
        import pickle
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({"clf": self.clf, "feature_cols": self.feature_cols,
                         "col_medians": self._col_medians}, f)
        logger.info("Model saved to %s", path)

    @classmethod
    def load(cls, path: Path) -> "XGBStockModel":
        """Load model from disk."""
        import pickle
        with open(path, "rb") as f:
            data = pickle.load(f)
        obj = cls.__new__(cls)
        obj.clf = data["clf"]
        obj.feature_cols = data["feature_cols"]
        obj._col_medians = data["col_medians"]
        obj._fitted = True
        return obj


# ── Combined ranking ──────────────────────────────────────────────────────────

def combined_scores(
    df: pd.DataFrame,
    xgb_model: XGBStockModel,
    xgb_weight: float = 0.5,
    chronos_col: str = "chronos_oc_return",
) -> pd.Series:
    """Blend XGB score with normalised Chronos2 rank.

    xgb_weight=1.0 → pure XGB
    xgb_weight=0.0 → pure Chronos2 rank
    xgb_weight=0.5 → equal blend (default)

    Chronos2 score is rank-normalised to [0,1] per day to remove
    scale differences across days.
    """
    xgb_scores = xgb_model.predict_scores(df)

    if chronos_col in df.columns and (df[chronos_col] != 0).any():
        chron_vals = df[chronos_col].fillna(0.0)
        # Rank-normalise per day
        if "date" in df.columns:
            chron_norm = chron_vals.groupby(df["date"]).rank(pct=True)
        else:
            chron_norm = chron_vals.rank(pct=True)
        chron_norm = chron_norm.fillna(0.5)
    else:
        chron_norm = pd.Series(0.5, index=df.index)

    combined = xgb_weight * xgb_scores + (1.0 - xgb_weight) * chron_norm
    return combined.rename("combined_score")


__all__ = ["XGBStockModel", "combined_scores"]
