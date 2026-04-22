"""XGBoost ranker for cross-sectional daily stock selection.

Trains an ``XGBRanker`` (LambdaMART with ``rank:ndcg``) where each day is
a group and the label is the decile of ``oc_return`` within that day
(0 = worst decile, 9 = best decile). This aligns the training objective
with the deployment objective (picking top-N stocks per day) far better
than the binary ``P(oc_return > 0)`` that the classification XGB uses.

The saved pickle follows the ``xgbnew.model_registry`` contract with
``family="xgb_rank"``. ``predict_scores`` returns a per-day-percentile-
rank-normalised score in [0, 1] so it slots into ``sweep_ensemble_grid``
without changing downstream thresholding.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

DEFAULT_N_DECILES = 10


def _check_xgb() -> None:
    try:
        import xgboost  # noqa: F401
    except ImportError as exc:
        raise ImportError("xgboost not installed") from exc


def _decile_label_per_day(df: pd.DataFrame, n_deciles: int = DEFAULT_N_DECILES) -> np.ndarray:
    """Per-day decile rank of the open-to-close return. 0 = worst, n-1 = best.

    Uses ``target_oc`` (the raw open-to-close return) which is the continuous
    version of the classification label ``target_oc_up`` produced by
    ``xgbnew.dataset.build_daily_dataset``.
    """
    if "target_oc" not in df.columns:
        raise ValueError("ranker training needs 'target_oc' column in train_df")
    if "date" not in df.columns:
        raise ValueError("ranker training needs 'date' column in train_df")

    out = df.groupby("date")["target_oc"].transform(
        lambda s: pd.qcut(s.rank(method="first"), q=n_deciles,
                          labels=False, duplicates="drop")
    )
    # qcut may return NaN for days with < n_deciles unique ranks; set to middle
    out = out.fillna(n_deciles // 2).astype(np.int32)
    return out.values


def _group_sizes(df: pd.DataFrame) -> np.ndarray:
    """Row counts per date, in date order. Expects df sorted by date."""
    sizes = df.groupby("date", sort=False).size().values
    return sizes.astype(np.int64)


class XGBRankerStockModel:
    """XGBoost LambdaMART-style ranker for cross-sectional daily picks."""

    family = "xgb_rank"

    DEFAULT_PARAMS = dict(
        n_estimators=400,
        max_depth=5,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.7,
        min_child_weight=20,
        gamma=0.1,
        reg_alpha=0.1,
        reg_lambda=1.0,
        objective="rank:ndcg",
        eval_metric="ndcg@5",
        random_state=42,
        n_jobs=-1,
    )

    def __init__(self, device: str | None = None, n_deciles: int = DEFAULT_N_DECILES,
                 sample_weight_mode: str = "none", sample_weight_clip: float = 0.05,
                 **kwargs) -> None:
        _check_xgb()
        from xgboost import XGBRanker, build_info
        params = {**self.DEFAULT_PARAMS, **kwargs}
        if device and device.startswith("cuda"):
            if not build_info().get("USE_CUDA"):
                logger.warning("xgboost built without USE_CUDA; falling back to CPU")
                device = None
        if device is not None:
            params["device"] = device
            if device.startswith("cuda"):
                params.setdefault("tree_method", "hist")
                params.setdefault("n_jobs", 1)
        self.device = device
        self.clf = XGBRanker(**params)
        self.feature_cols: list[str] = []
        self.n_deciles = int(n_deciles)
        self.sample_weight_mode = str(sample_weight_mode)
        self.sample_weight_clip = float(sample_weight_clip)
        self._fitted = False
        self._col_medians: np.ndarray | None = None

    def fit(self, train_df: pd.DataFrame,
            feature_cols: Sequence[str],
            val_df: pd.DataFrame | None = None,
            early_stopping_rounds: int = 30,
            verbose: bool = True) -> "XGBRankerStockModel":
        self.feature_cols = list(feature_cols)

        df = train_df.sort_values(["date", "symbol"]).reset_index(drop=True)
        X = df[self.feature_cols].values.astype(np.float32)
        col_medians = np.nanmedian(X, axis=0)
        nan_mask = np.isnan(X)
        X[nan_mask] = np.take(col_medians, np.where(nan_mask)[1])
        self._col_medians = col_medians

        y = _decile_label_per_day(df, n_deciles=self.n_deciles)
        group = _group_sizes(df)
        assert group.sum() == len(df), "group sizes do not match row count"

        sample_weight = self._compute_sample_weight(df)

        eval_kwargs = {}
        if sample_weight is not None:
            eval_kwargs["sample_weight"] = sample_weight
        if val_df is not None and len(val_df) > 0:
            vdf = val_df.sort_values(["date", "symbol"]).reset_index(drop=True)
            Xv = vdf[self.feature_cols].values.astype(np.float32)
            nvm = np.isnan(Xv)
            Xv[nvm] = np.take(col_medians, np.where(nvm)[1])
            yv = _decile_label_per_day(vdf, n_deciles=self.n_deciles)
            gv = _group_sizes(vdf)
            eval_kwargs["eval_set"] = [(Xv, yv)]
            eval_kwargs["eval_group"] = [gv]
            eval_kwargs["verbose"] = verbose
            try:
                self.clf.set_params(early_stopping_rounds=early_stopping_rounds)
            except Exception:
                eval_kwargs["early_stopping_rounds"] = early_stopping_rounds

        logger.info("Fitting XGBRanker on %d rows, %d features, %d groups...",
                    len(df), len(self.feature_cols), len(group))
        self.clf.fit(X, y, group=group, **eval_kwargs)
        self._fitted = True

        imps = self.feature_importances()
        logger.info("Top-5 features: %s", imps.head(5).to_dict())
        return self

    def _compute_sample_weight(self, df: pd.DataFrame) -> np.ndarray | None:
        """Per-GROUP (per-day) training weight — XGBRanker requires this
        shape. Modes:
        - "none": no sample weight
        - "abs_target": weight = day's max |target_oc| clipped at
          sample_weight_clip. Gives higher gradient mass to dispersive days
          where the top pick's margin matters most for the top-N objective.
        - "abs_target_sqrt": sqrt-damped variant of max |target_oc|.
        """
        mode = self.sample_weight_mode
        if mode == "none":
            return None
        if "target_oc" not in df.columns:
            raise ValueError(f"sample_weight_mode={mode!r} needs 'target_oc' column")
        if "date" not in df.columns:
            raise ValueError(f"sample_weight_mode={mode!r} needs 'date' column")
        per_day = df.assign(_abs=df["target_oc"].abs()).groupby("date", sort=False)["_abs"].max()
        a = np.nan_to_num(per_day.to_numpy(dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)
        if mode == "abs_target":
            w = np.clip(a, 0.0, self.sample_weight_clip) + 1e-6
        elif mode == "abs_target_sqrt":
            w = np.sqrt(a) + 1e-6
        else:
            raise ValueError(f"unknown sample_weight_mode {mode!r}")
        scale = float(w.mean())
        if scale > 0:
            w = w / scale
        return w.astype(np.float32)

    def predict_scores(self, df: pd.DataFrame) -> pd.Series:
        """Return per-day-rank-normalized scores in [0, 1] indexed like ``df``.

        Raw ranker output is unbounded; we rank-normalize WITHIN each day
        so the downstream ms-gate still sees a 0-1 score with top pick → 1.
        """
        if not self._fitted:
            raise RuntimeError("Ranker not fitted")

        X = df[self.feature_cols].values.astype(np.float32)
        nm = np.isnan(X)
        X[nm] = np.take(self._col_medians, np.where(nm)[1])

        dev = getattr(self, "device", None)
        if dev and str(dev).startswith("cuda"):
            try:
                import cupy as cp
                X_in = cp.asarray(X)
            except ImportError:
                X_in = X
        else:
            X_in = X
        raw = self.clf.predict(X_in)
        if hasattr(raw, "get"):
            raw = raw.get()
        raw_s = pd.Series(raw, index=df.index, name="raw")

        if "date" in df.columns:
            grouped = raw_s.groupby(df["date"])
            pct = grouped.rank(pct=True)
        else:
            pct = raw_s.rank(pct=True)
        return pct.rename("xgb_rank_score").astype(np.float64)

    def feature_importances(self) -> pd.Series:
        imps = self.clf.feature_importances_
        return pd.Series(imps, index=self.feature_cols).sort_values(ascending=False)

    def save(self, path: Path) -> None:
        import pickle
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({
                "family": "xgb_rank",
                "clf": self.clf,
                "feature_cols": self.feature_cols,
                "col_medians": self._col_medians,
                "device": getattr(self, "device", None),
                "n_deciles": self.n_deciles,
                "sample_weight_mode": self.sample_weight_mode,
                "sample_weight_clip": self.sample_weight_clip,
            }, f)
        logger.info("XGBRanker saved to %s", path)

    @classmethod
    def load(cls, path: Path) -> "XGBRankerStockModel":
        import pickle
        with open(path, "rb") as f:
            data = pickle.load(f)
        obj = cls.__new__(cls)
        obj.clf = data["clf"]
        obj.feature_cols = data["feature_cols"]
        obj._col_medians = data["col_medians"]
        obj.device = data.get("device")
        obj.n_deciles = int(data.get("n_deciles", DEFAULT_N_DECILES))
        obj.sample_weight_mode = str(data.get("sample_weight_mode", "none"))
        obj.sample_weight_clip = float(data.get("sample_weight_clip", 0.05))
        obj._fitted = True
        return obj


__all__ = ["XGBRankerStockModel"]
