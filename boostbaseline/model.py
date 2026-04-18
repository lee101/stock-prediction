from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd

from .backtest import BacktestResult, grid_search_sizing


MODELS_DIR = Path('boostbaseline/models')
MODELS_DIR.mkdir(parents=True, exist_ok=True)


@dataclass
class TrainedModel:
    model_name: str
    feature_cols: list[str]
    is_xgb: bool
    scaler_mean: Optional[np.ndarray]
    scaler_std: Optional[np.ndarray]
    # model is either xgboost Booster or sklearn estimator
    model: object
    # sizing params (tuned on validation split only — NOT on test)
    scale: float
    cap: float
    # early-stopping bookkeeping (None if no early stopping / not xgb)
    best_iteration: Optional[int] = None

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        X = X[self.feature_cols].astype(float)
        if self.scaler_mean is not None and self.scaler_std is not None:
            Xn = (X.values - self.scaler_mean) / np.maximum(self.scaler_std, 1e-8)
        else:
            Xn = X.values
        if self.is_xgb:
            import xgboost as xgb  # type: ignore
            d = xgb.DMatrix(Xn, feature_names=self.feature_cols)
            if self.best_iteration is not None:
                return self.model.predict(
                    d, iteration_range=(0, int(self.best_iteration) + 1),
                )
            return self.model.predict(d)
        else:
            return self.model.predict(Xn)

    def save(self, symbol: str):
        path = MODELS_DIR / f"{symbol}_boost.model"
        meta = {
            'model_name': self.model_name,
            'feature_cols': self.feature_cols,
            'is_xgb': self.is_xgb,
            'scaler_mean': self.scaler_mean.tolist() if self.scaler_mean is not None else None,
            'scaler_std': self.scaler_std.tolist() if self.scaler_std is not None else None,
            'scale': self.scale,
            'cap': self.cap,
            'best_iteration': (
                int(self.best_iteration) if self.best_iteration is not None else None
            ),
        }
        if self.is_xgb:
            import xgboost as xgb  # type: ignore
            model_path = str(path) + '.json'
            self.model.save_model(model_path)
            with open(path, 'w') as f:
                json.dump({**meta, 'xgb_json': Path(model_path).name}, f)
        else:
            import joblib  # type: ignore
            model_path = str(path) + '.joblib'
            joblib.dump(self.model, model_path)
            with open(path, 'w') as f:
                json.dump({**meta, 'sk_joblib': Path(model_path).name}, f)


def _xgb_cuda_available() -> bool:
    """True when xgboost is importable and built with USE_CUDA."""
    try:
        import xgboost as xgb  # type: ignore
    except Exception:
        return False
    try:
        return bool(xgb.build_info().get("USE_CUDA"))
    except Exception:
        return False


def _fit_model(
    X_tr: pd.DataFrame,
    y_tr: pd.Series,
    X_va: pd.DataFrame,
    y_va: pd.Series,
    *,
    device: Optional[str] = None,
    early_stopping_rounds: int = 200,
    num_boost_round: int = 4000,
) -> Tuple[object, bool, Optional[int]]:
    """Fit XGBoost with early stopping on the validation slice.

    Falls back to sklearn GradientBoostingRegressor if xgboost is not
    importable at runtime (best_iteration returned as None in that case).
    """
    try:
        import xgboost as xgb  # type: ignore
        # Auto-select GPU when available unless caller pinned a device.
        effective_device = device if device is not None else (
            "cuda" if _xgb_cuda_available() else "cpu"
        )
        feature_names = list(X_tr.columns)
        dtrain = xgb.QuantileDMatrix(
            X_tr.astype("float32"),
            y_tr.astype("float32"),
            feature_names=feature_names,
        )
        dvalid = xgb.QuantileDMatrix(
            X_va.astype("float32"),
            y_va.astype("float32"),
            feature_names=feature_names,
            ref=dtrain,
        )
        params = {
            "objective": "reg:pseudohubererror",
            "eval_metric": "mae",
            "device": effective_device,
            "tree_method": "hist",
            "max_depth": 6,
            "eta": 0.02,
            "subsample": 0.70,
            "colsample_bytree": 0.70,
            "min_child_weight": 8.0,
            "lambda": 2.0,
            "alpha": 0.5,
            "max_bin": 256,
            "seed": 42,
        }
        booster = xgb.train(
            params,
            dtrain,
            num_boost_round=num_boost_round,
            evals=[(dtrain, "train"), (dvalid, "valid")],
            early_stopping_rounds=early_stopping_rounds,
            verbose_eval=False,
        )
        best_iter = int(getattr(booster, "best_iteration", num_boost_round - 1))
        return booster, True, best_iter
    except Exception:
        from sklearn.ensemble import GradientBoostingRegressor  # type: ignore
        model = GradientBoostingRegressor(random_state=42)
        model.fit(X_tr.values, y_tr.values)
        return model, False, None


def train_and_optimize(
    df: pd.DataFrame,
    is_crypto: bool = True,
    fee: float = 0.0023,
    *,
    device: Optional[str] = None,
    turnover_proportional_fee: bool = True,
) -> TrainedModel:
    """Train on 60%, validate on 20%, hold out the final 20% as untouched test.

    Sizing (scale / cap) is chosen to maximise the validation-slice
    backtest only — the test split is reserved for honest OOS reporting
    by the caller (e.g. run_baseline.py).
    """
    # Select features
    feature_cols = [c for c in df.columns if c.startswith('feature_')]
    X = df[feature_cols].astype(float)
    y = df['y'].astype(float)

    n = len(df)
    if n < 50:
        raise ValueError(f"Need at least 50 rows for 60/20/20 split, got {n}")
    i1 = max(30, int(n * 0.60))
    i2 = max(i1 + 10, int(n * 0.80))
    i2 = min(i2, n - 1)

    X_tr, X_va, X_te = X.iloc[:i1], X.iloc[i1:i2], X.iloc[i2:]
    y_tr, y_va, y_te = y.iloc[:i1], y.iloc[i1:i2], y.iloc[i2:]

    # Trees don't need feature scaling; keep identity stats for forward compat
    # (``scaler_mean``/``scaler_std`` are consumed by ``TrainedModel.predict``).
    mean = np.zeros(X.shape[1], dtype=np.float32)
    std = np.ones(X.shape[1], dtype=np.float32)

    model, is_xgb, best_iter = _fit_model(X_tr, y_tr, X_va, y_va, device=device)

    # Predict on validation — size on this slice only, NOT on test.
    if is_xgb:
        import xgboost as xgb  # type: ignore
        dvalid = xgb.DMatrix(X_va.values, feature_names=feature_cols)
        iter_range = (0, (best_iter or 0) + 1) if best_iter is not None else None
        y_valid_pred = (
            model.predict(dvalid, iteration_range=iter_range)
            if iter_range is not None else model.predict(dvalid)
        )
    else:
        y_valid_pred = model.predict(X_va.values)

    # Grid-search sizing on validation only
    bt = grid_search_sizing(
        y_true=y_va.values,
        y_pred=y_valid_pred,
        is_crypto=is_crypto,
        fee=fee,
        turnover_proportional_fee=turnover_proportional_fee,
    )

    return TrainedModel(
        model_name='xgboost' if is_xgb else 'sklearn_gbr',
        feature_cols=feature_cols,
        is_xgb=is_xgb,
        scaler_mean=mean,
        scaler_std=std,
        model=model,
        scale=bt.scale,
        cap=bt.cap,
        best_iteration=best_iter,
    )

