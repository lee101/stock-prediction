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
    # sizing params
    scale: float
    cap: float

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        X = X[self.feature_cols].astype(float)
        if self.scaler_mean is not None and self.scaler_std is not None:
            Xn = (X.values - self.scaler_mean) / np.maximum(self.scaler_std, 1e-8)
        else:
            Xn = X.values
        if self.is_xgb:
            import xgboost as xgb  # type: ignore
            d = xgb.DMatrix(Xn)
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


def _fit_model(X: pd.DataFrame, y: pd.Series) -> Tuple[object, bool]:
    """Try to fit XGBoost; fallback to SKLearn GradientBoosting if xgboost unavailable."""
    # Standardize features to help tree models be stable across feature scales (optional)
    try:
        import xgboost as xgb  # type: ignore
        dtrain = xgb.DMatrix(X.values, label=y.values)
        params = {
            'objective': 'reg:squarederror',
            'max_depth': 4,
            'eta': 0.1,
            'subsample': 0.9,
            'colsample_bytree': 0.9,
            'min_child_weight': 1.0,
            'lambda': 1.0,
            'alpha': 0.0,
            'eval_metric': 'rmse',
        }
        model = xgb.train(params, dtrain, num_boost_round=200)
        return model, True
    except Exception:
        from sklearn.ensemble import GradientBoostingRegressor  # type: ignore
        model = GradientBoostingRegressor(random_state=42)
        model.fit(X.values, y.values)
        return model, False


def train_and_optimize(
    df: pd.DataFrame,
    is_crypto: bool = True,
    fee: float = 0.0023,
) -> TrainedModel:
    # Select features
    feature_cols = [
        c for c in df.columns if c.startswith('feature_')
    ]
    X = df[feature_cols].astype(float)
    y = df['y'].astype(float)

    # Time-based split (last 20% as test)
    n = len(df)
    split = max(10, int(n * 0.8))
    X_tr, X_te = X.iloc[:split], X.iloc[split:]
    y_tr, y_te = y.iloc[:split], y.iloc[split:]

    # Standardization parameters (optional for trees; keep for safety if fallback)
    mean = X_tr.mean().values
    std = X_tr.std(ddof=0).replace(0.0, 1.0).values

    X_tr_n = (X_tr.values - mean) / np.maximum(std, 1e-8)
    X_te_n = (X_te.values - mean) / np.maximum(std, 1e-8)

    # Fit model
    model, is_xgb = _fit_model(pd.DataFrame(X_tr_n, columns=feature_cols), y_tr)

    # Predict on test
    if is_xgb:
        import xgboost as xgb  # type: ignore
        dtest = xgb.DMatrix(X_te_n)
        y_pred = model.predict(dtest)
    else:
        y_pred = model.predict(X_te_n)

    # Backtest grid to pick sizing
    bt = grid_search_sizing(y_true=y_te.values, y_pred=y_pred, is_crypto=is_crypto, fee=fee)

    return TrainedModel(
        model_name='xgboost' if is_xgb else 'sklearn_gbr',
        feature_cols=feature_cols,
        is_xgb=is_xgb,
        scaler_mean=mean,
        scaler_std=std,
        model=model,
        scale=bt.scale,
        cap=bt.cap,
    )

