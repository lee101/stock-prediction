"""Pre-augmentation strategy implementations."""
from collections import deque
import numpy as np
import pandas as pd
from typing import List, Optional, Sequence

from .base import BaseAugmentation

PRICE_COLS = ["open", "high", "low", "close"]
VOLUME_COLS = ["volume", "amount"]


def _prediction_frame(predictions: np.ndarray, context: pd.DataFrame, columns: Optional[Sequence[str]] = None) -> pd.DataFrame:
    feature_columns = list(columns) if columns is not None else list(context.columns)
    if len(feature_columns) != predictions.shape[1]:
        raise ValueError(f"columns={len(feature_columns)} vs predictions={predictions.shape[1]}")
    return pd.DataFrame(predictions, columns=feature_columns)


class NoAugmentation(BaseAugmentation):
    def name(self) -> str:
        return "baseline"

    def transform_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        return df.copy()

    def inverse_transform_predictions(self, predictions: np.ndarray, context: pd.DataFrame, *, columns: Optional[Sequence[str]] = None) -> np.ndarray:
        return predictions


class PercentChangeAugmentation(BaseAugmentation):
    def name(self) -> str:
        return "percent_change"

    def transform_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        df_aug = df.copy()
        for col in PRICE_COLS:
            if col in df_aug.columns:
                first_val = df_aug[col].iloc[0]
                self.metadata[f"{col}_first"] = float(first_val)
                df_aug[col] = (df_aug[col] - first_val) / (first_val + 1e-8) * 100.0
        for col in VOLUME_COLS:
            if col in df_aug.columns:
                df_aug[col] = np.log1p(df_aug[col])
        return df_aug

    def inverse_transform_predictions(self, predictions: np.ndarray, context: pd.DataFrame, *, columns: Optional[Sequence[str]] = None) -> np.ndarray:
        pred_df = _prediction_frame(predictions, context, columns)
        for col in PRICE_COLS:
            if col not in pred_df.columns:
                continue
            if col in self.metadata:
                first_val = self.metadata[f"{col}_first"]
            else:
                series = context[col] if col in context.columns else None
                first_val = float(series.iloc[0]) if series is not None and not series.empty else 0.0
            pred_df[col] = (pred_df[col] / 100.0) * first_val + first_val
        for col in VOLUME_COLS:
            if col in pred_df.columns:
                pred_df[col] = np.expm1(pred_df[col])
        return pred_df.values


class LogReturnsAugmentation(BaseAugmentation):
    def name(self) -> str:
        return "log_returns"

    def transform_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        df_aug = df.copy()
        for col in PRICE_COLS:
            if col in df_aug.columns:
                self.metadata[f"{col}_initial"] = float(df_aug[col].iloc[0])
                log_prices = np.log(df_aug[col] + 1e-8)
                df_aug[col] = log_prices - log_prices.iloc[0]
        for col in VOLUME_COLS:
            if col in df_aug.columns:
                df_aug[col] = np.log1p(df_aug[col])
        return df_aug

    def inverse_transform_predictions(self, predictions: np.ndarray, context: pd.DataFrame, *, columns: Optional[Sequence[str]] = None) -> np.ndarray:
        pred_df = _prediction_frame(predictions, context, columns)
        for col in PRICE_COLS:
            if col not in pred_df.columns:
                continue
            if f"{col}_initial" in self.metadata:
                initial = self.metadata[f"{col}_initial"]
            else:
                series = context[col] if col in context.columns else None
                initial = float(series.iloc[0]) if series is not None and not series.empty else 0.0
            pred_df[col] = np.exp(pred_df[col]) * initial
        for col in VOLUME_COLS:
            if col in pred_df.columns:
                pred_df[col] = np.expm1(pred_df[col])
        return pred_df.values


class DifferencingAugmentation(BaseAugmentation):
    def __init__(self, order: int = 1, **kwargs):
        super().__init__(**kwargs)
        self.order = order

    def name(self) -> str:
        return f"differencing_order{self.order}"

    def transform_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        df_aug = df.copy()
        for col in PRICE_COLS:
            if col in df_aug.columns:
                series = df_aug[col].astype(float)
                self.metadata[f"{col}_tail"] = series.iloc[-self.order:].tolist()
                df_aug[col] = series.diff(self.order).fillna(0.0)
        for col in VOLUME_COLS:
            if col in df_aug.columns:
                log_vol = np.log1p(df_aug[col].astype(float))
                self.metadata[f"{col}_tail"] = log_vol.iloc[-self.order:].tolist()
                df_aug[col] = log_vol.diff(self.order).fillna(0.0)
        return df_aug

    def inverse_transform_predictions(self, predictions: np.ndarray, context: pd.DataFrame, *, columns: Optional[Sequence[str]] = None) -> np.ndarray:
        pred_df = _prediction_frame(predictions, context, columns)
        for col in PRICE_COLS:
            if col not in pred_df.columns:
                continue
            diffs = pred_df[col].to_numpy(dtype=float)
            tail = self.metadata.get(f"{col}_tail", [0.0])
            history = deque(tail[-self.order:], maxlen=self.order)
            restored = np.empty_like(diffs, dtype=float)
            for idx, value in enumerate(diffs):
                anchor = history[0] if history else 0.0
                next_val = anchor + float(value)
                restored[idx] = next_val
                history.append(next_val)
            pred_df[col] = restored
        for col in VOLUME_COLS:
            if col not in pred_df.columns:
                continue
            diffs = pred_df[col].to_numpy(dtype=float)
            tail = self.metadata.get(f"{col}_tail", [0.0])
            history = deque(tail[-self.order:], maxlen=self.order)
            restored = np.empty_like(diffs, dtype=float)
            for idx, value in enumerate(diffs):
                anchor = history[0] if history else 0.0
                next_val = anchor + float(value)
                restored[idx] = next_val
                history.append(next_val)
            pred_df[col] = np.expm1(restored)
        return pred_df.values


class RobustScalingAugmentation(BaseAugmentation):
    def name(self) -> str:
        return "robust_scaling"

    def transform_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        df_aug = df.copy()
        for col in PRICE_COLS:
            if col in df_aug.columns:
                median = df_aug[col].median()
                q25, q75 = df_aug[col].quantile(0.25), df_aug[col].quantile(0.75)
                iqr = q75 - q25
                self.metadata[f"{col}_median"] = float(median)
                self.metadata[f"{col}_iqr"] = float(iqr) if iqr > 0 else 1.0
                df_aug[col] = (df_aug[col] - median) / (iqr + 1e-8)
        for col in VOLUME_COLS:
            if col in df_aug.columns:
                log_vol = np.log1p(df_aug[col])
                median = log_vol.median()
                q25, q75 = log_vol.quantile(0.25), log_vol.quantile(0.75)
                iqr = q75 - q25
                self.metadata[f"{col}_median"] = float(median)
                self.metadata[f"{col}_iqr"] = float(iqr) if iqr > 0 else 1.0
                df_aug[col] = (log_vol - median) / (iqr + 1e-8)
        return df_aug

    def inverse_transform_predictions(self, predictions: np.ndarray, context: pd.DataFrame, *, columns: Optional[Sequence[str]] = None) -> np.ndarray:
        pred_df = _prediction_frame(predictions, context, columns)
        for col in PRICE_COLS:
            if col not in pred_df.columns:
                continue
            series = context[col] if col in context.columns else None
            median = self.metadata.get(f"{col}_median", float(series.median()) if series is not None else 0.0)
            iqr = self.metadata.get(f"{col}_iqr", 1.0)
            pred_df[col] = pred_df[col] * iqr + median
        for col in VOLUME_COLS:
            if col not in pred_df.columns:
                continue
            median = self.metadata.get(f"{col}_median", 0.0)
            iqr = self.metadata.get(f"{col}_iqr", 1.0)
            log_vol = pred_df[col] * iqr + median
            pred_df[col] = np.expm1(log_vol)
        return pred_df.values


class LogDiffAugmentation(BaseAugmentation):
    """Day-over-day log returns: y[t] = log(x[t]/x[t-1]). Inverse: x[t] = x[t-1]*exp(y[t])."""

    def name(self) -> str:
        return "log_diff"

    def transform_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        df_aug = df.copy()
        for col in PRICE_COLS:
            if col not in df_aug.columns:
                continue
            series = df_aug[col].astype(float).clip(lower=1e-8)
            log_prices = np.log(series)
            self.metadata[f"{col}_log_tail"] = float(log_prices.iloc[-1])
            df_aug[col] = log_prices.diff(1).fillna(0.0)
        for col in VOLUME_COLS:
            if col not in df_aug.columns:
                continue
            log_vol = np.log1p(df_aug[col].astype(float))
            self.metadata[f"{col}_log_tail"] = float(log_vol.iloc[-1])
            df_aug[col] = log_vol.diff(1).fillna(0.0)
        return df_aug

    def inverse_transform_predictions(self, predictions: np.ndarray, context: pd.DataFrame, *, columns: Optional[Sequence[str]] = None) -> np.ndarray:
        pred_df = _prediction_frame(predictions, context, columns)
        for col in PRICE_COLS:
            if col not in pred_df.columns:
                continue
            log_diffs = pred_df[col].to_numpy(dtype=float)
            if context is not None and col in context.columns:
                anchor_log = float(np.log(context[col].astype(float).clip(lower=1e-8).iloc[-1]))
            else:
                anchor_log = self.metadata.get(f"{col}_log_tail", 0.0)
            pred_df[col] = np.exp(anchor_log + np.cumsum(log_diffs))
        for col in VOLUME_COLS:
            if col not in pred_df.columns:
                continue
            log_diffs = pred_df[col].to_numpy(dtype=float)
            if context is not None and col in context.columns:
                anchor_log = float(np.log1p(context[col].astype(float).iloc[-1]))
            else:
                anchor_log = self.metadata.get(f"{col}_log_tail", 0.0)
            pred_df[col] = np.expm1(anchor_log + np.cumsum(log_diffs))
        return pred_df.values


class DiffNormAugmentation(BaseAugmentation):
    """Normalized differencing: y[t] = (x[t]-x[t-1]) / rolling_std(diffs, window). Inverse uses last scale."""

    def __init__(self, window: int = 20, **kwargs):
        super().__init__(**kwargs)
        self.window = window

    def name(self) -> str:
        return f"diff_norm_w{self.window}"

    def transform_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        df_aug = df.copy()
        for col in PRICE_COLS:
            if col not in df_aug.columns:
                continue
            series = df_aug[col].astype(float)
            diffs = series.diff(1).fillna(0.0)
            roll_std = diffs.rolling(self.window, min_periods=1).std().fillna(1.0).replace(0.0, 1.0)
            self.metadata[f"{col}_last_scale"] = float(roll_std.iloc[-1])
            self.metadata[f"{col}_tail"] = float(series.iloc[-1])
            df_aug[col] = diffs / roll_std
        for col in VOLUME_COLS:
            if col not in df_aug.columns:
                continue
            log_vol = np.log1p(df_aug[col].astype(float))
            diffs = log_vol.diff(1).fillna(0.0)
            roll_std = diffs.rolling(self.window, min_periods=1).std().fillna(1.0).replace(0.0, 1.0)
            self.metadata[f"{col}_last_scale"] = float(roll_std.iloc[-1])
            self.metadata[f"{col}_tail"] = float(log_vol.iloc[-1])
            df_aug[col] = diffs / roll_std
        return df_aug

    def inverse_transform_predictions(self, predictions: np.ndarray, context: pd.DataFrame, *, columns: Optional[Sequence[str]] = None) -> np.ndarray:
        pred_df = _prediction_frame(predictions, context, columns)
        for col in PRICE_COLS:
            if col not in pred_df.columns:
                continue
            scale = self.metadata.get(f"{col}_last_scale", 1.0)
            anchor = float(context[col].astype(float).iloc[-1]) if context is not None and col in context.columns else self.metadata.get(f"{col}_tail", 0.0)
            pred_df[col] = anchor + np.cumsum(pred_df[col].to_numpy(dtype=float) * scale)
        for col in VOLUME_COLS:
            if col not in pred_df.columns:
                continue
            scale = self.metadata.get(f"{col}_last_scale", 1.0)
            anchor = float(np.log1p(context[col].astype(float).iloc[-1])) if context is not None and col in context.columns else self.metadata.get(f"{col}_tail", 0.0)
            pred_df[col] = np.expm1(anchor + np.cumsum(pred_df[col].to_numpy(dtype=float) * scale))
        return pred_df.values


AUGMENTATION_REGISTRY = {
    "baseline": NoAugmentation,
    "percent_change": PercentChangeAugmentation,
    "log_returns": LogReturnsAugmentation,
    "differencing": DifferencingAugmentation,
    "robust_scaling": RobustScalingAugmentation,
    "log_diff": LogDiffAugmentation,
    "diff_norm": DiffNormAugmentation,
}


def get_augmentation(name: str, **kwargs) -> BaseAugmentation:
    if name not in AUGMENTATION_REGISTRY:
        raise ValueError(f"Unknown augmentation: {name}")
    return AUGMENTATION_REGISTRY[name](**kwargs)
