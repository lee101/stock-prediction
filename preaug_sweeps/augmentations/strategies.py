"""
Pre-augmentation strategy implementations.

Each strategy transforms the data in a different way to potentially improve learning.
"""

import numpy as np
import pandas as pd
from typing import List, Optional, Sequence

from .base_augmentation import BaseAugmentation


PRICE_COLS = ["open", "high", "low", "close"]
VOLUME_COLS = ["volume", "amount"]


def _prediction_frame(
    predictions: np.ndarray,
    context: pd.DataFrame,
    columns: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    feature_columns = list(columns) if columns is not None else list(context.columns)
    if len(feature_columns) != predictions.shape[1]:
        raise ValueError(
            "Prediction column count does not match provided columns; "
            f"columns={len(feature_columns)} vs predictions={predictions.shape[1]}"
        )
    return pd.DataFrame(predictions, columns=feature_columns)


class NoAugmentation(BaseAugmentation):
    """Baseline: no pre-augmentation applied."""

    def name(self) -> str:
        return "baseline"

    def transform_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Return dataframe unchanged."""
        return df.copy()

    def inverse_transform_predictions(
        self,
        predictions: np.ndarray,
        context: pd.DataFrame,
        *,
        columns: Optional[Sequence[str]] = None,
    ) -> np.ndarray:
        """Return predictions unchanged."""
        _ = context  # unused but kept for signature parity
        _ = columns
        return predictions


class PercentChangeAugmentation(BaseAugmentation):
    """
    Transform prices to percent changes from first value in each window.

    Formula: (price - price[0]) / price[0] * 100
    """

    def name(self) -> str:
        return f"percent_change"

    def transform_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform to percent changes."""
        df_aug = df.copy()

        # Store first values for each price column
        for col in PRICE_COLS:
            if col in df_aug.columns:
                first_val = df_aug[col].iloc[0]
                self.metadata[f"{col}_first"] = float(first_val)
                # Percent change from first value
                df_aug[col] = (df_aug[col] - first_val) / (first_val + 1e-8) * 100.0

        # Volume: use log1p transformation
        for col in VOLUME_COLS:
            if col in df_aug.columns:
                df_aug[col] = np.log1p(df_aug[col])

        return df_aug

    def inverse_transform_predictions(
        self,
        predictions: np.ndarray,
        context: pd.DataFrame,
        *,
        columns: Optional[Sequence[str]] = None,
    ) -> np.ndarray:
        """Convert percent changes back to actual prices."""
        pred_df = _prediction_frame(predictions, context, columns)

        # Reverse percent change for prices
        for col in PRICE_COLS:
            if col not in pred_df.columns:
                continue
            if col in self.metadata:
                first_val = self.metadata[f"{col}_first"]
            else:
                series = context[col] if col in context.columns else None
                first_val = float(series.iloc[0]) if series is not None and not series.empty else 0.0
            pred_df[col] = (pred_df[col] / 100.0) * first_val + first_val

        # Reverse log1p for volume
        for col in VOLUME_COLS:
            if col in pred_df.columns:
                pred_df[col] = np.expm1(pred_df[col])

        return pred_df.values


class LogReturnsAugmentation(BaseAugmentation):
    """
    Transform prices to log returns.

    Formula: log(price[t] / price[t-1])
    """

    def name(self) -> str:
        return "log_returns"

    def transform_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform to log returns."""
        df_aug = df.copy()

        # Store initial prices
        for col in PRICE_COLS:
            if col in df_aug.columns:
                self.metadata[f"{col}_initial"] = float(df_aug[col].iloc[0])
                # Log returns with first value as 0
                log_prices = np.log(df_aug[col] + 1e-8)
                df_aug[col] = log_prices - log_prices.iloc[0]

        # Volume: log1p
        for col in VOLUME_COLS:
            if col in df_aug.columns:
                df_aug[col] = np.log1p(df_aug[col])

        return df_aug

    def inverse_transform_predictions(
        self,
        predictions: np.ndarray,
        context: pd.DataFrame,
        *,
        columns: Optional[Sequence[str]] = None,
    ) -> np.ndarray:
        """Convert log returns back to prices."""
        pred_df = _prediction_frame(predictions, context, columns)

        # Get initial prices
        for col in PRICE_COLS:
            if col not in pred_df.columns:
                continue
            if f"{col}_initial" in self.metadata:
                initial = self.metadata[f"{col}_initial"]
            else:
                series = context[col] if col in context.columns else None
                initial = float(series.iloc[0]) if series is not None and not series.empty else 0.0

            # Convert log returns to prices
            pred_df[col] = np.exp(pred_df[col]) * initial

        # Reverse log1p for volume
        for col in VOLUME_COLS:
            if col in pred_df.columns:
                pred_df[col] = np.expm1(pred_df[col])

        return pred_df.values


class DifferencingAugmentation(BaseAugmentation):
    """
    First-order differencing: y[t] = x[t] - x[t-1]

    Makes the series more stationary.
    """

    def __init__(self, order: int = 1, **kwargs):
        super().__init__(**kwargs)
        self.order = order

    def name(self) -> str:
        return f"differencing_order{self.order}"

    def transform_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply differencing."""
        df_aug = df.copy()

        # Store initial values for reconstruction
        for col in PRICE_COLS:
            if col in df_aug.columns:
                self.metadata[f"{col}_initial"] = df_aug[col].iloc[:self.order].tolist()
                # Difference
                df_aug[col] = df_aug[col].diff(self.order).fillna(0.0)

        # Volume: log1p then difference
        for col in VOLUME_COLS:
            if col in df_aug.columns:
                log_vol = np.log1p(df_aug[col])
                self.metadata[f"{col}_initial"] = log_vol.iloc[:self.order].tolist()
                df_aug[col] = log_vol.diff(self.order).fillna(0.0)

        return df_aug

    def inverse_transform_predictions(
        self,
        predictions: np.ndarray,
        context: pd.DataFrame,
        *,
        columns: Optional[Sequence[str]] = None,
    ) -> np.ndarray:
        """Integrate differences back to levels."""
        pred_df = _prediction_frame(predictions, context, columns)

        # Reconstruct from differences for prices
        for col in PRICE_COLS:
            if col not in pred_df.columns:
                continue
            if f"{col}_initial" in self.metadata:
                initial_vals = self.metadata[f"{col}_initial"]
            else:
                series = context[col] if col in context.columns else None
                fallback = float(series.iloc[-1]) if series is not None and not series.empty else 0.0
                initial_vals = [fallback]

            reconstructed = np.cumsum(pred_df[col].values)
            reconstructed += initial_vals[-1]
            pred_df[col] = reconstructed

        # Volume
        for col in VOLUME_COLS:
            if col not in pred_df.columns:
                continue
            if f"{col}_initial" in self.metadata:
                initial_vals = self.metadata[f"{col}_initial"]
            else:
                series = context[col] if col in context.columns else None
                fallback = np.log1p(float(series.iloc[-1])) if series is not None and not series.empty else 0.0
                initial_vals = [fallback]

            reconstructed = np.cumsum(pred_df[col].values)
            reconstructed += initial_vals[-1]
            pred_df[col] = np.expm1(reconstructed)

        return pred_df.values


class DetrendingAugmentation(BaseAugmentation):
    """
    Remove linear trend from prices, train on residuals.

    Learns deviations from trend rather than absolute prices.
    """

    def name(self) -> str:
        return "detrending"

    def transform_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove linear trend."""
        df_aug = df.copy()
        n = len(df_aug)
        x = np.arange(n)

        # Detrend each price column
        for col in PRICE_COLS:
            if col in df_aug.columns:
                y = df_aug[col].values
                # Fit linear trend
                coeffs = np.polyfit(x, y, deg=1)
                trend = np.polyval(coeffs, x)

                # Store trend parameters
                self.metadata[f"{col}_trend"] = {
                    "slope": float(coeffs[0]),
                    "intercept": float(coeffs[1]),
                    "start_idx": 0,
                    "last_x": int(n - 1)
                }

                # Subtract trend
                df_aug[col] = y - trend

        # Volume: log1p
        for col in VOLUME_COLS:
            if col in df_aug.columns:
                df_aug[col] = np.log1p(df_aug[col])

        return df_aug

    def inverse_transform_predictions(
        self,
        predictions: np.ndarray,
        context: pd.DataFrame,
        *,
        columns: Optional[Sequence[str]] = None,
    ) -> np.ndarray:
        """Add trend back to predictions."""
        pred_df = _prediction_frame(predictions, context, columns)

        # Add trend back to prices
        for col in PRICE_COLS:
            if col not in pred_df.columns:
                continue
            if f"{col}_trend" in self.metadata:
                trend_info = self.metadata[f"{col}_trend"]
                last_x = trend_info["last_x"]
                future_x = np.arange(last_x + 1, last_x + 1 + len(pred_df))
                future_trend = trend_info["slope"] * future_x + trend_info["intercept"]

                pred_df[col] = pred_df[col].values + future_trend

        # Volume
        for col in VOLUME_COLS:
            if col in pred_df.columns:
                pred_df[col] = np.expm1(pred_df[col])

        return pred_df.values


class RobustScalingAugmentation(BaseAugmentation):
    """
    Scale using median and IQR instead of mean and std.

    More robust to outliers.
    """

    def name(self) -> str:
        return "robust_scaling"

    def transform_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply robust scaling."""
        df_aug = df.copy()

        # Robust scaling for prices
        for col in PRICE_COLS:
            if col in df_aug.columns:
                median = df_aug[col].median()
                q25 = df_aug[col].quantile(0.25)
                q75 = df_aug[col].quantile(0.75)
                iqr = q75 - q25

                self.metadata[f"{col}_median"] = float(median)
                self.metadata[f"{col}_iqr"] = float(iqr) if iqr > 0 else 1.0

                # Scale: (x - median) / IQR
                df_aug[col] = (df_aug[col] - median) / (iqr + 1e-8)

        # Volume: log1p then robust scale
        for col in VOLUME_COLS:
            if col in df_aug.columns:
                log_vol = np.log1p(df_aug[col])
                median = log_vol.median()
                q25 = log_vol.quantile(0.25)
                q75 = log_vol.quantile(0.75)
                iqr = q75 - q25

                self.metadata[f"{col}_median"] = float(median)
                self.metadata[f"{col}_iqr"] = float(iqr) if iqr > 0 else 1.0

                df_aug[col] = (log_vol - median) / (iqr + 1e-8)

        return df_aug

    def inverse_transform_predictions(
        self,
        predictions: np.ndarray,
        context: pd.DataFrame,
        *,
        columns: Optional[Sequence[str]] = None,
    ) -> np.ndarray:
        """Reverse robust scaling."""
        pred_df = _prediction_frame(predictions, context, columns)

        # Reverse for prices
        for col in PRICE_COLS:
            if col not in pred_df.columns:
                continue
            series = context[col] if col in context.columns else None
            median = self.metadata.get(f"{col}_median", float(series.median()) if series is not None else 0.0)
            iqr = self.metadata.get(f"{col}_iqr", 1.0)
            pred_df[col] = pred_df[col] * iqr + median

        # Volume
        for col in VOLUME_COLS:
            if col not in pred_df.columns:
                continue
            median = self.metadata.get(f"{col}_median", 0.0)
            iqr = self.metadata.get(f"{col}_iqr", 1.0)
            log_vol = pred_df[col] * iqr + median
            pred_df[col] = np.expm1(log_vol)

        return pred_df.values


class MinMaxStandardAugmentation(BaseAugmentation):
    """
    Min-max scaling to [0, 1] followed by standardization.

    Combines benefits of both normalization approaches.
    """

    def __init__(self, feature_range: tuple = (0, 1), **kwargs):
        super().__init__(**kwargs)
        self.feature_range = feature_range

    def name(self) -> str:
        return f"minmax_standard_{self.feature_range[0]}_{self.feature_range[1]}"

    def transform_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply min-max then standardization."""
        df_aug = df.copy()

        # Min-max for prices
        for col in PRICE_COLS:
            if col in df_aug.columns:
                min_val = df_aug[col].min()
                max_val = df_aug[col].max()

                self.metadata[f"{col}_min"] = float(min_val)
                self.metadata[f"{col}_max"] = float(max_val)

                # Min-max scale
                if max_val > min_val:
                    scaled = (df_aug[col] - min_val) / (max_val - min_val)
                    scaled = scaled * (self.feature_range[1] - self.feature_range[0]) + self.feature_range[0]
                else:
                    scaled = df_aug[col]

                # Then standardize
                mean = scaled.mean()
                std = scaled.std()
                self.metadata[f"{col}_mean"] = float(mean)
                self.metadata[f"{col}_std"] = float(std) if std > 0 else 1.0

                df_aug[col] = (scaled - mean) / (std + 1e-8)

        # Volume: log1p
        for col in VOLUME_COLS:
            if col in df_aug.columns:
                df_aug[col] = np.log1p(df_aug[col])

        return df_aug

    def inverse_transform_predictions(
        self,
        predictions: np.ndarray,
        context: pd.DataFrame,
        *,
        columns: Optional[Sequence[str]] = None,
    ) -> np.ndarray:
        """Reverse transformations."""
        pred_df = _prediction_frame(predictions, context, columns)

        range_span = self.feature_range[1] - self.feature_range[0] or 1.0

        # Reverse for prices
        for col in PRICE_COLS:
            if col not in pred_df.columns:
                continue
            mean = self.metadata.get(f"{col}_mean", 0.0)
            std = self.metadata.get(f"{col}_std", 1.0)
            unstandardized = pred_df[col] * std + mean

            series = context[col] if col in context.columns else None
            min_val = self.metadata.get(f"{col}_min", float(series.min()) if series is not None else 0.0)
            max_val = self.metadata.get(f"{col}_max", float(series.max()) if series is not None else 0.0)

            unscaled = (unstandardized - self.feature_range[0]) / range_span
            pred_df[col] = unscaled * (max_val - min_val) + min_val

        # Volume
        for col in VOLUME_COLS:
            if col in pred_df.columns:
                pred_df[col] = np.expm1(pred_df[col])

        return pred_df.values


class RollingWindowNormalization(BaseAugmentation):
    """
    Normalize using rolling window statistics instead of full window.

    Adapts to recent price movements.
    """

    def __init__(self, window_size: int = 20, **kwargs):
        super().__init__(**kwargs)
        self.window_size = window_size

    def name(self) -> str:
        return f"rolling_norm_w{self.window_size}"

    def transform_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply rolling window normalization."""
        df_aug = df.copy()

        # Rolling normalization for prices
        for col in PRICE_COLS:
            if col in df_aug.columns:
                rolling_mean = df_aug[col].rolling(window=self.window_size, min_periods=1).mean()
                rolling_std = df_aug[col].rolling(window=self.window_size, min_periods=1).std()

                # Fill NaN std with 1.0 to avoid division issues
                rolling_std = rolling_std.fillna(1.0)
                rolling_std = rolling_std.replace(0.0, 1.0)

                # Store final window stats for inverse transform
                self.metadata[f"{col}_final_mean"] = float(rolling_mean.iloc[-1])
                self.metadata[f"{col}_final_std"] = float(rolling_std.iloc[-1])

                df_aug[col] = (df_aug[col] - rolling_mean) / (rolling_std + 1e-8)

        # Volume: log1p
        for col in VOLUME_COLS:
            if col in df_aug.columns:
                df_aug[col] = np.log1p(df_aug[col])

        return df_aug

    def inverse_transform_predictions(
        self,
        predictions: np.ndarray,
        context: pd.DataFrame,
        *,
        columns: Optional[Sequence[str]] = None,
    ) -> np.ndarray:
        """Reverse rolling normalization using final window stats."""
        pred_df = _prediction_frame(predictions, context, columns)

        # Reverse for prices
        for col in PRICE_COLS:
            if col not in pred_df.columns:
                continue
            series = context[col] if col in context.columns else None
            fallback_mean = float(series.iloc[-self.window_size:].mean()) if series is not None and not series.empty else 0.0
            fallback_std = float(series.iloc[-self.window_size:].std()) if series is not None and not series.empty else 1.0
            std = self.metadata.get(f"{col}_final_std", fallback_std) or 1.0
            mean = self.metadata.get(f"{col}_final_mean", fallback_mean)
            pred_df[col] = pred_df[col] * std + mean

        # Volume
        for col in VOLUME_COLS:
            if col in pred_df.columns:
                pred_df[col] = np.expm1(pred_df[col])

        return pred_df.values


# Registry of all augmentation strategies
AUGMENTATION_REGISTRY = {
    "baseline": NoAugmentation,
    "percent_change": PercentChangeAugmentation,
    "log_returns": LogReturnsAugmentation,
    "differencing": DifferencingAugmentation,
    "detrending": DetrendingAugmentation,
    "robust_scaling": RobustScalingAugmentation,
    "minmax_standard": MinMaxStandardAugmentation,
    "rolling_norm": RollingWindowNormalization,
}


def get_augmentation(name: str, **kwargs) -> BaseAugmentation:
    """Factory function to create augmentation instances."""
    if name not in AUGMENTATION_REGISTRY:
        raise ValueError(f"Unknown augmentation: {name}. Available: {list(AUGMENTATION_REGISTRY.keys())}")
    return AUGMENTATION_REGISTRY[name](**kwargs)
