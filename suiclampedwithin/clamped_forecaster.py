"""Clamped forecaster: warp 24h hourly forecasts to fit within daily envelope."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from loguru import logger


@dataclass
class ClampedForecastConfig:
    symbol: str
    data_root: Path
    cache_root: Path
    hourly_horizon: int = 24
    daily_horizon: int = 1
    context_hours: int = 168
    quantile_levels: Tuple[float, ...] = (0.1, 0.5, 0.9)
    batch_size: int = 32
    model_id: str = "amazon/chronos-2"
    device_map: str = "cuda"
    clamp_mode: str = "scale"  # scale | clamp | affine


def aggregate_hourly_to_daily(hourly_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate hourly OHLCV to daily bars."""
    df = hourly_df.copy()
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.set_index("timestamp")
    df["date"] = df.index.date
    daily = df.groupby("date").agg({
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum" if "volume" in df.columns else "first",
    }).reset_index()
    daily["timestamp"] = pd.to_datetime(daily["date"])
    return daily.drop(columns=["date"])


def load_hourly_data(data_root: Path, symbol: str) -> pd.DataFrame:
    """Load hourly CSV data."""
    candidates = [
        data_root / f"{symbol}.csv",
        data_root / "binance_spot_hourly" / f"{symbol}.csv",
        Path("binance_spot_hourly") / f"{symbol}.csv",
    ]
    for p in candidates:
        if p.exists():
            df = pd.read_csv(p, parse_dates=["timestamp"])
            return df.sort_values("timestamp").reset_index(drop=True)
    raise FileNotFoundError(f"No hourly data found for {symbol} in {candidates}")


class ClampedForecaster:
    """Generate hourly forecasts clamped within daily forecast envelope."""

    def __init__(self, config: ClampedForecastConfig):
        self.config = config
        self._wrapper = None

    def _get_wrapper(self):
        if self._wrapper is None:
            from src.models.chronos2_wrapper import Chronos2OHLCWrapper
            self._wrapper = Chronos2OHLCWrapper.from_pretrained(
                model_id=self.config.model_id,
                device_map=self.config.device_map,
                default_context_length=self.config.context_hours,
                default_batch_size=self.config.batch_size,
                quantile_levels=self.config.quantile_levels,
            )
        return self._wrapper

    def forecast_daily(
        self,
        daily_df: pd.DataFrame,
        prediction_length: int = 1,
    ) -> dict:
        """Forecast daily OHLC bars."""
        wrapper = self._get_wrapper()
        result = wrapper.predict_ohlc(
            context_df=daily_df,
            prediction_length=prediction_length,
        )
        med = result.median
        q10 = result.quantile(0.1)
        q90 = result.quantile(0.9)
        return {
            "high_p50": med["high"].iloc[0] if prediction_length == 1 else med["high"].tolist(),
            "low_p50": med["low"].iloc[0] if prediction_length == 1 else med["low"].tolist(),
            "close_p50": med["close"].iloc[0] if prediction_length == 1 else med["close"].tolist(),
            "close_p10": q10["close"].iloc[0] if prediction_length == 1 else q10["close"].tolist(),
            "close_p90": q90["close"].iloc[0] if prediction_length == 1 else q90["close"].tolist(),
        }

    def forecast_hourly(
        self,
        hourly_df: pd.DataFrame,
        prediction_length: int = 24,
    ) -> dict:
        """Forecast hourly OHLC bars."""
        wrapper = self._get_wrapper()
        result = wrapper.predict_ohlc(
            context_df=hourly_df,
            prediction_length=prediction_length,
        )
        med = result.median
        q10 = result.quantile(0.1)
        q90 = result.quantile(0.9)
        return {
            "high_p50": med["high"].tolist(),
            "low_p50": med["low"].tolist(),
            "close_p50": med["close"].tolist(),
            "close_p10": q10["close"].tolist(),
            "close_p90": q90["close"].tolist(),
        }

    def clamp_hourly_to_daily(
        self,
        hourly_forecast: dict,
        daily_high: float,
        daily_low: float,
        mode: str = "scale",
    ) -> dict:
        """Warp hourly forecasts to fit within daily envelope."""
        result = {}
        for key in hourly_forecast:
            vals = np.array(hourly_forecast[key])
            if mode == "clamp":
                vals = np.clip(vals, daily_low, daily_high)
            elif mode == "scale":
                h_min, h_max = vals.min(), vals.max()
                if h_max > h_min:
                    normalized = (vals - h_min) / (h_max - h_min)
                    vals = daily_low + normalized * (daily_high - daily_low)
            elif mode == "affine":
                h_min, h_max = vals.min(), vals.max()
                if h_max > h_min:
                    # Scale to fit, preserving relative positions
                    scale = (daily_high - daily_low) / (h_max - h_min)
                    vals = daily_low + (vals - h_min) * scale
            result[key] = vals.tolist()
        return result

    def generate_clamped_forecast(
        self,
        hourly_df: pd.DataFrame,
        forecast_from_idx: int,
    ) -> dict:
        """Generate clamped 24h forecast at a specific index."""
        cfg = self.config

        # Get hourly context as dataframe
        start_idx = max(0, forecast_from_idx - cfg.context_hours)
        hourly_context_df = hourly_df.iloc[start_idx:forecast_from_idx].copy()
        if "timestamp" not in hourly_context_df.columns:
            hourly_context_df = hourly_context_df.reset_index()

        # Aggregate to daily for daily context
        daily_df = aggregate_hourly_to_daily(hourly_df.iloc[:forecast_from_idx])
        daily_context_df = daily_df.tail(cfg.context_hours // 24).copy()

        # Get unclamped hourly forecast
        hourly_fc = self.forecast_hourly(hourly_context_df, prediction_length=cfg.hourly_horizon)

        # Get daily envelope forecast
        daily_fc = self.forecast_daily(daily_context_df, prediction_length=cfg.daily_horizon)

        # Clamp hourly to daily bounds
        clamped_fc = self.clamp_hourly_to_daily(
            hourly_fc,
            daily_high=daily_fc["high_p50"],
            daily_low=daily_fc["low_p50"],
            mode=cfg.clamp_mode,
        )

        return {
            "unclamped": hourly_fc,
            "clamped": clamped_fc,
            "daily_envelope": daily_fc,
            "timestamp": hourly_df.iloc[forecast_from_idx]["timestamp"],
        }


def compute_mae_comparison(
    forecasts: Sequence[dict],
    actuals_df: pd.DataFrame,
    horizons: Sequence[int] = (1, 4, 8, 12, 24),
) -> pd.DataFrame:
    """Compare MAE of clamped vs unclamped forecasts at various horizons."""
    results = []
    for fc in forecasts:
        ts = pd.to_datetime(fc["timestamp"])
        for h in horizons:
            target_ts = ts + pd.Timedelta(hours=h)
            actual_row = actuals_df[actuals_df["timestamp"] == target_ts]
            if actual_row.empty:
                continue
            actual_close = actual_row["close"].values[0]

            unclamped_close = fc["unclamped"]["close_p50"][h - 1]
            clamped_close = fc["clamped"]["close_p50"][h - 1]

            results.append({
                "timestamp": ts,
                "horizon": h,
                "actual": actual_close,
                "unclamped_pred": unclamped_close,
                "clamped_pred": clamped_close,
                "unclamped_mae_pct": abs(unclamped_close - actual_close) / actual_close * 100,
                "clamped_mae_pct": abs(clamped_close - actual_close) / actual_close * 100,
            })

    return pd.DataFrame(results)


def run_mae_experiment(
    symbol: str = "SUIUSDT",
    data_root: Path = Path("binance_spot_hourly"),
    cache_root: Path = Path("suiclampedwithin/cache"),
    n_samples: int = 100,
    clamp_mode: str = "scale",
) -> pd.DataFrame:
    """Run full MAE comparison experiment."""
    config = ClampedForecastConfig(
        symbol=symbol,
        data_root=data_root,
        cache_root=cache_root,
        clamp_mode=clamp_mode,
    )
    forecaster = ClampedForecaster(config)
    hourly_df = load_hourly_data(data_root, symbol)

    # Sample forecast points from last portion of data
    total_rows = len(hourly_df)
    test_start = total_rows - n_samples * 2 - 24  # Leave room for actuals
    sample_indices = list(range(test_start, test_start + n_samples * 2, 2))

    forecasts = []
    for idx in sample_indices:
        if idx + 24 >= total_rows:
            break
        try:
            fc = forecaster.generate_clamped_forecast(hourly_df, idx)
            forecasts.append(fc)
            logger.info(f"Generated forecast {len(forecasts)}/{n_samples}")
        except Exception as e:
            logger.warning(f"Failed at idx {idx}: {e}")

    # Compute MAE comparison
    comparison = compute_mae_comparison(forecasts, hourly_df)

    if comparison.empty:
        logger.warning("No valid forecasts generated")
        return comparison

    # Aggregate by horizon
    summary = comparison.groupby("horizon").agg({
        "unclamped_mae_pct": ["mean", "std"],
        "clamped_mae_pct": ["mean", "std"],
    }).round(4)

    logger.info(f"\nMAE Comparison by Horizon:\n{summary}")
    return comparison


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", default="SUIUSDT")
    parser.add_argument("--n-samples", type=int, default=50)
    parser.add_argument("--clamp-mode", choices=["scale", "clamp", "affine"], default="scale")
    args = parser.parse_args()

    results = run_mae_experiment(
        symbol=args.symbol,
        n_samples=args.n_samples,
        clamp_mode=args.clamp_mode,
    )
    results.to_csv(f"suiclampedwithin/mae_comparison_{args.symbol}.csv", index=False)
