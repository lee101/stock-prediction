from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional, Sequence

from binanceneural.config import ForecastConfig
from binanceneural.forecasts import ChronosForecastManager


def build_forecast_bundle(
    *,
    symbol: str,
    data_root: Path,
    cache_root: Path,
    horizons: Sequence[int],
    context_hours: int,
    quantile_levels: Sequence[float],
    batch_size: int,
    model_id: str,
    device_map: str = "cuda",
    preaugmentation_dirs: Optional[Sequence[Path]] = None,
    cache_only: bool = False,
    start: Optional["pd.Timestamp"] = None,
    end: Optional["pd.Timestamp"] = None,
    wrapper_factory: Optional[Callable[[], object]] = None,
) -> "pd.DataFrame":
    """Build merged Chronos forecast frame with horizon-specific suffixes."""

    import pandas as pd

    if wrapper_factory is None:
        wrapper_factory = _build_wrapper_factory(
            model_id=model_id,
            device_map=device_map,
            context_hours=context_hours,
            batch_size=batch_size,
            quantile_levels=quantile_levels,
            preaugmentation_dirs=preaugmentation_dirs,
        )

    merged: Optional[pd.DataFrame] = None
    for horizon in horizons:
        horizon_dir = cache_root / f"h{int(horizon)}"
        cfg = ForecastConfig(
            symbol=symbol,
            data_root=data_root,
            context_hours=context_hours,
            prediction_horizon_hours=int(horizon),
            quantile_levels=tuple(quantile_levels),
            batch_size=batch_size,
            cache_dir=horizon_dir,
        )
        manager = ChronosForecastManager(cfg, wrapper_factory=wrapper_factory)
        forecast = manager.ensure_latest(start=start, end=end, cache_only=cache_only)
        if forecast.empty:
            continue
        suffix = f"_h{int(horizon)}"
        horizon_frame = forecast.rename(
            columns={
                "predicted_close_p50": f"predicted_close_p50{suffix}",
                "predicted_close_p10": f"predicted_close_p10{suffix}",
                "predicted_close_p90": f"predicted_close_p90{suffix}",
                "predicted_high_p50": f"predicted_high_p50{suffix}",
                "predicted_low_p50": f"predicted_low_p50{suffix}",
            }
        )
        horizon_frame = horizon_frame[[
            "timestamp",
            "symbol",
            f"predicted_close_p50{suffix}",
            f"predicted_high_p50{suffix}",
            f"predicted_low_p50{suffix}",
        ]]
        if merged is None:
            merged = horizon_frame
        else:
            merged = merged.merge(horizon_frame, on=["timestamp", "symbol"], how="inner")

    if merged is None:
        raise RuntimeError(
            f"No Chronos forecasts available for {symbol}. Generate caches first or disable cache_only."
        )
    return merged.sort_values("timestamp").reset_index(drop=True)


def _build_wrapper_factory(
    *,
    model_id: str,
    device_map: str,
    context_hours: int,
    batch_size: int,
    quantile_levels: Sequence[float],
    preaugmentation_dirs: Optional[Sequence[Path]],
) -> Callable[[], object]:
    wrapper: Optional[object] = None

    def _factory() -> object:
        nonlocal wrapper
        if wrapper is not None:
            return wrapper
        from src.models.chronos2_wrapper import Chronos2OHLCWrapper

        wrapper = Chronos2OHLCWrapper.from_pretrained(
            model_id=model_id,
            device_map=device_map,
            default_context_length=context_hours,
            default_batch_size=batch_size,
            quantile_levels=tuple(float(q) for q in quantile_levels),
            preaugmentation_dirs=preaugmentation_dirs,
        )
        return wrapper

    return _factory


__all__ = ["build_forecast_bundle"]
