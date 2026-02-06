from __future__ import annotations

import re
from dataclasses import is_dataclass, replace
from typing import List, Sequence, TypeVar


_T = TypeVar("_T")

_HOURS_SUFFIX_RE = re.compile(r"(\d+)h\b")
_HORIZON_SUFFIX_RE = re.compile(r"(?:^|_)h(\d+)\b")
_VOL_REGIME_RE = re.compile(r"\bvol_regime_(\d+)_(\d+)\b")


def _extract_hour_tokens(feature_name: str) -> List[int]:
    if not isinstance(feature_name, str):
        raise TypeError(f"feature_name must be str, got {type(feature_name).__name__}")
    hours: List[int] = []
    for match in _HOURS_SUFFIX_RE.finditer(feature_name):
        hours.append(int(match.group(1)))
    for match in _HORIZON_SUFFIX_RE.finditer(feature_name):
        hours.append(int(match.group(1)))
    for match in _VOL_REGIME_RE.finditer(feature_name):
        hours.append(int(match.group(1)))
        hours.append(int(match.group(2)))
    return hours


def filter_feature_columns_max_window(columns: Sequence[str], *, max_window_hours: int) -> List[str]:
    """Filter out feature columns that reference hour windows larger than max_window_hours.

    This is a pragmatic helper for short-history datasets (e.g. new Binance markets) where
    long rolling-window features create NaNs and collapse the usable frame.
    """
    max_window = int(max_window_hours)
    if max_window < 2:
        raise ValueError(f"max_window_hours must be >= 2, got {max_window_hours}.")
    kept: List[str] = []
    for col in columns:
        if not isinstance(col, str) or not col:
            continue
        hours = _extract_hour_tokens(col)
        if hours and max(hours) > max_window:
            continue
        kept.append(col)
    return kept


def apply_feature_max_window_hours(config: _T, *, max_window_hours: int) -> _T:
    """Apply a max window constraint to a DatasetConfig-like dataclass.

    - Truncates known rolling-window config fields (MA/EMA/ATR/etc) to <= max_window_hours.
    - Rebuilds feature_columns using binanceexp1 defaults and filters them by max_window_hours.
    """
    if not is_dataclass(config):
        raise TypeError(f"config must be a dataclass instance, got {type(config).__name__}")
    max_window = int(max_window_hours)
    if max_window < 2:
        raise ValueError(f"max_window_hours must be >= 2, got {max_window_hours}.")

    def _fields():
        return getattr(config, "__dataclass_fields__", {})

    def _replace(obj: _T, **updates) -> _T:
        allowed = set(_fields().keys())
        filtered = {k: v for k, v in updates.items() if k in allowed}
        if not filtered:
            return obj
        return replace(obj, **filtered)

    def _cap_window_tuple(values) -> tuple[int, ...]:
        cleaned: List[int] = []
        for v in values or ():
            try:
                n = int(v)
            except (TypeError, ValueError):
                continue
            if n < 2:
                continue
            if n <= max_window:
                cleaned.append(n)
        return tuple(cleaned)

    updates: dict = {}

    if hasattr(config, "moving_average_windows"):
        ma = _cap_window_tuple(getattr(config, "moving_average_windows"))
        if not ma:
            ma = (max(2, min(24, max_window)),)
        updates["moving_average_windows"] = ma

    if hasattr(config, "ema_windows"):
        ema = _cap_window_tuple(getattr(config, "ema_windows"))
        if not ema:
            ema = (max(2, min(24, max_window)),)
        updates["ema_windows"] = ema

    if hasattr(config, "atr_windows"):
        atr = _cap_window_tuple(getattr(config, "atr_windows"))
        if not atr:
            atr = (max(2, min(24, max_window)),)
        updates["atr_windows"] = atr

    if hasattr(config, "trend_windows"):
        updates["trend_windows"] = _cap_window_tuple(getattr(config, "trend_windows"))

    if hasattr(config, "drawdown_windows"):
        updates["drawdown_windows"] = _cap_window_tuple(getattr(config, "drawdown_windows"))

    if hasattr(config, "volume_z_window"):
        v = int(getattr(config, "volume_z_window") or 0)
        if v < 2:
            v = max_window
        updates["volume_z_window"] = max(2, min(v, max_window))

    if hasattr(config, "volume_shock_window"):
        v = int(getattr(config, "volume_shock_window") or 0)
        if v < 2:
            v = max(2, min(24, max_window))
        updates["volume_shock_window"] = max(2, min(v, max_window))

    if hasattr(config, "rsi_window"):
        v = int(getattr(config, "rsi_window") or 0)
        if v < 2:
            v = max(2, min(14, max_window))
        updates["rsi_window"] = max(2, min(v, max_window))

    if hasattr(config, "vol_regime_long") or hasattr(config, "vol_regime_short"):
        long_default = int(getattr(config, "vol_regime_long", max_window) or max_window)
        long_window = max(3, min(long_default, max_window))
        short_default = int(getattr(config, "vol_regime_short", 24) or 24)
        short_window = max(2, min(short_default, max(2, long_window // 2)))
        if short_window >= long_window:
            short_window = max(2, long_window - 1)
        updates["vol_regime_short"] = short_window
        updates["vol_regime_long"] = long_window

    capped = _replace(config, **updates)

    from binanceexp1.data import build_default_feature_columns  # Local import: heavy deps.

    columns = build_default_feature_columns(capped)
    filtered = filter_feature_columns_max_window(columns, max_window_hours=max_window)
    if not filtered:
        raise ValueError("Filtering feature columns produced an empty feature set.")

    capped = _replace(capped, feature_columns=tuple(filtered))
    return capped


__all__ = [
    "apply_feature_max_window_hours",
    "filter_feature_columns_max_window",
]

