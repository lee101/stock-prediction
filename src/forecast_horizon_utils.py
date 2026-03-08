from __future__ import annotations

import re
from collections.abc import Iterable, Sequence

_CHRONOS_HORIZON_RE = re.compile(r"_h(\d+)$")


def _coerce_horizons(values: Sequence[int] | Iterable[int] | None) -> tuple[int, ...]:
    if values is None:
        return ()
    seen: set[int] = set()
    horizons: list[int] = []
    for value in values:
        try:
            horizon = int(value)
        except (TypeError, ValueError):
            continue
        if horizon <= 0 or horizon in seen:
            continue
        seen.add(horizon)
        horizons.append(horizon)
    return tuple(sorted(horizons))


def infer_feature_horizons(feature_columns: Iterable[str] | None) -> tuple[int, ...]:
    """Infer required forecast horizons from Chronos feature column names."""
    if feature_columns is None:
        return ()
    discovered: list[int] = []
    seen: set[int] = set()
    for column in feature_columns:
        token = str(column or "")
        if not token.startswith("chronos_"):
            continue
        match = _CHRONOS_HORIZON_RE.search(token)
        if match is None:
            continue
        horizon = int(match.group(1))
        if horizon <= 0 or horizon in seen:
            continue
        seen.add(horizon)
        discovered.append(horizon)
    return tuple(sorted(discovered))


def resolve_required_forecast_horizons(
    requested_horizons: Sequence[int] | Iterable[int] | None,
    *,
    feature_columns: Iterable[str] | None = None,
    fallback_horizons: Sequence[int] | Iterable[int] | None = None,
) -> tuple[int, ...]:
    """Return the unique sorted union of requested and inferred forecast horizons."""
    requested = set(_coerce_horizons(requested_horizons))
    inferred = set(infer_feature_horizons(feature_columns))
    if not inferred and feature_columns is not None:
        inferred = set(_coerce_horizons(fallback_horizons))
    merged = requested | inferred
    return tuple(sorted(merged))


__all__ = [
    "infer_feature_horizons",
    "resolve_required_forecast_horizons",
]
