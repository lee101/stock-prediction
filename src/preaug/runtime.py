from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Sequence

from preaug_sweeps.augmentations import BaseAugmentation, get_augmentation

logger = logging.getLogger(__name__)

MetricMap = Dict[str, Any]


@dataclass(frozen=True)
class PreAugmentationChoice:
    """Concrete augmentation selection for a symbol."""

    symbol: str
    strategy: str
    params: Dict[str, Any]
    metric: str
    metric_value: float
    source_path: Path

    def instantiate(self) -> BaseAugmentation:
        return get_augmentation(self.strategy, **self.params)


class PreAugmentationSelector:
    """Resolve per-symbol augmentation strategies from saved sweep results."""

    def __init__(
        self,
        best_dirs: Optional[Sequence[str | Path]] = None,
        metric_priority: Sequence[str] = ("mae_percent", "mae", "rmse", "mape"),
    ) -> None:
        dirs = best_dirs or []
        self._best_dirs: tuple[Path, ...] = tuple(Path(d) for d in dirs if d)
        self._metric_priority = tuple(metric_priority)
        self._cache: Dict[str, Optional[PreAugmentationChoice]] = {}

    def get_choice(self, symbol: str) -> Optional[PreAugmentationChoice]:
        symbol_key = symbol.upper()
        if symbol_key in self._cache:
            return self._cache[symbol_key]

        for directory in self._best_dirs:
            path = directory / f"{symbol_key}.json"
            if not path.exists():
                continue
            choice = self._load_choice(symbol_key, path)
            self._cache[symbol_key] = choice
            return choice

        self._cache[symbol_key] = None
        return None

    def _load_choice(self, symbol: str, path: Path) -> Optional[PreAugmentationChoice]:
        try:
            payload = json.loads(path.read_text())
        except Exception as exc:
            logger.warning("Failed to parse pre-augmentation config %s: %s", path, exc)
            return None

        comparison: Dict[str, MetricMap] = payload.get("comparison", {})
        metric = self._resolve_metric(payload, comparison)
        if metric is None:
            return None

        strategy = self._resolve_strategy(payload, comparison, metric)
        if strategy is None:
            return None

        metric_value = self._extract_metric_value(payload, comparison, strategy, metric)
        if metric_value is None:
            return None

        params: Dict[str, Any] = {}
        config = payload.get("config") or {}
        if str(config.get("name")) == strategy:
            params = dict(config.get("params") or {})

        return PreAugmentationChoice(
            symbol=symbol,
            strategy=strategy,
            params=params,
            metric=metric,
            metric_value=float(metric_value),
            source_path=path,
        )

    def _resolve_metric(self, payload: Dict[str, Any], comparison: Dict[str, MetricMap]) -> Optional[str]:
        preferred = payload.get("selection_metric")
        if preferred and self._metric_available(preferred, payload, comparison):
            return preferred

        for metric in self._metric_priority:
            if self._metric_available(metric, payload, comparison):
                return metric
        return None

    @staticmethod
    def _metric_available(metric: str, payload: Dict[str, Any], comparison: Dict[str, MetricMap]) -> bool:
        if any(metric in entry for entry in comparison.values()):
            return True
        return metric in payload

    def _resolve_strategy(
        self,
        payload: Dict[str, Any],
        comparison: Dict[str, MetricMap],
        metric: str,
    ) -> Optional[str]:
        declared = payload.get("best_strategy")
        if declared and self._metric_defined(metric, comparison.get(declared), payload, declared):
            return declared

        candidate = self._argmin_metric(comparison.items(), metric)
        if candidate is not None:
            return candidate

        return declared

    def _argmin_metric(self, items: Iterable[tuple[str, MetricMap]], metric: str) -> Optional[str]:
        best_name: Optional[str] = None
        best_value: float = float("inf")
        for name, entry in items:
            value = entry.get(metric)
            if value is None:
                continue
            try:
                numeric = float(value)
            except (TypeError, ValueError):
                continue
            if numeric < best_value:
                best_value = numeric
                best_name = name
        return best_name

    def _metric_defined(
        self,
        metric: str,
        comparison_entry: Optional[MetricMap],
        payload: Dict[str, Any],
        strategy: str,
    ) -> bool:
        if comparison_entry and metric in comparison_entry:
            return True
        if payload.get("best_strategy") == strategy and metric in payload:
            return True
        if metric == payload.get("selection_metric") and payload.get("selection_value") is not None:
            return True
        return False

    def _extract_metric_value(
        self,
        payload: Dict[str, Any],
        comparison: Dict[str, MetricMap],
        strategy: str,
        metric: str,
    ) -> Optional[float]:
        entry = comparison.get(strategy)
        if entry and metric in entry:
            try:
                return float(entry[metric])
            except (TypeError, ValueError):
                return None

        if payload.get("best_strategy") == strategy:
            value = payload.get(metric)
            if value is None and payload.get("selection_metric") == metric:
                value = payload.get("selection_value")
            if value is not None:
                try:
                    return float(value)
                except (TypeError, ValueError):
                    return None
        return None


__all__ = ["PreAugmentationChoice", "PreAugmentationSelector"]
