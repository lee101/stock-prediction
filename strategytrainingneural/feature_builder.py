from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence

import numpy as np
import pandas as pd


def _ensure_columns(df: pd.DataFrame, columns: Iterable[str]) -> None:
    """Make sure columns exist by inserting zeros."""
    for column in columns:
        if column not in df.columns:
            df[column] = 0.0


@dataclass
class FeatureSpec:
    """Specification describing how to transform metrics into model features."""

    numeric_stats: Mapping[str, Mapping[str, float]]
    categorical_levels: Mapping[str, Sequence[str]]
    feature_names: Sequence[str]

    def to_dict(self) -> Dict[str, object]:
        return {
            "numeric_stats": {k: dict(v) for k, v in self.numeric_stats.items()},
            "categorical_levels": {k: list(v) for k, v in self.categorical_levels.items()},
            "feature_names": list(self.feature_names),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> "FeatureSpec":
        numeric_stats = {
            key: {"mean": float(value["mean"]), "std": float(value["std"])}
            for key, value in data["numeric_stats"].items()
        }
        categorical_levels = {
            key: list(levels) for key, levels in data["categorical_levels"].items()
        }
        feature_names = list(data["feature_names"])
        return cls(numeric_stats, categorical_levels, feature_names)


class FeatureBuilder:
    """Utility that normalises numeric columns and one-hot encodes categoricals."""

    def __init__(
        self,
        numeric_columns: Sequence[str],
        categorical_columns: Sequence[str],
        *,
        add_bias: bool = True,
    ) -> None:
        self.numeric_columns = list(numeric_columns)
        self.categorical_columns = list(categorical_columns)
        self.add_bias = add_bias
        self._spec: Optional[FeatureSpec] = None

    @property
    def spec(self) -> FeatureSpec:
        if self._spec is None:
            raise RuntimeError("FeatureBuilder.fit must run before accessing spec.")
        return self._spec

    def fit(self, df: pd.DataFrame) -> FeatureSpec:
        df = df.copy()
        _ensure_columns(df, self.numeric_columns)
        numeric_stats: Dict[str, Dict[str, float]] = {}
        for column in self.numeric_columns:
            series = pd.to_numeric(df[column], errors="coerce").fillna(0.0)
            numeric_stats[column] = {
                "mean": float(series.mean()),
                "std": float(series.std(ddof=1) or 1.0),
            }

        categorical_levels: Dict[str, List[str]] = {}
        for column in self.categorical_columns:
            series = df[column].fillna("UNKNOWN").astype(str)
            levels = sorted(series.unique().tolist())
            categorical_levels[column] = levels

        feature_order: List[str] = []
        if self.add_bias:
            feature_order.append("bias")

        for column in self.numeric_columns:
            feature_order.append(column)
        for column in self.categorical_columns:
            for level in categorical_levels[column]:
                feature_order.append(f"{column}__{level}")

        self._spec = FeatureSpec(numeric_stats, categorical_levels, feature_order)
        return self._spec

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        spec = self.spec
        df = df.copy()
        _ensure_columns(df, spec.numeric_stats.keys())
        num_rows = len(df)
        features: List[np.ndarray] = []

        if self.add_bias:
            features.append(np.ones((num_rows, 1), dtype=np.float32))

        for column, stats in spec.numeric_stats.items():
            series = pd.to_numeric(df[column], errors="coerce").fillna(0.0)
            arr = series.to_numpy(dtype=np.float32)
            arr = (arr - stats["mean"]) / (stats["std"] or 1.0)
            features.append(arr.reshape(num_rows, 1))

        for column, levels in spec.categorical_levels.items():
            series = df[column].fillna("UNKNOWN").astype(str)
            mapping = {level: idx for idx, level in enumerate(levels)}
            encoded = np.zeros((num_rows, len(levels)), dtype=np.float32)
            for row_idx, value in enumerate(series):
                level_idx = mapping.get(value)
                if level_idx is not None:
                    encoded[row_idx, level_idx] = 1.0
            features.append(encoded)

        if not features:
            return np.zeros((num_rows, 0), dtype=np.float32)

        return np.concatenate(features, axis=1)

    def fit_transform(self, df: pd.DataFrame) -> np.ndarray:
        self.fit(df)
        return self.transform(df)
