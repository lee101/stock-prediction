"""Lightweight runtime helpers for loading neural sizing policies in prod."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence

import pandas as pd
import torch

from .data import load_daily_metrics
from .feature_builder import FeatureBuilder, FeatureSpec
from .models import PolicyConfig, PortfolioPolicy

LOGGER = logging.getLogger(__name__)


def _resolve_device(preferred: Optional[str] = None) -> torch.device:
    if preferred:
        return torch.device(preferred)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@dataclass
class StrategyScore:
    name: str
    weight: float
    date: str


class NeuralStrategyEvaluator:
    """Run-time evaluator that scores sizing strategies via a trained policy."""

    def __init__(
        self,
        *,
        run_dir: str | Path,
        metrics_csv: str | Path,
        strategy_filter: Optional[Sequence[str]] = None,
        device: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        asset_class: str = "all",
    ) -> None:
        self.run_dir = Path(run_dir)
        self.metrics_csv = Path(metrics_csv)
        self.device = _resolve_device(device)
        self._feature_spec = self._load_spec()
        self._model = self._load_model()
        self._builder = self._init_builder()
        self._frame = load_daily_metrics(
            str(self.metrics_csv),
            strategy_filter=strategy_filter,
            start_date=start_date,
            end_date=end_date,
            asset_class_filter=asset_class,
        )
        if self._frame.empty:
            raise ValueError("Loaded sizing metrics are empty; cannot evaluate neural policy.")

    def _load_spec(self) -> FeatureSpec:
        spec_path = self.run_dir / "feature_spec.json"
        if not spec_path.exists():
            raise FileNotFoundError(f"Missing feature_spec.json in {self.run_dir}")
        data = json.loads(spec_path.read_text())
        return FeatureSpec.from_dict(data)

    def _load_model(self) -> PortfolioPolicy:
        state_path = self.run_dir / "sortino_policy.pt"
        if not state_path.exists():
            raise FileNotFoundError(f"Missing sortino_policy.pt in {self.run_dir}")
        input_dim = len(self._feature_spec.feature_names)
        config = PolicyConfig(input_dim=input_dim)
        model = PortfolioPolicy(config)
        state = torch.load(state_path, map_location=self.device)
        model.load_state_dict(state)
        model.to(self.device)
        model.eval()
        return model

    def _init_builder(self) -> FeatureBuilder:
        numeric_columns = list(self._feature_spec.numeric_stats.keys())
        categorical_columns = list(self._feature_spec.categorical_levels.keys())
        add_bias = "bias" in self._feature_spec.feature_names
        builder = FeatureBuilder(numeric_columns, categorical_columns, add_bias=add_bias)
        builder._spec = self._feature_spec
        return builder

    def _latest_slice(self, strategy: str, cutoff: Optional[pd.Timestamp]) -> Optional[pd.Series]:
        subset = self._frame[self._frame["strategy"] == strategy]
        if subset.empty:
            return None
        if cutoff is not None:
            subset = subset[subset["date"] <= cutoff]
        if subset.empty:
            return None
        return subset.iloc[-1]

    def score_strategies(
        self,
        strategies: Sequence[str],
        *,
        cutoff_date: Optional[str] = None,
    ) -> List[StrategyScore]:
        cutoff_ts = pd.to_datetime(cutoff_date) if cutoff_date else None
        rows: List[pd.Series] = []
        selected_names: List[str] = []

        for name in strategies:
            sample = self._latest_slice(name, cutoff_ts)
            if sample is None:
                LOGGER.debug("Neural evaluator skipping %s – no matching rows", name)
                continue
            rows.append(sample)
            selected_names.append(name)

        if not rows:
            return []

        frame = pd.DataFrame(rows).reset_index(drop=True)
        features = self._builder.transform(frame)
        with torch.inference_mode():
            tensor = torch.from_numpy(features.astype("float32")).to(self.device)
            weights = self._model(tensor).detach().cpu().numpy()

        scores: List[StrategyScore] = []
        for name, weight, (_, row) in zip(selected_names, weights, frame.iterrows()):
            row_date = getattr(row.get("date"), "date", None)
            iso_date = str(row.get("date")) if row_date is None else row.get("date").date().isoformat()
            scores.append(StrategyScore(name=name, weight=float(weight), date=iso_date))
        return scores

    def average_weight(
        self,
        strategies: Sequence[str],
        *,
        cutoff_date: Optional[str] = None,
    ) -> Optional[float]:
        scores = self.score_strategies(strategies, cutoff_date=cutoff_date)
        if not scores:
            return None
        return float(sum(score.weight for score in scores) / len(scores))


__all__ = ["NeuralStrategyEvaluator", "StrategyScore"]
