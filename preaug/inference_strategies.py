"""Inference-time ensemble strategies for Chronos2.

These wrap the prediction call to combine multiple inferences,
unlike pre-augmentation transforms which only modify input data.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np
from loguru import logger


def _median_quantile_index(quantiles: Sequence[float]) -> int:
    distances = [abs(float(q) - 0.5) for q in quantiles]
    return int(np.argmin(distances))


def _extract_median(pipeline: Any, predictions: list, quantile_index: Optional[int] = None) -> np.ndarray:
    quantiles = getattr(pipeline, "quantiles", [0.1, 0.5, 0.9])
    qi = quantile_index if quantile_index is not None else _median_quantile_index(quantiles)
    pred = predictions[0].detach().cpu().numpy()
    return pred[:, qi, :]


class BaseInferenceStrategy(ABC):
    @abstractmethod
    def name(self) -> str: ...

    @abstractmethod
    def predict(
        self,
        pipeline: Any,
        context: np.ndarray,
        prediction_length: int = 1,
        quantile_index: Optional[int] = None,
    ) -> np.ndarray:
        """
        Args:
            pipeline: Chronos2Pipeline
            context: (n_series, n_timesteps) e.g. (4, 332) for OHLC
            prediction_length: steps ahead
            quantile_index: into quantile array for median

        Returns: (n_series, prediction_length)
        """
        ...

    def get_config(self) -> Dict[str, Any]:
        return {"name": self.name()}


class SingleInference(BaseInferenceStrategy):
    """Baseline: single Chronos2 inference on raw context."""

    def __init__(self, max_context: int = 336):
        self.max_context = max_context

    def name(self) -> str:
        return "single"

    def predict(self, pipeline, context, prediction_length=1, quantile_index=None):
        if context.shape[1] > self.max_context:
            context = context[:, -self.max_context:]
        preds = pipeline.predict([context], prediction_length=prediction_length, batch_size=1)
        return _extract_median(pipeline, preds, quantile_index)


class TemporalDilationEnsemble(BaseInferenceStrategy):
    """Multiple inferences at different temporal resolutions, trimmed-mean combined.

    Each stride subsamples the context from the most recent point backwards:
      stride=1: every point (original resolution)
      stride=2: every 2nd point (2x temporal coverage per token)
      stride=4: every 4th point (4x temporal coverage)
      stride=8: every 8th point (8x temporal coverage)

    All inferences anchor at the current (most recent) timestep.
    Trimmed mean drops `trim` extremes from each end before averaging.
    """

    def __init__(
        self,
        strides: Tuple[int, ...] = (1, 2, 4, 8),
        trim: int = 1,
        target_points: int = 336,
    ):
        self.strides = strides
        self.trim = trim
        self.target_points = target_points

    def name(self) -> str:
        s = "_".join(str(s) for s in self.strides)
        return f"dilation_s{s}_t{self.trim}"

    def predict(self, pipeline, context, prediction_length=1, quantile_index=None):
        n_series, n_time = context.shape
        all_preds = []

        for stride in self.strides:
            # indices anchored at the last (most recent) timestep, stepping back by stride
            indices = np.arange(n_time - 1, -1, -stride)[::-1]
            # cap to target_points most recent subsampled points
            if len(indices) > self.target_points:
                indices = indices[-self.target_points:]
            if len(indices) < 4:
                continue

            dilated = context[:, indices]
            preds = pipeline.predict([dilated], prediction_length=prediction_length, batch_size=1)
            all_preds.append(_extract_median(pipeline, preds, quantile_index))

        if not all_preds:
            return np.zeros((n_series, prediction_length))
        if len(all_preds) == 1:
            return all_preds[0]

        stacked = np.stack(all_preds, axis=0)  # (n_inferences, n_series, pred_len)
        if self.trim > 0 and len(all_preds) > 2 * self.trim:
            sorted_preds = np.sort(stacked, axis=0)
            trimmed = sorted_preds[self.trim : -self.trim]
        else:
            trimmed = stacked
        return trimmed.mean(axis=0)

    def get_config(self) -> Dict[str, Any]:
        return {"name": self.name(), "strides": self.strides, "trim": self.trim, "target_points": self.target_points}


INFERENCE_STRATEGY_REGISTRY = {
    "single": SingleInference,
    "dilation": TemporalDilationEnsemble,
}


def get_inference_strategy(name: str, **kwargs) -> BaseInferenceStrategy:
    if name not in INFERENCE_STRATEGY_REGISTRY:
        raise ValueError(f"Unknown inference strategy: {name}")
    return INFERENCE_STRATEGY_REGISTRY[name](**kwargs)
