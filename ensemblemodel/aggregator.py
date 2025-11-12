"""Aggregators for ensemble forecasting."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence, Tuple

import numpy as np

from src.models.toto_aggregation import aggregate_with_spec

from .backends import BackendResult


def _coerce_sample_block(samples: np.ndarray) -> np.ndarray:
    arr = np.asarray(samples, dtype=np.float64)
    if arr.ndim == 1:
        return arr.reshape(-1, 1)
    if arr.ndim == 2:
        return arr
    return np.reshape(arr, (arr.shape[0], -1))


def _logsumexp(values: np.ndarray, axis: int | None = None, keepdims: bool = False) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    if axis is None:
        max_val = np.max(arr)
        total = np.sum(np.exp(arr - max_val))
        result = max_val + np.log(total)
        return np.array(result) if keepdims else result
    max_val = np.max(arr, axis=axis, keepdims=True)
    total = np.sum(np.exp(arr - max_val), axis=axis, keepdims=True)
    result = max_val + np.log(total)
    if not keepdims:
        result = np.squeeze(result, axis=axis)
    return result


@dataclass(slots=True)
class EnsembleColumnForecast:
    """Aggregated forecast metadata for a single column."""

    column: str
    prediction: np.ndarray
    combined_samples: np.ndarray
    backend_summaries: Dict[str, Dict[str, float]] = field(default_factory=dict)


@dataclass(slots=True)
class EnsembleForecast:
    """Container for aggregated ensemble output."""

    method: str
    clip_percentile: float
    clip_std: float
    columns: Dict[str, EnsembleColumnForecast]

    def to_dict(self) -> Dict[str, Dict[str, float]]:
        """Return lightweight dict for logging/JSON."""

        summary: Dict[str, Dict[str, float]] = {}
        for column, forecast in self.columns.items():
            summary[column] = {
                "prediction": float(np.asarray(forecast.prediction).ravel()[0]),
                "num_samples": float(forecast.combined_samples.shape[0]),
            }
        summary["_meta"] = {
            "method": self.method,
            "clip_percentile": self.clip_percentile,
            "clip_std": self.clip_std,
        }
        return summary


class ClippedMeanAggregator:
    """Apply clipping + Toto-style aggregation across backend samples."""

    def __init__(
        self,
        *,
        method: str = "trimmed_mean_10",
        clip_percentile: float = 0.10,
        clip_std: float = 2.5,
        weight_resolution: int = 8,
    ) -> None:
        if weight_resolution <= 0:
            raise ValueError("weight_resolution must be positive")
        self.method = method
        self.clip_percentile = float(max(0.0, min(0.5, clip_percentile)))
        self.clip_std = float(max(0.0, clip_std))
        self.weight_resolution = int(weight_resolution)

    def combine(
        self,
        backends: Sequence[BackendResult],
        columns: Sequence[str],
    ) -> EnsembleForecast:
        if not backends:
            raise RuntimeError("No backend results available for aggregation")

        aggregated: Dict[str, EnsembleColumnForecast] = {}
        for column in columns:
            column_blocks = []
            backend_meta: Dict[str, Dict[str, float]] = {}
            for result in backends:
                samples = result.samples.get(column)
                if samples is None or samples.size == 0:
                    continue
                normalized = self._normalize_block(samples)
                clipped = self._clip_block(normalized)
                repeats = max(1, int(round(result.weight * self.weight_resolution)))
                weighted = np.repeat(clipped, repeats, axis=0) if repeats > 1 else clipped
                column_blocks.append(weighted)
                backend_meta[result.name] = {
                    "weight": float(result.weight),
                    "latency_ms": float(result.latency_s * 1000.0),
                    "num_samples": float(normalized.shape[0]),
                    "mean": float(np.mean(normalized)),
                    "std": float(np.std(normalized)),
                }

            if not column_blocks:
                continue

            stacked = np.concatenate(column_blocks, axis=0)
            prediction = aggregate_with_spec(stacked, self.method)
            aggregated[column] = EnsembleColumnForecast(
                column=column,
                prediction=prediction,
                combined_samples=stacked,
                backend_summaries=backend_meta,
            )

        if not aggregated:
            raise RuntimeError("Aggregation produced no columns; verify backend outputs")

        return EnsembleForecast(
            method=self.method,
            clip_percentile=self.clip_percentile,
            clip_std=self.clip_std,
            columns=aggregated,
        )

    def _normalize_block(self, samples: np.ndarray) -> np.ndarray:
        return _coerce_sample_block(samples)

    def _clip_block(self, samples: np.ndarray) -> np.ndarray:
        clipped = samples
        if self.clip_percentile > 0:
            lower = np.quantile(samples, self.clip_percentile, axis=0, keepdims=True)
            upper = np.quantile(samples, 1.0 - self.clip_percentile, axis=0, keepdims=True)
            clipped = np.clip(clipped, lower, upper)
        if self.clip_std > 0:
            mean = np.mean(clipped, axis=0, keepdims=True)
            std = np.std(clipped, axis=0, keepdims=True)
            std = np.where(std == 0, 1.0, std)
            clipped = np.clip(clipped, mean - self.clip_std * std, mean + self.clip_std * std)
        return clipped


@dataclass(slots=True)
class _BackendColumnSummary:
    name: str
    weight: float
    samples: np.ndarray
    mean: np.ndarray
    std: np.ndarray


class PairwiseHMMVotingAggregator:
    """Blend two target backends with HMM-smoothed voting + fallback aggregation."""

    def __init__(
        self,
        *,
        pair: Sequence[str] = ("chronos2", "toto"),
        fallback: Optional[ClippedMeanAggregator] = None,
        switch_prob: float = 0.15,
        temperature: float = 0.65,
        vote_strength: float = 0.35,
        vote_temperature: float = 6.0,
        residual_scale: float = 0.5,
        min_pair_blend: float = 0.15,
        max_pair_blend: float = 0.75,
        min_std: float = 1e-3,
    ) -> None:
        if len(pair) != 2:
            raise ValueError("pair must contain exactly two backend names")
        self.pair: Tuple[str, str] = (str(pair[0]), str(pair[1]))
        self.fallback = fallback or ClippedMeanAggregator()
        self.switch_prob = float(np.clip(switch_prob, 1e-3, 0.49))
        self.temperature = float(max(1e-6, temperature))
        self.vote_strength = float(vote_strength)
        self.vote_temperature = float(max(1e-3, vote_temperature))
        self.residual_scale = float(max(0.0, residual_scale))
        self.min_pair_blend = float(np.clip(min_pair_blend, 0.0, 1.0))
        self.max_pair_blend = float(np.clip(max_pair_blend, self.min_pair_blend, 1.0))
        self.min_std = float(max(1e-6, min_std))
        self.method = f"hmm_vote_{self.pair[0]}_{self.pair[1]}"

    def combine(
        self,
        backends: Sequence[BackendResult],
        columns: Sequence[str],
    ) -> EnsembleForecast:
        baseline = self.fallback.combine(backends, columns)
        updated: Dict[str, EnsembleColumnForecast] = {}
        for column in columns:
            base_column = baseline.columns.get(column)
            if base_column is None:
                continue
            enriched = self._apply_pairwise_strategy(backends, column, base_column)
            updated[column] = enriched or base_column
        return EnsembleForecast(
            method=self.method,
            clip_percentile=self.fallback.clip_percentile,
            clip_std=self.fallback.clip_std,
            columns=updated,
        )

    def _apply_pairwise_strategy(
        self,
        backends: Sequence[BackendResult],
        column: str,
        base_column: EnsembleColumnForecast,
    ) -> Optional[EnsembleColumnForecast]:
        summaries: Dict[str, _BackendColumnSummary] = {}
        for backend in backends:
            summary = self._summarize_backend(backend, column)
            if summary is not None:
                summaries[backend.name] = summary

        first = summaries.get(self.pair[0])
        second = summaries.get(self.pair[1])
        if first is None or second is None:
            return None

        voters = [summary for name, summary in summaries.items() if name not in self.pair]
        base_prediction = np.asarray(base_column.prediction, dtype=np.float64).reshape(-1)
        pair_prediction, meta = self._compute_pairwise_prediction(first, second, voters, base_prediction)
        if pair_prediction is None:
            return None

        backend_meta = dict(base_column.backend_summaries)
        backend_meta["pairwise_hmm_vote"] = meta
        return EnsembleColumnForecast(
            column=base_column.column,
            prediction=pair_prediction,
            combined_samples=base_column.combined_samples,
            backend_summaries=backend_meta,
        )

    def _summarize_backend(self, backend: BackendResult, column: str) -> Optional[_BackendColumnSummary]:
        samples = backend.samples.get(column)
        if samples is None or samples.size == 0:
            return None
        normalized = _coerce_sample_block(samples)
        mean = np.mean(normalized, axis=0)
        std = np.std(normalized, axis=0)
        std = np.clip(std, self.min_std, None)
        weight = float(max(self.min_std, backend.weight))
        return _BackendColumnSummary(
            name=backend.name,
            weight=weight,
            samples=normalized,
            mean=mean,
            std=std,
        )

    def _compute_pairwise_prediction(
        self,
        first: _BackendColumnSummary,
        second: _BackendColumnSummary,
        voters: Sequence[_BackendColumnSummary],
        baseline: np.ndarray,
    ) -> Tuple[Optional[np.ndarray], Dict[str, float]]:
        if first.mean.shape != baseline.shape or second.mean.shape != baseline.shape:
            return None, {}

        proxy_first = self._mae_proxy(first, baseline)
        proxy_second = self._mae_proxy(second, baseline)
        vote_gap = self._compute_votes(voters, first, second)

        emissions = np.vstack(
            [
                self._emissions_from_proxy(proxy_first, vote_gap),
                self._emissions_from_proxy(proxy_second, -vote_gap),
            ]
        )
        state_prob = self._state_posterior(emissions)
        pair_prediction = state_prob * first.mean + (1.0 - state_prob) * second.mean
        confidence = np.clip(np.abs(state_prob - 0.5) * 2.0, 0.0, 1.0)
        blend = self.min_pair_blend + (self.max_pair_blend - self.min_pair_blend) * confidence
        final = blend * pair_prediction + (1.0 - blend) * baseline

        meta = {
            "pair": "->".join(self.pair),
            "state_prob_avg": float(np.mean(state_prob)),
            "confidence_avg": float(np.mean(confidence)),
            "blend_avg": float(np.mean(blend)),
            "vote_bias": float(np.mean(vote_gap)),
        }
        return final, meta

    def _mae_proxy(self, summary: _BackendColumnSummary, baseline: np.ndarray) -> np.ndarray:
        residual = np.abs(summary.mean - baseline)
        return summary.std + self.residual_scale * residual

    def _compute_votes(
        self,
        voters: Sequence[_BackendColumnSummary],
        first: _BackendColumnSummary,
        second: _BackendColumnSummary,
    ) -> np.ndarray:
        if not voters:
            return np.zeros_like(first.mean)
        vote = np.zeros_like(first.mean)
        for voter in voters:
            if voter.mean.shape != first.mean.shape:
                continue
            diff_first = np.abs(voter.mean - first.mean)
            diff_second = np.abs(voter.mean - second.mean)
            gap = diff_second - diff_first  # positive => favors first
            scale = np.clip(voter.std, self.min_std, None) * self.vote_temperature
            vote += voter.weight * np.tanh(gap / scale)
        return vote

    def _emissions_from_proxy(self, proxy: np.ndarray, vote_term: np.ndarray) -> np.ndarray:
        return (-proxy / self.temperature) + self.vote_strength * vote_term

    def _state_posterior(self, emissions: np.ndarray) -> np.ndarray:
        _, horizon = emissions.shape
        if horizon == 0:
            return np.array([])
        stay = np.log(1.0 - self.switch_prob)
        switch = np.log(self.switch_prob)
        transition = np.array([[stay, switch], [switch, stay]])
        log_alpha = np.empty_like(emissions)
        init = np.log(0.5)
        log_alpha[:, 0] = init + emissions[:, 0]
        for t in range(1, horizon):
            for state in range(2):
                log_alpha[state, t] = emissions[state, t] + _logsumexp(
                    log_alpha[:, t - 1] + transition[:, state], axis=0
                )

        log_beta = np.zeros_like(emissions)
        for t in range(horizon - 2, -1, -1):
            for state in range(2):
                log_beta[state, t] = _logsumexp(
                    transition[state, :] + emissions[:, t + 1] + log_beta[:, t + 1], axis=0
                )

        log_gamma = log_alpha + log_beta
        norm = _logsumexp(log_gamma, axis=0, keepdims=True)
        posterior = np.exp(log_gamma - norm)
        return posterior[0]
