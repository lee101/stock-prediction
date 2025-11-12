from __future__ import annotations

import numpy as np
import pandas as pd

from ensemblemodel.aggregator import ClippedMeanAggregator, PairwiseHMMVotingAggregator
from ensemblemodel.backends import BackendResult, EnsembleBackend, EnsembleRequest
from ensemblemodel.pipeline import EnsembleForecastPipeline


class _FakeBackend(EnsembleBackend):
    def __init__(self, name: str, samples: np.ndarray, *, weight: float = 1.0):
        super().__init__(name=name, weight=weight, enabled=True)
        self._samples = samples

    def run(self, request: EnsembleRequest) -> BackendResult:
        block = {col: self._samples.copy() for col in request.columns}
        return BackendResult(
            name=self.name,
            weight=self.weight,
            latency_s=0.001,
            samples=block,
        )


def test_clipped_mean_aggregator_downweights_outliers():
    agg = ClippedMeanAggregator(method="trimmed_mean_10", clip_percentile=0.10, clip_std=1.0, weight_resolution=4)
    base = BackendResult(
        name="stable",
        weight=1.0,
        latency_s=0.001,
        samples={"close": np.array([[1.0], [1.1], [0.9]])},
    )
    outlier = BackendResult(
        name="noisy",
        weight=1.0,
        latency_s=0.001,
        samples={"close": np.array([[15.0]])},
    )
    forecast = agg.combine([base, outlier], ["close"])
    value = float(forecast.columns["close"].prediction[0])
    assert value < 4.5  # trimmed aggregation dampens extreme outlier


def test_aggregator_respects_weights():
    agg = ClippedMeanAggregator(method="mean", clip_percentile=0.0, clip_std=0.0, weight_resolution=8)
    heavy = BackendResult(
        name="heavy",
        weight=2.0,
        latency_s=0.0,
        samples={"close": np.array([[1.0]])},
    )
    light = BackendResult(
        name="light",
        weight=0.5,
        latency_s=0.0,
        samples={"close": np.array([[5.0]])},
    )
    forecast = agg.combine([heavy, light], ["close"])
    value = float(forecast.columns["close"].prediction[0])
    assert value < 3.0  # weighted towards heavy backend


def test_pipeline_runs_with_fake_backends():
    df = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=10, freq="D"),
            "close": np.linspace(10, 19, num=10),
        }
    )
    backend_a = _FakeBackend("a", np.array([[10.0]]))
    backend_b = _FakeBackend("b", np.array([[20.0]]), weight=0.5)
    pipeline = EnsembleForecastPipeline([backend_a, backend_b], aggregator=ClippedMeanAggregator())
    output = pipeline.predict(df, columns=["close"], prediction_length=1)
    assert "close" in output.forecast.columns
    prediction = float(output.forecast.columns["close"].prediction[0])
    assert 10.0 <= prediction <= 20.0


def test_pairwise_hmm_voting_prefers_lower_variance_backend():
    chronos = BackendResult(
        name="chronos2",
        weight=1.0,
        latency_s=0.0,
        samples={"close": np.array([[0.98], [1.02], [1.01]])},
    )
    toto = BackendResult(
        name="toto",
        weight=1.0,
        latency_s=0.0,
        samples={"close": np.array([[1.8], [2.0], [2.2]])},
    )
    kronos = BackendResult(
        name="kronos",
        weight=0.8,
        latency_s=0.0,
        samples={"close": np.array([[1.05], [1.10], [0.95]])},
    )

    aggregator = PairwiseHMMVotingAggregator(
        pair=("chronos2", "toto"),
        fallback=ClippedMeanAggregator(method="mean", clip_percentile=0.0, clip_std=0.0),
        switch_prob=0.2,
        temperature=0.5,
        vote_strength=0.5,
        residual_scale=0.25,
        max_pair_blend=0.9,
    )

    forecast = aggregator.combine([chronos, toto, kronos], ["close"])
    column = forecast.columns["close"]
    value = float(column.prediction[0])
    assert abs(value - 1.0) < abs(value - 2.0)
    assert "pairwise_hmm_vote" in column.backend_summaries


def test_pairwise_hmm_voting_falls_back_without_full_pair():
    chronos = BackendResult(
        name="chronos2",
        weight=1.0,
        latency_s=0.0,
        samples={"close": np.array([[1.0]])},
    )
    aggregator = PairwiseHMMVotingAggregator(
        pair=("chronos2", "toto"),
        fallback=ClippedMeanAggregator(method="mean", clip_percentile=0.0, clip_std=0.0),
    )

    fallback_only = aggregator.fallback.combine([chronos], ["close"])
    combined = aggregator.combine([chronos], ["close"])
    assert np.allclose(
        fallback_only.columns["close"].prediction,
        combined.columns["close"].prediction,
    )
