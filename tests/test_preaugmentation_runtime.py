from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from preaug_sweeps.augmentations import get_augmentation
from src.models.chronos2_wrapper import Chronos2OHLCWrapper
from src.preaug import PreAugmentationSelector


class _DummyChronosPipeline:
    def __init__(self) -> None:
        self.last_context: pd.DataFrame | None = None

    def predict_df(
        self,
        context_df: pd.DataFrame,
        *,
        future_df=None,
        id_column,
        timestamp_column,
        target,
        prediction_length,
        quantile_levels,
        batch_size=None,
        **kwargs,
    ) -> pd.DataFrame:
        self.last_context = context_df.copy()
        start_ts = pd.to_datetime(context_df[timestamp_column]).iloc[-1]
        freq = pd.Timedelta(days=1)
        timestamps = [start_ts + freq * (i + 1) for i in range(prediction_length)]

        rows = []
        for ts in timestamps:
            for name in target:
                base = float(context_df[name].iloc[-1])
                payload = {
                    timestamp_column: ts,
                    "target_name": name,
                }
                for level in quantile_levels:
                    payload[format(level, "g")] = base
                rows.append(payload)
        return pd.DataFrame(rows)


def _write_best_config(directory: Path, symbol: str, strategy: str = "log_returns") -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    payload = {
        "symbol": symbol,
        "best_strategy": strategy,
        "mae": 0.01,
        "mae_percent": 0.5,
        "selection_metric": "mae_percent",
        "selection_value": 0.5,
        "config": {"name": strategy, "params": {}},
        "comparison": {
            strategy: {"mae_percent": 0.5},
            "baseline": {"mae_percent": 2.0},
        },
    }
    path = directory / f"{symbol}.json"
    path.write_text(json.dumps(payload))
    return path


def test_preaug_selector_prefers_mae_percent(tmp_path: Path) -> None:
    best_dir = tmp_path / "preaugstrategies"
    _write_best_config(best_dir, "BTCUSD")

    selector = PreAugmentationSelector([best_dir])
    choice = selector.get_choice("btcusd")
    assert choice is not None
    assert choice.strategy == "log_returns"
    assert choice.metric == "mae_percent"
    assert choice.metric_value == pytest.approx(0.5)


def test_chronos_wrapper_applies_preaug_and_restores_outputs(tmp_path: Path) -> None:
    best_dir = tmp_path / "preaugstrategies" / "chronos2"
    _write_best_config(best_dir, "TEST")

    pipeline = _DummyChronosPipeline()
    wrapper = Chronos2OHLCWrapper(
        pipeline=pipeline,
        id_column="symbol",
        timestamp_column="timestamp",
        target_columns=("open", "high", "low", "close"),
        default_context_length=4,
        preaugmentation_dirs=[best_dir],
    )

    timestamps = pd.date_range("2024-01-01", periods=6, freq="D", tz="UTC")
    data = pd.DataFrame(
        {
            "timestamp": timestamps,
            "symbol": ["TEST"] * 6,
            "open": np.linspace(100, 105, 6),
            "high": np.linspace(110, 115, 6),
            "low": np.linspace(90, 95, 6),
            "close": np.linspace(102, 107, 6),
        }
    )

    batch = wrapper.predict_ohlc(data, symbol="TEST", prediction_length=2, context_length=5)

    assert batch.applied_augmentation == "log_returns"
    assert pipeline.last_context is not None
    original_tail = data["open"].iloc[-5]
    assert not np.isclose(pipeline.last_context["open"].iloc[0], original_tail)

    median = batch.median
    assert pytest.approx(median["open"].iloc[0]) == data["open"].iloc[-1]
    assert pytest.approx(batch.raw_dataframe["0.5"].iloc[0]) == data["open"].iloc[-1]
    assert pytest.approx(batch.panel.context_df["open"].iloc[-1]) == data["open"].iloc[-1]


def test_percent_change_inverse_handles_subset_columns() -> None:
    augmentation = get_augmentation("percent_change")
    context = pd.DataFrame({"open": [100.0, 101.0, 102.0], "close": [99.0, 100.0, 101.0]})
    transformed = augmentation.transform_dataframe(context.copy())
    restored = augmentation.inverse_transform_predictions(
        transformed.to_numpy(),
        context,
        columns=context.columns,
    )
    assert np.allclose(restored, context.values)
