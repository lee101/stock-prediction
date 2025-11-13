from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from preaug_sweeps.augmentations import AUGMENTATION_REGISTRY, get_augmentation
from src.models.chronos2_wrapper import Chronos2OHLCWrapper, _default_preaug_dirs
from src.preaug import PreAugmentationSelector

_ALL_AUGMENTATIONS = tuple(AUGMENTATION_REGISTRY.keys())


def _sample_price_frame(length: int = 96) -> pd.DataFrame:
    rng = np.random.default_rng(12345)
    base = np.linspace(100.0, 130.0, length)
    noise = rng.normal(scale=0.5, size=length)
    frame = pd.DataFrame(
        {
            "open": base + noise,
            "high": base + 0.5 + rng.normal(scale=0.3, size=length),
            "low": base - 0.5 + rng.normal(scale=0.3, size=length),
            "close": base + rng.normal(scale=0.2, size=length),
            "volume": np.abs(rng.normal(loc=5_000.0, scale=500.0, size=length)) + 10.0,
            "amount": np.abs(rng.normal(loc=9_000.0, scale=900.0, size=length)) + 25.0,
        }
    )
    return frame.reset_index(drop=True)


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


@pytest.mark.parametrize("strategy_name", _ALL_AUGMENTATIONS)
def test_preaugmentation_roundtrip_matches_original(strategy_name: str) -> None:
    df = _sample_price_frame()
    augmentation = get_augmentation(strategy_name)

    transformed = augmentation.transform_dataframe(df.copy())
    restored = augmentation.inverse_transform_predictions(
        transformed.to_numpy(),
        context=df,
        columns=df.columns,
    )

    assert np.allclose(restored, df.to_numpy(), atol=1e-6, rtol=1e-6)


@pytest.mark.parametrize("strategy_name", _ALL_AUGMENTATIONS)
def test_preaugmentation_restores_future_window(strategy_name: str) -> None:
    df = _sample_price_frame(80)
    split = len(df) - 8
    context = df.iloc[:split].copy()
    future = df.iloc[split:].copy()

    augmentation = get_augmentation(strategy_name)
    transformed_full = augmentation.transform_dataframe(df.copy())
    future_aug = transformed_full.iloc[split:]

    restored = augmentation.inverse_transform_predictions(
        future_aug.to_numpy(),
        context=context,
        columns=df.columns,
    )

    assert np.allclose(restored, future.to_numpy(), atol=1e-5, rtol=1e-5)


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


def test_default_preaug_dirs_prioritize_frequency() -> None:
    dirs = _default_preaug_dirs("hourly")
    assert dirs[0] == Path("preaugstrategies") / "chronos2" / "hourly"
    assert dirs[1] == Path("preaugstrategies") / "best" / "hourly"
    assert dirs[2] == Path("preaugstrategies") / "chronos2"
    assert dirs[3] == Path("preaugstrategies") / "best"


def test_default_preaug_dirs_without_frequency() -> None:
    dirs = _default_preaug_dirs(None)
    assert dirs == (
        Path("preaugstrategies") / "chronos2",
        Path("preaugstrategies") / "best",
    )
