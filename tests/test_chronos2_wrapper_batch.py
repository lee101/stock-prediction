import numpy as np
import pandas as pd
import torch

from src.models.chronos2_wrapper import Chronos2OHLCWrapper


class _FakeModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("_dummy", torch.tensor(0.0))


class _FakePipeline:
    def __init__(self) -> None:
        self.model = _FakeModel()
        self.calls = 0
        self.last_kwargs = None

    def predict_df(
        self,
        df: pd.DataFrame,
        *,
        future_df: pd.DataFrame | None,
        id_column: str,
        timestamp_column: str,
        target: list[str],
        prediction_length: int,
        quantile_levels: list[float],
        batch_size: int,
        **_: dict,
    ) -> pd.DataFrame:
        self.calls += 1
        self.last_kwargs = dict(_)
        rows = []
        for symbol, group in df.groupby(id_column):
            last_ts = group[timestamp_column].max()
            forecast_ts = last_ts + pd.Timedelta(hours=1)
            for target_name in target:
                base_value = float(group[target_name].iloc[-1])
                row = {
                    id_column: symbol,
                    timestamp_column: forecast_ts,
                    "target_name": target_name,
                    "predictions": base_value,
                }
                for level in quantile_levels:
                    row[str(level)] = base_value + level
                rows.append(row)
        return pd.DataFrame(rows)


def _build_context(seed: float = 0.0) -> pd.DataFrame:
    timestamps = pd.date_range("2024-01-01", periods=12, freq="h", tz="UTC")
    base = np.linspace(seed, seed + 1.1, len(timestamps))
    frame = pd.DataFrame(
        {
            "timestamp": timestamps,
            "open": base + 0.1,
            "high": base + 0.2,
            "low": base - 0.2,
            "close": base,
            "volume": np.linspace(1_000, 2_000, len(timestamps)),
        }
    )
    return frame


def _build_wrapper() -> Chronos2OHLCWrapper:
    pipeline = _FakePipeline()
    return Chronos2OHLCWrapper(
        pipeline,
        device_hint="cpu",
        default_context_length=12,
        default_batch_size=8,
        quantile_levels=(0.1, 0.5, 0.9),
        torch_compile=False,
        preaugmentation_dirs=(),
    )


def test_batch_prediction_matches_single_calls():
    wrapper = _build_wrapper()
    contexts = [_build_context(1.0), _build_context(2.0)]
    singles = []
    for idx, ctx in enumerate(contexts):
        singles.append(
            wrapper.predict_ohlc(
                ctx,
                symbol=f"CTX_{idx}",
                prediction_length=1,
                context_length=8,
                batch_size=4,
            )
        )

    batched = wrapper.predict_ohlc_batch(
        contexts,
        symbols=("CTX_0", "CTX_1"),
        prediction_length=1,
        context_length=8,
        batch_size=4,
    )
    assert len(batched) == len(singles)
    for single, batch in zip(singles, batched):
        single_close = single.quantile_frames[0.5]["close"].iloc[-1]
        batch_close = batch.quantile_frames[0.5]["close"].iloc[-1]
        assert single_close == batch_close


def test_batch_predictions_use_cache():
    wrapper = _build_wrapper()
    contexts = [_build_context(3.0), _build_context(4.0)]
    pipeline = wrapper.pipeline
    assert isinstance(pipeline, _FakePipeline)

    wrapper.predict_ohlc_batch(
        contexts,
        symbols=("CTX_A", "CTX_B"),
        prediction_length=1,
        context_length=12,
        batch_size=2,
    )
    assert pipeline.calls == 1

    wrapper.predict_ohlc_batch(
        contexts,
        symbols=("CTX_A", "CTX_B"),
        prediction_length=1,
        context_length=12,
        batch_size=2,
    )
    assert pipeline.calls == 1


def test_batch_cross_learning_alias():
    wrapper = _build_wrapper()
    pipeline = wrapper.pipeline
    assert isinstance(pipeline, _FakePipeline)

    contexts = [_build_context(5.0), _build_context(6.0)]
    wrapper.predict_ohlc_batch(
        contexts,
        symbols=("CTX_X", "CTX_Y"),
        prediction_length=1,
        context_length=8,
        batch_size=4,
        predict_kwargs={"cross_learning": True},
    )

    assert pipeline.last_kwargs is not None
    assert pipeline.last_kwargs.get("predict_batches_jointly") is True
    assert "cross_learning" not in pipeline.last_kwargs
