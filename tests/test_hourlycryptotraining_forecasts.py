import numpy as np
import pandas as pd

from hourlycryptotraining import ForecastConfig
from hourlycryptotraining.forecasts import DailyChronosForecastManager
from src.models.chronos2_wrapper import Chronos2PredictionBatch, Chronos2PreparedPanel


def _build_history(hours: int = 72) -> pd.DataFrame:
    timestamps = pd.date_range("2024-03-01", periods=hours, freq="h", tz="UTC")
    prices = np.linspace(20.0, 25.0, hours)
    frame = pd.DataFrame(
        {
            "timestamp": timestamps,
            "open": prices - 0.1,
            "high": prices + 0.2,
            "low": prices - 0.3,
            "close": prices,
            "volume": np.linspace(1_000, 2_000, hours),
        }
    )
    return frame


class _StubChronosWrapper:
    def __init__(self) -> None:
        self.calls = 0

    def predict_ohlc_batch(
        self,
        contexts,
        *,
        symbols,
        prediction_length,
        context_length,
        batch_size,
        predict_kwargs=None,
    ):
        self.calls += 1
        batches = []
        for ctx, sym in zip(contexts, symbols):
            issued_at = ctx["timestamp"].max()
            target_ts = issued_at + pd.Timedelta(hours=1)
            base_close = float(ctx["close"].iloc[-1])
            mid = pd.DataFrame(
                {
                    "close": [base_close + 0.5],
                    "high": [base_close + 0.7],
                    "low": [base_close - 0.7],
                },
                index=[target_ts],
            )
            quantiles = {
                0.1: mid - 0.3,
                0.5: mid,
                0.9: mid + 0.3,
            }
            context_df = ctx.copy()
            context_df["symbol"] = sym
            panel = Chronos2PreparedPanel(
                symbol=sym,
                context_df=context_df.reset_index(drop=True),
                future_df=None,
                actual_df=pd.DataFrame(),
                context_length=len(context_df),
                prediction_length=prediction_length,
                id_column="symbol",
                timestamp_column="timestamp",
                target_columns=("open", "high", "low", "close"),
            )
            batches.append(
                Chronos2PredictionBatch(
                    panel=panel,
                    raw_dataframe=pd.DataFrame(),
                    quantile_frames=quantiles,
                )
            )
        return batches


def test_ensure_latest_batches_predictions(tmp_path):
    data_root = tmp_path / "data"
    data_root.mkdir()
    history = _build_history(48)
    history.to_csv(data_root / "LINKUSD.csv", index=False)
    cache_dir = tmp_path / "cache"
    stub = _StubChronosWrapper()
    manager = DailyChronosForecastManager(
            ForecastConfig(
                symbol="LINKUSD",
                data_root=data_root,
                cache_dir=cache_dir,
                context_hours=24,
                batch_size=4,
            ),
        wrapper_factory=lambda: stub,
    )
    start = history["timestamp"].iloc[30]
    end = history["timestamp"].iloc[40]
    result = manager.ensure_latest(start=start, end=end)
    assert not result.empty
    assert (cache_dir / "LINKUSD.parquet").exists()
    assert stub.calls >= 1
    assert (result["timestamp"] >= start).all()


def test_ensure_latest_skips_cached_ranges(tmp_path):
    data_root = tmp_path / "data"
    data_root.mkdir()
    history = _build_history(36)
    history.to_csv(data_root / "LINKUSD.csv", index=False)
    cache_dir = tmp_path / "cache"
    stub = _StubChronosWrapper()
    manager = DailyChronosForecastManager(
            ForecastConfig(
                symbol="LINKUSD",
                data_root=data_root,
                cache_dir=cache_dir,
                context_hours=24,
                batch_size=8,
            ),
        wrapper_factory=lambda: stub,
    )
    start = history["timestamp"].iloc[25]
    end = history["timestamp"].iloc[34]
    manager.ensure_latest(start=start, end=end)
    assert stub.calls >= 1

    first_calls = stub.calls
    result = manager.ensure_latest(start=start, end=end)
    assert stub.calls == first_calls
    assert not result.empty
