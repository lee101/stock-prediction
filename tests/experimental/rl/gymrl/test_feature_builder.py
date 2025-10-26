import numpy as np
import pandas as pd

from gymrl.feature_pipeline import FeatureBuilder, FeatureBuilderConfig


def _make_sample_frame(timestamps, price_offset: float = 0.0) -> pd.DataFrame:
    base_price = 100.0 + price_offset
    data = {
        "timestamp": timestamps,
        "open": np.linspace(base_price, base_price + 1.0, len(timestamps)),
        "high": np.linspace(base_price + 0.5, base_price + 1.5, len(timestamps)),
        "low": np.linspace(base_price - 0.5, base_price + 0.5, len(timestamps)),
        "close": np.linspace(base_price + 0.1, base_price + 1.1, len(timestamps)),
        "volume": np.linspace(1000.0, 2000.0, len(timestamps)),
    }
    return pd.DataFrame(data)


def test_feature_builder_handles_misaligned_indices(tmp_path):
    timestamps_a = pd.date_range("2023-01-01", periods=32, freq="D")
    timestamps_b = pd.date_range("2023-01-02", periods=32, freq="D")  # intentionally shifted

    frame_a = _make_sample_frame(timestamps_a)
    frame_b = _make_sample_frame(timestamps_b, price_offset=5.0)

    frame_a.to_csv(tmp_path / "AAPL.csv", index=False)
    frame_b.to_csv(tmp_path / "MSFT.csv", index=False)

    config = FeatureBuilderConfig(
        forecast_backend="bootstrap",
        num_samples=16,
        context_window=8,
        prediction_length=1,
        realized_horizon=1,
        min_history=8,
        enforce_common_index=False,
        fill_method="ffill",
    )
    builder = FeatureBuilder(config=config)
    cube = builder.build_from_directory(tmp_path)

    assert cube.features.shape[1] == 2  # two symbols
    assert cube.realized_returns.shape[0] == cube.features.shape[0]
    assert not np.isnan(cube.features).any()
    assert cube.symbols == sorted(["AAPL", "MSFT"])
