import csv
from datetime import datetime, timedelta

import numpy as np

from gymrl.config import FeatureBuilderConfig
from gymrl.feature_pipeline import FeatureBuilder


def _write_daily_csv(path, start_price=100.0, drift=0.01):
    start_time = datetime(2024, 1, 1)
    price = start_price
    with path.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["timestamp", "open", "high", "low", "close", "volume"])
        for day in range(90):
            timestamp = start_time + timedelta(days=day)
            open_price = price
            close_price = price * (1.0 + drift * 0.1)
            high_price = max(open_price, close_price) * 1.01
            low_price = min(open_price, close_price) * 0.99
            volume = 1_000_000 + 1000 * day
            writer.writerow([
                timestamp.isoformat(),
                f"{open_price:.4f}",
                f"{high_price:.4f}",
                f"{low_price:.4f}",
                f"{close_price:.4f}",
                volume,
            ])
            price = close_price


def test_no_forecast_mean_leakage(tmp_path):
    data_dir = tmp_path / "daily"
    data_dir.mkdir()
    _write_daily_csv(data_dir / "AAPL.csv", start_price=150.0, drift=0.02)

    config = FeatureBuilderConfig(
        forecast_backend="bootstrap",
        context_window=16,
        min_history=16,
        num_samples=32,
        realized_horizon=1,
        prediction_length=1,
        enforce_common_index=False,
        fill_method="ffill",
    )

    cube = FeatureBuilder(config=config).build_from_directory(data_dir)

    # Identify the forecast mean feature column
    fidx = cube.feature_names.index("forecast_mean_return")
    mu_forecast = cube.features[:, 0, fidx]
    realized = cube.realized_returns[:, 0]

    # The series should not be identical and correlation should be < 0.95 in typical bootstrap
    assert not np.allclose(mu_forecast, realized)
    if mu_forecast.std() > 1e-8 and realized.std() > 1e-8:
        corr = np.corrcoef(mu_forecast, realized, rowvar=False)[0, 1]
        assert corr < 0.95

