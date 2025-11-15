import pandas as pd
import pytest

from neuralpricingstrategy.data import (
    PricingDataset,
    build_pricing_dataset,
    split_dataset_by_date,
)


def _sample_frame(num_rows: int = 3) -> pd.DataFrame:
    timestamps = pd.date_range("2025-11-01", periods=num_rows, freq="D", tz="UTC")
    base_low = pd.Series([100.0, 102.0, 101.0], dtype=float).head(num_rows)
    base_high = pd.Series([110.0, 112.0, 111.0], dtype=float).head(num_rows)
    target_low = base_low * (1.0 + pd.Series([0.02, -0.01, 0.0]).head(num_rows))
    target_high = base_high * (1.0 + pd.Series([0.015, 0.005, -0.02]).head(num_rows))

    return pd.DataFrame(
        {
            "symbol": ["AAA"] * num_rows,
            "timestamp": timestamps,
            "maxdiffalwayson_low_price": base_low,
            "maxdiffalwayson_high_price": base_high,
            "maxdiffprofit_low_price": target_low,
            "maxdiffprofit_high_price": target_high,
            "maxdiff_return": [0.02, -0.01, 0.005][:num_rows],
            "maxdiffalwayson_return": [0.01, -0.02, -0.005][:num_rows],
            "close": 100.0,
            "predicted_close": 101.0,
        }
    )


def test_build_pricing_dataset_computes_targets():
    frame = _sample_frame(num_rows=2)
    dataset = build_pricing_dataset(frame, clamp_pct=0.05)

    assert isinstance(dataset, PricingDataset)
    assert dataset.targets.shape == (2, 3)

    expected_low = [
        pytest.approx(0.02, rel=1e-5),
        pytest.approx(-0.01, rel=1e-5),
    ]
    expected_high = [
        pytest.approx(0.015, rel=1e-5),
        pytest.approx(0.005, rel=1e-5),
    ]
    expected_pnl = [
        pytest.approx(0.01, rel=1e-5),
        pytest.approx(0.01, rel=1e-5),
    ]
    for idx, target_row in enumerate(dataset.targets.tolist()):
        assert target_row[0] == expected_low[idx]
        assert target_row[1] == expected_high[idx]
        assert target_row[2] == expected_pnl[idx]


def test_split_dataset_by_date_requires_two_dates():
    dataset = build_pricing_dataset(_sample_frame(num_rows=2))
    train_ds, val_ds = split_dataset_by_date(dataset, validation_fraction=0.5)
    assert train_ds.features.shape[0] > 0
    assert val_ds.features.shape[0] > 0

    single_row = build_pricing_dataset(_sample_frame(num_rows=1))
    with pytest.raises(ValueError):
        split_dataset_by_date(single_row)
