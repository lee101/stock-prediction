from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from hfshared.features import compute_compact_features, compute_training_style_features
from hftraining.data_utils import StockDataProcessor


def _frame_with_price_gap() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "open": [100.0, 110.0, 112.0, 121.0, 150.0, 153.0],
            "high": [101.0, 111.0, 113.0, 122.0, 151.0, 154.0],
            "low": [99.0, 109.0, 111.0, 120.0, 149.0, 152.0],
            "close": [100.0, 110.0, np.nan, 121.0, 150.0, 153.0],
            "volume": [1000.0, 1100.0, 1200.0, 1300.0, 1400.0, 1500.0],
        }
    )


def test_training_style_features_match_stock_data_processor_on_price_gaps() -> None:
    frame = _frame_with_price_gap()
    processor = StockDataProcessor()

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        shared = compute_training_style_features(frame)
        processor_features = processor.add_technical_indicators(frame)

    pct_change_warnings = [
        warning
        for warning in caught
        if issubclass(warning.category, FutureWarning) and "fill_method" in str(warning.message)
    ]
    assert not pct_change_warnings

    for column in ("price_change", "price_change_2", "price_change_5"):
        np.testing.assert_allclose(shared[column].to_numpy(), processor_features[column].to_numpy())


def test_compute_compact_features_pct_change_stays_finite() -> None:
    frame = _frame_with_price_gap()

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        features = compute_compact_features(frame, use_pct_change=True)

    pct_change_warnings = [
        warning
        for warning in caught
        if issubclass(warning.category, FutureWarning) and "fill_method" in str(warning.message)
    ]
    assert not pct_change_warnings
    assert features.shape == (len(frame), 5)
    assert np.isfinite(features).all()
