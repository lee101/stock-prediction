#!/usr/bin/env python3
"""Targeted tests for Toto feature integration in StockDataProcessor."""

import numpy as np
import pandas as pd
from unittest.mock import patch
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from hftraining.data_utils import StockDataProcessor


def test_prepare_features_uppercase_columns_with_toto():
    """Toto forecasts should tolerate uppercase OHLCV source columns."""
    df = pd.DataFrame(
        {
            'Open': [100.0, 101.0, 102.0, 103.0],
            'High': [105.0, 106.0, 107.0, 108.0],
            'Low': [95.0, 96.0, 97.0, 98.0],
            'Close': [102.0, 103.0, 104.0, 105.0],
            'Volume': [1000, 1100, 1200, 1300],
        }
    )

    calls = {}

    class DummyGenerator:
        def __init__(self, options):
            self.options = options

        def compute_features(self, price_matrix, price_columns, symbol_prefix):
            calls['price_columns'] = list(price_columns)
            calls['symbol_prefix'] = symbol_prefix
            close_idx = price_columns.index('close')
            close_vals = price_matrix[:, close_idx : close_idx + 1]
            std_vals = np.full_like(close_vals, 0.5, dtype=np.float32)
            features = np.concatenate([close_vals, std_vals], axis=1)
            return features, [
                f"{symbol_prefix}_close_toto_mean_t+1",
                f"{symbol_prefix}_close_toto_std_t+1",
            ]

    with patch('hftraining.data_utils.TotoFeatureGenerator', DummyGenerator):
        processor = StockDataProcessor(use_toto_forecasts=True)
        feature_matrix = processor.prepare_features(df, symbol="AAPL")

    assert calls['price_columns'] == ['open', 'high', 'low', 'close', 'volume']
    assert calls['symbol_prefix'] == 'aapl'

    feature_df = pd.DataFrame(feature_matrix, columns=processor.feature_names)
    mean_col = 'aapl_close_toto_mean_t+1'
    std_col = 'aapl_close_toto_std_t+1'
    residual_col = 'aapl_close_toto_residual'

    assert mean_col in feature_df.columns
    assert std_col in feature_df.columns
    assert residual_col in feature_df.columns
    assert np.allclose(feature_df[residual_col].to_numpy(), 0.0)


def test_toto_prediction_zero_fill_for_missing_history():
    """Ensure Toto prediction columns remain present when a symbol lacks history."""
    dates = pd.date_range("2024-01-01", periods=4, freq="D")
    base_df = pd.DataFrame(
        {
            "date": dates,
            "open": np.linspace(100.0, 103.0, len(dates)),
            "high": np.linspace(101.0, 104.0, len(dates)),
            "low": np.linspace(99.0, 102.0, len(dates)),
            "close": np.linspace(100.5, 103.5, len(dates)),
            "volume": np.linspace(1_000_000, 1_300_000, len(dates)),
        }
    )

    predictions = pd.DataFrame(
        {
            "toto_pred_signal": [0.1, 0.2, 0.3, 0.4],
            "toto_pred_confidence": [0.6, 0.5, 0.7, 0.8],
        },
        index=dates,
    )

    processor = StockDataProcessor(
        toto_prediction_features={"AAPL": predictions},
        toto_prediction_columns=["toto_pred_confidence", "toto_pred_signal"],
    )

    # Symbol with history should retain declared columns and availability flag.
    features_aapl = processor.prepare_features(base_df.copy(), symbol="AAPL")
    columns_aapl = list(processor.feature_names)
    aapl_df = pd.DataFrame(features_aapl, columns=columns_aapl)

    assert "toto_pred_confidence" in aapl_df.columns
    assert "toto_pred_signal" in aapl_df.columns
    assert "toto_pred_available" in aapl_df.columns
    assert np.allclose(aapl_df["toto_pred_available"].to_numpy(), 1.0)

    # Symbol without history should still expose zero-filled prediction columns.
    features_msft = processor.prepare_features(base_df.copy(), symbol="MSFT")
    columns_msft = list(processor.feature_names)
    msft_df = pd.DataFrame(features_msft, columns=columns_msft)

    assert columns_msft == columns_aapl
    assert np.allclose(msft_df["toto_pred_confidence"].to_numpy(), 0.0)
    assert np.allclose(msft_df["toto_pred_signal"].to_numpy(), 0.0)
    assert np.allclose(msft_df["toto_pred_available"].to_numpy(), 0.0)
