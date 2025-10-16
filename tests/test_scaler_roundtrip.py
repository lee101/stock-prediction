from __future__ import annotations

import logging
from types import SimpleNamespace

import numpy as np
import pytest

import hfshared
from hfinference.production_engine import ProductionTradingEngine
from hftraining.data_utils import StockDataProcessor


def _fit_processor_with_basic_ohlc(train_matrix: np.ndarray, feature_names: list[str]):
    processor = StockDataProcessor(sequence_length=train_matrix.shape[0], prediction_horizon=1)
    processor.fit_scalers(train_matrix)
    processor.feature_names = feature_names
    return processor


def test_load_processor_exposes_standard_scaler(tmp_path):
    feature_names = ['open', 'high', 'low', 'close']
    training_values = np.array(
        [
            [2000.0, 2010.0, 1990.0, 2005.0],
            [1980.0, 1995.0, 1975.0, 1988.0],
            [2050.0, 2075.0, 2035.0, 2060.0],
        ],
        dtype=np.float32,
    )
    processor = _fit_processor_with_basic_ohlc(training_values, feature_names)
    dump_path = tmp_path / "processor.pkl"
    processor.save_scalers(str(dump_path))

    payload = hfshared.load_processor(str(dump_path))
    assert payload['feature_names'] == feature_names
    assert 'standard' in payload['scalers']

    scaler = payload['scalers']['standard']
    sample = np.array([[2100.0, 2120.0, 2085.0, 2105.0]], dtype=np.float32)
    normalized = scaler.transform(sample)[0]

    idx_close = feature_names.index('close')
    idx_high = feature_names.index('high')
    idx_low = feature_names.index('low')

    denorm_close = hfshared.denormalize_with_scaler(
        normalized[idx_close],
        scaler,
        feature_names,
        column_name='close',
    )
    denorm_high = hfshared.denormalize_with_scaler(
        normalized[idx_high],
        scaler,
        feature_names,
        column_name='high',
    )
    denorm_low = hfshared.denormalize_with_scaler(
        normalized[idx_low],
        scaler,
        feature_names,
        column_name='low',
    )

    assert denorm_close == pytest.approx(sample[0, idx_close], rel=1e-5)
    assert denorm_high == pytest.approx(sample[0, idx_high], rel=1e-5)
    assert denorm_low == pytest.approx(sample[0, idx_low], rel=1e-5)

    # Production engine helper should respect the scaler as well.
    engine = ProductionTradingEngine.__new__(ProductionTradingEngine)
    engine.data_processor = SimpleNamespace(scalers={'standard': scaler})
    engine.feature_names = feature_names
    engine.logger = logging.getLogger(__name__)

    current_price = 2095.0
    price_from_engine = ProductionTradingEngine._denormalize_price(engine, normalized[idx_close], current_price)
    assert price_from_engine == pytest.approx(sample[0, idx_close], rel=1e-5)

    # If the scaler is unavailable, fallback should behave like a return-based prediction.
    engine.data_processor = SimpleNamespace(scalers={})
    fallback_pred = 0.0125
    fallback_price = ProductionTradingEngine._denormalize_price(engine, fallback_pred, current_price)
    assert fallback_price == pytest.approx(current_price * (1 + fallback_pred), rel=1e-9)
