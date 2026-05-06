from __future__ import annotations

import numpy as np

from pufferlib_market.hourly_replay import MktdData
from scripts.search_binance33_linear_symbol_filter import _goodness, _masked_data


def test_masked_data_marks_excluded_symbol_untradable() -> None:
    features = np.zeros((3, 2, 16), dtype=np.float32)
    prices = np.ones((3, 2, 5), dtype=np.float32)
    tradable = np.ones((3, 2), dtype=np.uint8)
    data = MktdData(version=2, symbols=["BTCUSD", "ETHUSD"], features=features, prices=prices, tradable=tradable)

    masked = _masked_data(data, {"ETHUSD"})

    assert masked.tradable is not None
    assert masked.tradable[:, 0].sum() == 3
    assert masked.tradable[:, 1].sum() == 0


def test_goodness_penalizes_negative_windows_and_drawdown() -> None:
    row = {
        "median_monthly_pct": 30.0,
        "p10_monthly_pct": 10.0,
        "p90_dd_pct": 20.0,
        "neg_windows": 1,
    }

    assert _goodness(row) == 30.0 + 0.75 * 10.0 - 1.25 * 20.0 - 18.0
