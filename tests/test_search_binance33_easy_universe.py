import numpy as np

from pufferlib_market.hourly_replay import MktdData
from scripts.search_binance33_easy_universe import _masked_data, _safe_corr


def test_masked_data_restricts_tradable_symbols() -> None:
    data = MktdData(
        version=2,
        symbols=["AAAUSD", "BBBUSD", "CCCUSD"],
        features=np.zeros((4, 3, 2), dtype=np.float32),
        prices=np.ones((4, 3, 5), dtype=np.float32),
        tradable=np.ones((4, 3), dtype=np.uint8),
    )

    masked = _masked_data(data, ["BBBUSD"])

    assert masked.symbols == data.symbols
    assert masked.tradable is not None
    assert masked.tradable.tolist() == [[0, 1, 0]] * 4


def test_safe_corr_handles_constant_or_short_inputs() -> None:
    assert _safe_corr(np.ones(10), np.arange(10, dtype=float)) == 0.0

    x = np.arange(40, dtype=float)
    y = x * 2.0 + 1.0
    assert _safe_corr(x, y) > 0.99
