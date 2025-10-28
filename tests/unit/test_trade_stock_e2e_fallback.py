import pandas as pd
import pytest

from trade_stock_e2e import _compute_fallback_signal, FallbackSignalUnavailable


def _build_trend_frame(start: float, step: float, periods: int = 30) -> pd.DataFrame:
    closes = [start + step * idx for idx in range(periods)]
    frame = pd.DataFrame(
        {
            "Close": closes,
            "Open": [value - 0.5 * step for value in closes],
            "High": [value + abs(step) for value in closes],
            "Low": [value - abs(step) for value in closes],
        }
    )
    return frame


@pytest.mark.parametrize("step", [1.0, -1.5])
def test_compute_fallback_signal_always_raises(step: float) -> None:
    frame = _build_trend_frame(100.0, step)
    with pytest.raises(FallbackSignalUnavailable):
        _compute_fallback_signal("TEST", frame=frame)
