import pandas as pd

from src.tradinglib.direction_filter import (
    compute_predicted_close_return,
    filter_forecasts_by_predicted_close_return,
    should_trade_predicted_close_return,
)


def test_compute_predicted_close_return_basic() -> None:
    assert compute_predicted_close_return(None, 100.0) is None
    assert compute_predicted_close_return(101.0, None) is None
    assert compute_predicted_close_return(101.0, 0.0) is None
    assert compute_predicted_close_return(101.0, -1.0) is None
    assert compute_predicted_close_return(105.0, 100.0) == 0.05


def test_should_trade_predicted_close_return_threshold_none() -> None:
    assert should_trade_predicted_close_return(None, min_return_pct=None) is True
    assert should_trade_predicted_close_return(0.0, min_return_pct=None) is True


def test_should_trade_predicted_close_return_threshold_blocks_missing() -> None:
    assert should_trade_predicted_close_return(None, min_return_pct=0.0) is False


def test_should_trade_predicted_close_return_threshold_comparison() -> None:
    assert should_trade_predicted_close_return(-0.001, min_return_pct=0.0) is False
    assert should_trade_predicted_close_return(0.0, min_return_pct=0.0) is True
    assert should_trade_predicted_close_return(0.001, min_return_pct=0.0) is True
    assert should_trade_predicted_close_return(0.001, min_return_pct=0.002) is False


def test_filter_forecasts_by_predicted_close_return() -> None:
    daily = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                ["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04"],
                utc=True,
            ),
            "close": [100.0, 100.0, 100.0, 100.0],
        }
    )
    forecasts = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"], utc=True),
            "predicted_close_p50": [110.0, 90.0, 100.0],
            "predicted_low_p35": [95.0, 95.0, 95.0],
            "predicted_high_p50": [105.0, 105.0, 105.0],
        }
    )

    kept = filter_forecasts_by_predicted_close_return(forecasts, daily, min_return_pct=0.0)
    assert list(kept["timestamp"].dt.strftime("%Y-%m-%d")) == ["2024-01-02", "2024-01-04"]

