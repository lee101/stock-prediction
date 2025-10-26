import pandas as pd
import pytest

from stock_data_utils import add_ohlc_percent_change


def test_add_ohlc_percent_change_basic():
    df = pd.DataFrame(
        {
            "open": [100, 105],
            "high": [110, 112],
            "low": [95, 104],
            "close": [105, 108],
        },
        index=pd.to_datetime(["2024-01-01", "2024-01-02"]),
    )

    pct_df = add_ohlc_percent_change(df)
    first = pct_df.iloc[0]
    assert first["open_pct"] == 0.0
    assert first["close_pct"] == 0.0

    second = pct_df.iloc[1]
    assert pytest.approx(second["open_pct"], rel=1e-6) == (105 - 105) / 105
    assert pytest.approx(second["close_pct"], rel=1e-6) == (108 - 105) / 105


def test_add_ohlc_percent_change_handles_zero_baseline():
    df = pd.DataFrame(
        {"open": [0.0, 1.0], "close": [0.0, 2.0]},
        index=pd.to_datetime(["2024-01-01", "2024-01-02"]),
    )

    pct_df = add_ohlc_percent_change(df, price_columns=("open", "close"))
    assert pct_df.iloc[0]["open_pct"] == 0.0
    assert pct_df.iloc[1]["open_pct"] == 0.0


def test_add_ohlc_percent_change_missing_baseline_raises():
    df = pd.DataFrame({"open": [1, 2]})
    with pytest.raises(ValueError):
        add_ohlc_percent_change(df)
