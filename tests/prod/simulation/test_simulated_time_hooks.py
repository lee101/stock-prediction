from datetime import datetime, timezone
import os
from typing import Dict

import pandas as pd
import pytest
import pytz

from marketsimulator import environment
from marketsimulator.state import PriceSeries


def _build_frame(start: datetime, periods: int = 24 * 6) -> pd.DataFrame:
    index = pd.date_range(start, periods=periods, freq="h")
    frame = pd.DataFrame(
        {
            "timestamp": index.tz_convert("UTC") if index.tz is not None else index.tz_localize("UTC"),
            "Open": 100.0,
            "High": 101.0,
            "Low": 99.0,
            "Close": 100.5,
            "Volume": 1_000,
        }
    )
    return frame


def test_activate_simulation_patches_trading_day(monkeypatch):
    start_ts = datetime(2024, 1, 2, 15, 0, tzinfo=timezone.utc)

    def fake_load_price_series(symbols, data_root=None) -> Dict[str, PriceSeries]:
        frame = _build_frame(start_ts)
        return {symbol: PriceSeries(symbol=symbol, frame=frame.copy()) for symbol in symbols}

    monkeypatch.setattr(environment, "load_price_series", fake_load_price_series)

    import trade_stock_e2e as trade_module
    from src import date_utils

    original_trade_now = trade_module.is_nyse_trading_day_now
    original_trade_ending = trade_module.is_nyse_trading_day_ending
    original_utils_now = date_utils.is_nyse_trading_day_now
    original_utils_ending = date_utils.is_nyse_trading_day_ending

    monkeypatch.delenv("MARKETSIM_SKIP_CLOSED_EQUITY", raising=False)

    with environment.activate_simulation(symbols=["AAPL"], initial_cash=10_000.0) as controller:
        # Functions should be patched to simulation-aware versions
        assert trade_module.is_nyse_trading_day_now is not original_trade_now
        assert date_utils.is_nyse_trading_day_now is not original_utils_now

        current = controller.current_time()
        assert trade_module.is_nyse_trading_day_now() == date_utils.is_nyse_trading_day_now(current)
        assert trade_module.is_nyse_trading_day_now() is True

        # Advance until the simulated clock reaches a weekend
        while controller.current_time().astimezone(pytz.timezone("US/Eastern")).weekday() < 5:
            controller.advance_steps(1)

        weekend_time = controller.current_time()
        assert trade_module.is_nyse_trading_day_now() == date_utils.is_nyse_trading_day_now(weekend_time)
        assert trade_module.is_nyse_trading_day_now() is False

    # Patches should be fully restored
    from src import date_utils as restored_utils

    assert trade_module.is_nyse_trading_day_now is original_trade_now
    assert trade_module.is_nyse_trading_day_ending is original_trade_ending
    assert restored_utils.is_nyse_trading_day_now is original_utils_now
    assert restored_utils.is_nyse_trading_day_ending is original_utils_ending
    assert "MARKETSIM_SKIP_CLOSED_EQUITY" not in os.environ
