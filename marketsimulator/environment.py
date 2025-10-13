from __future__ import annotations

import importlib
import sys
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Iterable, Optional

from . import alpaca_wrapper_mock
from . import backtest_test3_inline as backtest_module
from . import data_curate_daily_mock
from . import predict_stock_forecasting_mock
from . import process_utils_mock
from .data_feed import DEFAULT_DATA_ROOT, load_price_series
from .state import SimulationState, SimulatedClock, set_state


def _install_env_stub() -> None:
    if "env_real" in sys.modules:
        pass
    else:
        import types

        env_stub = types.ModuleType("env_real")
        env_stub.ALP_KEY_ID = "SIM-KEY"
        env_stub.ALP_SECRET_KEY = "SIM-SECRET"
        env_stub.ALP_KEY_ID_PROD = "SIM-KEY"
        env_stub.ALP_SECRET_KEY_PROD = "SIM-SECRET"
        env_stub.ALP_ENDPOINT = "paper"
        env_stub.PAPER = True
        env_stub.ADD_LATEST = False
        env_stub.BINANCE_API_KEY = "SIM"
        env_stub.BINANCE_SECRET = "SIM"
        env_stub.CLAUDE_API_KEY = "SIM"
        env_stub.SIMULATE = True
        sys.modules["env_real"] = env_stub

    if "loguru" not in sys.modules:
        import sys as _sys
        import types
        from datetime import datetime

        class _Logger:
            def __init__(self) -> None:
                self._sinks = [(_sys.stdout, "{time} | {level} | {message}")]

            def _record(self, level: str, message: str) -> None:
                record = {
                    "time": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
                    "level": level,
                    "message": message,
                }
                active_sinks = self._sinks or [(_sys.stdout, "{time} | {level} | {message}")]
                for sink, fmt in active_sinks:
                    sink.write(fmt.format(**record) + "\n")
                    sink.flush()

            def add(self, sink, format="{time} | {level} | {message}", **kwargs):
                self._sinks.append((sink, format))
                return len(self._sinks) - 1

            def remove(self, handler_id=None):
                if handler_id is None:
                    self._sinks = []
                elif 0 <= handler_id < len(self._sinks):
                    self._sinks.pop(handler_id)

            def info(self, message: str):
                self._record("INFO", message)

            def warning(self, message: str):
                self._record("WARNING", message)

            def error(self, message: str):
                self._record("ERROR", message)

            def debug(self, message: str):
                self._record("DEBUG", message)

        loguru_mod = types.ModuleType("loguru")
        loguru_mod.logger = _Logger()
        sys.modules["loguru"] = loguru_mod

    if "alpaca" not in sys.modules:
        import types
        from unittest.mock import MagicMock

        alpaca_mod = types.ModuleType("alpaca")
        alpaca_data = types.ModuleType("alpaca.data")
        alpaca_trading = types.ModuleType("alpaca.trading")
        alpaca_trading.client = types.ModuleType("client")
        alpaca_trading.enums = types.ModuleType("enums")
        alpaca_trading.requests = types.ModuleType("requests")

        mock_class = MagicMock()
        alpaca_data.StockLatestQuoteRequest = mock_class
        alpaca_data.StockHistoricalDataClient = MagicMock()
        alpaca_data.CryptoHistoricalDataClient = MagicMock()
        alpaca_data.CryptoLatestQuoteRequest = mock_class
        alpaca_data.CryptoBarsRequest = MagicMock()
        alpaca_data.StockBarsRequest = MagicMock()
        alpaca_data.TimeFrame = MagicMock()
        alpaca_data.TimeFrameUnit = MagicMock()

        alpaca_trading.OrderType = MagicMock()
        alpaca_trading.LimitOrderRequest = MagicMock()
        alpaca_trading.GetOrdersRequest = MagicMock()
        alpaca_trading.Order = MagicMock()
        alpaca_trading.client.TradingClient = MagicMock()
        alpaca_trading.TradingClient = MagicMock()
        alpaca_trading.enums.OrderSide = MagicMock()
        alpaca_trading.requests.MarketOrderRequest = MagicMock()

        sys.modules["alpaca"] = alpaca_mod
        sys.modules["alpaca.data"] = alpaca_data
        sys.modules["alpaca.trading"] = alpaca_trading
        sys.modules["alpaca.trading.client"] = alpaca_trading.client
        sys.modules["alpaca.trading.enums"] = alpaca_trading.enums
        sys.modules["alpaca.trading.requests"] = alpaca_trading.requests

    if "alpaca_trade_api" not in sys.modules:
        import types

        rest_mod = types.ModuleType("alpaca_trade_api.rest")
        rest_mod.APIError = Exception
        main_mod = types.ModuleType("alpaca_trade_api")
        sys.modules["alpaca_trade_api.rest"] = rest_mod
        sys.modules["alpaca_trade_api"] = main_mod

    if not hasattr(sys.modules["alpaca_trade_api"], "REST"):
        from unittest.mock import MagicMock

        sys.modules["alpaca_trade_api"].REST = MagicMock()

    if "typer" not in sys.modules:
        import types

        typer_mod = types.ModuleType("typer")

        def run(func):
            return func()

        typer_mod.run = run
        sys.modules["typer"] = typer_mod


def _patch_third_party():
    sys.modules["alpaca_wrapper"] = alpaca_wrapper_mock
    sys.modules["backtest_test3_inline"] = backtest_module
    sys.modules["data_curate_daily"] = data_curate_daily_mock
    sys.modules["predict_stock_forecasting"] = predict_stock_forecasting_mock

    # Ensure downstream modules reuse the patched alpaca_wrapper.
    importlib.invalidate_caches()

    process_utils = importlib.import_module("src.process_utils")
    original = (
        process_utils.backout_near_market,
        process_utils.ramp_into_position,
        process_utils.spawn_close_position_at_takeprofit,
    )
    process_utils.backout_near_market = process_utils_mock.backout_near_market
    process_utils.ramp_into_position = process_utils_mock.ramp_into_position
    process_utils.spawn_close_position_at_takeprofit = process_utils_mock.spawn_close_position_at_takeprofit
    return {"process_utils": (process_utils, original)}


@dataclass
class SimulationController:
    state: SimulationState

    def advance_steps(self, steps: int = 1):
        self.state.advance_time(steps)
        alpaca_wrapper_mock._sync_account_metrics(self.state)  # type: ignore[attr-defined]
        return self.state.clock.current

    def advance_minutes(self, minutes: int) -> None:
        steps = max(1, minutes // 60)
        self.advance_steps(steps)

    def current_time(self):
        return self.state.clock.current

    def summary(self):
        return {
            "cash": self.state.cash,
            "equity": self.state.equity,
            "positions": {symbol: pos.qty for symbol, pos in self.state.positions.items()},
        }


@contextmanager
def activate_simulation(
    symbols: Optional[Iterable[str]] = None,
    initial_cash: float = 100_000.0,
    data_root=DEFAULT_DATA_ROOT,
):
    _install_env_stub()
    if not symbols:
        symbols = ["AAPL", "MSFT", "NVDA"]
    prices = load_price_series(symbols, data_root=data_root)
    first_timestamp = min(series.timestamp for series in prices.values())
    clock = SimulatedClock(first_timestamp)
    state = SimulationState(
        clock=clock,
        prices=prices,
        cash=initial_cash,
        buying_power=initial_cash * alpaca_wrapper_mock.margin_multiplier,
        equity=initial_cash,
    )
    set_state(state)
    alpaca_wrapper_mock.reset_account(initial_cash)
    restore_handles = _patch_third_party()
    controller = SimulationController(state)
    try:
        yield controller
    finally:
        process_utils, originals = restore_handles["process_utils"]
        (
            process_utils.backout_near_market,
            process_utils.ramp_into_position,
            process_utils.spawn_close_position_at_takeprofit,
        ) = originals
        # Best-effort clean-up so other runs can import real modules if needed.
        for name in [
            "alpaca_wrapper",
            "backtest_test3_inline",
            "data_curate_daily",
            "predict_stock_forecasting",
        ]:
            sys.modules.pop(name, None)
