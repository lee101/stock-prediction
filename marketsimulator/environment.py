from __future__ import annotations

import importlib
import sys
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Iterable, Optional

import os

from src.leverage_settings import get_leverage_settings
from . import alpaca_wrapper_mock
from . import data_curate_daily_mock
from . import process_utils_mock
from .logging_utils import logger
from .data_feed import DEFAULT_DATA_ROOT, load_price_series
from .state import SimulationState, SimulatedClock, set_state


def _install_env_stub() -> None:
    os.environ.setdefault("MARKETSIM_ALLOW_MOCK_ANALYTICS", "1")
    os.environ.setdefault("MARKETSIM_SKIP_REAL_IMPORT", "0")
    os.environ.setdefault("MARKETSIM_RELAX_SPREAD", "1")

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

    # Pre-seed frequently imported trading modules with simulator stubs to avoid
    # hitting live APIs before the full patching phase installs mocks.
    if "alpaca_wrapper" not in sys.modules:
        sys.modules["alpaca_wrapper"] = alpaca_wrapper_mock
    if "data_curate_daily" not in sys.modules:
        sys.modules["data_curate_daily"] = data_curate_daily_mock
    os.environ.setdefault("MARKETSIM_ALLOW_MOCK_ANALYTICS", "1")
    os.environ.setdefault("MARKETSIM_SKIP_REAL_IMPORT", "1")


def _load_mock_backtest_module():
    from . import backtest_test3_inline as module

    return module


def _load_mock_forecasting_module():
    from . import predict_stock_forecasting_proxy as module

    return module


def _patch_third_party(use_mock_analytics: bool, force_kronos: bool):
    replaced_modules = {}

    def replace_module(name: str, module):
        replaced_modules[name] = sys.modules.get(name)
        if module is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = module

    replace_module("alpaca_wrapper", alpaca_wrapper_mock)
    replace_module("data_curate_daily", data_curate_daily_mock)

    if use_mock_analytics:
        mock_backtest = _load_mock_backtest_module()
        mock_forecasting = _load_mock_forecasting_module()
        replace_module("backtest_test3_inline", mock_backtest)
        replace_module("predict_stock_forecasting", mock_forecasting)
    else:
        try:
            real_backtest = importlib.import_module("backtest_test3_inline")
        except Exception:
            real_backtest = _load_mock_backtest_module()
        replace_module("backtest_test3_inline", real_backtest)

        try:
            real_forecasting = importlib.import_module("predict_stock_forecasting")
        except Exception:
            real_forecasting = _load_mock_forecasting_module()
        replace_module("predict_stock_forecasting", real_forecasting)

    # Ensure downstream modules reuse the patched modules.
    importlib.invalidate_caches()

    patched_alpaca = sys.modules.get("alpaca_wrapper", alpaca_wrapper_mock)
    for module_name in (
        "predict_stock_forecasting",
        "predict_stock_forecasting_proxy",
        "trade_stock_e2e",
        "trade_stock_e2e_trained",
        "backtest_test3_inline",
    ):
        module = sys.modules.get(module_name)
        if module is not None:
            setattr(module, "alpaca_wrapper", patched_alpaca)

    process_utils = importlib.import_module("src.process_utils")
    original = (
        process_utils.backout_near_market,
        process_utils.ramp_into_position,
        process_utils.spawn_close_position_at_takeprofit,
    )
    process_utils.backout_near_market = process_utils_mock.backout_near_market
    process_utils.ramp_into_position = process_utils_mock.ramp_into_position
    process_utils.spawn_close_position_at_takeprofit = process_utils_mock.spawn_close_position_at_takeprofit
    if force_kronos:
        logger.info("[sim] Kronos-only forecasting flag active for simulation environment.")

    return {"process_utils": (process_utils, original), "replaced_modules": replaced_modules}


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
    use_mock_analytics: Optional[bool] = None,
    force_kronos: Optional[bool] = None,
):
    if use_mock_analytics is None:
        env_flag = os.getenv("MARKETSIM_USE_MOCK_ANALYTICS", "0").lower()
        use_mock_analytics = env_flag in {"1", "true", "yes"}

    allow_env_key = "MARKETSIM_ALLOW_MOCK_ANALYTICS"
    skip_env_key = "MARKETSIM_SKIP_REAL_IMPORT"
    previous_allow_value = os.environ.get(allow_env_key)
    previous_skip_value = os.environ.get(skip_env_key)
    had_allow_env = allow_env_key in os.environ
    had_skip_env = skip_env_key in os.environ

    if use_mock_analytics:
        os.environ[allow_env_key] = "1"
        os.environ[skip_env_key] = "1"
    else:
        if not had_allow_env:
            os.environ[allow_env_key] = "1"
        os.environ[skip_env_key] = "0"

    relax_spread_key = "MARKETSIM_RELAX_SPREAD"
    previous_relax_value = os.environ.get(relax_spread_key)
    had_relax_env = relax_spread_key in os.environ
    if use_mock_analytics:
        os.environ[relax_spread_key] = "1"

    env_force_key = "MARKETSIM_FORCE_KRONOS"
    previous_force_value = os.environ.get(env_force_key)
    had_force_env = env_force_key in os.environ
    override_force_env = force_kronos is not None
    if override_force_env:
        if force_kronos:
            os.environ[env_force_key] = "1"
        else:
            os.environ.pop(env_force_key, None)
    else:
        force_kronos = (
            previous_force_value is not None
            and str(previous_force_value).lower() in {"1", "true", "yes", "on"}
        )

    kronos_sample_key = "MARKETSIM_KRONOS_SAMPLE_COUNT"
    had_sample_env = kronos_sample_key in os.environ
    sample_override_applied = False
    if force_kronos and not had_sample_env:
        default_sample_raw = os.getenv("MARKETSIM_FORCE_KRONOS_SAMPLE_COUNT", "64")
        try:
            default_sample = max(1, int(default_sample_raw))
        except ValueError:
            default_sample = 64
        os.environ[kronos_sample_key] = str(default_sample)
        sample_override_applied = True
        logger.info(
            f"[sim] Kronos sample_count override set to {default_sample} via MARKETSIM_KRONOS_SAMPLE_COUNT."
        )

    backtest_sim_key = "MARKETSIM_BACKTEST_SIMULATIONS"
    had_backtest_env = backtest_sim_key in os.environ
    backtest_override_applied = False
    if force_kronos and not had_backtest_env:
        default_backtest_raw = os.getenv("MARKETSIM_FORCE_KRONOS_BACKTEST_SIMULATIONS", "20")
        try:
            default_backtest = max(1, int(default_backtest_raw))
        except ValueError:
            default_backtest = 20
        os.environ[backtest_sim_key] = str(default_backtest)
        backtest_override_applied = True
        logger.info(
            f"[sim] Backtest simulation count override set to {default_backtest} via MARKETSIM_BACKTEST_SIMULATIONS."
        )

    if force_kronos:
        logger.info("[sim] Kronos-only forecasting enabled for this simulation.")

    _install_env_stub()
    if not symbols:
        symbols = ["AAPL", "MSFT", "NVDA"]
    prices = load_price_series(symbols, data_root=data_root)
    first_timestamp = min(series.timestamp for series in prices.values())
    clock = SimulatedClock(first_timestamp)
    leverage_settings = get_leverage_settings()
    alpaca_wrapper_mock.margin_multiplier = leverage_settings.max_gross_leverage
    state = SimulationState(
        clock=clock,
        prices=prices,
        cash=initial_cash,
        equity=initial_cash,
        leverage_settings=leverage_settings,
    )
    set_state(state)
    alpaca_wrapper_mock.reset_account(initial_cash)
    restore_handles = _patch_third_party(
        use_mock_analytics=use_mock_analytics,
        force_kronos=bool(force_kronos),
    )
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
        for name, original in restore_handles["replaced_modules"].items():
            if original is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = original
        if sample_override_applied and not had_sample_env:
            os.environ.pop(kronos_sample_key, None)
        if backtest_override_applied and not had_backtest_env:
            os.environ.pop(backtest_sim_key, None)
        if override_force_env:
            if had_force_env:
                if previous_force_value is not None:
                    os.environ[env_force_key] = previous_force_value
                else:
                    # Original environment had the key set to an empty string.
                    os.environ[env_force_key] = ""
            else:
                os.environ.pop(env_force_key, None)
        if had_allow_env:
            if previous_allow_value is not None:
                os.environ[allow_env_key] = previous_allow_value
            else:
                os.environ.pop(allow_env_key, None)
        else:
            os.environ.pop(allow_env_key, None)
        if had_skip_env:
            if previous_skip_value is not None:
                os.environ[skip_env_key] = previous_skip_value
            else:
                os.environ.pop(skip_env_key, None)
        else:
            os.environ.pop(skip_env_key, None)
        if had_relax_env:
            if previous_relax_value is not None:
                os.environ[relax_spread_key] = previous_relax_value
            else:
                os.environ.pop(relax_spread_key, None)
        else:
            os.environ.pop(relax_spread_key, None)
