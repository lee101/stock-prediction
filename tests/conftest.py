#!/usr/bin/env python3
"""Pytest configuration for environments with real PyTorch installed."""

import os
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock

os.environ.setdefault("MARKETSIM_ALLOW_MOCK_ANALYTICS", "1")
os.environ.setdefault("MARKETSIM_SKIP_REAL_IMPORT", "1")
os.environ.setdefault("MARKETSIM_ALLOW_CPU_FALLBACK", "1")

import pytest

# Provide a harmless env_real stub during tests so we never import the real
# credentials or accidentally place live trades. Set USE_REAL_ENV=1 to bypass.
if os.getenv("USE_REAL_ENV", "0") not in ("1", "true", "TRUE", "yes", "YES"):
    env_stub = types.ModuleType("env_real")
    env_stub.ALP_KEY_ID = "test-key"
    env_stub.ALP_SECRET_KEY = "test-secret"
    env_stub.ALP_KEY_ID_PROD = "test-key-prod"
    env_stub.ALP_SECRET_KEY_PROD = "test-secret-prod"
    env_stub.ALP_ENDPOINT = "paper"
    env_stub.PAPER = True
    env_stub.ADD_LATEST = False
    env_stub.BINANCE_API_KEY = "test-binance-key"
    env_stub.BINANCE_SECRET = "test-binance-secret"
    env_stub.CLAUDE_API_KEY = "test-claude-key"
    env_stub.SIMULATE = True
    sys.modules["env_real"] = env_stub

# Lightweight stubs for optional third-party dependencies so unit tests never
# reach external services when the packages are missing locally.
if "loguru" not in sys.modules:
    loguru_mod = types.ModuleType("loguru")
    loguru_mod.logger = MagicMock()
    sys.modules["loguru"] = loguru_mod

if "cachetools" not in sys.modules:
    cachetools_mod = types.ModuleType("cachetools")

    def cached(**kwargs):
        def decorator(func):
            return func

        return decorator

    class TTLCache(dict):
        def __init__(self, maxsize, ttl):
            super().__init__()

    cachetools_mod.cached = cached
    cachetools_mod.TTLCache = TTLCache
    sys.modules["cachetools"] = cachetools_mod

try:
    import requests as requests_mod  # type: ignore
    from requests import exceptions as requests_exceptions  # type: ignore
except Exception:
    requests_mod = sys.modules.setdefault("requests", types.ModuleType("requests"))
    requests_exceptions = sys.modules.setdefault(
        "requests.exceptions", types.ModuleType("requests.exceptions")
    )

    class _RequestException(Exception):
        """Lightweight stand-in for requests.RequestException."""

    class _HTTPError(_RequestException):
        """HTTP error placeholder matching requests semantics."""

    class _ConnectionError(_RequestException):
        """Connection error placeholder matching requests semantics."""

    class _Timeout(_RequestException):
        """Timeout placeholder matching requests semantics."""

    class _Response:
        """Minimal Response stub used by tests expecting requests.Response."""

        status_code = 200

        def __init__(self, content=None, headers=None):
            self.content = content
            self.headers = headers or {}

        def json(self):
            raise NotImplementedError("Response.json() stubbed for tests")

    requests_mod.RequestException = _RequestException
    requests_mod.HTTPError = _HTTPError
    requests_mod.ConnectionError = _ConnectionError
    requests_mod.Timeout = _Timeout
    requests_mod.Response = _Response

    requests_exceptions.RequestException = _RequestException
    requests_exceptions.HTTPError = _HTTPError
    requests_exceptions.ConnectionError = _ConnectionError
    requests_exceptions.Timeout = _Timeout

if "retry" not in sys.modules:
    retry_mod = types.ModuleType("retry")

    def _retry(*args, **kwargs):
        def decorator(func):
            return func

        return decorator

    retry_mod.retry = _retry
    sys.modules["retry"] = retry_mod

if "alpaca" not in sys.modules:
    alpaca_mod = types.ModuleType("alpaca")
    alpaca_data = types.ModuleType("alpaca.data")
    alpaca_data_enums = types.ModuleType("alpaca.data.enums")
    alpaca_trading = types.ModuleType("alpaca.trading")
    alpaca_trading.client = types.ModuleType("client")
    alpaca_trading.enums = types.ModuleType("enums")
    alpaca_trading.requests = types.ModuleType("requests")

    alpaca_data.StockLatestQuoteRequest = MagicMock()
    alpaca_data.StockHistoricalDataClient = MagicMock()
    alpaca_data.CryptoHistoricalDataClient = MagicMock()
    alpaca_data.CryptoLatestQuoteRequest = MagicMock()
    alpaca_data.CryptoBarsRequest = MagicMock()
    alpaca_data.StockBarsRequest = MagicMock()
    alpaca_data.TimeFrame = MagicMock()
    alpaca_data.TimeFrameUnit = MagicMock()
    alpaca_data_enums.DataFeed = MagicMock()
    alpaca_data_historical = types.ModuleType("alpaca.data.historical")
    alpaca_data_historical.StockHistoricalDataClient = MagicMock()
    alpaca_data_historical.CryptoHistoricalDataClient = MagicMock()
    sys.modules["alpaca.data.historical"] = alpaca_data_historical

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
    sys.modules["alpaca.data.enums"] = alpaca_data_enums
    sys.modules["alpaca.trading"] = alpaca_trading
    sys.modules["alpaca.trading.client"] = alpaca_trading.client
    sys.modules["alpaca.trading.enums"] = alpaca_trading.enums
    sys.modules["alpaca.trading.requests"] = alpaca_trading.requests
else:
    alpaca_trading_mod = sys.modules.get("alpaca.trading")
    if alpaca_trading_mod is None or not isinstance(alpaca_trading_mod, types.ModuleType):
        alpaca_trading_mod = types.ModuleType("alpaca.trading")
        sys.modules["alpaca.trading"] = alpaca_trading_mod

    if not hasattr(alpaca_trading_mod, "Position"):
        class _PositionStub:
            """Minimal Alpaca Position stub used in tests."""

            symbol: str
            qty: str
            side: str
            market_value: str

            def __init__(self, symbol="TEST", qty="0", side="long", market_value="0"):
                self.symbol = symbol
                self.qty = qty
                self.side = side
                self.market_value = market_value

        alpaca_trading_mod.Position = _PositionStub  # type: ignore[attr-defined]

sys.modules.setdefault("alpaca_trade_api", types.ModuleType("alpaca_trade_api"))
alpaca_rest = sys.modules.setdefault(
    "alpaca_trade_api.rest", types.ModuleType("alpaca_trade_api.rest")
)

if not hasattr(alpaca_rest, "APIError"):
    alpaca_rest.APIError = Exception

tradeapi_mod = sys.modules["alpaca_trade_api"]
if not hasattr(tradeapi_mod, "REST"):
    class _DummyREST:
        def __init__(self, *args, **kwargs):
            self._orders = []

        def get_all_positions(self):
            return []

        def get_account(self):
            return types.SimpleNamespace(
                equity=1.0,
                cash=1.0,
                multiplier=1,
                buying_power=1.0,
            )

        def get_clock(self):
            return types.SimpleNamespace(is_open=True)


def pytest_addoption(parser):
    """Register custom CLI options for this repository."""
    parser.addoption(
        "--run-experimental",
        action="store_true",
        default=False,
        help="Run tests under tests/experimental (skipped by default).",
    )


def pytest_collection_modifyitems(config, items):
    """Automatically mark and optionally skip experimental tests.

    Also applies CI-specific test filtering based on environment variables.
    """
    run_experimental = config.getoption("--run-experimental")
    mark_experimental = pytest.mark.experimental
    skip_marker = pytest.mark.skip(reason="experimental suite disabled; pass --run-experimental to include")
    experimental_root = Path(config.rootpath, "tests", "experimental").resolve()

    # CI mode detection
    is_ci = os.getenv("CI", "0") in ("1", "true", "TRUE", "yes", "YES")
    is_fast_ci = os.getenv("FAST_CI", "0") in ("1", "true", "TRUE", "yes", "YES")
    cpu_only = os.getenv("CPU_ONLY", "0") in ("1", "true", "TRUE", "yes", "YES")

    # Check for CUDA availability
    has_cuda = False
    try:
        import torch
        has_cuda = torch.cuda.is_available()
    except Exception:
        pass

    for item in items:
        path = Path(str(item.fspath)).resolve()
        try:
            path.relative_to(experimental_root)
            is_experimental = True
        except ValueError:
            is_experimental = False

        if is_experimental:
            item.add_marker(mark_experimental)
            if not run_experimental:
                item.add_marker(skip_marker)

        # Skip slow tests in fast CI mode
        if is_fast_ci and "slow" in item.keywords:
            item.add_marker(pytest.mark.skip(reason="Slow test skipped in FAST_CI mode"))

        # Skip model-required tests in fast CI unless specifically marked as smoke test
        if is_fast_ci and "model_required" in item.keywords and "smoke" not in item.keywords:
            item.add_marker(pytest.mark.skip(reason="Model test skipped in FAST_CI mode (only smoke tests run)"))

        # Skip CUDA tests when running on CPU-only or no CUDA available
        if (cpu_only or not has_cuda) and ("cuda_required" in item.keywords or "gpu_required" in item.keywords):
            item.add_marker(pytest.mark.skip(reason="GPU test skipped on CPU-only environment"))

        # Skip self-hosted-only tests on non-self-hosted runners
        if is_ci and "self_hosted_only" in item.keywords:
            if not os.getenv("SELF_HOSTED", "0") in ("1", "true", "TRUE", "yes", "YES"):
                item.add_marker(pytest.mark.skip(reason="Self-hosted only test skipped on standard runner"))

        # Skip external/network tests in CI unless explicitly enabled
        if is_ci and not os.getenv("RUN_EXTERNAL_TESTS", "0") in ("1", "true", "TRUE", "yes", "YES"):
            if "external" in item.keywords or "network_required" in item.keywords:
                item.add_marker(pytest.mark.skip(reason="External/network test skipped in CI"))

        def cancel_orders(self):
            self._orders.clear()
            return []

        def submit_order(self, *args, **kwargs):
            self._orders.append((args, kwargs))
            return types.SimpleNamespace(id=len(self._orders))

    tradeapi_mod.REST = _DummyREST

if "data_curate_daily" not in sys.modules:
    data_curate_daily_stub = types.ModuleType("data_curate_daily")
    _latest_prices = {}

    def download_exchange_latest_data(client, symbol):
        # store deterministic bid/ask defaults for tests
        _latest_prices[symbol] = {
            "bid": _latest_prices.get(symbol, {}).get("bid", 99.0),
            "ask": _latest_prices.get(symbol, {}).get("ask", 101.0),
        }

    def get_bid(symbol):
        return _latest_prices.get(symbol, {}).get("bid", 99.0)

    def get_ask(symbol):
        return _latest_prices.get(symbol, {}).get("ask", 101.0)

    def get_spread(symbol):
        prices = _latest_prices.get(symbol, {})
        bid = prices.get("bid", 99.0)
        ask = prices.get("ask", 101.0)
        return ask - bid

    def download_daily_stock_data(current_time, symbols):
        import pandas as pd

        dates = pd.date_range(start="2023-01-01", periods=30, freq="D")
        data = {
            "Open": [100.0] * len(dates),
            "High": [101.0] * len(dates),
            "Low": [99.0] * len(dates),
            "Close": [100.5] * len(dates),
        }
        return pd.DataFrame(data, index=dates)

    def fetch_spread(symbol):
        return 1.001

    data_curate_daily_stub.download_exchange_latest_data = download_exchange_latest_data
    data_curate_daily_stub.get_bid = get_bid
    data_curate_daily_stub.get_ask = get_ask
    data_curate_daily_stub.get_spread = get_spread
    data_curate_daily_stub.download_daily_stock_data = download_daily_stock_data
    data_curate_daily_stub.fetch_spread = fetch_spread
    sys.modules["data_curate_daily"] = data_curate_daily_stub

if "backtest_test3_inline" not in sys.modules:
    try:
        # Use the real module when available so that strategy logic is exercised.
        import backtest_test3_inline  # noqa: F401
    except Exception as exc:
        backtest_stub = types.ModuleType("backtest_test3_inline")

        def backtest_forecasts(symbol, num_simulations=10):
            import pandas as pd

            return pd.DataFrame(
                {
                    "simple_strategy_return": [0.01] * num_simulations,
                    "simple_strategy_avg_daily_return": [0.01] * num_simulations,
                    "simple_strategy_annual_return": [0.01 * 252] * num_simulations,
                    "all_signals_strategy_return": [0.01] * num_simulations,
                    "all_signals_strategy_avg_daily_return": [0.01] * num_simulations,
                    "all_signals_strategy_annual_return": [0.01 * 252] * num_simulations,
                    "entry_takeprofit_return": [0.01] * num_simulations,
                    "entry_takeprofit_avg_daily_return": [0.01] * num_simulations,
                    "entry_takeprofit_annual_return": [0.01 * 252] * num_simulations,
                    "highlow_return": [0.01] * num_simulations,
                    "highlow_avg_daily_return": [0.01] * num_simulations,
                    "highlow_annual_return": [0.01 * 252] * num_simulations,
                    "maxdiff_return": [0.01] * num_simulations,
                    "maxdiff_avg_daily_return": [0.01] * num_simulations,
                    "maxdiff_annual_return": [0.01 * 252] * num_simulations,
                    "maxdiff_sharpe": [1.2] * num_simulations,
                    "maxdiffprofit_high_price": [1.1] * num_simulations,
                    "maxdiffprofit_low_price": [0.9] * num_simulations,
                    "maxdiffprofit_profit_high_multiplier": [0.02] * num_simulations,
                    "maxdiffprofit_profit_low_multiplier": [-0.02] * num_simulations,
                    "maxdiffprofit_profit": [0.01] * num_simulations,
                    "maxdiffprofit_profit_values": ["[0.01]"] * num_simulations,
                    "maxdiffalwayson_return": [0.009] * num_simulations,
                    "maxdiffalwayson_avg_daily_return": [0.009] * num_simulations,
                    "maxdiffalwayson_annual_return": [0.009 * 252] * num_simulations,
                    "maxdiffalwayson_sharpe": [1.1] * num_simulations,
                    "maxdiffalwayson_turnover": [0.012] * num_simulations,
                    "maxdiffalwayson_profit": [0.009] * num_simulations,
                    "maxdiffalwayson_profit_values": ["[0.009]"] * num_simulations,
                    "maxdiffalwayson_high_multiplier": [0.015] * num_simulations,
                    "maxdiffalwayson_low_multiplier": [-0.015] * num_simulations,
                    "maxdiffalwayson_high_price": [1.12] * num_simulations,
                    "maxdiffalwayson_low_price": [0.88] * num_simulations,
                    "maxdiffalwayson_buy_contribution": [0.005] * num_simulations,
                    "maxdiffalwayson_sell_contribution": [0.004] * num_simulations,
                    "maxdiffalwayson_filled_buy_trades": [5] * num_simulations,
                    "maxdiffalwayson_filled_sell_trades": [4] * num_simulations,
                    "maxdiffalwayson_trades_total": [9] * num_simulations,
                    "maxdiffalwayson_trade_bias": [0.1] * num_simulations,
                    "pctdiff_return": [0.008] * num_simulations,
                    "pctdiff_avg_daily_return": [0.008] * num_simulations,
                    "pctdiff_annual_return": [0.008 * 252] * num_simulations,
                    "pctdiff_sharpe": [1.05] * num_simulations,
                    "pctdiff_turnover": [0.011] * num_simulations,
                    "pctdiff_profit": [0.008] * num_simulations,
                    "pctdiff_profit_values": ["[0.008]"] * num_simulations,
                    "pctdiff_entry_low_multiplier": [-0.01] * num_simulations,
                    "pctdiff_entry_high_multiplier": [0.01] * num_simulations,
                    "pctdiff_long_pct": [0.02] * num_simulations,
                    "pctdiff_short_pct": [0.015] * num_simulations,
                    "pctdiff_entry_low_price": [0.9] * num_simulations,
                    "pctdiff_entry_high_price": [1.1] * num_simulations,
                    "pctdiff_takeprofit_high_price": [0.9 * 1.02] * num_simulations,
                    "pctdiff_takeprofit_low_price": [1.1 * (1 - 0.015)] * num_simulations,
                    "pctdiff_primary_side": ["buy"] * num_simulations,
                    "pctdiff_trade_bias": [0.2] * num_simulations,
                    "pctdiff_trades_positive": [6] * num_simulations,
                    "pctdiff_trades_negative": [3] * num_simulations,
                    "pctdiff_trades_total": [9] * num_simulations,
                    "predicted_close": [1.0] * num_simulations,
                    "predicted_high": [1.2] * num_simulations,
                    "predicted_low": [0.8] * num_simulations,
                    "close": [1.0] * num_simulations,
                }
            )

        backtest_stub.backtest_forecasts = backtest_forecasts

        def _compute_toto_forecast(*args, **kwargs):
            import torch

            if "current_last_price" in kwargs:
                last_price = kwargs["current_last_price"]
            elif len(args) >= 2:
                last_price = args[-2]
            else:
                last_price = 0.0

            predictions = torch.zeros(1, dtype=torch.float32)
            band = torch.zeros_like(predictions)
            return predictions, band, float(last_price or 0.0)

        backtest_stub._compute_toto_forecast = _compute_toto_forecast

        def pre_process_data(frame, price_column="Close"):
            return frame.copy()

        def resolve_toto_params(symbol):
            return {"num_samples": 64, "samples_per_batch": 32}

        def release_model_resources():
            return None

        backtest_stub.pre_process_data = pre_process_data
        backtest_stub.resolve_toto_params = resolve_toto_params
        backtest_stub.release_model_resources = release_model_resources
        backtest_stub.__import_error__ = exc  # expose failure reason for debugging
        sys.modules["backtest_test3_inline"] = backtest_stub

# Allow skipping the hard PyTorch requirement for lightweight coverage runs.
if os.getenv("SKIP_TORCH_CHECK", "0") not in ("1", "true", "TRUE", "yes", "YES"):
    # Ensure PyTorch is available; fail fast if not.
    try:
        import torch  # noqa: F401
    except Exception as e:
        raise RuntimeError(
            "PyTorch must be installed for this test suite."
        ) from e


# Backwards compatibility for chronos pipelines that used the old `context` keyword
try:  # pragma: no cover - best-effort compatibility shim
    from chronos import ChronosPipeline
    import inspect

    _predict_sig = inspect.signature(ChronosPipeline.predict)

    if "context" not in _predict_sig.parameters:
        _chronos_predict = ChronosPipeline.predict

        def _predict_with_context(self, *args, **kwargs):
            if "context" in kwargs:
                ctx = kwargs.pop("context")
                if not args:
                    args = (ctx,)
                else:
                    args = (ctx,) + args
            return _chronos_predict(self, *args, **kwargs)

        setattr(ChronosPipeline, "predict", _predict_with_context)
except Exception:
    pass


# Minimal stubs for fal cloud runtime APIs used by integration tests.
if "fal" not in sys.modules:
    fal_mod = types.ModuleType("fal")

    class _FalApp:
        def __init_subclass__(cls, **kwargs):  # swallow keyword-only configuration
            super().__init_subclass__()

        def __init__(self, *args, **kwargs):
            pass

    fal_mod.App = _FalApp
    fal_mod.endpoint = lambda *a, **k: (lambda fn: fn)
    sys.modules["fal"] = fal_mod


# ============================================================================
# CI Mode Fixtures
# ============================================================================


@pytest.fixture(scope="session")
def ci_mode():
    """Returns True if running in CI environment."""
    return os.getenv("CI", "0") in ("1", "true", "TRUE", "yes", "YES")


@pytest.fixture(scope="session")
def fast_ci_mode():
    """Returns True if running in fast CI mode (limited test suite)."""
    return os.getenv("FAST_CI", "0") in ("1", "true", "TRUE", "yes", "YES")


@pytest.fixture(scope="session")
def cpu_only_mode():
    """Returns True if running on CPU-only (no GPU available)."""
    return os.getenv("CPU_ONLY", "0") in ("1", "true", "TRUE", "yes", "YES")


@pytest.fixture(scope="session")
def fast_simulate_mode():
    """Returns True if fast simulation mode is enabled (minimal iterations)."""
    return os.getenv("FAST_SIMULATE", "0") in ("1", "true", "TRUE", "yes", "YES")


@pytest.fixture(scope="function")
def fast_model_config():
    """Provides fast model configuration for CI testing.

    Returns dict with minimal settings for quick model tests:
    - Small context lengths
    - Few prediction steps
    - Minimal samples
    """
    return {
        "context_length": 8,
        "prediction_length": 2,
        "num_samples": 2,
        "batch_size": 1,
        "device": "cpu",
    }


@pytest.fixture(scope="function")
def fast_simulation_config():
    """Provides fast simulation configuration for CI testing.

    Returns dict with minimal settings for quick simulation tests:
    - Few timesteps
    - Single environment
    - Minimal episodes
    """
    return {
        "total_timesteps": 100,
        "num_envs": 1,
        "num_episodes": 1,
        "max_steps": 50,
        "device": "cpu",
    }


@pytest.fixture(autouse=True, scope="function")
def setup_fast_simulate_env(fast_simulate_mode):
    """Auto-apply FAST_SIMULATE environment variable to each test if enabled."""
    if fast_simulate_mode:
        original = os.environ.get("FAST_SIMULATE")
        os.environ["FAST_SIMULATE"] = "1"
        yield
        if original is None:
            os.environ.pop("FAST_SIMULATE", None)
        else:
            os.environ["FAST_SIMULATE"] = original
    else:
        yield


@pytest.fixture(scope="function")
def torch_device(cpu_only_mode):
    """Returns appropriate torch device based on environment.

    Returns 'cpu' in CPU_ONLY mode, otherwise 'cuda' if available.
    """
    try:
        import torch
        if cpu_only_mode:
            return "cpu"
        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"
