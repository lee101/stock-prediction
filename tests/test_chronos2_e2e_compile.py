"""E2E smoke tests for Chronos2 wrapper creation + prediction (eager + compiled)."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.chronos2_wrapper import Chronos2OHLCWrapper

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("chronos2_e2e")


def create_test_data(n_points: int = 128) -> pd.DataFrame:
    """Create realistic OHLC data for testing."""
    np.random.seed(42)
    base_price = 100.0
    returns = np.random.randn(n_points) * 0.02
    prices = base_price * np.exp(np.cumsum(returns))

    return pd.DataFrame(
        {
            "timestamp": pd.date_range(start="2024-01-01", periods=n_points, freq="D"),
            "open": prices * (1 + np.random.randn(n_points) * 0.005),
            "high": prices * (1 + np.abs(np.random.randn(n_points)) * 0.01),
            "low": prices * (1 - np.abs(np.random.randn(n_points)) * 0.01),
            "close": prices,
            "symbol": "TEST",
        }
    )


@pytest.fixture(scope="module")
def data() -> pd.DataFrame:
    return create_test_data()


def _make_wrapper(*, compile_enabled: bool) -> Chronos2OHLCWrapper:
    mode = "compiled" if compile_enabled else "eager"
    logger.info("Creating Chronos2OHLCWrapper (%s)...", mode)
    wrapper = Chronos2OHLCWrapper.from_pretrained(
        model_id="amazon/chronos-2",
        device_map="cpu",
        default_context_length=64,
        torch_compile=compile_enabled,
        compile_mode="reduce-overhead" if compile_enabled else None,
        compile_backend="inductor" if compile_enabled else None,
    )
    logger.info("✓ Wrapper created (%s)", mode)
    return wrapper


@pytest.fixture(scope="module")
def eager_wrapper() -> Chronos2OHLCWrapper:
    wrapper = _make_wrapper(compile_enabled=False)
    yield wrapper
    wrapper.unload()


@pytest.fixture(scope="module")
def compiled_wrapper() -> Chronos2OHLCWrapper:
    wrapper = _make_wrapper(compile_enabled=True)
    yield wrapper
    wrapper.unload()


def test_import():
    """Verify Chronos2Pipeline can be imported."""
    logger.info("Testing Chronos2Pipeline import...")
    from chronos import Chronos2Pipeline

    assert Chronos2Pipeline is not None


@pytest.mark.parametrize(
    "wrapper_fixture, mode",
    [
        ("eager_wrapper", "eager"),
        ("compiled_wrapper", "compiled"),
    ],
)
def test_prediction(wrapper_fixture: str, mode: str, request: pytest.FixtureRequest, data: pd.DataFrame):
    """Test making predictions."""
    wrapper = request.getfixturevalue(wrapper_fixture)
    logger.info("Testing prediction (%s)...", mode)

    context_length = 64
    prediction_length = 7

    context = data.iloc[:-prediction_length]
    result = wrapper.predict_ohlc(
        context_df=context,
        symbol="TEST",
        prediction_length=prediction_length,
        context_length=context_length,
    )

    assert result is not None
    assert hasattr(result, "median")
    median = result.median
    assert len(median) == prediction_length
    assert "close" in median.columns
    close = median["close"].to_numpy()
    assert close.shape == (prediction_length,)
    assert np.isfinite(close).all()
