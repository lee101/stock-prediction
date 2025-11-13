"""E2E test for Chronos2 with torch.compile to debug import and compile issues."""
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.chronos2_wrapper import Chronos2OHLCWrapper

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("chronos2_e2e")


def create_test_data(n_points=128):
    """Create realistic OHLC data for testing."""
    np.random.seed(42)
    base_price = 100.0
    returns = np.random.randn(n_points) * 0.02
    prices = base_price * np.exp(np.cumsum(returns))

    return pd.DataFrame({
        'timestamp': pd.date_range(start='2024-01-01', periods=n_points, freq='D'),
        'open': prices * (1 + np.random.randn(n_points) * 0.005),
        'high': prices * (1 + np.abs(np.random.randn(n_points)) * 0.01),
        'low': prices * (1 - np.abs(np.random.randn(n_points)) * 0.01),
        'close': prices,
        'symbol': 'TEST'
    })


def test_import():
    """Verify Chronos2Pipeline can be imported."""
    logger.info("Testing Chronos2Pipeline import...")
    try:
        from chronos import Chronos2Pipeline
        logger.info("✓ Chronos2Pipeline imported successfully")
        return True
    except Exception as exc:
        logger.error(f"✗ Failed to import Chronos2Pipeline: {exc}")
        return False


def test_wrapper_creation(compile_enabled=False):
    """Test creating Chronos2OHLCWrapper with/without compilation."""
    mode = "compiled" if compile_enabled else "eager"
    logger.info(f"Testing wrapper creation ({mode})...")

    try:
        wrapper = Chronos2OHLCWrapper.from_pretrained(
            model_id="amazon/chronos-2",
            device_map="cpu",
            default_context_length=64,
            torch_compile=compile_enabled,
            compile_mode="reduce-overhead" if compile_enabled else None,
            compile_backend="inductor" if compile_enabled else None,
        )
        logger.info(f"✓ Wrapper created successfully ({mode})")
        return wrapper
    except Exception as exc:
        logger.error(f"✗ Failed to create wrapper ({mode}): {exc}")
        raise


def test_prediction(wrapper, data, compile_mode="unknown"):
    """Test making predictions."""
    logger.info(f"Testing prediction ({compile_mode})...")

    context_length = 64
    prediction_length = 7

    context = data.iloc[:-prediction_length]
    holdout = data.iloc[-prediction_length:]

    try:
        result = wrapper.predict_ohlc(
            context_df=context,
            symbol="TEST",
            prediction_length=prediction_length,
            context_length=context_length,
        )

        assert result is not None, "Result is None"
        assert hasattr(result, 'median'), "Result missing median attribute"
        median = result.median
        assert len(median) == prediction_length, f"Expected {prediction_length} predictions, got {len(median)}"
        assert 'close' in median.columns, "Missing close column"

        logger.info(f"✓ Prediction successful ({compile_mode})")
        logger.info(f"  Predicted close values: {median['close'].values[:3]}...")
        return result
    except Exception as exc:
        logger.error(f"✗ Prediction failed ({compile_mode}): {exc}")
        raise


def run_e2e_test():
    """Run full e2e test sequence."""
    logger.info("=" * 60)
    logger.info("Chronos2 E2E Test with torch.compile")
    logger.info("=" * 60)

    # Test 1: Import
    if not test_import():
        logger.error("FAILED: Cannot import Chronos2Pipeline")
        return False

    # Create test data
    logger.info("\nCreating test data...")
    data = create_test_data()
    logger.info(f"Created {len(data)} rows of OHLC data")

    # Test 2: Eager mode (no compilation)
    logger.info("\n--- Testing Eager Mode ---")
    try:
        wrapper_eager = test_wrapper_creation(compile_enabled=False)
        result_eager = test_prediction(wrapper_eager, data, compile_mode="eager")
    except Exception:
        logger.error("FAILED: Eager mode failed")
        return False

    # Test 3: Compiled mode
    logger.info("\n--- Testing Compiled Mode ---")
    try:
        wrapper_compiled = test_wrapper_creation(compile_enabled=True)
        result_compiled = test_prediction(wrapper_compiled, data, compile_mode="compiled")
    except Exception as exc:
        logger.error(f"FAILED: Compiled mode failed: {exc}")
        logger.exception("Full traceback:")
        return False

    # Compare results
    logger.info("\n--- Comparing Results ---")
    eager_close = result_eager.median['close'].values
    compiled_close = result_compiled.median['close'].values
    mae_diff = np.mean(np.abs(eager_close - compiled_close))
    logger.info(f"MAE difference between eager and compiled: {mae_diff:.6f}")

    if mae_diff > 1e-2:
        logger.warning(f"Large difference detected: {mae_diff}")
    else:
        logger.info("✓ Results are consistent")

    logger.info("\n" + "=" * 60)
    logger.info("SUCCESS: All tests passed")
    logger.info("=" * 60)
    return True


if __name__ == "__main__":
    try:
        success = run_e2e_test()
        sys.exit(0 if success else 1)
    except Exception as exc:
        logger.exception(f"E2E test crashed: {exc}")
        sys.exit(1)
