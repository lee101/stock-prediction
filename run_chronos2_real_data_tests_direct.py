"""Run Chronos2 real data tests directly without pytest."""
import os
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

# Set environment variables BEFORE any imports
os.environ["ONLY_CHRONOS2"] = "1"
os.environ["REAL_TESTING"] = "1"
os.environ["TORCH_COMPILED"] = "0"  # Disable torch.compile to avoid SDPA issues

print("Loading modules...")
from backtest_test3_inline import (
    load_chronos2_wrapper,
    resolve_best_model,
    resolve_chronos2_params,
)

print("✓ Successfully imported functions")
print(f"  - resolve_best_model: {resolve_best_model}")
print(f"  - resolve_chronos2_params: {resolve_chronos2_params}")
print(f"  - load_chronos2_wrapper: {load_chronos2_wrapper}")

# Test 1: resolve_best_model returns chronos2
print("\nTest 1: resolve_best_model returns chronos2")
model = resolve_best_model("BTCUSD")
assert model == "chronos2", f"Expected 'chronos2', got '{model}'"
print(f"✓ BTCUSD model: {model}")

model = resolve_best_model("AAPL")
assert model == "chronos2", f"Expected 'chronos2', got '{model}'"
print(f"✓ AAPL model: {model}")

# Test 2: resolve_chronos2_params
print("\nTest 2: resolve_chronos2_params")
params = resolve_chronos2_params("BTCUSD")
print(f"✓ Got params: {list(params.keys())}")
assert isinstance(params, dict)
assert "model_id" in params
assert "context_length" in params
assert "prediction_length" in params
assert "quantile_levels" in params
assert "batch_size" in params
print("✓ All required params present")

# Test 3: load_chronos2_wrapper
print("\nTest 3: load_chronos2_wrapper")
wrapper = load_chronos2_wrapper(params)
assert wrapper is not None
assert hasattr(wrapper, "predict_ohlc")
assert hasattr(wrapper, "pipeline")
print("✓ Wrapper loaded successfully")

# Test 4: Prediction with real data
print("\nTest 4: Prediction with real BTCUSD data")
import pandas as pd
import numpy as np

data_path = Path(__file__).parent / "trainingdata" / "BTCUSD.csv"
if not data_path.exists():
    print(f"⚠ Training data not found: {data_path}")
    print("  Skipping prediction test")
else:
    df = pd.read_csv(data_path)
    df = df.tail(200).copy()
    df = df.reset_index(drop=True)
    df.columns = [col.lower() for col in df.columns]
    df["timestamp"] = pd.date_range(start="2024-01-01", periods=len(df), freq="D")
    df["symbol"] = "BTCUSD"

    print(f"  Data shape: {df.shape}")
    print(f"  Columns: {list(df.columns)}")

    try:
        result = wrapper.predict_ohlc(
            context_df=df,
            symbol="BTCUSD",
            prediction_length=7,
            context_length=min(params["context_length"], len(df)),
            batch_size=params["batch_size"],
        )

        assert result is not None
        assert 0.5 in result.quantile_frames

        median_frame = result.quantile_frames[0.5]
        for col in ["open", "high", "low", "close"]:
            assert col in median_frame.columns, f"Missing column: {col}"

        assert len(median_frame) == 7

        # Check for valid values
        for col in ["open", "high", "low", "close"]:
            values = median_frame[col].values
            assert not np.any(np.isnan(values)), f"{col} contains NaN"
            assert not np.any(np.isinf(values)), f"{col} contains inf"
            assert np.all(values > 0), f"{col} contains non-positive values"

        print("✓ Prediction successful")
        print(f"  Predicted closes: {median_frame['close'].values}")

    except Exception as e:
        print(f"✗ Prediction failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

# Test 5: No "Invalid backend" error (run multiple times)
print("\nTest 5: No 'Invalid backend' error (3 attempts)")
for i in range(3):
    try:
        result = wrapper.predict_ohlc(
            context_df=df.copy(),
            symbol="BTCUSD",
            prediction_length=7,
            context_length=min(params["context_length"], len(df)),
            batch_size=params["batch_size"],
        )
        assert result is not None
        print(f"✓ Attempt {i+1}/3 successful")
    except RuntimeError as e:
        if "Invalid backend" in str(e):
            print(f"✗ Got 'Invalid backend' error on attempt {i+1}: {e}")
            sys.exit(1)
        else:
            raise

print("\n" + "="*60)
print("ALL TESTS PASSED!")
print("="*60)
