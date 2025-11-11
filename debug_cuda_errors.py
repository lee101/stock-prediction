#!/usr/bin/env python
"""
Debug CUDA errors in model inference.

Usage:
    CUDA_LAUNCH_BLOCKING=1 python debug_cuda_errors.py
"""

import os
import sys
import torch
import logging

# Enable synchronous CUDA for better error messages
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["TORCH_USE_CUDA_DSA"] = "1"  # Device-side assertions

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def check_cuda_health():
    """Check CUDA is working properly"""
    print("="*80)
    print("CUDA Health Check")
    print("="*80)

    if not torch.cuda.is_available():
        print("❌ CUDA not available")
        return False

    print(f"✅ CUDA available: {torch.cuda.get_device_name(0)}")
    print(f"   CUDA version: {torch.version.cuda}")
    print(f"   PyTorch version: {torch.__version__}")

    # Check memory
    try:
        mem_allocated = torch.cuda.memory_allocated(0) / 1024**3
        mem_reserved = torch.cuda.memory_reserved(0) / 1024**3
        mem_total = torch.cuda.get_device_properties(0).total_memory / 1024**3

        print(f"\n   Memory:")
        print(f"   - Allocated: {mem_allocated:.2f} GB")
        print(f"   - Reserved: {mem_reserved:.2f} GB")
        print(f"   - Total: {mem_total:.2f} GB")
        print(f"   - Free: {mem_total - mem_reserved:.2f} GB")
    except Exception as e:
        print(f"   ⚠️  Could not get memory info: {e}")

    # Test basic operations
    print("\n   Testing basic CUDA operations...")
    try:
        x = torch.randn(1000, 1000, device='cuda')
        y = torch.matmul(x, x)
        z = y.cpu()
        print("   ✅ Basic CUDA operations work")
    except Exception as e:
        print(f"   ❌ Basic CUDA operations failed: {e}")
        return False

    # Clean up
    del x, y, z
    torch.cuda.empty_cache()

    return True


def test_model_loading():
    """Test loading models"""
    print("\n" + "="*80)
    print("Model Loading Test")
    print("="*80)

    # Test Toto
    print("\n1. Testing Toto...")
    try:
        sys.path.insert(0, "toto")
        from src.models.toto_wrapper import TotoPipeline

        print("   Loading Toto pipeline...")
        pipeline = TotoPipeline.from_pretrained(
            model_id="Datadog/Toto-Open-Base-1.0",
            device_map="cuda",
            torch_dtype=torch.float32,
            torch_compile=False,  # Disable for testing
            warmup_sequence=0,  # Skip warmup for testing
        )
        print("   ✅ Toto loaded successfully")

        # Test prediction
        print("   Testing Toto prediction...")
        context = torch.randn(1, 64, device='cuda', dtype=torch.float32)
        with torch.no_grad():
            results = pipeline.predict(
                context=context,
                prediction_length=10,
                num_samples=32,
            )
        print(f"   ✅ Toto prediction successful: {len(results)} results")

        # Clean up
        del pipeline, context, results
        torch.cuda.empty_cache()

    except Exception as e:
        print(f"   ❌ Toto test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Test Kronos
    print("\n2. Testing Kronos...")
    try:
        from src.models.kronos_wrapper import KronosForecastingWrapper

        print("   Loading Kronos wrapper...")
        wrapper = KronosForecastingWrapper(
            model_name="NeoQuasar/Kronos-base",
            device="cuda",
            torch_dtype="float32",
            compile_model=False,  # Disable for testing
        )
        print("   ✅ Kronos loaded successfully")

        # Test prediction
        print("   Testing Kronos prediction...")
        import pandas as pd
        import numpy as np

        # Create test data
        dates = pd.date_range('2024-01-01', periods=100, freq='1h')
        df = pd.DataFrame({
            'timestamp': dates,
            'close': np.random.randn(100).cumsum() + 100
        })

        results = wrapper.predict_series(
            data=df,
            timestamp_col='timestamp',
            columns=['close'],
            pred_len=5,
            lookback=50,
            temperature=0.7,
            sample_count=10,
        )
        print(f"   ✅ Kronos prediction successful: {len(results)} results")

        # Clean up
        del wrapper, df, results
        torch.cuda.empty_cache()

    except Exception as e:
        print(f"   ❌ Kronos test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    return True


def test_both_models_together():
    """Test loading and using both models"""
    print("\n" + "="*80)
    print("Combined Model Test (Toto + Kronos)")
    print("="*80)

    print("\nThis tests if both models can work together without CUDA errors...")

    try:
        sys.path.insert(0, "toto")
        from src.models.toto_wrapper import TotoPipeline
        from src.models.kronos_wrapper import KronosForecastingWrapper
        import pandas as pd
        import numpy as np

        # Load Toto
        print("\n1. Loading Toto...")
        toto_pipeline = TotoPipeline.from_pretrained(
            model_id="Datadog/Toto-Open-Base-1.0",
            device_map="cuda",
            torch_dtype=torch.float32,
            torch_compile=False,
            warmup_sequence=0,
        )
        print("   ✅ Toto loaded")

        # Run Toto prediction
        print("\n2. Running Toto prediction...")
        context = torch.randn(1, 64, device='cuda', dtype=torch.float32)
        with torch.no_grad():
            toto_results = toto_pipeline.predict(
                context=context,
                prediction_length=10,
                num_samples=32,
            )
        print(f"   ✅ Toto prediction done: {len(toto_results)} results")

        # Clear CUDA cache
        print("\n3. Clearing CUDA cache...")
        del context, toto_results
        torch.cuda.empty_cache()
        print("   ✅ Cache cleared")

        # Load Kronos
        print("\n4. Loading Kronos...")
        kronos_wrapper = KronosForecastingWrapper(
            model_name="NeoQuasar/Kronos-base",
            device="cuda",
            torch_dtype="float32",
            compile_model=False,
        )
        print("   ✅ Kronos loaded")

        # Run Kronos prediction
        print("\n5. Running Kronos prediction...")
        dates = pd.date_range('2024-01-01', periods=100, freq='1h')
        df = pd.DataFrame({
            'timestamp': dates,
            'close': np.random.randn(100).cumsum() + 100
        })
        kronos_results = kronos_wrapper.predict_series(
            data=df,
            timestamp_col='timestamp',
            columns=['close'],
            pred_len=5,
            lookback=50,
            sample_count=10,
        )
        print(f"   ✅ Kronos prediction done: {len(kronos_results)} results")

        # Clean up
        print("\n6. Cleaning up...")
        del toto_pipeline, kronos_wrapper, df, kronos_results
        torch.cuda.empty_cache()
        print("   ✅ Cleanup done")

        print("\n✅ SUCCESS: Both models work together without errors")
        return True

    except Exception as e:
        print(f"\n❌ Combined test failed: {e}")
        import traceback
        traceback.print_exc()

        # Try to clean up
        try:
            torch.cuda.empty_cache()
        except:
            pass

        return False


if __name__ == "__main__":
    print("CUDA Error Debugging Script")
    print("="*80)
    print("This script helps debug CUDA errors in model inference.")
    print("Running with CUDA_LAUNCH_BLOCKING=1 for better error messages.")
    print("="*80)

    # Run tests
    try:
        if not check_cuda_health():
            print("\n❌ CUDA health check failed")
            sys.exit(1)

        if not test_model_loading():
            print("\n❌ Model loading test failed")
            sys.exit(1)

        if not test_both_models_together():
            print("\n❌ Combined model test failed")
            sys.exit(1)

        print("\n" + "="*80)
        print("✅ ALL TESTS PASSED")
        print("="*80)
        print("\nYour CUDA setup is working correctly!")
        print("If you're still seeing errors in backtest, try:")
        print("  1. Reducing batch sizes")
        print("  2. Using float32 instead of bfloat16")
        print("  3. Disabling torch.compile temporarily")
        print("  4. Running with CUDA_LAUNCH_BLOCKING=1 for better error messages")

    except KeyboardInterrupt:
        print("\n\nTest interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
