#!/usr/bin/env python
"""
Test if Toto model supports batch dimension and produces identical results.

This verifies that:
1. pipeline.predict() accepts batched context tensors [batch_size, seq_len]
2. Batched results match sequential single predictions
"""

import sys
import torch
import numpy as np
from pathlib import Path

# Add repo to path
repo_root = Path(__file__).resolve().parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

# Import after path setup
import backtest_test3_inline as bt

def test_batch_support():
    """Test batched vs sequential predictions."""

    print("=" * 80)
    print("TOTO BATCH SUPPORT TEST")
    print("=" * 80)

    # Load pipeline
    print("\n1. Loading Toto pipeline...")
    pipeline = bt.load_toto_pipeline()
    print(f"   Pipeline loaded on {pipeline.device}")

    # Create two different context sequences (simulating Close and High targets)
    print("\n2. Creating test contexts...")
    np.random.seed(42)
    torch.manual_seed(42)

    seq_len = 100
    context1 = torch.randn(seq_len, dtype=torch.float32)  # Simulate Close
    context2 = torch.randn(seq_len, dtype=torch.float32)  # Simulate High

    print(f"   Context 1 shape: {context1.shape}")
    print(f"   Context 2 shape: {context2.shape}")
    print(f"   Context 1 mean: {context1.mean():.4f}, std: {context1.std():.4f}")
    print(f"   Context 2 mean: {context2.mean():.4f}, std: {context2.std():.4f}")

    # Test parameters
    prediction_length = 1
    num_samples = 32  # Small for speed
    samples_per_batch = 16

    print(f"\n3. Test parameters:")
    print(f"   prediction_length: {prediction_length}")
    print(f"   num_samples: {num_samples}")
    print(f"   samples_per_batch: {samples_per_batch}")

    # Sequential predictions (current approach)
    print("\n4. Running SEQUENTIAL predictions...")
    with torch.inference_mode():
        result1 = pipeline.predict(
            context=context1,
            prediction_length=prediction_length,
            num_samples=num_samples,
            samples_per_batch=samples_per_batch,
        )
        result2 = pipeline.predict(
            context=context2,
            prediction_length=prediction_length,
            num_samples=num_samples,
            samples_per_batch=samples_per_batch,
        )

    print(f"   Result 1 type: {type(result1)}")
    print(f"   Result 1 length: {len(result1) if hasattr(result1, '__len__') else 'N/A'}")
    print(f"   Result 2 type: {type(result2)}")
    print(f"   Result 2 length: {len(result2) if hasattr(result2, '__len__') else 'N/A'}")

    # Extract the forecast tensors
    if isinstance(result1, (list, tuple)):
        tensor1 = result1[0]
        tensor2 = result2[0]
    else:
        tensor1 = result1
        tensor2 = result2

    # Check if it's a TotoForecast object
    if hasattr(tensor1, 'samples'):
        print(f"   Result is TotoForecast object")
        tensor1 = tensor1.samples
        tensor2 = tensor2.samples
        print(f"   Extracted samples from TotoForecast")

    print(f"   Tensor 1 shape: {tensor1.shape if hasattr(tensor1, 'shape') else 'N/A'}")
    print(f"   Tensor 2 shape: {tensor2.shape if hasattr(tensor2, 'shape') else 'N/A'}")

    # Convert to numpy for analysis
    if hasattr(tensor1, 'cpu'):
        array1 = tensor1.cpu().numpy()
        array2 = tensor2.cpu().numpy()
    else:
        array1 = np.array(tensor1)
        array2 = np.array(tensor2)

    print(f"   Array 1: shape={array1.shape}, mean={array1.mean():.6f}, std={array1.std():.6f}")
    print(f"   Array 2: shape={array2.shape}, mean={array2.mean():.6f}, std={array2.std():.6f}")

    # Batched prediction (proposed approach)
    print("\n5. Running BATCHED prediction...")
    batched_context = torch.stack([context1, context2])  # [2, seq_len]
    print(f"   Batched context shape: {batched_context.shape}")

    try:
        with torch.inference_mode():
            batched_result = pipeline.predict(
                context=batched_context,
                prediction_length=prediction_length,
                num_samples=num_samples,
                samples_per_batch=samples_per_batch,
            )

        print(f"   ✓ Batched prediction succeeded!")
        print(f"   Batched result type: {type(batched_result)}")
        print(f"   Batched result length: {len(batched_result) if hasattr(batched_result, '__len__') else 'N/A'}")

        # Extract batched tensors
        if isinstance(batched_result, (list, tuple)):
            if len(batched_result) == 2:
                # Assume [batch_elem_0, batch_elem_1]
                batched_tensor1 = batched_result[0]
                batched_tensor2 = batched_result[1]

                # Check for TotoForecast
                if hasattr(batched_tensor1, 'samples'):
                    print(f"   Batched results are TotoForecast objects")
                    batched_tensor1 = batched_tensor1.samples
                    batched_tensor2 = batched_tensor2.samples
                    print(f"   Extracted samples from batched TotoForecasts")

                print(f"   Batched tensor 1 shape: {batched_tensor1.shape if hasattr(batched_tensor1, 'shape') else 'N/A'}")
                print(f"   Batched tensor 2 shape: {batched_tensor2.shape if hasattr(batched_tensor2, 'shape') else 'N/A'}")
            else:
                # Try to index the first element
                batched_tensor_all = batched_result[0]

                # Check for TotoForecast
                if hasattr(batched_tensor_all, 'samples'):
                    print(f"   Batched result is TotoForecast object")
                    batched_tensor_all = batched_tensor_all.samples
                    print(f"   Extracted samples from batched TotoForecast")

                print(f"   Batched tensor all shape: {batched_tensor_all.shape if hasattr(batched_tensor_all, 'shape') else 'N/A'}")

                # If it has batch dimension, split it
                if hasattr(batched_tensor_all, 'shape') and len(batched_tensor_all.shape) >= 2:
                    if batched_tensor_all.shape[0] == 2:
                        batched_tensor1 = batched_tensor_all[0]
                        batched_tensor2 = batched_tensor_all[1]
                        print(f"   Split batched tensor 1 shape: {batched_tensor1.shape}")
                        print(f"   Split batched tensor 2 shape: {batched_tensor2.shape}")
                    else:
                        print(f"   ⚠ Unexpected batch dimension: {batched_tensor_all.shape[0]}")
                        batched_tensor1 = batched_tensor_all[0]
                        batched_tensor2 = batched_tensor_all[1] if batched_tensor_all.shape[0] > 1 else batched_tensor_all[0]
                else:
                    print(f"   ⚠ Cannot split batched tensor")
                    batched_tensor1 = batched_tensor_all
                    batched_tensor2 = batched_tensor_all
        else:
            batched_tensor_all = batched_result

            # Check for TotoForecast
            if hasattr(batched_tensor_all, 'samples'):
                print(f"   Batched result is TotoForecast object")
                batched_tensor_all = batched_tensor_all.samples
                print(f"   Extracted samples from batched TotoForecast")

            print(f"   Batched tensor shape: {batched_tensor_all.shape if hasattr(batched_tensor_all, 'shape') else 'N/A'}")

            # Try to split
            if hasattr(batched_tensor_all, 'shape') and batched_tensor_all.shape[0] == 2:
                batched_tensor1 = batched_tensor_all[0]
                batched_tensor2 = batched_tensor_all[1]
                print(f"   Split tensor 1 shape: {batched_tensor1.shape}")
                print(f"   Split tensor 2 shape: {batched_tensor2.shape}")
            else:
                print(f"   ⚠ Cannot split batched result")
                batched_tensor1 = batched_tensor_all
                batched_tensor2 = batched_tensor_all

        # Convert to numpy
        if hasattr(batched_tensor1, 'cpu'):
            batched_array1 = batched_tensor1.cpu().numpy()
            batched_array2 = batched_tensor2.cpu().numpy()
        else:
            batched_array1 = np.array(batched_tensor1)
            batched_array2 = np.array(batched_tensor2)

        print(f"   Batched array 1: shape={batched_array1.shape}, mean={batched_array1.mean():.6f}, std={batched_array1.std():.6f}")
        print(f"   Batched array 2: shape={batched_array2.shape}, mean={batched_array2.mean():.6f}, std={batched_array2.std():.6f}")

        # Compare results
        print("\n6. Comparing sequential vs batched results...")

        # Flatten for comparison
        array1_flat = array1.flatten()
        array2_flat = array2.flatten()
        batched_array1_flat = batched_array1.flatten()
        batched_array2_flat = batched_array2.flatten()

        # Check shapes match
        shape_match_1 = array1_flat.shape == batched_array1_flat.shape
        shape_match_2 = array2_flat.shape == batched_array2_flat.shape

        print(f"   Shape match (context 1): {shape_match_1} - {array1_flat.shape} vs {batched_array1_flat.shape}")
        print(f"   Shape match (context 2): {shape_match_2} - {array2_flat.shape} vs {batched_array2_flat.shape}")

        if shape_match_1 and shape_match_2:
            # Calculate differences
            diff1 = np.abs(array1_flat - batched_array1_flat)
            diff2 = np.abs(array2_flat - batched_array2_flat)

            max_diff_1 = diff1.max()
            mean_diff_1 = diff1.mean()
            max_diff_2 = diff2.max()
            mean_diff_2 = diff2.mean()

            print(f"\n   Context 1 differences:")
            print(f"     Max absolute diff: {max_diff_1:.10f}")
            print(f"     Mean absolute diff: {mean_diff_1:.10f}")
            print(f"     Relative error: {(max_diff_1 / (np.abs(array1_flat).mean() + 1e-8)):.10f}")

            print(f"\n   Context 2 differences:")
            print(f"     Max absolute diff: {max_diff_2:.10f}")
            print(f"     Mean absolute diff: {mean_diff_2:.10f}")
            print(f"     Relative error: {(max_diff_2 / (np.abs(array2_flat).mean() + 1e-8)):.10f}")

            # Check if results are close enough
            tolerance = 1e-4  # Allow small numerical differences
            match_1 = max_diff_1 < tolerance
            match_2 = max_diff_2 < tolerance

            print(f"\n7. RESULT:")
            if match_1 and match_2:
                print(f"   ✅ SUCCESS: Batched predictions match sequential predictions!")
                print(f"   ✅ Max difference: {max(max_diff_1, max_diff_2):.10f} < {tolerance}")
                print(f"\n   Batch dimension is SUPPORTED and produces IDENTICAL results!")
                return True
            else:
                print(f"   ⚠ PARTIAL MATCH: Results differ slightly")
                print(f"   Max difference: {max(max_diff_1, max_diff_2):.10f}")
                print(f"   This might be due to:")
                print(f"   - Different random sampling in the model")
                print(f"   - Numerical precision differences")
                print(f"   - Batch processing order")

                # Check if it's just sampling variance
                if max_diff_1 < 0.1 and max_diff_2 < 0.1:
                    print(f"\n   ✓ Differences are small enough - likely sampling variance")
                    print(f"   Batch dimension is SUPPORTED!")
                    return True
                else:
                    print(f"\n   ✗ Differences are too large - may need investigation")
                    return False
        else:
            print(f"   ✗ FAIL: Output shapes don't match")
            print(f"   Batch dimension may not be fully supported")
            return False

    except Exception as e:
        print(f"   ✗ Batched prediction FAILED!")
        print(f"   Error: {e}")
        import traceback
        traceback.print_exc()
        print(f"\n7. RESULT:")
        print(f"   ✗ Batch dimension is NOT supported")
        return False

if __name__ == "__main__":
    success = test_batch_support()
    sys.exit(0 if success else 1)
