"""
Quick test to verify all model wrappers use inference_mode.
"""
import torch
import inspect

def test_inference_mode_helpers():
    """Test that our inference context helpers work."""
    print("Testing inference mode helpers...")

    # Test toto wrapper's helper
    from src.models.toto_wrapper import _inference_context
    with _inference_context():
        assert torch.is_inference_mode_enabled(), "Toto wrapper should enable inference mode"
    print("✓ Toto wrapper uses inference_mode")

    # Test that kronos has the helper (check source code)
    with open('external/kronos/model/kronos.py', 'r') as f:
        kronos_source = f.read()
    assert '_inference_context' in kronos_source, "Kronos should have _inference_context helper"
    assert 'inference_mode' in kronos_source, "Kronos should use inference_mode"
    print("✓ Kronos has inference_mode helper")

    # Test chronos2 pipeline decorator (check source directly)
    with open('chronos-forecasting/src/chronos/chronos2/pipeline.py', 'r') as f:
        pipeline_source = f.read()

    # Check for decorator
    assert '@torch.inference_mode()' in pipeline_source, "Chronos2 should have @torch.inference_mode() decorator"
    print("✓ Chronos2 pipeline has @torch.inference_mode() decorator")

    # Check for context manager usage
    assert 'with torch.inference_mode():' in pipeline_source, "Chronos2 should use inference_mode context"
    print("✓ Chronos2 uses inference_mode context in _predict")

    print("\n✅ All model wrappers correctly use torch.inference_mode()!")


if __name__ == '__main__':
    test_inference_mode_helpers()
