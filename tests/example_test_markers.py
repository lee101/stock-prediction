"""
Example test file demonstrating proper use of pytest markers.

This file shows best practices for categorizing and marking tests
for efficient CI execution.
"""

import pytest


# ============================================================================
# Unit Tests - Fast, no external dependencies
# ============================================================================


@pytest.mark.unit
def test_simple_calculation():
    """Fast unit test with no dependencies."""
    result = 2 + 2
    assert result == 4


@pytest.mark.unit
def test_string_operations():
    """Another fast unit test."""
    text = "hello world"
    assert text.upper() == "HELLO WORLD"


# ============================================================================
# Integration Tests - May use models or external services
# ============================================================================


@pytest.mark.integration
@pytest.mark.model_required
def test_model_integration(fast_ci_mode, fast_model_config):
    """Integration test that adapts to CI mode.

    This test uses different configurations based on whether
    it's running in Fast CI or Full CI.
    """
    if fast_ci_mode:
        # This branch won't run in Fast CI because of @pytest.mark.model_required
        # But if it did, it would use minimal config
        config = fast_model_config
        pytest.skip("Model tests skipped in Fast CI")
    else:
        # Full CI uses production settings
        config = {
            "context_length": 512,
            "prediction_length": 96,
        }

    # Model testing logic here
    assert config is not None


@pytest.mark.integration
@pytest.mark.slow
def test_long_running_integration():
    """Integration test that takes more than 10 seconds.

    Marked as 'slow' so it's skipped in Fast CI.
    """
    # Simulate long-running operation
    import time

    if pytest.ci_mode:  # type: ignore
        time.sleep(0.1)  # Fast in CI
    else:
        time.sleep(1.0)  # Slower locally for realistic testing


# ============================================================================
# Smoke Tests - Quick validation, can include minimal model tests
# ============================================================================


@pytest.mark.smoke
@pytest.mark.model_required
def test_model_loads(torch_device):
    """Smoke test to verify model can be initialized.

    Even though this requires a model, it's marked as 'smoke'
    so it can run in Fast CI for quick validation.
    """
    # Quick check that dependencies work
    assert torch_device in ["cpu", "cuda"]

    # In real test, would check model initialization
    # model = MyModel(device=torch_device)
    # assert model is not None


# ============================================================================
# GPU/CUDA Tests
# ============================================================================


@pytest.mark.cuda_required
@pytest.mark.integration
def test_gpu_training():
    """Test that requires CUDA GPU.

    Automatically skipped on CPU-only runners.
    """
    import torch

    assert torch.cuda.is_available()
    device = torch.device("cuda")
    tensor = torch.ones(10, device=device)
    assert tensor.is_cuda


@pytest.mark.cpu_only
@pytest.mark.unit
def test_cpu_fallback():
    """Test CPU fallback path.

    Ensures code works without GPU.
    """
    import os

    # This test only runs when CPU_ONLY=1
    assert os.getenv("CPU_ONLY", "0") == "1"


# ============================================================================
# External Service Tests
# ============================================================================


@pytest.mark.external
@pytest.mark.requires_openai
def test_openai_api():
    """Test requiring OpenAI API access.

    Skipped in CI unless RUN_EXTERNAL_TESTS=1.
    """
    # Would make real API call
    pytest.skip("External API test - run with RUN_EXTERNAL_TESTS=1")


@pytest.mark.external
@pytest.mark.network_required
def test_data_download():
    """Test requiring internet connection.

    Skipped in CI by default.
    """
    # Would download data from internet
    pytest.skip("Network test - run with RUN_EXTERNAL_TESTS=1")


# ============================================================================
# Benchmark Tests
# ============================================================================


@pytest.mark.benchmark
@pytest.mark.slow
def test_performance_benchmark():
    """Performance benchmark test.

    Measures execution time of critical operations.
    """
    import time

    start = time.time()
    # Simulate operation
    _ = [i**2 for i in range(1000)]
    elapsed = time.time() - start

    # Should complete in reasonable time
    assert elapsed < 1.0


# ============================================================================
# Self-Hosted Only Tests
# ============================================================================


@pytest.mark.self_hosted_only
@pytest.mark.cuda_required
@pytest.mark.slow
def test_multi_gpu_feature():
    """Test requiring multiple GPUs.

    Only runs on self-hosted runners with GPU support.
    """
    import torch

    if not torch.cuda.is_available():
        pytest.skip("No CUDA available")

    gpu_count = torch.cuda.device_count()
    # This test needs specific hardware
    assert gpu_count >= 1


# ============================================================================
# Parametrized Tests
# ============================================================================


@pytest.mark.unit
@pytest.mark.parametrize(
    "input_value,expected",
    [
        (1, 2),
        (2, 4),
        (3, 6),
        (4, 8),
    ],
)
def test_doubling(input_value, expected):
    """Parametrized unit test.

    Efficiently tests multiple cases.
    """
    result = input_value * 2
    assert result == expected


# ============================================================================
# Tests Using CI Fixtures
# ============================================================================


@pytest.mark.unit
def test_using_fast_simulate(fast_simulate_mode):
    """Test that adapts to FAST_SIMULATE mode."""
    if fast_simulate_mode:
        iterations = 10
    else:
        iterations = 1000

    # Run fewer iterations in fast mode
    results = [i for i in range(iterations)]
    assert len(results) == iterations


@pytest.mark.integration
def test_using_fast_config(fast_simulation_config, ci_mode):
    """Test using fast simulation config in CI."""
    config = fast_simulation_config

    if ci_mode:
        # In CI, use minimal settings
        assert config["total_timesteps"] == 100
        assert config["num_envs"] == 1
    else:
        # Locally, could use different settings
        # (but fixture provides same values for consistency)
        pass

    assert config["device"] == "cpu"


# ============================================================================
# Tests to Skip in CI
# ============================================================================


@pytest.mark.ci_skip
def test_manual_verification():
    """Test requiring manual verification.

    Always skipped in CI environments.
    """
    # This test needs human interaction
    pytest.skip("Manual verification required")


# ============================================================================
# Experimental Tests
# ============================================================================


@pytest.mark.experimental
@pytest.mark.integration
def test_new_experimental_feature():
    """Experimental feature test.

    Only runs with --run-experimental flag.
    """
    # Test new feature under development
    pass
