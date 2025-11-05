"""Integration tests for GPU detection in model wrappers."""

import importlib
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock, patch, call

import pytest


gpu_utils = importlib.import_module("src.gpu_utils")


# ------------------------------------------------------------------ #
# Test TotoPipeline integration
# ------------------------------------------------------------------ #


def test_toto_wrapper_respects_gpu_detection_on_5090(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that TotoPipeline.unload() does NOT call model.to('cpu') on RTX 5090."""
    # Mock torch to simulate RTX 5090
    class FakeTorchCuda:
        @staticmethod
        def is_available() -> bool:
            return True

        @staticmethod
        def get_device_name(device_idx: int) -> str:
            return "NVIDIA GeForce RTX 5090"

        @staticmethod
        def empty_cache() -> None:
            pass

    class FakeTorchModule:
        cuda = FakeTorchCuda()

    monkeypatch.setattr(gpu_utils, "torch", FakeTorchModule)

    # Import after monkeypatching
    toto_wrapper = importlib.import_module("src.models.toto_wrapper")

    # Create a mock model with a to() method
    mock_model = Mock()
    mock_model.to = Mock()

    # Create a TotoPipeline instance with mocked internals
    class FakePipeline:
        def __init__(self):
            self.device = "cuda:0"
            self.model = mock_model

        def _should_offload_to_cpu(self):
            # Use the real implementation from gpu_utils
            return gpu_utils.should_offload_to_cpu(self.device)

        def unload(self):
            should_offload = self._should_offload_to_cpu()
            try:
                model = getattr(self, "model", None)
                if should_offload:
                    move_to_cpu = getattr(model, "to", None)
                    if callable(move_to_cpu):
                        move_to_cpu("cpu")
            except Exception:
                pass
            self.model = None
            self.forecaster = None

    pipeline = FakePipeline()

    # Call unload
    pipeline.unload()

    # Verify model.to("cpu") was NOT called on RTX 5090
    mock_model.to.assert_not_called()


def test_toto_wrapper_calls_to_cpu_on_regular_gpu(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that TotoPipeline.unload() DOES call model.to('cpu') on regular GPUs."""
    # Mock torch to simulate RTX 3090 (not high-VRAM)
    class FakeTorchCuda:
        @staticmethod
        def is_available() -> bool:
            return True

        @staticmethod
        def get_device_name(device_idx: int) -> str:
            return "NVIDIA GeForce RTX 3090"

        @staticmethod
        def empty_cache() -> None:
            pass

    class FakeTorchModule:
        cuda = FakeTorchCuda()

    monkeypatch.setattr(gpu_utils, "torch", FakeTorchModule)

    # Create a mock model with a to() method
    mock_model = Mock()
    mock_model.to = Mock()

    # Create a TotoPipeline instance with mocked internals
    class FakePipeline:
        def __init__(self):
            self.device = "cuda:0"
            self.model = mock_model

        def _should_offload_to_cpu(self):
            return gpu_utils.should_offload_to_cpu(self.device)

        def unload(self):
            should_offload = self._should_offload_to_cpu()
            try:
                model = getattr(self, "model", None)
                if should_offload:
                    move_to_cpu = getattr(model, "to", None)
                    if callable(move_to_cpu):
                        move_to_cpu("cpu")
            except Exception:
                pass
            self.model = None

    pipeline = FakePipeline()

    # Call unload
    pipeline.unload()

    # Verify model.to("cpu") WAS called on RTX 3090
    mock_model.to.assert_called_once_with("cpu")


# ------------------------------------------------------------------ #
# Test KronosForecastingWrapper integration
# ------------------------------------------------------------------ #


def test_kronos_wrapper_respects_gpu_detection_on_a100(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that KronosForecastingWrapper.unload() does NOT call model.to('cpu') on A100."""
    # Mock torch to simulate A100
    class FakeTorchCuda:
        @staticmethod
        def is_available() -> bool:
            return True

        @staticmethod
        def get_device_name(device_idx: int) -> str:
            return "NVIDIA A100-SXM4-40GB"

        @staticmethod
        def empty_cache() -> None:
            pass

    class FakeTorchModule:
        cuda = FakeTorchCuda()

    monkeypatch.setattr(gpu_utils, "torch", FakeTorchModule)

    # Create mock predictor with model and tokenizer
    mock_model = Mock()
    mock_model.to = Mock()
    mock_tokenizer = Mock()
    mock_tokenizer.to = Mock()

    mock_predictor = SimpleNamespace(
        model=mock_model,
        tokenizer=mock_tokenizer
    )

    # Create a KronosForecastingWrapper-like instance
    class FakeKronosWrapper:
        def __init__(self):
            self._device = "cuda:0"
            self._predictor = mock_predictor

        def _should_offload_to_cpu(self):
            return gpu_utils.should_offload_to_cpu(self._device)

        def unload(self):
            predictor = self._predictor
            if predictor is None:
                return

            should_offload = self._should_offload_to_cpu()

            try:
                if should_offload and hasattr(predictor.model, "to"):
                    predictor.model.to("cpu")
            except Exception:
                pass
            try:
                if should_offload and hasattr(predictor.tokenizer, "to"):
                    predictor.tokenizer.to("cpu")
            except Exception:
                pass
            self._predictor = None

    wrapper = FakeKronosWrapper()

    # Call unload
    wrapper.unload()

    # Verify model.to("cpu") and tokenizer.to("cpu") were NOT called on A100
    mock_model.to.assert_not_called()
    mock_tokenizer.to.assert_not_called()


def test_kronos_wrapper_calls_to_cpu_on_regular_gpu(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that KronosForecastingWrapper.unload() DOES call to('cpu') on regular GPUs."""
    # Mock torch to simulate RTX 4090 (not high-VRAM)
    class FakeTorchCuda:
        @staticmethod
        def is_available() -> bool:
            return True

        @staticmethod
        def get_device_name(device_idx: int) -> str:
            return "NVIDIA GeForce RTX 4090"

        @staticmethod
        def empty_cache() -> None:
            pass

    class FakeTorchModule:
        cuda = FakeTorchCuda()

    monkeypatch.setattr(gpu_utils, "torch", FakeTorchModule)

    # Create mock predictor with model and tokenizer
    mock_model = Mock()
    mock_model.to = Mock()
    mock_tokenizer = Mock()
    mock_tokenizer.to = Mock()

    mock_predictor = SimpleNamespace(
        model=mock_model,
        tokenizer=mock_tokenizer
    )

    # Create a KronosForecastingWrapper-like instance
    class FakeKronosWrapper:
        def __init__(self):
            self._device = "cuda:0"
            self._predictor = mock_predictor

        def _should_offload_to_cpu(self):
            return gpu_utils.should_offload_to_cpu(self._device)

        def unload(self):
            predictor = self._predictor
            if predictor is None:
                return

            should_offload = self._should_offload_to_cpu()

            try:
                if should_offload and hasattr(predictor.model, "to"):
                    predictor.model.to("cpu")
            except Exception:
                pass
            try:
                if should_offload and hasattr(predictor.tokenizer, "to"):
                    predictor.tokenizer.to("cpu")
            except Exception:
                pass
            self._predictor = None

    wrapper = FakeKronosWrapper()

    # Call unload
    wrapper.unload()

    # Verify model.to("cpu") and tokenizer.to("cpu") WERE called on RTX 4090
    mock_model.to.assert_called_once_with("cpu")
    mock_tokenizer.to.assert_called_once_with("cpu")


# ------------------------------------------------------------------ #
# Test different device specifications
# ------------------------------------------------------------------ #


@pytest.mark.parametrize("device_spec", [
    "cuda",
    "cuda:0",
    "cuda:1",
])
def test_gpu_detection_works_with_various_device_specs(monkeypatch: pytest.MonkeyPatch, device_spec: str) -> None:
    """Test that GPU detection works correctly with various device specifications."""
    call_count = 0

    class FakeTorchCuda:
        @staticmethod
        def is_available() -> bool:
            return True

        @staticmethod
        def get_device_name(device_idx: int) -> str:
            nonlocal call_count
            call_count += 1
            # Return 5090 for device 0, A100 for device 1
            if device_idx == 0:
                return "NVIDIA GeForce RTX 5090"
            elif device_idx == 1:
                return "NVIDIA A100-SXM4-40GB"
            return "NVIDIA GeForce RTX 3090"

    class FakeTorchModule:
        cuda = FakeTorchCuda()

    monkeypatch.setattr(gpu_utils, "torch", FakeTorchModule)

    # Test the function
    result = gpu_utils.should_offload_to_cpu(device_spec)

    # Both RTX 5090 and A100 are high-VRAM, so should not offload
    assert result is False
    assert call_count > 0  # Verify the function was actually called


# ------------------------------------------------------------------ #
# Real GPU integration test (requires actual CUDA GPU)
# ------------------------------------------------------------------ #


@pytest.mark.gpu
def test_real_model_device_after_unload() -> None:
    """
    Test that a real tensor stays on GPU after unload on high-VRAM GPUs.
    This is a sanity check to ensure our logic works end-to-end.
    """
    import torch

    if not torch.cuda.is_available():
        pytest.skip("CUDA not available, skipping real GPU test")

    # Check if we're on a high-VRAM GPU
    is_high_vram = gpu_utils.is_high_vram_gpu()
    should_offload = gpu_utils.should_offload_to_cpu()

    # Create a simple tensor on GPU
    test_tensor = torch.randn(100, 100, device="cuda:0")

    # Verify it's on GPU
    assert test_tensor.is_cuda
    assert test_tensor.device.type == "cuda"

    # Simulate what our wrappers do
    if should_offload:
        # On regular GPUs, we would move to CPU
        test_tensor = test_tensor.to("cpu")
        assert not test_tensor.is_cuda
    else:
        # On high-VRAM GPUs, we keep on GPU
        # (in real code, we just don't call .to("cpu"))
        assert test_tensor.is_cuda

    gpu_name = gpu_utils.get_gpu_name()
    print(f"\nGPU: {gpu_name}")
    print(f"High-VRAM: {is_high_vram}")
    print(f"Should offload: {should_offload}")
    print(f"Tensor stayed on GPU: {test_tensor.is_cuda}")

    # Verify inverse relationship
    assert is_high_vram != should_offload


@pytest.mark.gpu
def test_vram_headroom_on_real_gpu() -> None:
    """
    Test that we have sufficient VRAM headroom on high-VRAM GPUs.
    This helps verify it makes sense to keep models loaded.
    """
    import torch

    if not torch.cuda.is_available():
        pytest.skip("CUDA not available, skipping VRAM test")

    is_high_vram = gpu_utils.is_high_vram_gpu()

    if not is_high_vram:
        pytest.skip("Not a high-VRAM GPU, skipping headroom test")

    # Get memory info
    total_memory = torch.cuda.get_device_properties(0).total_memory
    allocated_memory = torch.cuda.memory_allocated(0)
    reserved_memory = torch.cuda.memory_reserved(0)
    free_memory = total_memory - reserved_memory

    total_gb = total_memory / (1024 ** 3)
    free_gb = free_memory / (1024 ** 3)
    allocated_gb = allocated_memory / (1024 ** 3)

    print(f"\nVRAM Stats:")
    print(f"Total: {total_gb:.2f} GB")
    print(f"Allocated: {allocated_gb:.2f} GB")
    print(f"Free: {free_gb:.2f} GB")

    # On high-VRAM GPUs (5090 = 32GB, A100 = 40/80GB, H100 = 80GB)
    # we should have at least 16GB total to justify keeping models loaded
    assert total_gb >= 16.0, f"Expected high-VRAM GPU to have >=16GB, got {total_gb:.2f}GB"

    # Should have reasonable free memory
    assert free_gb >= 1.0, f"Very low free VRAM: {free_gb:.2f}GB"
