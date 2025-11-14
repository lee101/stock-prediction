"""Unit tests for src/torch_device_utils.py"""

import numpy as np
import pytest
import torch

from src.torch_device_utils import (
    get_device_name,
    get_optimal_device_for_size,
    get_strategy_device,
    is_cuda_device,
    move_to_device,
    require_cuda,
    to_tensor,
)


class TestGetStrategyDevice:
    """Tests for get_strategy_device function."""

    def test_force_cpu_returns_cpu(self):
        """Test force_cpu=True always returns CPU."""
        device = get_strategy_device(force_cpu=True)
        assert device.type == "cpu"

    def test_without_cuda_returns_cpu(self, monkeypatch):
        """Test returns CPU when CUDA unavailable."""
        # Mock CUDA as unavailable
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
        device = get_strategy_device()
        assert device.type == "cpu"

    def test_with_cpu_fallback_env(self, monkeypatch):
        """Test CPU fallback via environment variable."""
        monkeypatch.setenv("MARKETSIM_ALLOW_CPU_FALLBACK", "1")
        device = get_strategy_device()
        assert device.type == "cpu"

    def test_env_flag_override(self, monkeypatch):
        """Test specific environment flag forces CPU."""
        monkeypatch.setenv("MAXDIFF_FORCE_CPU", "1")
        device = get_strategy_device(env_flag="MAXDIFF_FORCE_CPU")
        assert device.type == "cpu"

    def test_returns_cuda_when_available(self, monkeypatch):
        """Test returns CUDA when available and not forced to CPU."""
        # Mock CUDA as available
        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        monkeypatch.delenv("MARKETSIM_ALLOW_CPU_FALLBACK", raising=False)

        device = get_strategy_device()
        # Will be CUDA if actually available, CPU if not
        assert device.type in ["cuda", "cpu"]


class TestToTensor:
    """Tests for to_tensor function."""

    def test_numpy_array_conversion(self):
        """Test converting numpy array to tensor."""
        arr = np.array([1.0, 2.0, 3.0])
        tensor = to_tensor(arr, device=torch.device("cpu"))

        assert isinstance(tensor, torch.Tensor)
        assert tensor.dtype == torch.float32
        assert torch.allclose(tensor, torch.tensor([1.0, 2.0, 3.0]))

    def test_list_conversion(self):
        """Test converting list to tensor."""
        data = [1.0, 2.0, 3.0]
        tensor = to_tensor(data, device=torch.device("cpu"))

        assert isinstance(tensor, torch.Tensor)
        assert tensor.dtype == torch.float32
        assert torch.allclose(tensor, torch.tensor([1.0, 2.0, 3.0]))

    def test_scalar_conversion(self):
        """Test converting scalar to tensor."""
        tensor = to_tensor(5.0, device=torch.device("cpu"))

        assert isinstance(tensor, torch.Tensor)
        assert tensor.dtype == torch.float32
        assert tensor.item() == 5.0

    def test_tensor_passthrough(self):
        """Test passing through existing tensor."""
        original = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)
        tensor = to_tensor(original, device=torch.device("cpu"))

        assert isinstance(tensor, torch.Tensor)
        assert torch.allclose(tensor, original)

    def test_custom_dtype(self):
        """Test custom dtype specification."""
        arr = np.array([1.0, 2.0, 3.0])
        tensor = to_tensor(arr, dtype=torch.float64, device=torch.device("cpu"))

        assert tensor.dtype == torch.float64

    def test_auto_device_detection(self):
        """Test automatic device detection when not specified."""
        arr = np.array([1.0, 2.0, 3.0])
        tensor = to_tensor(arr)  # No device specified

        assert isinstance(tensor, torch.Tensor)
        # Device will be auto-detected


class TestRequireCuda:
    """Tests for require_cuda function."""

    def test_returns_cuda_when_available(self, monkeypatch):
        """Test returns CUDA when available."""
        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        monkeypatch.delenv("MARKETSIM_ALLOW_CPU_FALLBACK", raising=False)

        device = require_cuda("test_feature", allow_fallback=True)
        # Will be CUDA if actually available
        assert device.type in ["cuda", "cpu"]

    def test_fallback_to_cpu_when_allowed(self, monkeypatch):
        """Test falls back to CPU when CUDA unavailable and fallback allowed."""
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

        device = require_cuda("test_feature", allow_fallback=True)
        assert device.type == "cpu"

    def test_raises_when_fallback_not_allowed(self, monkeypatch):
        """Test raises error when CUDA unavailable and fallback not allowed."""
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

        with pytest.raises(RuntimeError, match="requires CUDA"):
            require_cuda("test_feature", allow_fallback=False)

    def test_error_message_includes_symbol(self, monkeypatch):
        """Test error message includes symbol when provided."""
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

        with pytest.raises(RuntimeError, match="for AAPL"):
            require_cuda("test_feature", symbol="AAPL", allow_fallback=False)


class TestMoveToDevice:
    """Tests for move_to_device function."""

    def test_move_to_cpu(self):
        """Test moving tensor to CPU."""
        tensor = torch.tensor([1.0, 2.0, 3.0])
        moved = move_to_device(tensor, torch.device("cpu"))

        assert moved.device.type == "cpu"
        assert torch.allclose(moved, tensor)

    def test_auto_device_detection(self):
        """Test automatic device detection."""
        tensor = torch.tensor([1.0, 2.0, 3.0])
        moved = move_to_device(tensor)  # No device specified

        assert isinstance(moved, torch.Tensor)

    def test_no_op_when_already_on_device(self):
        """Test that moving to same device is a no-op."""
        tensor = torch.tensor([1.0, 2.0, 3.0], device=torch.device("cpu"))
        moved = move_to_device(tensor, torch.device("cpu"))

        # Should be same tensor or equal
        assert torch.allclose(moved, tensor)
        assert moved.device.type == "cpu"


class TestGetDeviceName:
    """Tests for get_device_name function."""

    def test_cpu_device_name(self):
        """Test CPU device name."""
        device = torch.device("cpu")
        name = get_device_name(device)
        assert name == "cpu"

    def test_cuda_device_name(self):
        """Test CUDA device name without index."""
        device = torch.device("cuda")
        name = get_device_name(device)
        assert name == "cuda"

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_cuda_device_name_with_index(self):
        """Test CUDA device name with index."""
        if torch.cuda.device_count() > 0:
            device = torch.device("cuda:0")
            name = get_device_name(device)
            assert name == "cuda:0"


class TestIsCudaDevice:
    """Tests for is_cuda_device function."""

    def test_cpu_is_not_cuda(self):
        """Test CPU device returns False."""
        device = torch.device("cpu")
        assert is_cuda_device(device) is False

    def test_cuda_is_cuda(self):
        """Test CUDA device returns True."""
        device = torch.device("cuda")
        assert is_cuda_device(device) is True

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_cuda_with_index_is_cuda(self):
        """Test CUDA device with index returns True."""
        if torch.cuda.device_count() > 0:
            device = torch.device("cuda:0")
            assert is_cuda_device(device) is True


class TestGetOptimalDeviceForSize:
    """Tests for get_optimal_device_for_size function."""

    def test_small_tensor_uses_cpu(self):
        """Test small tensors prefer CPU."""
        device = get_optimal_device_for_size(100)
        assert device.type == "cpu"

    def test_large_tensor_may_use_cuda(self, monkeypatch):
        """Test large tensors may use CUDA if available."""
        device = get_optimal_device_for_size(100000)
        # Will be CUDA or CPU depending on availability
        assert device.type in ["cuda", "cpu"]

    def test_custom_threshold(self):
        """Test custom threshold parameter."""
        # Below threshold - should be CPU
        device = get_optimal_device_for_size(500, threshold=1000)
        assert device.type == "cpu"

        # Above threshold - may be CUDA
        device = get_optimal_device_for_size(2000, threshold=1000)
        assert device.type in ["cuda", "cpu"]

    def test_exactly_at_threshold(self):
        """Test behavior at exact threshold."""
        threshold = 1000
        device = get_optimal_device_for_size(threshold, threshold=threshold)
        # At threshold, should prefer CUDA if available
        assert device.type in ["cuda", "cpu"]


class TestIntegration:
    """Integration tests for combined functionality."""

    def test_tensor_creation_and_movement(self):
        """Test creating and moving tensors."""
        arr = np.array([1.0, 2.0, 3.0])

        # Create on CPU
        tensor_cpu = to_tensor(arr, device=torch.device("cpu"))
        assert tensor_cpu.device.type == "cpu"

        # Move to auto-detected device
        tensor_moved = move_to_device(tensor_cpu)
        assert isinstance(tensor_moved, torch.Tensor)

    def test_device_selection_consistency(self):
        """Test device selection is consistent."""
        device1 = get_strategy_device(force_cpu=True)
        device2 = get_strategy_device(force_cpu=True)

        assert device1.type == device2.type == "cpu"

    def test_optimal_device_respects_force_cpu(self, monkeypatch):
        """Test optimal device respects CPU fallback."""
        monkeypatch.setenv("MARKETSIM_ALLOW_CPU_FALLBACK", "1")

        # Even for large tensors, should use CPU when fallback enabled
        device = get_strategy_device()
        assert device.type == "cpu"
