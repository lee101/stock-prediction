import importlib
from types import SimpleNamespace
from typing import List

import pytest


gpu_utils = importlib.import_module("src.gpu_utils")


@pytest.mark.parametrize("thresholds,expected", [
    ([(8, 2), (16, 4), (24, 6)], 4),
    ([(8, 2), (16, 4), (32, 8)], 4),
])
def test_recommend_batch_size_increase(thresholds: List[tuple[float, int]], expected: int) -> None:
    total_vram_bytes = 17 * 1024 ** 3
    result = gpu_utils.recommend_batch_size(total_vram_bytes, default_batch_size=2, thresholds=thresholds)
    assert result == expected


def test_recommend_batch_size_no_increase_when_disabled() -> None:
    total_vram_bytes = 24 * 1024 ** 3
    result = gpu_utils.recommend_batch_size(
        total_vram_bytes,
        default_batch_size=2,
        thresholds=[(8, 4), (16, 6)],
        allow_increase=False,
    )
    assert result == 2


@pytest.mark.parametrize(
    "argv,flag_name,expected",
    [
        (("--batch-size", "8"), "--batch-size", True),
        (("--batch-size=16",), "--batch-size", True),
        (("--other", "1"), "--batch-size", False),
    ],
)
def test_cli_flag_detection(argv, flag_name: str, expected: bool) -> None:
    assert gpu_utils.cli_flag_was_provided(flag_name, argv=argv) is expected


def test_detect_total_vram_bytes_normalizes_visible_device_for_torch(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "1")
    fake_calls: List[str] = []

    class FakeDevice:
        def __init__(self, spec: str) -> None:
            fake_calls.append(spec)
            self.spec = spec

    class FakeTorchCuda:
        @staticmethod
        def is_available() -> bool:
            return True

        @staticmethod
        def get_device_properties(device: FakeDevice) -> SimpleNamespace:
            assert device.spec == "cuda:0"
            return SimpleNamespace(total_memory=16 * 1024 ** 3)

    class FakeTorchModule:
        cuda = FakeTorchCuda()

        @staticmethod
        def device(spec: str) -> FakeDevice:
            return FakeDevice(spec)

    monkeypatch.setattr(gpu_utils, "torch", FakeTorchModule)
    monkeypatch.setattr(gpu_utils, "pynvml", None)

    total = gpu_utils.detect_total_vram_bytes()
    assert total == 16 * 1024 ** 3
    assert fake_calls == ["cuda:0"]


def test_detect_total_vram_bytes_respects_cuda_visible_devices_for_nvml(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "1,3")
    monkeypatch.setattr(gpu_utils, "torch", None)

    class FakePynvml:
        def __init__(self) -> None:
            self.init_called = False
            self.shutdown_called = False
            self.handles: List[int] = []

        def nvmlInit(self) -> None:
            self.init_called = True

        def nvmlShutdown(self) -> None:
            self.shutdown_called = True

        def nvmlDeviceGetHandleByIndex(self, index: int) -> str:
            self.handles.append(index)
            return f"handle-{index}"

        def nvmlDeviceGetHandleByPciBusId(self, bus_id: str) -> str:
            raise AssertionError(f"Unexpected PCI bus id lookup: {bus_id}")

        def nvmlDeviceGetMemoryInfo(self, handle: str) -> SimpleNamespace:
            assert handle == "handle-1"
            return SimpleNamespace(total=8 * 1024 ** 3)

    fake_pynvml = FakePynvml()
    monkeypatch.setattr(gpu_utils, "pynvml", fake_pynvml)

    total = gpu_utils.detect_total_vram_bytes()
    assert total == 8 * 1024 ** 3
    assert fake_pynvml.init_called is True
    assert fake_pynvml.shutdown_called is True
    assert fake_pynvml.handles == [1]


# ------------------------------------------------------------------ #
# Tests for GPU detection and offloading logic
# ------------------------------------------------------------------ #


def test_get_gpu_name_returns_none_when_torch_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that get_gpu_name returns None when torch is not available."""
    monkeypatch.setattr(gpu_utils, "torch", None)
    result = gpu_utils.get_gpu_name()
    assert result is None


def test_get_gpu_name_returns_none_when_cuda_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that get_gpu_name returns None when CUDA is not available."""
    class FakeTorchCuda:
        @staticmethod
        def is_available() -> bool:
            return False

    class FakeTorchModule:
        cuda = FakeTorchCuda()

    monkeypatch.setattr(gpu_utils, "torch", FakeTorchModule)
    result = gpu_utils.get_gpu_name()
    assert result is None


def test_get_gpu_name_returns_device_name(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that get_gpu_name returns the GPU device name."""
    class FakeTorchCuda:
        @staticmethod
        def is_available() -> bool:
            return True

        @staticmethod
        def get_device_name(device_idx: int) -> str:
            assert device_idx == 0
            return "NVIDIA GeForce RTX 5090"

    class FakeTorchModule:
        cuda = FakeTorchCuda()

    monkeypatch.setattr(gpu_utils, "torch", FakeTorchModule)
    result = gpu_utils.get_gpu_name()
    assert result == "NVIDIA GeForce RTX 5090"


def test_get_gpu_name_with_device_specification(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that get_gpu_name works with device specification."""
    class FakeTorchCuda:
        @staticmethod
        def is_available() -> bool:
            return True

        @staticmethod
        def get_device_name(device_idx: int) -> str:
            assert device_idx == 1
            return "NVIDIA A100"

    class FakeTorchModule:
        cuda = FakeTorchCuda()

    monkeypatch.setattr(gpu_utils, "torch", FakeTorchModule)
    result = gpu_utils.get_gpu_name("cuda:1")
    assert result == "NVIDIA A100"


@pytest.mark.parametrize("gpu_name,expected", [
    ("NVIDIA GeForce RTX 5090", True),
    ("NVIDIA A100-SXM4-40GB", True),
    ("NVIDIA H100 80GB HBM3", True),
    ("NVIDIA GeForce RTX 4090", False),
    ("NVIDIA GeForce RTX 3090", False),
    ("NVIDIA Tesla V100", False),
])
def test_is_high_vram_gpu_detection(monkeypatch: pytest.MonkeyPatch, gpu_name: str, expected: bool) -> None:
    """Test that is_high_vram_gpu correctly identifies high-VRAM GPUs."""
    class FakeTorchCuda:
        @staticmethod
        def is_available() -> bool:
            return True

        @staticmethod
        def get_device_name(device_idx: int) -> str:
            return gpu_name

    class FakeTorchModule:
        cuda = FakeTorchCuda()

    monkeypatch.setattr(gpu_utils, "torch", FakeTorchModule)
    result = gpu_utils.is_high_vram_gpu()
    assert result is expected


def test_is_high_vram_gpu_returns_false_when_no_gpu(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that is_high_vram_gpu returns False when no GPU is available."""
    monkeypatch.setattr(gpu_utils, "torch", None)
    result = gpu_utils.is_high_vram_gpu()
    assert result is False


@pytest.mark.parametrize("gpu_name,expected_offload", [
    ("NVIDIA GeForce RTX 5090", False),  # High-VRAM GPU, no offload
    ("NVIDIA A100-SXM4-40GB", False),    # High-VRAM GPU, no offload
    ("NVIDIA H100 80GB HBM3", False),    # High-VRAM GPU, no offload
    ("NVIDIA GeForce RTX 4090", True),   # Regular GPU, offload
    ("NVIDIA GeForce RTX 3090", True),   # Regular GPU, offload
])
def test_should_offload_to_cpu_based_on_gpu(monkeypatch: pytest.MonkeyPatch, gpu_name: str, expected_offload: bool) -> None:
    """Test that should_offload_to_cpu makes correct decision based on GPU type."""
    class FakeTorchCuda:
        @staticmethod
        def is_available() -> bool:
            return True

        @staticmethod
        def get_device_name(device_idx: int) -> str:
            return gpu_name

    class FakeTorchModule:
        cuda = FakeTorchCuda()

    monkeypatch.setattr(gpu_utils, "torch", FakeTorchModule)
    result = gpu_utils.should_offload_to_cpu("cuda:0")
    assert result is expected_offload


def test_should_offload_to_cpu_returns_false_for_cpu_device(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that should_offload_to_cpu returns False for CPU devices."""
    class FakeTorchModule:
        pass

    monkeypatch.setattr(gpu_utils, "torch", FakeTorchModule)
    result = gpu_utils.should_offload_to_cpu("cpu")
    assert result is False


def test_should_offload_to_cpu_defaults_to_true_on_exception(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that should_offload_to_cpu defaults to True (safe behavior) on exceptions."""
    class FakeTorchCuda:
        @staticmethod
        def is_available() -> bool:
            raise RuntimeError("Simulated error")

    class FakeTorchModule:
        cuda = FakeTorchCuda()

    monkeypatch.setattr(gpu_utils, "torch", FakeTorchModule)
    result = gpu_utils.should_offload_to_cpu("cuda:0")
    assert result is True


# ------------------------------------------------------------------ #
# Integration test with actual GPU (if available)
# ------------------------------------------------------------------ #


@pytest.mark.gpu
def test_gpu_detection_on_real_hardware() -> None:
    """
    Test GPU detection on actual hardware (requires CUDA-capable GPU).
    This test is marked with @pytest.mark.gpu and will only run when explicitly requested.
    """
    import torch

    if not torch.cuda.is_available():
        pytest.skip("CUDA not available, skipping real GPU test")

    # Get the actual GPU name
    gpu_name = gpu_utils.get_gpu_name()
    assert gpu_name is not None, "GPU name should be detected"

    # Test high-VRAM detection
    is_high_vram = gpu_utils.is_high_vram_gpu()

    # Test offloading decision
    should_offload = gpu_utils.should_offload_to_cpu()

    # Log the results for verification
    print(f"\nDetected GPU: {gpu_name}")
    print(f"Is high-VRAM GPU: {is_high_vram}")
    print(f"Should offload to CPU: {should_offload}")

    # On RTX 5090, should be high-VRAM and should NOT offload
    if "5090" in gpu_name.lower():
        assert is_high_vram is True, "RTX 5090 should be detected as high-VRAM GPU"
        assert should_offload is False, "RTX 5090 should NOT offload to CPU"

    # Verify inverse relationship
    assert is_high_vram != should_offload, "High-VRAM GPUs should not offload to CPU"
