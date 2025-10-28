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
