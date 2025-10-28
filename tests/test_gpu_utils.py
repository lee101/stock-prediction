import importlib
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
