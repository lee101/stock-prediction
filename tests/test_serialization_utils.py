from __future__ import annotations

from pathlib import Path

from binanceneural.config import TrainingConfig
from src.serialization_utils import serialize_for_checkpoint


def _contains_path(value: object) -> bool:
    if isinstance(value, Path):
        return True
    if isinstance(value, dict):
        return any(_contains_path(v) for v in value.values())
    if isinstance(value, (list, tuple, set)):
        return any(_contains_path(v) for v in value)
    return False


def test_serialize_for_checkpoint_removes_paths_and_dataclasses() -> None:
    cfg = TrainingConfig()
    payload = serialize_for_checkpoint(cfg)
    assert isinstance(payload, dict)
    assert not _contains_path(payload)
    # Nested configs should be mappings, not dataclass instances.
    assert isinstance(payload.get("forecast_config"), dict)
    assert isinstance(payload.get("dataset"), dict)

