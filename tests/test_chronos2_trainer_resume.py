from __future__ import annotations

from pathlib import Path

from chronos2_trainer import _find_last_checkpoint


def test_find_last_checkpoint_picks_highest_step(tmp_path: Path) -> None:
    (tmp_path / "checkpoint-1").mkdir()
    (tmp_path / "checkpoint-12").mkdir()
    (tmp_path / "checkpoint-3").mkdir()
    (tmp_path / "checkpoint-foo").mkdir()
    (tmp_path / "not-a-checkpoint-99").mkdir()

    found = _find_last_checkpoint(tmp_path)
    assert found is not None
    assert Path(found).name == "checkpoint-12"


def test_find_last_checkpoint_none_when_no_checkpoints(tmp_path: Path) -> None:
    (tmp_path / "something-else").mkdir()
    assert _find_last_checkpoint(tmp_path) is None

