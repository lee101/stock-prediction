from __future__ import annotations

from pathlib import Path

import wandboard as wandboard_module


class _FakeRun:
    def __init__(self) -> None:
        self.logged = []

    def log(self, payload, **kwargs):
        self.logged.append((dict(payload), dict(kwargs)))

    def finish(self):
        return None


class _FakeWandb:
    def __init__(self) -> None:
        self.init_calls = []
        self.run = _FakeRun()

    def init(self, **kwargs):
        self.init_calls.append(dict(kwargs))
        return self.run


def test_wandboard_logger_passes_run_id_and_resume(monkeypatch, tmp_path: Path) -> None:
    fake = _FakeWandb()
    monkeypatch.setattr(wandboard_module, "wandb", fake)
    monkeypatch.setattr(wandboard_module, "_WANDB_AVAILABLE", True)

    with wandboard_module.WandBoardLogger(
        run_name="resume-test",
        run_id="run123",
        resume="allow",
        project="stock",
        entity="team",
        mode="offline",
        log_dir=tmp_path / "tb",
    ) as logger:
        logger.log({"metric/value": 1.0}, step=3)

    assert fake.init_calls
    init_kwargs = fake.init_calls[0]
    assert init_kwargs["id"] == "run123"
    assert init_kwargs["resume"] == "allow"
    assert init_kwargs["project"] == "stock"
