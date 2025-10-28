from __future__ import annotations

import json
from pathlib import Path

import yaml

from pufferlibtraining2.config import load_plan


def test_load_plan_default(tmp_path: Path) -> None:
    overrides = {
        "data": {"symbols": ["AAPL", "MSFT"]},
        "logging": {
            "tensorboard_dir": str(tmp_path / "tb"),
            "checkpoint_dir": str(tmp_path / "ckpt"),
            "summary_path": str(tmp_path / "summary.json"),
        },
    }
    plan = load_plan(overrides=overrides)
    assert plan.data.validated_symbols() == ["AAPL", "MSFT"]
    assert plan.logging.tensorboard_dir.exists()
    assert plan.logging.checkpoint_dir.exists()


def test_load_plan_from_yaml(tmp_path: Path) -> None:
    cfg_path = tmp_path / "config.yaml"
    cfg = {
        "train": {"total_timesteps": 1_000_000, "learning_rate": 1e-4},
        "logging": {
            "tensorboard_dir": str(tmp_path / "tb"),
            "checkpoint_dir": str(tmp_path / "ckpt"),
            "summary_path": str(tmp_path / "summary.json"),
        },
    }
    cfg_path.write_text(yaml.safe_dump(cfg))
    plan = load_plan(cfg_path)
    assert plan.train.total_timesteps == 1_000_000
    assert abs(plan.train.learning_rate - 1e-4) < 1e-12
