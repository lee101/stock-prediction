from __future__ import annotations

import json
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional

import types

import torch

from .config import TrainingConfig
from .data import FeatureNormalizer


@dataclass
class CheckpointRecord:
    path: Path
    val_loss: float
    epoch: int
    timestamp: float


def save_checkpoint(
    path: Path,
    *,
    state_dict: Dict[str, torch.Tensor],
    normalizer: FeatureNormalizer,
    feature_columns: List[str],
    metrics: Dict[str, float],
    config: TrainingConfig,
) -> Path:
    payload = {
        "state_dict": {k: v.detach().cpu() for k, v in state_dict.items()},
        "normalizer": normalizer.to_dict(),
        "feature_columns": list(feature_columns),
        "metrics": metrics,
        "config": asdict(config),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)
    return path


def _json_default(obj):
    if isinstance(obj, Path):
        return str(obj)
    return obj


def write_manifest(
    directory: Path,
    records: List[CheckpointRecord],
    config: TrainingConfig,
    feature_columns: List[str],
) -> None:
    manifest = {
        "config": asdict(config),
        "feature_columns": list(feature_columns),
        "checkpoints": [
            {
                "path": record.path.name,
                "val_loss": record.val_loss,
                "epoch": record.epoch,
                "timestamp": record.timestamp,
            }
            for record in records
        ],
    }
    with (directory / "manifest.json").open("w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2, default=_json_default)


def find_best_checkpoint(root: Path) -> Optional[Path]:
    best_path: Optional[Path] = None
    best_loss = float("inf")
    if not root.exists():
        return None
    for manifest_path in root.glob("*/manifest.json"):
        try:
            data = json.loads(manifest_path.read_text())
        except Exception:
            continue
        checkpoints = data.get("checkpoints", [])
        for record in checkpoints:
            val_loss = record.get("val_loss")
            rel_path = record.get("path")
            if rel_path is None or val_loss is None:
                continue
            abs_path = manifest_path.parent / rel_path
            if not abs_path.exists():
                continue
            if val_loss < best_loss:
                best_loss = val_loss
                best_path = abs_path
    return best_path


def _ensure_legacy_pathlib_module() -> None:
    """Work around pickles that reference pathlib._local.Path from legacy envs."""
    if "pathlib._local" in sys.modules:
        return
    import pathlib

    module = types.ModuleType("pathlib._local")
    module.Path = pathlib.Path
    module.PosixPath = pathlib.PosixPath
    module.WindowsPath = getattr(pathlib, "WindowsPath", pathlib.Path)
    module.PurePath = pathlib.PurePath
    module.PurePosixPath = pathlib.PurePosixPath
    module.PureWindowsPath = getattr(pathlib, "PureWindowsPath", pathlib.PurePath)
    sys.modules["pathlib._local"] = module


def load_checkpoint(path: Path) -> Dict[str, object]:
    _ensure_legacy_pathlib_module()
    payload = torch.load(path, map_location="cpu", weights_only=False)
    normalizer = FeatureNormalizer.from_dict(payload["normalizer"])
    return {
        "state_dict": payload["state_dict"],
        "normalizer": normalizer,
        "feature_columns": payload.get("feature_columns", []),
        "metrics": payload.get("metrics", {}),
        "config": payload.get("config"),
    }


__all__ = [
    "CheckpointRecord",
    "save_checkpoint",
    "write_manifest",
    "find_best_checkpoint",
    "load_checkpoint",
]
