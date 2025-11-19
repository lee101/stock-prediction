from __future__ import annotations

import json
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional

import torch

from .config import DailyTrainingConfig
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
    config: DailyTrainingConfig,
) -> Path:
    payload = {
        "state_dict": {k: v.detach().cpu() for k, v in state_dict.items()},
        "normalizer": normalizer.to_dict(),
        "feature_columns": list(feature_columns),
        "metrics": dict(metrics),
        "config": asdict(config),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)
    return path


def write_manifest(
    directory: Path,
    records: List[CheckpointRecord],
    config: DailyTrainingConfig,
    feature_columns: List[str],
) -> None:
    def _json_default(obj):
        if isinstance(obj, Path):
            return str(obj)
        return obj

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
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / "manifest.json").open("w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2, default=_json_default)


def load_checkpoint(path: Path) -> Dict[str, object]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    normalizer = FeatureNormalizer.from_dict(payload["normalizer"])
    return {
        "state_dict": payload["state_dict"],
        "normalizer": normalizer,
        "feature_columns": payload.get("feature_columns", []),
        "metrics": payload.get("metrics", {}),
        "config": payload.get("config"),
    }


def find_best_checkpoint(root: Path) -> Optional[Path]:
    if not root.exists():
        return None
    best_loss = float("inf")
    best_path: Optional[Path] = None
    for manifest_path in root.glob("*/manifest.json"):
        try:
            data = json.loads(manifest_path.read_text())
        except Exception:
            continue
        for record in data.get("checkpoints", []):
            val_loss = record.get("val_loss")
            rel = record.get("path")
            if val_loss is None or rel is None:
                continue
            candidate = manifest_path.parent / rel
            if not candidate.exists():
                continue
            # val_loss < best_loss correctly selects most negative (best) since loss = -score
            if val_loss < best_loss:
                best_loss = val_loss
                best_path = candidate
    return best_path


__all__ = ["CheckpointRecord", "save_checkpoint", "write_manifest", "load_checkpoint", "find_best_checkpoint"]
