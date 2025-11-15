import numpy as np
import torch

from hourlycryptotraining.checkpoints import (
    CheckpointRecord,
    find_best_checkpoint,
    load_checkpoint,
    save_checkpoint,
    write_manifest,
)
from hourlycryptotraining.config import TrainingConfig
from hourlycryptotraining.data import FeatureNormalizer


def test_checkpoint_save_and_find_best(tmp_path):
    config = TrainingConfig(run_name="testrun")
    root = tmp_path / "checkpoints"
    run_dir = root / config.run_name
    normalizer = FeatureNormalizer(mean=np.array([0.0], dtype=np.float32), std=np.array([1.0], dtype=np.float32))
    records = []
    for epoch, loss in [(1, 0.2), (2, 0.1)]:
        path = run_dir / f"epoch{epoch:02d}.pt"
        save_checkpoint(
            path,
            state_dict={"layer": torch.tensor([loss], dtype=torch.float32)},
            normalizer=normalizer,
            feature_columns=["return_1h"],
            metrics={"loss": loss},
            config=config,
        )
        records.append(CheckpointRecord(path=path, val_loss=loss, epoch=epoch, timestamp=epoch))
    write_manifest(run_dir, records, config, ["return_1h"])
    best_path = find_best_checkpoint(root)
    assert best_path is not None
    payload = load_checkpoint(best_path)
    assert payload["metrics"]["loss"] == 0.1
    assert payload["feature_columns"] == ["return_1h"]
