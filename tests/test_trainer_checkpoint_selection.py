from __future__ import annotations

import json
from pathlib import Path
import tempfile

import numpy as np

from binanceneural.config import TrainingConfig
from binanceneural.data import FeatureNormalizer
from binanceneural.trainer import BinanceHourlyTrainer


class _TinyDataModule:
    def __init__(self) -> None:
        self.feature_columns = [f"feat_{idx}" for idx in range(4)]
        self.normalizer = FeatureNormalizer(
            mean=np.zeros(4, dtype=np.float32),
            std=np.ones(4, dtype=np.float32),
        )

    def train_dataloader(self, batch_size: int, num_workers: int = 0):
        return [None]

    def val_dataloader(self, batch_size: int, num_workers: int = 0):
        return [None]


def test_trainer_prefers_robust_checkpoint_metric(monkeypatch) -> None:
    epoch_metrics = iter(
        [
            {"loss": -1.0, "score": 120.0, "sortino": 110.0, "return": 8.0},
            {"loss": -1.0, "score": 100.0, "sortino": 95.0, "return": 6.0},
            {"loss": -1.0, "score": 105.0, "sortino": 100.0, "return": 7.0},
            {"loss": -1.0, "score": 95.0, "sortino": 94.0, "return": 5.0},
        ]
    )

    def _fake_run_epoch(self, model, loader, optimizer, *, train, global_step, current_epoch=1):
        return dict(next(epoch_metrics)), global_step

    def _fake_save_checkpoint(self, model, epoch: int, metrics: dict[str, float]) -> Path:
        path = self.checkpoint_dir / f"epoch_{epoch:03d}.pt"
        path.write_text(json.dumps({"epoch": epoch, "metrics": metrics}))
        return path

    monkeypatch.setattr(BinanceHourlyTrainer, "_run_epoch", _fake_run_epoch)
    monkeypatch.setattr(BinanceHourlyTrainer, "_save_checkpoint", _fake_save_checkpoint)

    with tempfile.TemporaryDirectory() as tmpdir:
        cfg = TrainingConfig(
            epochs=2,
            batch_size=2,
            sequence_length=8,
            transformer_dim=16,
            transformer_layers=1,
            transformer_heads=4,
            transformer_dropout=0.0,
            use_compile=False,
            use_amp=False,
            checkpoint_root=Path(tmpdir) / "ckpts",
            run_name="robust_ckpt_test",
            checkpoint_metric="robust_score",
            checkpoint_gap_penalty=1.0,
            top_k_checkpoints=2,
        )
        trainer = BinanceHourlyTrainer(cfg, _TinyDataModule())
        artifacts = trainer.train()

        expected_best = trainer.checkpoint_dir / "epoch_002.pt"
        assert artifacts.best_checkpoint == expected_best

        alias_path = trainer.checkpoint_dir / "best.pt"
        assert alias_path.exists()
        if alias_path.is_symlink():
            assert alias_path.resolve() == expected_best.resolve()
        else:
            assert alias_path.read_text() == expected_best.read_text()

        manifest = json.loads((trainer.checkpoint_dir / ".topk_manifest.json").read_text())
        metrics_by_epoch = {int(row["epoch"]): float(row["metric"]) for row in manifest}
        assert metrics_by_epoch[1] == 80.0
        assert metrics_by_epoch[2] == 85.0

        progress = json.loads((trainer.checkpoint_dir / "training_progress.json").read_text())
        assert progress["checkpoint_metric_name"] == "robust_score"
        assert progress["checkpoint_metric"] == 85.0
        assert progress["generalization_gap"] == 10.0
        assert progress["best_checkpoint"].endswith("epoch_002.pt")
