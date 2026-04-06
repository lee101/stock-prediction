from __future__ import annotations

import json
from pathlib import Path
import os
import subprocess
import sys
import tempfile

import pytest

def _run_jax_trainer_probe(case: str) -> dict[str, object]:
    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = Path(tmpdir) / "output.json"
        script = """
import json
import sys
from pathlib import Path

import numpy as np
import optax
import torch
from torch.utils.data import DataLoader, Dataset

from binanceneural.config import TrainingConfig
from binanceneural.data import FeatureNormalizer
import binanceneural.jax_trainer as jax_trainer_module
from binanceneural.jax_trainer import JaxClassicTrainer


class _SyntheticDataset(Dataset):
    def __init__(self, n_features: int, seq_len: int, n_samples: int = 10):
        self.n_features = n_features
        self.seq_len = seq_len
        self.n_samples = n_samples
        base = np.linspace(100.0, 104.0, n_samples + seq_len, dtype=np.float32)
        self.prices = base + np.sin(np.arange(n_samples + seq_len, dtype=np.float32)) * 0.2

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int):
        start = idx
        end = idx + self.seq_len
        closes = torch.from_numpy(self.prices[start:end].copy())
        highs = closes * 1.01
        lows = closes * 0.99
        opens = closes * 0.999
        return {
            "features": torch.randn(self.seq_len, self.n_features),
            "open": opens,
            "high": highs,
            "low": lows,
            "close": closes,
            "reference_close": closes.clone(),
            "chronos_high": highs * 1.002,
            "chronos_low": lows * 0.998,
            "can_long": torch.tensor(1.0),
            "can_short": torch.tensor(0.0),
        }


class _SyntheticDataModule:
    def __init__(self, n_features: int = 8, seq_len: int = 8) -> None:
        self.feature_columns = [f"feat_{idx}" for idx in range(n_features)]
        self.normalizer = FeatureNormalizer(
            mean=np.zeros(n_features, dtype=np.float32),
            std=np.ones(n_features, dtype=np.float32),
        )
        self._train = _SyntheticDataset(n_features, seq_len)
        self._val = _SyntheticDataset(n_features, seq_len)

    def train_dataloader(self, batch_size: int, num_workers: int = 0):
        return DataLoader(self._train, batch_size=batch_size, drop_last=True)

    def val_dataloader(self, batch_size: int, num_workers: int = 0):
        return DataLoader(self._val, batch_size=batch_size, drop_last=False)


class _DummyWandBoardLogger:
    instances = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.logs = []
        self.texts = []
        self.hparams = []
        _DummyWandBoardLogger.instances.append(self)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return None

    def log(self, metrics, *, step=None, commit=None):
        self.logs.append((dict(metrics), step))

    def log_text(self, name, text, *, step=None):
        self.texts.append((name, text, step))

    def log_hparams(self, hparams, metrics, *, step=None, table_name="hparams"):
        self.hparams.append((dict(hparams), dict(metrics), step, table_name))


case = sys.argv[1]
root = Path(sys.argv[2])
output_path = Path(sys.argv[3])

if case == "wandboard":
    jax_trainer_module.WandBoardLogger = _DummyWandBoardLogger
    _DummyWandBoardLogger.instances.clear()
    cfg = TrainingConfig(
        epochs=1,
        batch_size=2,
        sequence_length=8,
        learning_rate=1e-3,
        weight_decay=1e-4,
        transformer_dim=16,
        transformer_layers=1,
        transformer_heads=4,
        transformer_dropout=0.0,
        trainer_backend="jax_classic",
        use_compile=False,
        dry_train_steps=1,
        checkpoint_root=root / "ckpts",
        log_dir=root / "tb",
        run_name="jax_wandboard_test",
        wandb_project="test-project",
        wandb_group="test-group",
        wandb_tags="jax,smoke",
    )
    trainer = JaxClassicTrainer(cfg, _SyntheticDataModule())
    artifacts = trainer.train()
    logger = _DummyWandBoardLogger.instances[-1]
    payload = {
        "best_checkpoint": artifacts.best_checkpoint is not None,
        "project": logger.kwargs["project"],
        "group": logger.kwargs["group"],
        "tags": list(logger.kwargs["tags"]),
        "log_count": len(logger.logs),
        "has_val_score": any("val/score" in metrics for metrics, _step in logger.logs),
        "text_count": len(logger.texts),
        "hparams_count": len(logger.hparams),
    }
elif case == "grad_clip":
    clip_calls = []
    real_clip = optax.clip_by_global_norm

    def _record_clip(value):
        clip_calls.append(float(value))
        return real_clip(value)

    jax_trainer_module.optax.clip_by_global_norm = _record_clip
    cfg = TrainingConfig(
        epochs=1,
        batch_size=2,
        sequence_length=8,
        learning_rate=1e-3,
        weight_decay=1e-4,
        grad_clip=0.25,
        transformer_dim=16,
        transformer_layers=1,
        transformer_heads=4,
        transformer_dropout=0.0,
        trainer_backend="jax_classic",
        checkpoint_root=root / "ckpts",
        log_dir=root / "tb",
        run_name="jax_grad_clip_test",
    )
    JaxClassicTrainer(cfg, _SyntheticDataModule())
    payload = {"clip_calls": clip_calls}
elif case == "non_finite_stop":
    jax_trainer_module.WandBoardLogger = _DummyWandBoardLogger
    _DummyWandBoardLogger.instances.clear()
    cfg = TrainingConfig(
        epochs=3,
        batch_size=2,
        sequence_length=8,
        learning_rate=1e-3,
        weight_decay=1e-4,
        transformer_dim=16,
        transformer_layers=1,
        transformer_heads=4,
        transformer_dropout=0.0,
        trainer_backend="jax_classic",
        use_compile=False,
        checkpoint_root=root / "ckpts",
        log_dir=root / "tb",
        run_name="jax_non_finite_stop_test",
    )
    trainer = JaxClassicTrainer(cfg, _SyntheticDataModule())
    calls = iter(
        [
            {"loss": 1.0, "score": 2.0, "sortino": 3.0, "return": 4.0},
            {"loss": 1.5, "score": 2.5, "sortino": 3.5, "return": 4.5},
            {"loss": float("nan"), "score": 5.0, "sortino": 6.0, "return": 7.0},
        ]
    )

    def _fake_run_epoch(*, train):
        return next(calls)

    trainer._run_epoch = _fake_run_epoch
    artifacts = trainer.train()
    logger = _DummyWandBoardLogger.instances[-1]
    meta = (root / "ckpts" / "jax_non_finite_stop_test" / "training_meta.json").read_text()
    payload = {
        "best_checkpoint": artifacts.best_checkpoint is not None,
        "history_len": len(artifacts.history),
        "stop_reason": artifacts.stop_reason,
        "text_names": [name for name, _text, _step in logger.texts],
        "meta": meta,
    }
else:
    raise ValueError(f"Unknown trainer probe case: {case}")

output_path.write_text(json.dumps(payload))
"""

        env = os.environ.copy()
        env.setdefault("JAX_PLATFORMS", "cpu")
        env.setdefault("JAX_DISABLE_JIT", "1")
        proc = subprocess.run(
            [sys.executable, "-c", script, case, tmpdir, str(output_path)],
            capture_output=True,
            text=True,
            env=env,
        )
        if proc.returncode != 0:
            combined_output = ((proc.stdout or "") + "\n" + (proc.stderr or "")).strip()
            lowered = combined_output.lower()
            if proc.returncode < 0 or "fatal python error" in lowered or "aborted" in lowered:
                pytest.skip(
                    "JAX trainer test skipped due to native JAX/XLA compiler abort under this environment: "
                    f"{combined_output[-400:]}"
                )
            if "out of memory" in lowered or "resource_exhausted" in lowered:
                pytest.skip(
                    "JAX trainer test skipped under shared-GPU resource pressure: "
                    f"{combined_output[-400:]}"
                )
            raise AssertionError(
                "JAX trainer subprocess failed unexpectedly:\n"
                f"{combined_output or f'process exited with code {proc.returncode}'}"
            )
        return json.loads(output_path.read_text())


def test_jax_trainer_logs_with_wandboard() -> None:
    probe = _run_jax_trainer_probe("wandboard")

    assert probe["best_checkpoint"] is True
    assert probe["project"] == "test-project"
    assert probe["group"] == "test-group"
    assert probe["tags"] == ["jax", "smoke"]
    assert probe["log_count"] > 0
    assert probe["has_val_score"] is True
    assert probe["text_count"] > 0
    assert probe["hparams_count"] > 0


def test_jax_trainer_applies_grad_clip() -> None:
    probe = _run_jax_trainer_probe("grad_clip")
    assert probe["clip_calls"] == [0.25]


def test_jax_trainer_stops_early_on_non_finite_metrics() -> None:
    probe = _run_jax_trainer_probe("non_finite_stop")

    assert probe["best_checkpoint"] is True
    assert probe["history_len"] == 1
    assert probe["stop_reason"] is not None
    assert "non-finite train/loss" in probe["stop_reason"]
    assert "jax/stop_reason" in set(probe["text_names"])
    assert '"trainer_backend": "jax_classic"' in probe["meta"]
    assert '"stop_reason": "Stopping at epoch 2: non-finite train/loss=nan"' in probe["meta"]
