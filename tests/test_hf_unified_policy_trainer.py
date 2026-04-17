from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch
from torch.utils.data import Dataset

from binanceneural.config import TrainingConfig
from binanceneural.data import FeatureNormalizer
from binanceneural.hf_trainer_bridge import (
    _EpochMetricCallback,
    UnifiedPolicyHFModel,
    UnifiedPolicyHFTrainer,
    compute_unified_policy_eval_metrics,
    make_training_arguments,
    write_run_metadata,
)
import unified_hourly_experiment.train_hf_trainer_policy as train_hf_trainer_policy


class _TinySequenceDataset(Dataset):
    def __init__(self, rows: int = 4, seq_len: int = 6, input_dim: int = 4) -> None:
        base = torch.linspace(-0.5, 0.5, steps=seq_len * input_dim, dtype=torch.float32).reshape(seq_len, input_dim)
        price = torch.linspace(100.0, 101.0, steps=seq_len, dtype=torch.float32)
        self._items = []
        for idx in range(rows):
            shift = float(idx) * 0.01
            close = price + shift
            item = {
                "features": base + shift,
                "open": close - 0.05,
                "high": close + 0.20,
                "low": close - 0.20,
                "close": close,
                "reference_close": close,
                "chronos_high": close + 0.30,
                "chronos_low": close - 0.30,
                "can_long": torch.tensor(1.0, dtype=torch.float32),
                "can_short": torch.tensor(0.0, dtype=torch.float32),
            }
            self._items.append(item)

    def __len__(self) -> int:
        return len(self._items)

    def __getitem__(self, idx: int):
        return self._items[idx]


class _TinyDataModule:
    def __init__(self, dataset: Dataset) -> None:
        self.train_dataset = dataset
        self.val_dataset = dataset
        self.feature_columns = [f"feat_{idx}" for idx in range(4)]
        self.normalizer = FeatureNormalizer(
            mean=np.zeros(4, dtype=np.float32),
            std=np.ones(4, dtype=np.float32),
        )


class _DummyWandBoardLogger:
    instances: list["_DummyWandBoardLogger"] = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.logs: list[tuple[dict[str, float], int | None]] = []
        self.texts: list[tuple[str, str, int | None]] = []
        self.hparams: list[tuple[dict[str, object], dict[str, float], int | None, str]] = []
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


def test_hf_trainer_exports_portable_checkpoints(tmp_path: Path) -> None:
    dataset = _TinySequenceDataset()
    data_module = _TinyDataModule(dataset)
    checkpoint_dir = tmp_path / "hf_unified"
    train_config = TrainingConfig(
        epochs=1,
        batch_size=2,
        sequence_length=6,
        learning_rate=1e-4,
        weight_decay=0.0,
        grad_clip=1.0,
        fill_temperature=1e-3,
        transformer_dim=16,
        transformer_layers=1,
        transformer_heads=4,
        transformer_dropout=0.0,
        use_compile=False,
        use_amp=False,
        checkpoint_root=tmp_path,
        run_name="hf_unified",
        top_k_checkpoints=2,
        checkpoint_metric="robust_score",
        checkpoint_gap_penalty=0.25,
        num_workers=0,
    )
    args = make_training_arguments(
        output_dir=checkpoint_dir,
        run_name="hf_unified",
        batch_size=2,
        epochs=1,
        max_steps=-1,
        learning_rate=1e-4,
        weight_decay=0.0,
        warmup_steps=0,
        grad_clip=1.0,
        accumulation_steps=1,
        bf16=False,
        fp16=False,
        tf32=False,
        torch_compile=False,
        num_workers=0,
        logging_steps=1,
        optim_name="adamw_torch",
        report_to=["none"],
    )
    model = UnifiedPolicyHFModel(train_config, input_dim=len(data_module.feature_columns))
    trainer = UnifiedPolicyHFTrainer(
        model=model,
        args=args,
        train_dataset=data_module.train_dataset,
        eval_dataset=data_module.val_dataset,
        compute_metrics=compute_unified_policy_eval_metrics,
        callbacks=[],
        train_config=train_config,
        data_module=data_module,  # type: ignore[arg-type]
        checkpoint_dir=checkpoint_dir,
    )
    trainer.add_callback(_EpochMetricCallback(trainer))

    write_run_metadata(
        checkpoint_dir=checkpoint_dir,
        train_config=train_config,
        data_module=data_module,  # type: ignore[arg-type]
        symbols=["AAPL", "TSLA"],
    )

    trainer.train()
    assert trainer.best_checkpoint is not None

    exported = checkpoint_dir / "epoch_001.pt"
    assert exported.exists()
    assert (checkpoint_dir / "best.pt").exists()
    assert (checkpoint_dir / ".topk_manifest.json").exists()
    progress = json.loads((checkpoint_dir / "training_progress.json").read_text())
    assert progress["epoch"] == 1
    assert progress["trainer_backend"] == "transformers_trainer"
    assert progress["checkpoint_metric_name"] == "robust_score"
    assert progress["best_checkpoint"].endswith("epoch_001.pt")

    payload = torch.load(exported, map_location="cpu", weights_only=False)
    assert payload["trainer_backend"] == "transformers_trainer"
    assert payload["feature_columns"] == data_module.feature_columns
    assert "state_dict" in payload


def test_make_training_arguments_falls_back_to_cpu_on_cuda_oom(
    tmp_path: Path,
    monkeypatch,
) -> None:
    calls: list[dict[str, object]] = []

    class _FakeTrainingArguments:
        def __init__(self, **kwargs):
            calls.append(dict(kwargs))
            if not kwargs.get("use_cpu"):
                raise torch.AcceleratorError("CUDA error: out of memory")
            self.kwargs = kwargs

    monkeypatch.setattr("binanceneural.hf_trainer_bridge.TrainingArguments", _FakeTrainingArguments)

    args = make_training_arguments(
        output_dir=tmp_path / "hf_cpu_fallback",
        run_name="hf_cpu_fallback",
        batch_size=2,
        epochs=1,
        max_steps=-1,
        learning_rate=1e-4,
        weight_decay=0.0,
        warmup_steps=0,
        grad_clip=1.0,
        accumulation_steps=1,
        bf16=True,
        fp16=False,
        tf32=True,
        torch_compile=False,
        num_workers=0,
        logging_steps=1,
        optim_name="adamw_torch",
        report_to=["none"],
    )

    assert isinstance(args, _FakeTrainingArguments)
    assert len(calls) == 2
    assert calls[0]["use_cpu"] is False
    assert calls[1]["use_cpu"] is True
    assert calls[1]["dataloader_pin_memory"] is False
    assert calls[1]["bf16"] is False
    assert calls[1]["fp16"] is False
    assert calls[1]["tf32"] is False


def test_hf_trainer_logs_metrics_via_wandboard(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr("binanceneural.hf_trainer_bridge.WandBoardLogger", _DummyWandBoardLogger)
    _DummyWandBoardLogger.instances.clear()

    dataset = _TinySequenceDataset()
    data_module = _TinyDataModule(dataset)
    checkpoint_dir = tmp_path / "hf_unified_wandboard"
    train_config = TrainingConfig(
        epochs=1,
        batch_size=2,
        sequence_length=6,
        learning_rate=1e-4,
        weight_decay=0.0,
        grad_clip=1.0,
        fill_temperature=1e-3,
        transformer_dim=16,
        transformer_layers=1,
        transformer_heads=4,
        transformer_dropout=0.0,
        use_compile=False,
        use_amp=False,
        checkpoint_root=tmp_path,
        log_dir=tmp_path / "tb",
        run_name="hf_unified_wandboard",
        wandb_project="test-project",
        wandb_group="hf-group",
        wandb_tags="hf,smoke",
        top_k_checkpoints=2,
        checkpoint_metric="robust_score",
        checkpoint_gap_penalty=0.25,
        num_workers=0,
    )
    args = make_training_arguments(
        output_dir=checkpoint_dir,
        run_name="hf_unified_wandboard",
        batch_size=2,
        epochs=1,
        max_steps=1,
        learning_rate=1e-4,
        weight_decay=0.0,
        warmup_steps=0,
        grad_clip=1.0,
        accumulation_steps=1,
        bf16=False,
        fp16=False,
        tf32=False,
        torch_compile=False,
        num_workers=0,
        logging_steps=1,
        optim_name="adamw_torch",
        report_to=["none"],
    )
    model = UnifiedPolicyHFModel(train_config, input_dim=len(data_module.feature_columns))
    trainer = UnifiedPolicyHFTrainer(
        model=model,
        args=args,
        train_dataset=data_module.train_dataset,
        eval_dataset=data_module.val_dataset,
        compute_metrics=compute_unified_policy_eval_metrics,
        callbacks=[],
        train_config=train_config,
        data_module=data_module,  # type: ignore[arg-type]
        checkpoint_dir=checkpoint_dir,
    )
    trainer.add_callback(_EpochMetricCallback(trainer))
    write_run_metadata(
        checkpoint_dir=checkpoint_dir,
        train_config=train_config,
        data_module=data_module,  # type: ignore[arg-type]
        symbols=["AAPL", "TSLA"],
    )

    trainer.train()

    assert _DummyWandBoardLogger.instances
    logger = _DummyWandBoardLogger.instances[-1]
    assert logger.kwargs["project"] == "test-project"
    assert logger.kwargs["group"] == "hf-group"
    assert logger.kwargs["tags"] == ("hf", "smoke")
    assert any("val/score" in payload for payload, _step in logger.logs)
    assert any(name == "train/feature_columns" for name, _text, _step in logger.texts)
    assert any(table_name == "hf_train_summary" for _hp, _metrics, _step, table_name in logger.hparams)


@pytest.mark.parametrize(
    ("argv", "expected_message"),
    [
        (["prog", "--symbols", ""], "At least one symbol is required."),
        (["prog", "--symbols", "../../etc/passwd"], "Unsupported symbol"),
        (["prog", "--forecast-horizons", " , "], "At least one forecast horizon is required."),
    ],
)
def test_train_hf_trainer_policy_rejects_invalid_plan_inputs_before_data_loading(
    monkeypatch,
    argv: list[str],
    expected_message: str,
) -> None:
    def _unexpected(*args, **kwargs):
        raise AssertionError("Invalid plan input should fail before touching the data module")

    monkeypatch.setattr(train_hf_trainer_policy.sys, "argv", argv)
    monkeypatch.setattr(train_hf_trainer_policy, "MultiSymbolDataModule", _unexpected)

    with pytest.raises(SystemExit, match="^Plan error:") as excinfo:
        train_hf_trainer_policy.main()
    assert expected_message in str(excinfo.value)
