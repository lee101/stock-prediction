from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from neuraldailytraining import DailyDatasetConfig, DailyTrainingConfig

PACKAGE_ROOT = Path(__file__).resolve().parent
CHECKPOINT_ROOT = PACKAGE_ROOT / "checkpoints"
LOG_ROOT = PACKAGE_ROOT / "tensorboard_logs"
RESULTS_ROOT = PACKAGE_ROOT / "results"


@dataclass(frozen=True)
class DailyExperimentSpec:
    name: str
    description: str
    training_overrides: dict[str, Any] = field(default_factory=dict)
    dataset_overrides: dict[str, Any] = field(default_factory=dict)


DEFAULT_EXPERIMENT_ORDER = (
    "muon_baseline",
    "muon_ema",
    "muon_small_stable",
    "adamw_small_stable",
)


def _default_dataset_config() -> DailyDatasetConfig:
    return DailyDatasetConfig(
        data_root=Path("trainingdata") / "train",
        forecast_cache_dir=Path("strategytraining") / "forecast_cache",
        grouping_strategy="correlation",
        symbol_dropout_rate=0.1,
        require_forecasts=True,
        forecast_fill_strategy="persistence",
        forecast_cache_writeback=True,
    )



def _default_training_config(dataset_config: DailyDatasetConfig) -> DailyTrainingConfig:
    return DailyTrainingConfig(
        epochs=1,
        batch_size=32,
        sequence_length=256,
        learning_rate=3e-4,
        weight_decay=1e-4,
        grad_clip=1.0,
        optimizer_name="muon",
        warmup_steps=16,
        ema_decay=0.0,
        checkpoint_root=CHECKPOINT_ROOT,
        log_dir=LOG_ROOT,
        wandb_project=None,
        use_compile=False,
        use_amp=True,
        num_workers=0,
        max_train_batches=96,
        max_val_batches=24,
        dataset=dataset_config,
    )



def _experiment_specs() -> dict[str, DailyExperimentSpec]:
    return {
        "muon_baseline": DailyExperimentSpec(
            name="muon_baseline",
            description="Current large Muon setup with short warmup for fair capped runs.",
            training_overrides={
                "optimizer_name": "muon",
                "learning_rate": 3e-4,
                "weight_decay": 1e-4,
                "warmup_steps": 16,
                "ema_decay": 0.0,
                "transformer_dim": 256,
                "transformer_layers": 4,
                "transformer_heads": 8,
                "transformer_dropout": 0.1,
                "grad_clip": 1.0,
                "permutation_rate": 0.5,
                "price_scale_probability": 0.2,
            },
            dataset_overrides={"symbol_dropout_rate": 0.1},
        ),
        "muon_ema": DailyExperimentSpec(
            name="muon_ema",
            description="Muon kept large, but with EMA and gentler augmentation for more stable small-data fitting.",
            training_overrides={
                "optimizer_name": "muon",
                "learning_rate": 2e-4,
                "weight_decay": 1e-4,
                "warmup_steps": 24,
                "ema_decay": 0.995,
                "transformer_dim": 256,
                "transformer_layers": 4,
                "transformer_heads": 8,
                "transformer_dropout": 0.08,
                "grad_clip": 0.75,
                "permutation_rate": 0.25,
                "price_scale_probability": 0.1,
            },
            dataset_overrides={"symbol_dropout_rate": 0.05},
        ),
        "muon_small_stable": DailyExperimentSpec(
            name="muon_small_stable",
            description="Smaller Muon model with EMA and reduced augmentation to match limited daily history.",
            training_overrides={
                "optimizer_name": "muon",
                "learning_rate": 2e-4,
                "weight_decay": 1e-4,
                "warmup_steps": 24,
                "ema_decay": 0.995,
                "transformer_dim": 192,
                "transformer_layers": 3,
                "transformer_heads": 6,
                "transformer_dropout": 0.05,
                "grad_clip": 0.75,
                "permutation_rate": 0.2,
                "price_scale_probability": 0.1,
            },
            dataset_overrides={"symbol_dropout_rate": 0.05},
        ),
        "adamw_small_stable": DailyExperimentSpec(
            name="adamw_small_stable",
            description="Same smaller model but with AdamW, EMA, and a more cautious LR / WD pair.",
            training_overrides={
                "optimizer_name": "adamw",
                "learning_rate": 1.5e-4,
                "weight_decay": 2e-4,
                "warmup_steps": 24,
                "ema_decay": 0.995,
                "transformer_dim": 192,
                "transformer_layers": 3,
                "transformer_heads": 6,
                "transformer_dropout": 0.05,
                "grad_clip": 0.75,
                "permutation_rate": 0.2,
                "price_scale_probability": 0.1,
            },
            dataset_overrides={"symbol_dropout_rate": 0.05},
        ),
        "adamw_small_cautious": DailyExperimentSpec(
            name="adamw_small_cautious",
            description="Extra-cautious AdamW variant for tiny-data stability checks.",
            training_overrides={
                "optimizer_name": "adamw",
                "learning_rate": 1e-4,
                "weight_decay": 3e-4,
                "warmup_steps": 32,
                "ema_decay": 0.997,
                "transformer_dim": 192,
                "transformer_layers": 3,
                "transformer_heads": 6,
                "transformer_dropout": 0.05,
                "grad_clip": 0.5,
                "permutation_rate": 0.15,
                "price_scale_probability": 0.05,
            },
            dataset_overrides={"symbol_dropout_rate": 0.03},
        ),
    }



def experiment_names() -> tuple[str, ...]:
    return tuple(_experiment_specs().keys())



def build_experiment(
    name: str,
    *,
    seed: int | None = None,
    device: str | None = None,
    dataset_overrides: dict[str, Any] | None = None,
    training_overrides: dict[str, Any] | None = None,
) -> tuple[DailyExperimentSpec, DailyDatasetConfig, DailyTrainingConfig]:
    specs = _experiment_specs()
    if name not in specs:
        raise KeyError(f"Unknown experiment '{name}'. Available: {', '.join(sorted(specs))}")
    spec = specs[name]
    merged_dataset = dict(spec.dataset_overrides)
    if dataset_overrides:
        merged_dataset.update(dataset_overrides)
    dataset_config = replace(_default_dataset_config(), **merged_dataset)

    merged_training = dict(spec.training_overrides)
    if training_overrides:
        merged_training.update(training_overrides)
    if seed is not None:
        merged_training["seed"] = seed
    if device is not None:
        merged_training["device"] = device

    training_config = replace(_default_training_config(dataset_config), dataset=dataset_config, **merged_training)
    return spec, dataset_config, training_config
