from __future__ import annotations

from typing import Any

from .config import TrainingConfig
from .data import BinanceHourlyDataModule, MultiSymbolDataModule
from .trainer import BinanceHourlyTrainer


def build_trainer(
    config: TrainingConfig,
    data_module: BinanceHourlyDataModule | MultiSymbolDataModule,
) -> Any:
    backend = str(config.trainer_backend or "torch")
    if backend == "torch":
        return BinanceHourlyTrainer(config, data_module)
    if backend == "jax_classic":
        if (config.model_arch or "classic").lower() != "classic":
            raise ValueError("trainer_backend='jax_classic' requires model_arch='classic'")
        from .jax_trainer import JaxClassicTrainer

        return JaxClassicTrainer(config, data_module)
    raise ValueError(f"Unsupported trainer_backend {config.trainer_backend!r}")


__all__ = ["build_trainer"]
