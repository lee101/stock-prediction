from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Tuple

from differentiable_market.config import (
    DataConfig,
    EnvironmentConfig,
    EvaluationConfig,
    TrainingConfig,
)


@dataclass(slots=True)
class TotoEmbeddingConfig:
    """
    Configuration for generating frozen Toto embeddings that augment the market
    features consumed by the differentiable trainer.
    """

    context_length: int = 128
    input_feature_dim: int | None = None
    use_toto: bool = True
    freeze_backbone: bool = True
    embedding_dim: int | None = None
    toto_model_id: str = "Datadog/Toto-Open-Base-1.0"
    toto_device: str = "cuda"
    toto_horizon: int = 8
    toto_num_samples: int = 2048
    batch_size: int = 256
    pretrained_model_path: Path | None = None
    cache_dir: Path | None = Path("differentiable_market_totoembedding") / "cache"
    reuse_cache: bool = True
    detach: bool = True
    market_regime_thresholds: Tuple[float, float] = (0.003, 0.015)
    pad_mode: Literal["edge", "repeat"] = "edge"


@dataclass(slots=True)
class TotoTrainingConfig(TrainingConfig):
    """Training configuration extended with Toto embedding controls."""

    toto: TotoEmbeddingConfig = field(default_factory=TotoEmbeddingConfig)


__all__ = [
    "DataConfig",
    "EnvironmentConfig",
    "EvaluationConfig",
    "TotoEmbeddingConfig",
    "TotoTrainingConfig",
]
