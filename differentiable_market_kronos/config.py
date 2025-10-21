from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal


@dataclass(slots=True)
class KronosFeatureConfig:
    """Configuration for extracting frozen Kronos embeddings as market features."""

    model_path: str = "NeoQuasar/Kronos-small"
    tokenizer_path: str = "NeoQuasar/Kronos-Tokenizer-base"
    context_length: int = 192
    clip: float = 5.0
    batch_size: int = 64
    device: str = "auto"
    embedding_mode: Literal["context", "bits", "both"] = "context"
    cache_dir: Path | None = None
    precision: Literal["float32", "bfloat16"] = "float32"

    def __post_init__(self) -> None:
        if self.context_length <= 0:
            raise ValueError("context_length must be positive")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
