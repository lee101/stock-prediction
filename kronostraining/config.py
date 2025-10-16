from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Tuple


@dataclass
class KronosTrainingConfig:
    """
    Centralised configuration for Kronos fine-tuning on the local dataset.

    The defaults are calibrated for the synthetic equities set under
    ``trainingdata/`` and target a manageable runtime while still exercising
    the Kronos-small weights.
    """

    data_dir: Path = Path("trainingdata")
    output_dir: Path = Path("kronostraining") / "artifacts"
    model_name: str = "NeoQuasar/Kronos-small"
    tokenizer_name: str = "NeoQuasar/Kronos-Tokenizer-base"

    lookback_window: int = 64
    prediction_length: int = 30
    validation_days: int = 30
    batch_size: int = 16
    epochs: int = 3
    num_workers: int = 4

    learning_rate: float = 4e-5
    weight_decay: float = 0.01
    betas: Tuple[float, float] = (0.9, 0.95)
    grad_clip_norm: float = 3.0

    clip_value: float = 5.0
    log_interval: int = 10
    seed: int = 1337
    min_symbol_length: int = 180

    eval_temperature: float = 1.0
    eval_top_p: float = 0.9
    eval_sample_count: int = 4

    device: str | None = None

    _checkpoint_dir: Path = field(init=False, repr=False)
    _metrics_dir: Path = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.data_dir = Path(self.data_dir)
        self.output_dir = Path(self.output_dir)
        self._checkpoint_dir = self.output_dir / "checkpoints"
        self._metrics_dir = self.output_dir / "metrics"

    @property
    def checkpoint_dir(self) -> Path:
        return self._checkpoint_dir

    @property
    def best_model_path(self) -> Path:
        return self.checkpoint_dir / "best_model"

    @property
    def last_model_path(self) -> Path:
        return self.checkpoint_dir / "last_model"

    @property
    def metrics_dir(self) -> Path:
        return self._metrics_dir

    @property
    def metrics_file(self) -> Path:
        return self.metrics_dir / "evaluation.json"

    def ensure_output_dirs(self) -> None:
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_dir.mkdir(parents=True, exist_ok=True)

    def resolved_device(self) -> str:
        if self.device:
            return self.device
        import torch

        if torch.cuda.is_available():
            return "cuda:0"
        return "cpu"

    def as_dict(self) -> Dict[str, object]:
        return {
            "data_dir": str(self.data_dir),
            "output_dir": str(self.output_dir),
            "model_name": self.model_name,
            "tokenizer_name": self.tokenizer_name,
            "lookback_window": self.lookback_window,
            "prediction_length": self.prediction_length,
            "validation_days": self.validation_days,
            "batch_size": self.batch_size,
            "epochs": self.epochs,
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
            "betas": self.betas,
            "grad_clip_norm": self.grad_clip_norm,
            "clip_value": self.clip_value,
            "seed": self.seed,
            "device": self.resolved_device(),
        }
