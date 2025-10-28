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
    grad_accum_steps: int = 1

    clip_value: float = 5.0
    log_interval: int = 10
    seed: int = 1337
    min_symbol_length: int = 180

    max_tokens_per_batch: int = 262_144
    length_buckets: Tuple[int, ...] = (128, 256, 512)
    horizon_buckets: Tuple[int, ...] = (20, 32, 64)
    window_stride: int = 20
    pack_windows: bool = True
    bucket_warmup_steps: int = 0

    torch_compile: bool = True
    compile_mode: str = "max-autotune"
    use_fused_optimizer: bool = True
    precision: str = "bf16"

    eval_temperature: float = 1.0
    eval_top_p: float = 0.9
    eval_sample_count: int = 4

    device: str | None = None

    adapter_type: str = "none"
    adapter_rank: int = 8
    adapter_alpha: float = 16.0
    adapter_dropout: float = 0.05
    adapter_targets: Tuple[str, ...] = (
        "embedding.fusion_proj",
        "transformer",
        "dep_layer",
        "head",
    )
    adapter_output_dir: Path | None = None
    adapter_name: str | None = None
    freeze_backbone: bool = True

    _checkpoint_dir: Path = field(init=False, repr=False)
    _metrics_dir: Path = field(init=False, repr=False)

    _adapter_dir: Path = field(init=False, repr=False)
    _adapter_file: Path = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.data_dir = Path(self.data_dir)
        self.output_dir = Path(self.output_dir)
        self._checkpoint_dir = self.output_dir / "checkpoints"
        self._metrics_dir = self.output_dir / "metrics"
        if isinstance(self.adapter_targets, list):
            self.adapter_targets = tuple(self.adapter_targets)
        adapter_root = Path(self.adapter_output_dir) if self.adapter_output_dir else (self.output_dir / "adapters")
        if self.adapter_type != "none":
            adapter_name = self.adapter_name or self.model_name.replace("/", "_")
            self._adapter_dir = adapter_root / adapter_name
            self._adapter_file = self._adapter_dir / "adapter.pt"
        else:
            self._adapter_dir = adapter_root
            self._adapter_file = self._adapter_dir / "adapter.pt"
        self.length_buckets = self._normalise_bucket(self.length_buckets, "length_buckets")
        self.horizon_buckets = self._normalise_bucket(self.horizon_buckets, "horizon_buckets")
        if self.grad_accum_steps < 1:
            raise ValueError("grad_accum_steps must be >= 1")
        if self.max_tokens_per_batch <= 0:
            raise ValueError("max_tokens_per_batch must be > 0")
        precision_norm = self.precision.lower()
        if precision_norm not in {"bf16", "fp16", "fp32"}:
            raise ValueError("precision must be one of {'bf16', 'fp16', 'fp32'}.")
        self.precision = precision_norm

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
        if self.adapter_type != "none":
            self.adapter_dir.mkdir(parents=True, exist_ok=True)

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
            "grad_accum_steps": self.grad_accum_steps,
            "max_tokens_per_batch": self.max_tokens_per_batch,
            "length_buckets": self.length_buckets,
            "horizon_buckets": self.horizon_buckets,
            "window_stride": self.window_stride,
            "pack_windows": self.pack_windows,
            "bucket_warmup_steps": self.bucket_warmup_steps,
            "torch_compile": self.torch_compile,
            "compile_mode": self.compile_mode,
            "use_fused_optimizer": self.use_fused_optimizer,
            "precision": self.precision,
            "adapter_type": self.adapter_type,
            "adapter_rank": self.adapter_rank,
            "adapter_alpha": self.adapter_alpha,
            "adapter_dropout": self.adapter_dropout,
            "adapter_targets": self.adapter_targets,
            "adapter_output_dir": str(self.adapter_dir),
            "adapter_name": self.adapter_name or self.model_name.replace("/", "_"),
            "freeze_backbone": self.freeze_backbone,
        }

    @property
    def adapter_dir(self) -> Path:
        return self._adapter_dir

    @property
    def adapter_file(self) -> Path:
        return self._adapter_file

    @staticmethod
    def _normalise_bucket(bucket: Tuple[int, ...] | list[int], name: str) -> Tuple[int, ...]:
        values = {int(v) for v in bucket if int(v) > 0}
        if not values:
            raise ValueError(f"{name} must contain at least one positive integer.")
        return tuple(sorted(values))
