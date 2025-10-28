from __future__ import annotations

import uuid
from typing import Dict

from torch.utils.tensorboard import SummaryWriter

from ..config import LoggingConfig


class CompositeLogger:
    """TensorBoard + optional Weights & Biases logger compatible with PuffeRL."""

    def __init__(self, cfg: LoggingConfig, run_config: Dict[str, object]):
        self.cfg = cfg
        self.writer = SummaryWriter(
            log_dir=str(cfg.tensorboard_dir),
            flush_secs=cfg.flush_interval,
        )
        self._wandb = None
        self._wandb_run = None
        self._last_logs: Dict[str, float] = {}
        self.run_id = cfg.wandb_run_name or f"pufferlibtraining2-{uuid.uuid4().hex[:8]}"
        self.should_upload_model = False

        if cfg.wandb_project:
            try:
                import wandb  # type: ignore[import-not-found]
            except ImportError as exc:  # pragma: no cover - environment constraint
                raise RuntimeError(
                    "wandb_project configured but the wandb package is unavailable. "
                    "Install wandb or disable wandb logging."
                ) from exc

            self._wandb = wandb
            self._wandb_run = wandb.init(
                project=cfg.wandb_project,
                entity=cfg.wandb_entity,
                name=cfg.wandb_run_name or self.run_id,
                tags=list(cfg.wandb_tags),
                config=run_config,
                resume="allow",
                allow_val_change=True,
            )
            self.run_id = self._wandb_run.id
            self.should_upload_model = True

    def log(self, logs: Dict[str, float], step: int) -> None:
        numeric_logs = {}
        for key, value in logs.items():
            if isinstance(value, (int, float)):
                self.writer.add_scalar(key, value, step)
                numeric_logs[key] = float(value)
        self.writer.flush()
        self._last_logs = numeric_logs
        if self._wandb_run is not None:
            self._wandb_run.log(logs, step=step)

    def close(self, model_path: str) -> None:
        if self._wandb_run is not None and self.should_upload_model and model_path:
            artifact = self._wandb.Artifact(f"{self.run_id}-checkpoint", type="model")
            artifact.add_file(model_path)
            self._wandb_run.log_artifact(artifact)
            self._wandb_run.finish()
        self.writer.close()

    def download(self) -> str:
        if self._wandb is None or self._wandb_run is None:
            raise RuntimeError("download() is only available when wandb logging is enabled.")
        artifact = self._wandb_run.use_artifact(f"{self.run_id}-checkpoint:latest")
        return artifact.download()

    @property
    def last_logs(self) -> Dict[str, float]:
        return self._last_logs


__all__ = ["CompositeLogger"]
