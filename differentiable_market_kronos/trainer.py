from __future__ import annotations

import json
from typing import Dict, Literal

import torch

from differentiable_market.config import DataConfig, EnvironmentConfig, EvaluationConfig, TrainingConfig
from differentiable_market.trainer import DifferentiableMarketTrainer

from .adapter import KronosFeatureAdapter
from .config import KronosFeatureConfig


class DifferentiableMarketKronosTrainer(DifferentiableMarketTrainer):
    """Augments differentiable market training with frozen Kronos path-summary features."""

    def __init__(
        self,
        data_cfg: DataConfig,
        env_cfg: EnvironmentConfig,
        train_cfg: TrainingConfig,
        eval_cfg: EvaluationConfig | None,
        kronos_cfg: KronosFeatureConfig,
    ) -> None:
        self.kronos_cfg = kronos_cfg
        self._kronos_adapter: KronosFeatureAdapter | None = None
        self._kronos_features_full: torch.Tensor | None = None
        self._train_timesteps: int | None = None
        super().__init__(data_cfg, env_cfg, train_cfg, eval_cfg)

    def _ensure_adapter(self) -> KronosFeatureAdapter:
        if self._kronos_adapter is None:
            self._kronos_adapter = KronosFeatureAdapter(
                cfg=self.kronos_cfg,
                data_cfg=self.data_cfg,
                symbols=self.symbols,
                index=self.index,
            )
        return self._kronos_adapter

    def _ensure_full_features(self, dtype: torch.dtype) -> torch.Tensor:
        if self._kronos_features_full is None:
            adapter = self._ensure_adapter()
            features = adapter.features_tensor(add_cash=False, dtype=dtype)
            if features.numel() == 0:
                raise ValueError("Kronos features tensor is empty; check context length and data availability")
            self._kronos_features_full = features
        return self._kronos_features_full

    def _slice_kronos(self, start: int, end: int, device: torch.device, dtype: torch.dtype, add_cash: bool) -> torch.Tensor:
        full = self._ensure_full_features(dtype=dtype).to(device=device, dtype=dtype)
        if add_cash:
            zeros = torch.zeros(full.shape[0], 1, full.shape[2], dtype=dtype, device=device)
            full = torch.cat([full, zeros], dim=1)
        if end > full.shape[0]:
            raise ValueError(f"Requested Kronos slice {start}:{end} exceeds feature length {full.shape[0]}")
        segment = full[start:end]
        if segment.shape[0] <= 1:
            return torch.zeros((0, segment.shape[1], segment.shape[2]), dtype=dtype, device=device)
        return segment[1:].contiguous()

    def _build_features(
        self,
        ohlc_tensor: torch.Tensor,
        add_cash: bool,
        phase: Literal["train", "eval"],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        base_features, forward_returns = super()._build_features(ohlc_tensor, add_cash, phase)
        dtype = base_features.dtype
        device = base_features.device

        if phase == "train":
            start = 0
            end = ohlc_tensor.shape[0]
            self._train_timesteps = end
        elif phase == "eval":
            if self._train_timesteps is None:
                raise RuntimeError("Training features must be initialised before evaluation features")
            start = self._train_timesteps
            end = start + ohlc_tensor.shape[0]
        else:  # pragma: no cover
            raise ValueError(f"Unknown phase {phase}")

        kronos_features = self._slice_kronos(start, end, device=device, dtype=dtype, add_cash=add_cash)
        if kronos_features.shape[0] != base_features.shape[0]:
            raise ValueError(
                f"Kronos features length {kronos_features.shape[0]} does not match base features {base_features.shape[0]}"
            )
        augmented = torch.cat([base_features, kronos_features], dim=-1)
        return augmented, forward_returns

    def _write_config_snapshot(self, data_preview: Dict[str, object]) -> None:
        super()._write_config_snapshot(data_preview)
        config_path = self.run_dir / "config.json"
        payload = json.loads(config_path.read_text())
        payload["kronos"] = {
            "model_path": self.kronos_cfg.model_path,
            "tokenizer_path": self.kronos_cfg.tokenizer_path,
            "context_length": self.kronos_cfg.context_length,
            "horizons": list(self.kronos_cfg.horizons),
            "quantiles": list(self.kronos_cfg.quantiles),
            "sample_count": self.kronos_cfg.sample_count,
            "temperature": self.kronos_cfg.temperature,
            "top_p": self.kronos_cfg.top_p,
            "bf16": self.kronos_cfg.bf16,
        }
        config_path.write_text(json.dumps(payload, indent=2))
        self._config_snapshot = payload
