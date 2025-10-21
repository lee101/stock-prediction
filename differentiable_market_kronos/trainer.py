from __future__ import annotations

import json
from typing import Dict, Literal

import torch

from differentiable_market.config import DataConfig, EnvironmentConfig, EvaluationConfig, TrainingConfig
from differentiable_market.trainer import DifferentiableMarketTrainer

from .config import KronosFeatureConfig
from .embedding import KronosEmbeddingAdapter


class DifferentiableMarketKronosTrainer(DifferentiableMarketTrainer):
    """Differentiable market trainer that augments state with frozen Kronos embeddings."""

    def __init__(
        self,
        data_cfg: DataConfig,
        env_cfg: EnvironmentConfig,
        train_cfg: TrainingConfig,
        eval_cfg: EvaluationConfig | None,
        kronos_cfg: KronosFeatureConfig,
    ) -> None:
        self.kronos_cfg = kronos_cfg
        self.kronos_adapter: KronosEmbeddingAdapter | None = None
        self._kronos_train_length: int | None = None
        super().__init__(data_cfg, env_cfg, train_cfg, eval_cfg)

    def _build_features(
        self,
        ohlc_tensor: torch.Tensor,
        add_cash: bool,
        phase: Literal["train", "eval"],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        base_features, forward_returns = super()._build_features(ohlc_tensor, add_cash, phase)
        if self.kronos_adapter is None:
            self.kronos_adapter = KronosEmbeddingAdapter(
                self.kronos_cfg,
                self.data_cfg,
                self.symbols,
                self.index,
            )
        start = 0
        if phase == "train":
            self._kronos_train_length = ohlc_tensor.shape[0]
        elif phase == "eval":
            if self._kronos_train_length is None:
                raise RuntimeError("Training features must be initialised before evaluation features.")
            start = self._kronos_train_length
        else:
            raise ValueError(f"Unknown phase {phase}")

        embeddings = self.kronos_adapter.embed_slice(start, ohlc_tensor.shape[0], add_cash=add_cash)
        if embeddings.shape[0] != base_features.shape[0]:
            raise ValueError(
                f"Kronos embeddings length {embeddings.shape[0]} does not match base features {base_features.shape[0]}"
            )
        augmented = torch.cat([base_features, embeddings.to(base_features.dtype)], dim=-1)
        return augmented.contiguous(), forward_returns

    def _write_config_snapshot(self, data_preview: Dict[str, object]) -> None:
        super()._write_config_snapshot(data_preview)
        config_path = self.run_dir / "config.json"
        payload = json.loads(config_path.read_text())
        payload["kronos"] = self._serialize_config(self.kronos_cfg)
        config_path.write_text(json.dumps(payload, indent=2))
        self._config_snapshot = payload
