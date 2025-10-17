from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from torch import nn
from torch.distributions import Dirichlet

from .config import DataConfig, EnvironmentConfig, TrainingConfig, EvaluationConfig
from .data import load_aligned_ohlc, log_data_preview, split_train_eval
from .env import DifferentiableMarketEnv, smooth_abs
from .features import ohlc_to_features
from .losses import dirichlet_kl
from .policy import DirichletGRUPolicy
from .utils import append_jsonl, ensure_dir, resolve_device, resolve_dtype, set_seed


@dataclass(slots=True)
class TrainingState:
    step: int = 0
    best_eval_loss: float = math.inf
    best_step: int = -1


class DifferentiableMarketTrainer:
    def __init__(
        self,
        data_cfg: DataConfig,
        env_cfg: EnvironmentConfig,
        train_cfg: TrainingConfig,
        eval_cfg: EvaluationConfig | None = None,
    ):
        self.data_cfg = data_cfg
        self.env_cfg = env_cfg
        self.train_cfg = train_cfg
        self.eval_cfg = eval_cfg or EvaluationConfig()

        set_seed(train_cfg.seed)
        self.device = resolve_device(train_cfg.device)
        self.dtype = resolve_dtype(train_cfg.dtype, self.device)
        self.autocast_enabled = self.device.type == "cuda" and train_cfg.bf16_autocast

        self.env = DifferentiableMarketEnv(env_cfg)

        ohlc_all, symbols, index = load_aligned_ohlc(data_cfg)
        self.symbols = symbols
        self.index = index
        self._log_data_preview(ohlc_all)

        train_tensor, eval_tensor = split_train_eval(ohlc_all)
        self.train_features, self.train_returns = ohlc_to_features(train_tensor)
        self.eval_features, self.eval_returns = ohlc_to_features(eval_tensor)

        self.asset_count = self.train_features.shape[1]
        self.feature_dim = self.train_features.shape[2]

        self.policy = DirichletGRUPolicy(
            n_assets=self.asset_count,
            feature_dim=self.feature_dim,
            gradient_checkpointing=train_cfg.gradient_checkpointing,
        ).to(self.device)

        self.ref_policy = DirichletGRUPolicy(
            n_assets=self.asset_count,
            feature_dim=self.feature_dim,
            gradient_checkpointing=False,
        ).to(self.device)
        self.ref_policy.load_state_dict(self.policy.state_dict())
        for param in self.ref_policy.parameters():
            param.requires_grad_(False)

        self.optimizer = self._make_optimizer()

        self.state = TrainingState()
        self.run_dir = self._prepare_run_dir()
        self.ckpt_dir = ensure_dir(self.run_dir / "checkpoints")
        self.metrics_path = self.run_dir / "metrics.jsonl"

        self._write_config_snapshot()
        self._train_step = self._build_train_step()
        if train_cfg.use_compile and hasattr(torch, "compile"):
            self._train_step = torch.compile(self._train_step, mode=train_cfg.torch_compile_mode)

    def _prepare_run_dir(self) -> Path:
        base = ensure_dir(self.train_cfg.save_dir)
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        return ensure_dir(base / timestamp)

    def _write_config_snapshot(self) -> None:
        payload = {
            "data": self._serialize_config(self.data_cfg),
            "env": self._serialize_config(self.env_cfg),
            "train": self._serialize_config(self.train_cfg),
            "eval": self._serialize_config(self.eval_cfg),
            "symbols": self.symbols,
        }
        config_path = self.run_dir / "config.json"
        config_path.write_text(torch.tensor([]).to(torch.float32).device.type)  # placeholder to ensure directory exists?
