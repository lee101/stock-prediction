#!/usr/bin/env python3
"""
Common experiment abstractions for neural trading strategies.

We centralise device / dtype handling here so individual strategies can focus
on model specifics without duplicating boilerplate.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch


@dataclass
class ExperimentResult:
    """Container for experiment outcomes."""

    name: str
    metrics: Dict[str, float]
    config_path: Optional[Path] = None

    def to_json(self) -> str:
        return json.dumps(
            {
                "name": self.name,
                "metrics": self.metrics,
                "config_path": str(self.config_path) if self.config_path else None,
            },
            indent=2,
        )


class StrategyExperiment:
    """
    Base class for GPU-aware neural trading experiments.

    Subclasses override data / model hooks while this class handles device
    selection, bf16 support detection, and bookkeeping.
    """

    def __init__(self, config: Dict[str, Any], config_path: Optional[Path] = None):
        self.config = config
        self.config_path = config_path
        self.device = self._select_device()
        self.dtype = self._select_dtype(config.get("training", {}).get("dtype", "fp32"))
        self.gradient_checkpointing = bool(
            config.get("training", {}).get("gradient_checkpointing", False)
        )
        self._rng = torch.Generator(device=self.device if self.device.type == "cuda" else "cpu")
        seed = config.get("training", {}).get("seed")
        if seed is not None:
            self._rng.manual_seed(int(seed))

    # --------------------------------------------------------------------- #
    # Public API                                                            #
    # --------------------------------------------------------------------- #
    def run(self) -> ExperimentResult:
        """End-to-end execution hook used by the CLI runner."""
        self._log_device_banner()
        dataset = self.prepare_data()
        model, optim, criterion = self.build_model(dataset)
        metrics = self.train_and_evaluate(model, optim, criterion, dataset)
        return ExperimentResult(
            name=self.config.get("name", self.__class__.__name__),
            metrics=metrics,
            config_path=self.config_path,
        )

    # --------------------------------------------------------------------- #
    # Abstract hooks                                                        #
    # --------------------------------------------------------------------- #
    def prepare_data(self) -> Any:  # pragma: no cover - abstract in practice
        raise NotImplementedError

    def build_model(
        self, dataset: Any
    ) -> Tuple[torch.nn.Module, torch.optim.Optimizer, torch.nn.Module]:  # pragma: no cover
        raise NotImplementedError

    def train_and_evaluate(  # pragma: no cover - abstract in practice
        self,
        model: torch.nn.Module,
        optim: torch.optim.Optimizer,
        criterion: torch.nn.Module,
        dataset: Any,
    ) -> Dict[str, float]:
        raise NotImplementedError

    # --------------------------------------------------------------------- #
    # Utilities                                                             #
    # --------------------------------------------------------------------- #
    def _select_device(self) -> torch.device:
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")

    def _select_dtype(self, dtype_cfg: str) -> torch.dtype:
        desired = dtype_cfg.lower()
        if desired == "bf16" and self.device.type == "cuda":
            if torch.cuda.is_bf16_supported():
                return torch.bfloat16
            # Fall back gracefully if bf16 is unavailable on the current GPU.
        if desired in {"fp16", "float16"} and self.device.type == "cuda":
            return torch.float16
        return torch.float32

    def _log_device_banner(self) -> None:
        gpu = torch.cuda.get_device_name(self.device) if self.device.type == "cuda" else "CPU"
        dtype_name = str(self.dtype).replace("torch.", "")
        print(
            f"[Experiment:{self.config.get('name', self.__class__.__name__)}] "
            f"device={gpu} dtype={dtype_name} "
            f"grad_checkpointing={self.gradient_checkpointing}"
        )
