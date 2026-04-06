"""Lightweight signal calibration layer for final-stage order tuning.

Sits on top of any base signal (RL, Chronos2, Gemini) and learns
feature-dependent adjustments to buy_price, sell_price, buy_intensity,
and sell_intensity. Bounded via tanh/sigmoid so adjustments stay within
configurable limits.

Supports two modes:
- 3-output (legacy): shared intensity for buy/sell
- 4-output (directional): separate buy/sell intensity for directional trading

Training uses the differentiable sim from differentiable_loss_utils.py
with Sortino objective. Validation uses binary fills.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn


@dataclass
class CalibrationConfig:
    n_features: int = 16
    hidden: int = 32
    max_price_adj_bps: float = 25.0
    max_amount_adj: float = 0.30
    base_buy_offset: float = -0.001
    base_sell_offset: float = 0.008
    base_intensity: float = 0.5
    directional: bool = False


class SignalCalibrator(nn.Module):
    def __init__(self, config: CalibrationConfig | None = None):
        super().__init__()
        cfg = config or CalibrationConfig()
        self.max_price_adj = cfg.max_price_adj_bps / 10_000.0
        self.max_amount_adj = cfg.max_amount_adj
        self.base_buy_offset = cfg.base_buy_offset
        self.base_sell_offset = cfg.base_sell_offset
        self.base_intensity = cfg.base_intensity
        self.directional = cfg.directional

        n_out = 4 if cfg.directional else 3
        self.net = nn.Sequential(
            nn.Linear(cfg.n_features, cfg.hidden),
            nn.ReLU(),
            nn.Linear(cfg.hidden, n_out),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(
        self,
        features: torch.Tensor,
        base_buy_offset: torch.Tensor | None = None,
        base_sell_offset: torch.Tensor | None = None,
        base_intensity: torch.Tensor | None = None,
    ) -> (
        tuple[torch.Tensor, torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
    ):
        raw = self.net(features)
        buy_adj = torch.tanh(raw[..., 0]) * self.max_price_adj
        sell_adj = torch.tanh(raw[..., 1]) * self.max_price_adj

        if base_buy_offset is None:
            base_buy_offset = torch.full_like(buy_adj, self.base_buy_offset)
        if base_sell_offset is None:
            base_sell_offset = torch.full_like(sell_adj, self.base_sell_offset)
        if base_intensity is None:
            base_intensity = torch.full_like(buy_adj, self.base_intensity)

        cal_buy = base_buy_offset + buy_adj
        cal_sell = base_sell_offset + sell_adj

        if self.directional:
            buy_int = torch.sigmoid(raw[..., 2]) * base_intensity * 2.0
            sell_int = torch.sigmoid(raw[..., 3]) * base_intensity * 2.0
            buy_int = torch.clamp(buy_int, 0.0, 1.0)
            sell_int = torch.clamp(sell_int, 0.0, 1.0)
            return cal_buy, cal_sell, buy_int, sell_int
        else:
            amount_adj = torch.tanh(raw[..., 2]) * self.max_amount_adj
            cal_intensity = torch.clamp(base_intensity * (1.0 + amount_adj), 0.0, 1.0)
            return cal_buy, cal_sell, cal_intensity

    def to_prices(
        self,
        features: torch.Tensor,
        close: torch.Tensor,
        base_buy_offset: torch.Tensor | None = None,
        base_sell_offset: torch.Tensor | None = None,
        base_intensity: torch.Tensor | None = None,
    ) -> tuple:
        result = self.forward(features, base_buy_offset, base_sell_offset, base_intensity)
        buy_off = result[0]
        sell_off = result[1]
        buy_price = close * (1.0 + buy_off)
        sell_price = close * (1.0 + sell_off)
        if self.directional:
            return buy_price, sell_price, result[2], result[3]
        else:
            return buy_price, sell_price, result[2]


def save_calibrator(
    model: SignalCalibrator, path: str | Path, config: CalibrationConfig, metadata: dict | None = None
) -> None:
    data = {
        "state_dict": model.state_dict(),
        "config": {
            "n_features": config.n_features,
            "hidden": config.hidden,
            "max_price_adj_bps": config.max_price_adj_bps,
            "max_amount_adj": config.max_amount_adj,
            "base_buy_offset": config.base_buy_offset,
            "base_sell_offset": config.base_sell_offset,
            "base_intensity": config.base_intensity,
            "directional": config.directional,
        },
    }
    if metadata:
        data["metadata"] = metadata
    torch.save(data, str(path))


def load_calibrator(path: str | Path, device: str = "cpu") -> tuple[SignalCalibrator, CalibrationConfig]:
    ckpt = torch.load(str(path), map_location=device, weights_only=False)
    cfg_dict = ckpt["config"]
    if "directional" not in cfg_dict:
        cfg_dict["directional"] = False
    cfg = CalibrationConfig(**cfg_dict)
    model = SignalCalibrator(cfg)
    model.load_state_dict(ckpt["state_dict"])
    model.to(device).eval()
    return model, cfg
