#!/usr/bin/env python3
"""Live signal generator for SAPP per-symbol models.

Loads best checkpoint per symbol from leaderboard, runs hourly inference,
outputs buy/sell signals with prices and intensity.

Usage:
    from sharpnessadjustedproximalpolicy.live_signal import SAPPPortfolioSignal

    signal = SAPPPortfolioSignal(leaderboard_path="best_leaderboard_xxx.csv")
    actions = signal.generate_signals(market_data)
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from binanceneural.config import DatasetConfig
from binanceneural.data import BinanceHourlyDataModule, FeatureNormalizer
from binanceneural.model import PolicyConfig, build_policy
from src.torch_load_utils import torch_load_compat

logger = logging.getLogger(__name__)


@dataclass
class SymbolSignal:
    symbol: str
    buy_price: float
    sell_price: float
    trade_intensity: float  # 0-1, how strongly to trade
    buy_intensity: float
    sell_intensity: float
    weight: float  # portfolio allocation weight


@dataclass
class PortfolioSignals:
    signals: dict[str, SymbolSignal]
    timestamp: str
    total_weight: float


class SAPPModelLoader:
    """Loads and manages a single per-symbol SAPP model."""

    def __init__(self, ckpt_path: Path, symbol: str, device: torch.device):
        self.symbol = symbol
        self.device = device
        self.ckpt_path = ckpt_path

        ckpt = torch_load_compat(ckpt_path, map_location="cpu", weights_only=False)
        self.config = ckpt.get("config", {})
        state_dict = ckpt.get("state_dict", {})
        self.feature_columns = ckpt.get("feature_columns", [])
        self.normalizer_dict = ckpt.get("normalizer", {})

        input_dim = 0
        for key in ("embed.weight", "_orig_mod.embed.weight"):
            if key in state_dict:
                input_dim = state_dict[key].shape[1]
                break

        horizons = set()
        for col in self.feature_columns:
            if col.startswith("chronos_") and "_h" in col:
                try:
                    horizons.add(int(col.split("_h")[-1]))
                except ValueError:
                    pass
        self.horizons = tuple(sorted(horizons)) if horizons else (1, 24)

        pc = PolicyConfig(
            input_dim=input_dim,
            hidden_dim=self.config.get("transformer_dim", 256),
            num_heads=self.config.get("transformer_heads", 8),
            num_layers=self.config.get("transformer_layers", 4),
            model_arch=self.config.get("model_arch", "classic"),
            max_len=max(self.config.get("sequence_length", 72), 32),
            use_flex_attention=False,
            moe_num_experts=self.config.get("moe_num_experts", 0),
        )
        self.model = build_policy(pc)
        sd = {k.replace("_orig_mod.", "", 1): v for k, v in state_dict.items()}
        self.model.load_state_dict(sd, strict=False)
        self.model.eval().to(device)
        self.seq_len = self.config.get("sequence_length", 72)
        self.scale = self.config.get("trade_amount_scale", 100.0)

        # Build normalizer from checkpoint
        self.normalizer = None
        if self.normalizer_dict:
            self.normalizer = FeatureNormalizer.from_dict(self.normalizer_dict)

    @torch.inference_mode()
    def predict(self, features: torch.Tensor, ref_close: float,
                chronos_high: float, chronos_low: float) -> dict:
        """Run inference on a feature window. Returns action dict."""
        feat = features.unsqueeze(0).to(self.device)
        rc = torch.tensor([[ref_close]], device=self.device)
        ch = torch.tensor([[chronos_high]], device=self.device)
        cl = torch.tensor([[chronos_low]], device=self.device)

        outputs = self.model(feat)
        actions = self.model.decode_actions(outputs, reference_close=rc,
                                            chronos_high=ch, chronos_low=cl)

        return {
            "buy_price": actions["buy_price"][:, -1].item(),
            "sell_price": actions["sell_price"][:, -1].item(),
            "trade_intensity": actions["trade_amount"][:, -1].item() / self.scale,
            "buy_intensity": actions["buy_amount"][:, -1].item() / self.scale,
            "sell_intensity": actions["sell_amount"][:, -1].item() / self.scale,
        }


class SAPPPortfolioSignal:
    """Portfolio-level signal generator using per-symbol SAPP models."""

    def __init__(
        self,
        leaderboard_path: str | Path,
        min_sortino: float = 50.0,
        alloc_method: str = "sqrt_sortino",
        device: str = "cuda",
        symbols: Optional[list[str]] = None,
    ):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.alloc_method = alloc_method
        self.models: dict[str, SAPPModelLoader] = {}
        self.weights: dict[str, float] = {}

        # Load leaderboard
        import csv
        lb_path = Path(leaderboard_path)
        entries = {}
        with open(lb_path) as f:
            for row in csv.DictReader(f):
                sym = row["symbol"]
                if row.get("error"):
                    continue
                sort_val = float(row.get("val_sortino", 0))
                if sort_val < min_sortino:
                    continue
                if symbols and sym not in symbols:
                    continue
                entries[sym] = {
                    "sortino": sort_val,
                    "config": row["config"],
                    "epoch": int(row["best_epoch"]),
                }

        # Load models
        ckpt_root = Path("sharpnessadjustedproximalpolicy/checkpoints")
        for sym, info in sorted(entries.items(), key=lambda x: x[1]["sortino"], reverse=True):
            ckpt_path = self._find_checkpoint(ckpt_root, sym, info["config"], info["epoch"])
            if not ckpt_path:
                logger.warning(f"No checkpoint for {sym} {info['config']} ep{info['epoch']}")
                continue
            try:
                self.models[sym] = SAPPModelLoader(ckpt_path, sym, self.device)
                logger.info(f"Loaded {sym}: sort={info['sortino']:.0f} cfg={info['config']}")
            except Exception as e:
                logger.error(f"Failed to load {sym}: {e}")

        # Compute allocation weights
        self._compute_weights(entries)
        logger.info(f"Portfolio: {len(self.models)} symbols, weights sum={sum(self.weights.values()):.3f}")

    def _find_checkpoint(self, ckpt_root: Path, symbol: str, config: str, epoch: int) -> Optional[Path]:
        candidates = sorted(ckpt_root.glob(f"sap_{config}_{symbol}_*"))
        for d in reversed(candidates):
            p = d / f"epoch_{epoch:03d}.pt"
            if p.exists():
                return p
        return None

    def _compute_weights(self, entries: dict):
        syms = [s for s in self.models]
        if not syms:
            return

        if self.alloc_method == "sqrt_sortino":
            raw = {s: np.sqrt(max(entries[s]["sortino"], 0.01)) for s in syms}
        elif self.alloc_method == "equal":
            raw = {s: 1.0 for s in syms}
        else:
            raw = {s: 1.0 for s in syms}

        total = sum(raw.values())
        self.weights = {s: v / total for s, v in raw.items()}

    def get_symbols(self) -> list[str]:
        return list(self.models.keys())

    def get_weights(self) -> dict[str, float]:
        return dict(self.weights)

    def generate_signal(self, symbol: str, features: torch.Tensor,
                        ref_close: float, chronos_high: float,
                        chronos_low: float) -> Optional[SymbolSignal]:
        """Generate signal for a single symbol."""
        if symbol not in self.models:
            return None

        model = self.models[symbol]
        actions = model.predict(features, ref_close, chronos_high, chronos_low)

        return SymbolSignal(
            symbol=symbol,
            buy_price=actions["buy_price"],
            sell_price=actions["sell_price"],
            trade_intensity=actions["trade_intensity"],
            buy_intensity=actions["buy_intensity"],
            sell_intensity=actions["sell_intensity"],
            weight=self.weights.get(symbol, 0.0),
        )
