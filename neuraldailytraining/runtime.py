from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import pandas as pd
import numpy as np
import torch

from .checkpoints import load_checkpoint
from .config import DailyDatasetConfig, DailyTrainingConfig
from .data import FeatureNormalizer, SymbolFrameBuilder
from .model import DailyMultiAssetPolicy, DailyPolicyConfig

LOGGER = logging.getLogger(__name__)


def _reconstruct_training_config(payload: Optional[Dict]) -> Optional[DailyTrainingConfig]:
    if not payload:
        return None
    data = dict(payload)
    dataset_payload = data.pop("dataset", None)
    dataset_cfg = DailyDatasetConfig(**dataset_payload) if isinstance(dataset_payload, dict) else DailyDatasetConfig()
    return DailyTrainingConfig(**data, dataset=dataset_cfg)


@dataclass
class TradingPlan:
    symbol: str
    timestamp: str
    buy_price: float
    sell_price: float
    trade_amount: float
    reference_close: float


class DailyTradingRuntime:
    """Utility for generating neural trading plans from saved checkpoints."""

    def __init__(
        self,
        checkpoint_path: str | Path,
        *,
        dataset_config: Optional[DailyDatasetConfig] = None,
        device: Optional[str] = None,
        risk_threshold: Optional[float] = None,
    ) -> None:
        path = Path(checkpoint_path)
        if not path.exists():
            raise FileNotFoundError(path)
        payload = load_checkpoint(path)
        feature_columns = payload.get("feature_columns")
        if not feature_columns:
            raise ValueError("Checkpoint is missing feature_columns metadata.")
        normalizer = payload.get("normalizer")
        if not isinstance(normalizer, FeatureNormalizer):
            raise ValueError("Checkpoint normalizer is invalid.")
        ckpt_config = _reconstruct_training_config(payload.get("config"))
        self.sequence_length = ckpt_config.sequence_length if ckpt_config else len(feature_columns)
        self.dataset_config = dataset_config or (ckpt_config.dataset if ckpt_config else DailyDatasetConfig())
        self.dataset_config.sequence_length = self.sequence_length
        self.feature_columns = tuple(feature_columns)
        self.normalizer = normalizer
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        if risk_threshold is None:
            inferred = ckpt_config.risk_threshold if ckpt_config else 1.0
            self.risk_threshold = float(max(0.0, inferred))
        else:
            self.risk_threshold = float(max(0.0, risk_threshold))
        policy_config = DailyPolicyConfig(
            input_dim=len(self.feature_columns),
            hidden_dim=(ckpt_config.transformer_dim if ckpt_config else 256),
            dropout=(ckpt_config.transformer_dropout if ckpt_config else 0.1),
            price_offset_pct=(ckpt_config.price_offset_pct if ckpt_config else 0.025),
            max_trade_qty=(ckpt_config.max_trade_qty if ckpt_config else 3.0),
            min_price_gap_pct=(ckpt_config.min_price_gap_pct if ckpt_config else 0.0005),
            num_heads=(ckpt_config.transformer_heads if ckpt_config else 8),
            num_layers=(ckpt_config.transformer_layers if ckpt_config else 4),
        )
        self.model = DailyMultiAssetPolicy(policy_config).to(self.device)
        self.model.load_state_dict(payload["state_dict"], strict=False)
        self.model.eval()
        self._builder = SymbolFrameBuilder(self.dataset_config, self.feature_columns)

    def plan_for_symbol(
        self,
        symbol: str,
        *,
        as_of: Optional[str | pd.Timestamp] = None,
    ) -> Optional[TradingPlan]:
        frame = self._builder.build(symbol)
        if as_of is not None:
            cutoff = pd.to_datetime(as_of, utc=True)
            frame = frame[frame["date"] <= cutoff]
        if len(frame) < self.sequence_length:
            LOGGER.warning("Symbol %s only has %d rows < sequence length %d", symbol, len(frame), self.sequence_length)
            return None
        window = frame.iloc[-self.sequence_length :].reset_index(drop=True)
        features = window[list(self.feature_columns)].to_numpy(dtype=np.float32)
        norm = self.normalizer.transform(features)
        batch = torch.from_numpy(norm).unsqueeze(0).to(self.device)
        reference = torch.from_numpy(window["reference_close"].to_numpy(dtype=np.float32)).unsqueeze(0).to(self.device)
        c_high = torch.from_numpy(window["chronos_high"].to_numpy(dtype=np.float32)).unsqueeze(0).to(self.device)
        c_low = torch.from_numpy(window["chronos_low"].to_numpy(dtype=np.float32)).unsqueeze(0).to(self.device)
        asset_flag = float(window["asset_class"].iloc[-1]) if "asset_class" in window.columns else (
            1.0 if symbol.upper().endswith("-USD") else 0.0
        )
        asset_tensor = torch.tensor([asset_flag], dtype=batch.dtype, device=self.device)
        with torch.inference_mode():
            outputs = self.model(batch)
            actions = self.model.decode_actions(
                outputs,
                reference_close=reference,
                chronos_high=c_high,
                chronos_low=c_low,
                asset_class=asset_tensor,
            )
        buy = float(actions["buy_price"][0, -1].item())
        sell = float(actions["sell_price"][0, -1].item())
        trade_amount = float(actions["trade_amount"][0, -1].item())
        trade_amount = min(trade_amount, self.risk_threshold)
        timestamp = str(window["timestamp"].iloc[-1]) if "timestamp" in window.columns else str(window["date"].iloc[-1])
        return TradingPlan(
            symbol=symbol,
            timestamp=timestamp,
            buy_price=buy,
            sell_price=sell,
            trade_amount=trade_amount,
            reference_close=float(window["reference_close"].iloc[-1]),
        )

    def generate_plans(self, symbols: Iterable[str]) -> List[TradingPlan]:
        plans: List[TradingPlan] = []
        for symbol in symbols:
            plan = self.plan_for_symbol(symbol)
            if plan:
                plans.append(plan)
        return plans


__all__ = ["DailyTradingRuntime", "TradingPlan"]
