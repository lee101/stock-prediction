from __future__ import annotations

import json
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
from .symbol_groups import get_group_id
from .model import DailyMultiAssetPolicy, DailyPolicyConfig, MultiSymbolDailyPolicy

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
        non_tradable: Optional[Iterable[str]] = None,
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
        self.symbol_to_group_id = payload.get("symbol_to_group_id", {}) if isinstance(payload, dict) else {}
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
            use_cross_attention=(ckpt_config.use_cross_attention if ckpt_config else True),
        )
        if policy_config.use_cross_attention:
            self.model = MultiSymbolDailyPolicy(policy_config).to(self.device)
        else:
            self.model = DailyMultiAssetPolicy(policy_config).to(self.device)
        self.model.load_state_dict(payload["state_dict"], strict=False)
        self.model.eval()
        self._builder = SymbolFrameBuilder(self.dataset_config, self.feature_columns)
        self.non_tradable = self._load_non_tradable(non_tradable, default_path=path.parent / "non_tradable.json")

    def plan_for_symbol(
        self,
        symbol: str,
        *,
        as_of: Optional[str | pd.Timestamp] = None,
    ) -> Optional[TradingPlan]:
        plans = self.plan_batch([symbol], as_of=as_of)
        return plans[0] if plans else None

    def generate_plans(self, symbols: Iterable[str]) -> List[TradingPlan]:
        return self.plan_batch(list(symbols))

    def plan_batch(
        self,
        symbols: Sequence[str],
        *,
        as_of: Optional[str | pd.Timestamp] = None,
        non_tradable_override: Optional[Iterable[str]] = None,
    ) -> List[TradingPlan]:
        prepared = [self._prepare_symbol_window(sym, as_of=as_of) for sym in symbols]
        prepared = [item for item in prepared if item is not None]
        if not prepared:
            return []
        override_set = {sym.upper() for sym in non_tradable_override} if non_tradable_override else set()
        combined_non_tradable = override_set or self.non_tradable

        batch_feats = torch.stack([item["features"] for item in prepared], dim=0).to(self.device)
        reference = torch.stack([item["reference"] for item in prepared], dim=0).to(self.device)
        c_high = torch.stack([item["c_high"] for item in prepared], dim=0).to(self.device)
        c_low = torch.stack([item["c_low"] for item in prepared], dim=0).to(self.device)
        asset_class = torch.as_tensor([item["asset_flag"] for item in prepared], dtype=batch_feats.dtype, device=self.device)
        group_ids = torch.as_tensor([item["group_id"] for item in prepared], dtype=torch.long, device=self.device)

        group_mask = None
        if hasattr(self.model, "blocks") and len(prepared) > 1:
            group_mask = group_ids.unsqueeze(0) == group_ids.unsqueeze(1)

        with torch.inference_mode():
            if group_mask is not None:
                outputs = self.model(batch_feats, group_mask=group_mask)
            else:
                outputs = self.model(batch_feats)
            actions = self.model.decode_actions(
                outputs,
                reference_close=reference,
                chronos_high=c_high,
                chronos_low=c_low,
                asset_class=asset_class,
            )

        plans: List[TradingPlan] = []
        for idx, item in enumerate(prepared):
            trade_amount = float(actions["trade_amount"][idx, -1].item())
            trade_amount = min(trade_amount, self.risk_threshold)
            if item["symbol"].upper() in combined_non_tradable:
                trade_amount = 0.0

            # Extract prices from model output
            buy_price = float(actions["buy_price"][idx, -1].item())
            sell_price = float(actions["sell_price"][idx, -1].item())

            # SAFETY CHECK: Ensure sell_price > buy_price with minimum 0.03% spread (3 bps)
            # This prevents overtrading and ensures profitable execution
            MIN_SPREAD_PCT = 0.0003  # 0.03%
            required_min_sell = buy_price * (1.0 + MIN_SPREAD_PCT)

            if sell_price <= buy_price or sell_price < required_min_sell:
                LOGGER.warning(
                    "Invalid price spread for %s: buy=%.4f sell=%.4f (spread=%.6f%%). "
                    "Required min spread: %.6f%%. Skipping trade.",
                    item["symbol"],
                    buy_price,
                    sell_price,
                    ((sell_price - buy_price) / buy_price * 100) if buy_price > 0 else 0,
                    MIN_SPREAD_PCT * 100,
                )
                trade_amount = 0.0

            plans.append(
                TradingPlan(
                    symbol=item["symbol"],
                    timestamp=item["timestamp"],
                    buy_price=buy_price,
                    sell_price=sell_price,
                    trade_amount=trade_amount,
                    reference_close=float(item["reference_close"]),
                )
            )
        return plans

    def _prepare_symbol_window(
        self,
        symbol: str,
        *,
        as_of: Optional[str | pd.Timestamp] = None,
    ) -> Optional[Dict[str, object]]:
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
        reference = torch.from_numpy(window["reference_close"].to_numpy(dtype=np.float32))
        c_high = torch.from_numpy(window["chronos_high"].to_numpy(dtype=np.float32))
        c_low = torch.from_numpy(window["chronos_low"].to_numpy(dtype=np.float32))
        asset_flag = float(window["asset_class"].iloc[-1]) if "asset_class" in window.columns else (
            1.0 if symbol.upper().endswith("-USD") else 0.0
        )
        if self.symbol_to_group_id:
            group_id = int(self.symbol_to_group_id.get(symbol.upper(), 0))
        else:
            group_id = get_group_id(symbol)
        timestamp = str(window["timestamp"].iloc[-1]) if "timestamp" in window.columns else str(window["date"].iloc[-1])
        return {
            "symbol": symbol,
            "features": torch.from_numpy(norm),
            "reference": reference,
            "c_high": c_high,
            "c_low": c_low,
            "asset_flag": asset_flag,
            "group_id": group_id,
            "timestamp": timestamp,
            "reference_close": float(window["reference_close"].iloc[-1]),
        }

    @staticmethod
    def _load_non_tradable(
        external: Optional[Iterable[str]],
        *,
        default_path: Path,
    ) -> set[str]:
        if external:
            return {sym.upper() for sym in external}
        if default_path.exists():
            try:
                data = json.loads(default_path.read_text())
                entries = data.get("non_tradable", [])
                if isinstance(entries, list):
                    return {str(item["symbol"] if isinstance(item, dict) else item).upper() for item in entries}
            except Exception:
                LOGGER.warning("Failed to load non_tradable file at %s", default_path)
        return set()


__all__ = ["DailyTradingRuntime", "TradingPlan"]
