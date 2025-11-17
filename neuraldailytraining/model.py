from __future__ import annotations

from dataclasses import dataclass

import torch

from hourlycryptotraining.model import HourlyCryptoPolicy, PolicyHeadConfig


@dataclass
class DailyPolicyConfig(PolicyHeadConfig):
    """Policy head configuration with daily-friendly defaults."""

    price_offset_pct: float = 0.025
    min_price_gap_pct: float = 0.0005
    max_len: int = 4096
    equity_max_leverage: float = 2.0
    crypto_max_leverage: float = 1.0


class DailyMultiAssetPolicy(HourlyCryptoPolicy):
    """Thin wrapper around HourlyCryptoPolicy for clarity."""

    def __init__(self, config: DailyPolicyConfig) -> None:
        super().__init__(config)
        self.equity_max_leverage = float(config.equity_max_leverage)
        self.crypto_max_leverage = float(config.crypto_max_leverage)

    def decode_actions(
        self,
        outputs,
        *,
        reference_close,
        chronos_high,
        chronos_low,
        asset_class: Optional[torch.Tensor] = None,
    ):
        decoded = super().decode_actions(
            outputs,
            reference_close=reference_close,
            chronos_high=chronos_high,
            chronos_low=chronos_low,
        )
        if asset_class is not None:
            mask = asset_class.to(decoded["trade_amount"].device, dtype=decoded["trade_amount"].dtype)
            mask = mask.view(-1)
            limits = torch.where(
                mask > 0.5,
                torch.as_tensor(self.crypto_max_leverage, device=decoded["trade_amount"].device, dtype=decoded["trade_amount"].dtype),
                torch.as_tensor(self.equity_max_leverage, device=decoded["trade_amount"].device, dtype=decoded["trade_amount"].dtype),
            )
            decoded["trade_amount"] = torch.minimum(decoded["trade_amount"], limits.unsqueeze(-1))
        return decoded


__all__ = ["DailyPolicyConfig", "DailyMultiAssetPolicy"]
