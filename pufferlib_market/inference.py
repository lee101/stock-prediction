#!/usr/bin/env python3
"""
Inference script for trained PPO trading model.
Loads checkpoint and generates trading signals for live/paper trading.
"""

from __future__ import annotations

import argparse
import struct
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


class ResidualBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.net = nn.Sequential(nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, dim))

    def forward(self, x):
        return x + self.net(self.norm(x))


class Policy(nn.Module):
    def __init__(self, obs_size: int, num_actions: int, hidden: int = 512, num_blocks: int = 3):
        super().__init__()
        self.input_proj = nn.Linear(obs_size, hidden)
        self.blocks = nn.Sequential(*[ResidualBlock(hidden) for _ in range(num_blocks)])
        self.out_norm = nn.LayerNorm(hidden)
        self.actor = nn.Sequential(nn.Linear(hidden, hidden // 2), nn.GELU(), nn.Linear(hidden // 2, num_actions))
        self.critic = nn.Sequential(nn.Linear(hidden, hidden // 2), nn.GELU(), nn.Linear(hidden // 2, 1))

    def forward(self, x):
        h = self.input_proj(x)
        h = self.blocks(h)
        h = self.out_norm(h)
        return self.actor(h), self.critic(h).squeeze(-1)


@dataclass
class TradingSignal:
    action: str  # "flat", "long_BTCUSD", "short_ETHUSD", etc.
    symbol: Optional[str]
    direction: Optional[str]  # "long" or "short"
    confidence: float  # softmax probability
    value_estimate: float  # critic value


class PPOTrader:
    """Inference wrapper for trained PPO model."""

    SYMBOLS = ["BTCUSD", "ETHUSD", "SOLUSD", "LINKUSD"]

    def __init__(self, checkpoint_path: str, device: str = "cpu", long_only: bool = False):
        self.device = torch.device(device)
        self.checkpoint_path = checkpoint_path
        self.long_only = long_only

        # Load checkpoint
        ckpt = torch.load(checkpoint_path, map_location=self.device, weights_only=False)

        # Handle both formats: raw state dict or dict with "model" key
        if "model" in ckpt:
            state_dict = ckpt["model"]
            config = ckpt.get("config", {})
        else:
            state_dict = ckpt
            config = {}

        self.num_symbols = len(self.SYMBOLS)
        self.obs_size = self.num_symbols * 16 + 5 + self.num_symbols
        if long_only:
            self.num_actions = 1 + self.num_symbols  # flat + longs only
        else:
            self.num_actions = 1 + 2 * self.num_symbols  # flat + long + short

        hidden = config.get("hidden_size", 512)
        blocks = config.get("num_blocks", 3)

        self.policy = Policy(self.obs_size, self.num_actions, hidden, blocks)
        self.policy.load_state_dict(state_dict)
        self.policy.to(self.device)
        self.policy.eval()

        self.current_position = None
        self.cash = 10000.0
        self.position_qty = 0.0
        self.entry_price = 0.0
        self.hold_hours = 0
        self.step = 0
        self.max_steps = config.get("max_steps", 720)

        print(f"Loaded model from {checkpoint_path}")

    def build_observation(self, features: np.ndarray, prices: dict) -> np.ndarray:
        """
        Build observation vector from current market data.

        Args:
            features: [num_symbols, 16] feature array (from compute_hourly_features)
            prices: dict of {symbol: current_price}

        Returns:
            obs: [obs_size] observation vector
        """
        obs = np.zeros(self.obs_size, dtype=np.float32)

        # Per-symbol features
        obs[:self.num_symbols * 16] = features.flatten()

        # Portfolio state
        base = self.num_symbols * 16
        pos_val = 0.0
        if self.current_position is not None:
            sym_idx = self.current_position % self.num_symbols
            sym = self.SYMBOLS[sym_idx]
            cur_price = prices.get(sym, 0)
            is_short = self.current_position >= self.num_symbols
            if is_short:
                pos_val = -self.position_qty * cur_price
            else:
                pos_val = self.position_qty * cur_price

        obs[base + 0] = self.cash / 10000.0
        obs[base + 1] = pos_val / 10000.0
        obs[base + 2] = 0  # unrealized pnl (simplified)
        obs[base + 3] = self.hold_hours / self.max_steps
        obs[base + 4] = self.step / self.max_steps

        # One-hot position
        if self.current_position is not None:
            sym_idx = self.current_position % self.num_symbols
            is_short = self.current_position >= self.num_symbols
            obs[base + 5 + sym_idx] = -1.0 if is_short else 1.0

        return obs

    def get_signal(self, features: np.ndarray, prices: dict) -> TradingSignal:
        """
        Get trading signal from current market state.

        Args:
            features: [num_symbols, 16] feature array
            prices: dict of {symbol: current_price}

        Returns:
            TradingSignal with action recommendation
        """
        obs = self.build_observation(features, prices)
        obs_t = torch.from_numpy(obs).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits, value = self.policy(obs_t)
            probs = torch.softmax(logits, dim=-1)
            action = logits.argmax(dim=-1).item()
            confidence = probs[0, action].item()
            value_est = value.item()

        # Decode action
        if action == 0:
            return TradingSignal("flat", None, None, confidence, value_est)
        else:
            if self.long_only:
                sym_idx = action - 1
                symbol = self.SYMBOLS[sym_idx]
                return TradingSignal(f"long_{symbol}", symbol, "long", confidence, value_est)
            else:
                action_idx = action - 1
                is_short = action_idx >= self.num_symbols
                sym_idx = action_idx % self.num_symbols
                symbol = self.SYMBOLS[sym_idx]
                direction = "short" if is_short else "long"
                action_str = f"{direction}_{symbol}"
                return TradingSignal(action_str, symbol, direction, confidence, value_est)

    def update_state(self, action: int, fill_price: float, symbol: str):
        """Update internal state after trade execution."""
        if action == 0:
            self.current_position = None
            self.position_qty = 0.0
            self.entry_price = 0.0
            self.hold_hours = 0
        else:
            action_idx = action - 1
            is_short = action_idx >= self.num_symbols
            sym_idx = self.SYMBOLS.index(symbol)
            self.current_position = (self.num_symbols + sym_idx) if is_short else sym_idx
            self.entry_price = fill_price
            self.hold_hours = 0

        self.step += 1

    def reset(self):
        """Reset state for new trading session."""
        self.current_position = None
        self.cash = 10000.0
        self.position_qty = 0.0
        self.entry_price = 0.0
        self.hold_hours = 0
        self.step = 0


def compute_hourly_features(df) -> np.ndarray:
    """
    Compute 16 features from OHLCV dataframe.
    Must match pufferlib_market/export_data_hourly_price.py
    """
    import pandas as pd

    close = df["close"].values
    high = df["high"].values
    low = df["low"].values
    volume = df["volume"].values

    # Returns
    ret_1h = np.diff(close, prepend=close[0]) / np.maximum(close, 1e-8)
    ret_4h = (close - np.roll(close, 4)) / np.maximum(np.roll(close, 4), 1e-8)
    ret_24h = (close - np.roll(close, 24)) / np.maximum(np.roll(close, 24), 1e-8)

    # Moving averages
    ma_24 = pd.Series(close).rolling(24, min_periods=1).mean().values
    ma_72 = pd.Series(close).rolling(72, min_periods=1).mean().values
    ema_24 = pd.Series(close).ewm(span=24, min_periods=1).mean().values

    # Volatility
    atr_24 = pd.Series(high - low).rolling(24, min_periods=1).mean().values
    vol_ratio = volume / (pd.Series(volume).rolling(24, min_periods=1).mean().values + 1e-8)

    # Price position
    range_pos = (close - low) / (high - low + 1e-8)
    ma_ratio = close / (ma_24 + 1e-8)

    # Momentum
    rsi_proxy = ret_24h  # simplified
    macd_proxy = ema_24 - ma_72

    # Normalize
    def zscore(x, window=72):
        s = pd.Series(x)
        return ((s - s.rolling(window, min_periods=1).mean()) /
                (s.rolling(window, min_periods=1).std() + 1e-8)).values

    features = np.stack([
        zscore(ret_1h),
        zscore(ret_4h),
        zscore(ret_24h),
        zscore(ma_ratio - 1),
        zscore(range_pos - 0.5),
        zscore(vol_ratio - 1),
        zscore(atr_24 / close),
        zscore(macd_proxy / close),
        np.zeros_like(close),  # placeholder
        np.zeros_like(close),
        np.zeros_like(close),
        np.zeros_like(close),
        np.zeros_like(close),
        np.zeros_like(close),
        np.zeros_like(close),
        np.zeros_like(close),
    ], axis=-1)

    return features[-1].astype(np.float32)  # Return latest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    trader = PPOTrader(args.checkpoint, args.device)

    # Example usage
    print("\nExample signal generation:")
    dummy_features = np.random.randn(4, 16).astype(np.float32) * 0.1
    dummy_prices = {"BTCUSD": 45000, "ETHUSD": 2500, "SOLUSD": 100, "LINKUSD": 15}

    signal = trader.get_signal(dummy_features, dummy_prices)
    print(f"  Action: {signal.action}")
    print(f"  Confidence: {signal.confidence:.2%}")
    print(f"  Value estimate: {signal.value_estimate:.4f}")


if __name__ == "__main__":
    main()
