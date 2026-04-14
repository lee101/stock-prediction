#!/usr/bin/env python3
"""
Daily RL inference for the trade_pen_05 model.
Loads checkpoint and generates daily trading signals from daily OHLCV bars.

Features must match pufferlib_market/export_data_daily.py (16 per symbol).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Optional, Sequence

from pufferlib_market.inference import PPOTrader, TradingSignal


def compute_daily_features(df: pd.DataFrame) -> np.ndarray:
    """
    Compute 16 daily features from OHLCV dataframe.
    Must match pufferlib_market/export_data_daily.py compute_daily_features().

    Args:
        df: DataFrame with columns [open, high, low, close, volume], indexed by date.
            Must have enough history (60+ rows) for rolling features.

    Returns:
        features: [16] float32 array for the latest bar
    """
    close = df["close"].astype(float)
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    volume = df["volume"].astype(float)

    ret_1d_raw = close.pct_change(1).fillna(0.0)  # unclipped — matches export_data_daily volatility calc
    ret_1d = ret_1d_raw.clip(-0.5, 0.5)
    ret_5d = close.pct_change(5).fillna(0.0).clip(-1.0, 1.0)
    ret_20d = close.pct_change(20).fillna(0.0).clip(-2.0, 2.0)

    vol_5d = ret_1d_raw.rolling(5, min_periods=1).std(ddof=0).fillna(0.01).clip(0.0, 1.0)
    vol_20d = ret_1d_raw.rolling(20, min_periods=1).std(ddof=0).fillna(0.01).clip(0.0, 1.0)

    ma5 = close.rolling(5, min_periods=1).mean()
    ma20 = close.rolling(20, min_periods=1).mean()
    ma60 = close.rolling(60, min_periods=1).mean()
    ma_delta_5d = ((close - ma5) / ma5.clip(lower=1e-8)).fillna(0.0).clip(-0.5, 0.5)
    ma_delta_20d = ((close - ma20) / ma20.clip(lower=1e-8)).fillna(0.0).clip(-0.5, 0.5)
    ma_delta_60d = ((close - ma60) / ma60.clip(lower=1e-8)).fillna(0.0).clip(-0.5, 0.5)

    tr = pd.concat(
        [high - low, (high - close.shift(1)).abs(), (low - close.shift(1)).abs()],
        axis=1,
    ).max(axis=1)
    atr14 = tr.rolling(14, min_periods=1).mean()
    atr_pct_14d = (atr14 / close.clip(lower=1e-8)).fillna(0.0).clip(0.0, 0.5)
    range_pct_1d = ((high - low) / close.clip(lower=1e-8)).fillna(0.0).clip(0.0, 0.5)

    # RSI(14) normalized to [-1, 1] — replaces duplicate trend_20d (was identical to ret_20d)
    delta = close.diff()
    gain = delta.clip(lower=0.0).rolling(14, min_periods=1).mean()
    loss = (-delta.clip(upper=0.0)).rolling(14, min_periods=1).mean()
    rs = gain / loss.clip(lower=1e-8)
    rsi_14 = (2.0 * (100.0 - 100.0 / (1.0 + rs)) / 100.0 - 1.0).fillna(0.0).clip(-1.0, 1.0)
    trend_60d = close.pct_change(60).fillna(0.0).clip(-3.0, 3.0)

    roll_max_20 = close.rolling(20, min_periods=1).max()
    roll_max_60 = close.rolling(60, min_periods=1).max()
    drawdown_20d = ((close - roll_max_20) / roll_max_20.clip(lower=1e-8)).fillna(0.0).clip(-1.0, 0.0)
    drawdown_60d = ((close - roll_max_60) / roll_max_60.clip(lower=1e-8)).fillna(0.0).clip(-1.0, 0.0)

    log_vol = np.log1p(volume.clip(lower=0.0))
    log_vol_mean20 = log_vol.rolling(20, min_periods=1).mean()
    log_vol_std20 = log_vol.rolling(20, min_periods=1).std(ddof=0).replace(0.0, 1.0)
    log_volume_z20d = ((log_vol - log_vol_mean20) / log_vol_std20).fillna(0.0).clip(-5.0, 5.0)
    log_volume_delta_5d = (log_vol - log_vol.rolling(5, min_periods=1).mean()).fillna(0.0).clip(-10.0, 10.0)

    features = np.array([
        float(ret_1d.iloc[-1]),
        float(ret_5d.iloc[-1]),
        float(ret_20d.iloc[-1]),
        float(vol_5d.iloc[-1]),
        float(vol_20d.iloc[-1]),
        float(ma_delta_5d.iloc[-1]),
        float(ma_delta_20d.iloc[-1]),
        float(ma_delta_60d.iloc[-1]),
        float(atr_pct_14d.iloc[-1]),
        float(range_pct_1d.iloc[-1]),
        float(rsi_14.iloc[-1]),
        float(trend_60d.iloc[-1]),
        float(drawdown_20d.iloc[-1]),
        float(drawdown_60d.iloc[-1]),
        float(log_volume_z20d.iloc[-1]),
        float(log_volume_delta_5d.iloc[-1]),
    ], dtype=np.float32)

    return features


class DailyPPOTrader(PPOTrader):
    """Daily trading variant of PPOTrader."""

    def __init__(
        self,
        checkpoint_path: str,
        device: str = "cpu",
        long_only: bool = False,
        symbols: Optional[Sequence[str]] = None,
        allow_unsafe_checkpoint_loading: bool = False,
    ):
        if symbols is None:
            symbols = ["BTCUSD", "ETHUSD", "SOLUSD", "LTCUSD", "AVAXUSD"]
        super().__init__(
            checkpoint_path,
            device,
            long_only,
            symbols,
            allow_unsafe_checkpoint_loading=allow_unsafe_checkpoint_loading,
        )
        # Use max_steps from the checkpoint if available (new checkpoints save
        # the training max_steps via action_meta so obs[base+3] is correctly
        # normalised).  Fall back to 90 for legacy checkpoints that don't have it
        # (the historical default for this class — training typically used 252 but
        # we kept 90 to avoid changing val/prod parity on existing models).
        if self.max_steps == 720:
            # PPOTrader used the hourly default (720) because max_steps was absent
            # from the checkpoint.  Apply the daily legacy fallback of 90.
            self.max_steps = 90
        self.hold_days = 0

    def get_daily_signal(
        self,
        daily_dfs: dict[str, pd.DataFrame],
        prices: dict[str, float],
    ) -> TradingSignal:
        """
        Get trading signal from daily OHLCV data.

        Args:
            daily_dfs: {symbol: DataFrame with [open,high,low,close,volume]}
                       Must have 60+ rows of history.
            prices: {symbol: current_price}

        Returns:
            TradingSignal
        """
        fps = self.features_per_sym  # 16 for standard, 20 for cross-features models
        features = np.zeros((self.num_symbols, fps), dtype=np.float32)
        for i, sym in enumerate(self.SYMBOLS):
            if sym in daily_dfs:
                features[i, :16] = compute_daily_features(daily_dfs[sym])

        if fps > 16:
            # Append cross-symbol features (corr, beta, relative_return, breadth_rank).
            # These require prices from ALL symbols simultaneously.
            from pufferlib_market.cross_symbol_features import compute_cross_features
            # Build aligned close-price matrix [T, S] — use longest common length
            max_len = max((len(daily_dfs[s]) for s in self.SYMBOLS if s in daily_dfs), default=0)
            close_mat = np.zeros((max_len, self.num_symbols), dtype=np.float64)
            for i, sym in enumerate(self.SYMBOLS):
                if sym in daily_dfs:
                    col = daily_dfs[sym]["close"].astype(float).values[-max_len:]
                    close_mat[-len(col):, i] = col
            cross = compute_cross_features(close_mat, self.SYMBOLS, window=20, anchor_symbol="SPY")
            # cross[-1] is the cross-feature vector for the latest bar: [S, 4]
            features[:, 16:fps] = cross[-1]

        return self.get_signal(features, prices)

    def update_state(self, action: int, fill_price: float, symbol: str, qty: float = 0.0):
        """Override to maintain hold_days in sync with C env hold_hours semantics.

        C env: open_long() resets hold_hours=0; the first obs WHILE HOLDING
        (next step) also sees hold_hours=0 because build_observation() runs
        BEFORE any HOLD action increments it.

        We match this by setting hold_days=-1 on a new open.  step_day() then
        increments to 0, so the next obs correctly sees hold_hours=0.
        On a close (action=0) we reset hold_days=0 immediately so that obs
        after closing sees 0 (no stale value from the previous position).
        """
        super().update_state(action, fill_price, symbol, qty=qty)
        if action == 0:
            self.hold_days = 0
        else:
            # -1 so step_day() increments to 0; first obs while holding → 0
            self.hold_days = -1

    def update_state_daily(self, action: int, fill_price: float, symbol: str, qty: float = 0.0):
        """Update internal state after daily trade execution."""
        self.update_state(action, fill_price, symbol, qty=qty)

    def step_day(self):
        """Advance one day."""
        self.step += 1
        if self.current_position is not None:
            self.hold_days += 1
        else:
            # Flat: clear any stale value left from a previous position.
            self.hold_days = 0
        # Update hold_hours to hold_days for obs
        self.hold_hours = self.hold_days


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Daily RL inference demo")
    parser.add_argument("--checkpoint", type=str,
                        default="pufferlib_market/checkpoints/autoresearch_daily/trade_pen_05/best.pt")
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    trader = DailyPPOTrader(args.checkpoint, args.device)

    # Demo with random data
    print("\nDemo daily signal:")
    dummy_features = np.random.randn(5, 16).astype(np.float32) * 0.1
    dummy_prices = {
        "BTCUSD": 85000, "ETHUSD": 2000, "SOLUSD": 130,
        "LTCUSD": 90, "AVAXUSD": 25,
    }
    signal = trader.get_signal(dummy_features, dummy_prices)
    print(f"  Action: {signal.action}")
    print(f"  Confidence: {signal.confidence:.2%}")
    print(f"  Value estimate: {signal.value_estimate:.4f}")


if __name__ == "__main__":
    main()
