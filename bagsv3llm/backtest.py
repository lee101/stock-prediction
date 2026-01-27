#!/usr/bin/env python3
"""Backtesting script for BagsV3LLM.

Compares V3 transformer model against V1, V2, and buy-and-hold.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Optional, Any, Tuple

import numpy as np
import pandas as pd
import torch

from bagsv3llm.model import BagsV3Config, BagsV3Transformer
from bagsv3llm.dataset import (
    load_ohlc_dataframe,
    build_bar_features,
    build_aggregate_features,
    build_chronos_features,
    FeatureNormalizerV3,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


@dataclass
class Trade:
    """Single trade record."""
    entry_time: pd.Timestamp
    exit_time: pd.Timestamp
    entry_price: float
    exit_price: float
    position_size: float  # As fraction of capital
    pnl_pct: float
    signal_prob: float


@dataclass
class BacktestResult:
    """Backtest results."""
    model_name: str
    total_return: float
    num_trades: int
    win_rate: float
    avg_trade_return: float
    max_drawdown: float
    sharpe_ratio: float
    trades: List[Trade]
    equity_curve: np.ndarray


class V3Backtester:
    """Backtester for BagsV3LLM transformer model."""

    def __init__(
        self,
        model: BagsV3Transformer,
        config: BagsV3Config,
        normalizers: Optional[Dict[str, FeatureNormalizerV3]] = None,
        buy_threshold: float = 0.5,
        sell_threshold: float = 0.45,
        cost_bps: float = 130.0,
        max_position: float = 1.0,
        device: str = "cuda",
        chronos_wrapper=None,
    ):
        self.model = model
        self.config = config
        self.normalizers = normalizers or {}
        self.buy_threshold = buy_threshold
        self.sell_threshold = sell_threshold
        self.cost_bps = cost_bps
        self.cost_frac = cost_bps / 10000.0
        self.max_position = max_position
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()
        self.chronos_wrapper = chronos_wrapper

    @torch.no_grad()
    def predict(
        self,
        bar_features: np.ndarray,
        chronos_features: np.ndarray,
        agg_features: np.ndarray,
    ) -> Tuple[float, float]:
        """Get signal probability and size from model."""
        # Normalize if normalizers available
        if "bar" in self.normalizers:
            bar_flat = bar_features.reshape(-1, bar_features.shape[-1])
            bar_flat = self.normalizers["bar"].transform(bar_flat)
            bar_features = bar_flat.reshape(bar_features.shape)

        if "chronos" in self.normalizers:
            chronos_flat = chronos_features.reshape(-1, chronos_features.shape[-1])
            chronos_flat = self.normalizers["chronos"].transform(chronos_flat)
            chronos_features = chronos_flat.reshape(chronos_features.shape)

        if "agg" in self.normalizers:
            agg_features = self.normalizers["agg"].transform(agg_features)

        # Convert to tensors
        bar_t = torch.tensor(bar_features, dtype=torch.float32).unsqueeze(0).to(self.device)
        chronos_t = torch.tensor(chronos_features, dtype=torch.float32).unsqueeze(0).to(self.device)
        agg_t = torch.tensor(agg_features, dtype=torch.float32).unsqueeze(0).to(self.device)

        # Forward pass
        signal_logit, size_logit = self.model(bar_t, chronos_t, agg_t)

        prob = torch.sigmoid(signal_logit).item()
        size = torch.sigmoid(size_logit).item()

        return prob, size

    def _get_chronos_features(
        self,
        opens: np.ndarray,
        highs: np.ndarray,
        lows: np.ndarray,
        closes: np.ndarray,
        timestamps: np.ndarray,
        context_length: int,
    ) -> np.ndarray:
        """Get Chronos2 forecast features for a window."""
        try:
            # Build context dataframe for Chronos
            context_df = pd.DataFrame({
                "timestamp": timestamps,
                "open": opens,
                "high": highs,
                "low": lows,
                "close": closes,
            })

            batch = self.chronos_wrapper.predict_ohlc(
                context_df,
                symbol="CODEX",
                prediction_length=3,
                context_length=min(512, len(context_df)),
            )

            # Build features for each position in context
            chronos_features = []
            for i in range(context_length):
                current_price = closes[i]
                features = build_chronos_features(
                    predicted_prices=batch["close_median"].numpy(),
                    predicted_p10=batch["close_p10"].numpy(),
                    predicted_p90=batch["close_p90"].numpy(),
                    current_price=current_price,
                    num_horizons=3,
                )
                chronos_features.append(features)

            return np.array(chronos_features, dtype=np.float32)
        except Exception as e:
            logger.warning(f"Chronos2 feature extraction failed: {e}")
            return np.zeros((context_length, 12), dtype=np.float32)

    def run(
        self,
        df: pd.DataFrame,
        start_idx: int,
        end_idx: Optional[int] = None,
    ) -> BacktestResult:
        """Run backtest on OHLC data.

        Args:
            df: OHLC dataframe
            start_idx: Start index (must be >= context_length)
            end_idx: End index (defaults to end of data)
        """
        if end_idx is None:
            end_idx = len(df)

        opens = df["open"].to_numpy(dtype=np.float32)
        highs = df["high"].to_numpy(dtype=np.float32)
        lows = df["low"].to_numpy(dtype=np.float32)
        closes = df["close"].to_numpy(dtype=np.float32)
        timestamps = df["timestamp"].to_numpy()

        trades = []
        equity = [1.0]
        position = 0.0  # Current position (0 = no position, 1 = full)
        entry_price = 0.0
        entry_time = None
        entry_prob = 0.0
        entry_equity = 1.0  # Equity at time of entry

        context_length = self.config.context_length

        for idx in range(start_idx, end_idx):
            if idx < context_length:
                equity.append(equity[-1])
                continue

            start = idx - context_length
            current_price = closes[idx]

            # Build features
            bar_features = build_bar_features(
                opens[start:idx],
                highs[start:idx],
                lows[start:idx],
                closes[start:idx],
            )

            agg_features = build_aggregate_features(
                opens[start:idx],
                highs[start:idx],
                lows[start:idx],
                closes[start:idx],
            )

            # Chronos features
            if self.chronos_wrapper is not None:
                chronos_features = self._get_chronos_features(
                    opens[start:idx], highs[start:idx], lows[start:idx], closes[start:idx],
                    timestamps[start:idx], context_length
                )
            else:
                chronos_features = np.zeros((context_length, 12), dtype=np.float32)

            # Get prediction
            prob, size = self.predict(bar_features, chronos_features, agg_features)

            # Trading logic
            if position == 0:
                # No position - check for buy signal
                if prob >= self.buy_threshold:
                    position = min(size, self.max_position)
                    entry_price = current_price * (1 + self.cost_frac)
                    entry_time = timestamps[idx]
                    entry_prob = prob
                    entry_equity = equity[-1]  # Save equity at entry
            else:
                # Have position - check for sell signal
                if prob < self.sell_threshold:
                    exit_price = current_price * (1 - self.cost_frac)
                    pnl_pct = (exit_price / entry_price - 1) * position

                    trades.append(Trade(
                        entry_time=pd.Timestamp(entry_time),
                        exit_time=pd.Timestamp(timestamps[idx]),
                        entry_price=entry_price,
                        exit_price=exit_price,
                        position_size=position,
                        pnl_pct=pnl_pct,
                        signal_prob=entry_prob,
                    ))

                    equity.append(entry_equity * (1 + pnl_pct))
                    position = 0.0
                    entry_price = 0.0
                    continue

            # Update equity for unrealized PnL
            if position > 0:
                unrealized = (current_price / entry_price - 1) * position
                equity.append(entry_equity * (1 + unrealized))
            else:
                equity.append(equity[-1])

        # Close any open position at end
        if position > 0:
            exit_price = closes[end_idx - 1] * (1 - self.cost_frac)
            pnl_pct = (exit_price / entry_price - 1) * position
            trades.append(Trade(
                entry_time=pd.Timestamp(entry_time),
                exit_time=pd.Timestamp(timestamps[end_idx - 1]),
                entry_price=entry_price,
                exit_price=exit_price,
                position_size=position,
                pnl_pct=pnl_pct,
                signal_prob=entry_prob,
            ))
            equity[-1] = equity[-2] * (1 + pnl_pct)

        equity_arr = np.array(equity)

        # Calculate metrics
        total_return = equity_arr[-1] / equity_arr[0] - 1
        num_trades = len(trades)

        if num_trades > 0:
            win_rate = sum(1 for t in trades if t.pnl_pct > 0) / num_trades
            avg_trade_return = np.mean([t.pnl_pct for t in trades])
        else:
            win_rate = 0.0
            avg_trade_return = 0.0

        # Max drawdown
        peak = equity_arr[0]
        max_dd = 0.0
        for val in equity_arr:
            if val > peak:
                peak = val
            dd = (peak - val) / peak
            if dd > max_dd:
                max_dd = dd

        # Sharpe ratio (simplified)
        returns = np.diff(equity_arr) / equity_arr[:-1]
        if len(returns) > 0 and np.std(returns) > 0:
            sharpe = np.mean(returns) / np.std(returns) * np.sqrt(252)  # Annualized
        else:
            sharpe = 0.0

        return BacktestResult(
            model_name="BagsV3LLM",
            total_return=total_return,
            num_trades=num_trades,
            win_rate=win_rate,
            avg_trade_return=avg_trade_return,
            max_drawdown=max_dd,
            sharpe_ratio=sharpe,
            trades=trades,
            equity_curve=equity_arr,
        )


def buy_and_hold_backtest(
    df: pd.DataFrame,
    start_idx: int,
    end_idx: Optional[int] = None,
    cost_bps: float = 130.0,
) -> BacktestResult:
    """Simple buy and hold benchmark."""
    if end_idx is None:
        end_idx = len(df)

    closes = df["close"].to_numpy(dtype=np.float32)
    timestamps = df["timestamp"].to_numpy()

    cost_frac = cost_bps / 10000.0
    entry_price = closes[start_idx] * (1 + cost_frac)
    exit_price = closes[end_idx - 1] * (1 - cost_frac)
    total_return = exit_price / entry_price - 1

    # Build equity curve
    equity = closes[start_idx:end_idx] / closes[start_idx]

    # Max drawdown
    peak = equity[0]
    max_dd = 0.0
    for val in equity:
        if val > peak:
            peak = val
        dd = (peak - val) / peak
        if dd > max_dd:
            max_dd = dd

    return BacktestResult(
        model_name="Buy & Hold",
        total_return=total_return,
        num_trades=1,
        win_rate=1.0 if total_return > 0 else 0.0,
        avg_trade_return=total_return,
        max_drawdown=max_dd,
        sharpe_ratio=0.0,  # Not meaningful for B&H
        trades=[],
        equity_curve=equity,
    )


def load_v3_model(
    checkpoint_path: Path,
    device: str = "cuda",
) -> Tuple[BagsV3Transformer, BagsV3Config, Dict[str, FeatureNormalizerV3]]:
    """Load V3 model from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    config_dict = checkpoint.get("config", {})
    config = BagsV3Config(
        context_length=config_dict.get("context_length", 256),
        n_layer=config_dict.get("n_layer", 6),
        n_head=config_dict.get("n_head", 8),
        n_embd=config_dict.get("n_embd", 128),
        dropout=config_dict.get("dropout", 0.1),
    )

    model = BagsV3Transformer(config)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    # Load normalizers
    normalizers = {}
    if "normalizers" in checkpoint:
        for key, norm_dict in checkpoint["normalizers"].items():
            normalizers[key] = FeatureNormalizerV3.from_dict(norm_dict)

    return model, config, normalizers


def compare_models(
    ohlc_path: Path,
    mint: str,
    v3_checkpoint: Optional[Path] = None,
    v2_checkpoint: Optional[Path] = None,
    v1_checkpoint: Optional[Path] = None,
    test_split: float = 0.2,
    buy_threshold: float = 0.5,
    sell_threshold: float = 0.45,
    cost_bps: float = 130.0,
    device: str = "cuda",
    chronos_wrapper=None,
) -> Dict[str, BacktestResult]:
    """Compare multiple model versions."""
    # Load data
    df = load_ohlc_dataframe(ohlc_path, mint)
    logger.info(f"Loaded {len(df)} bars for {mint[:8]}...")

    # Test on last portion
    test_start = int(len(df) * (1 - test_split))
    logger.info(f"Testing on bars {test_start} to {len(df)} ({len(df) - test_start} bars)")

    results = {}

    # V3 backtest
    if v3_checkpoint and v3_checkpoint.exists():
        logger.info(f"Loading V3 model from {v3_checkpoint}")
        model, config, normalizers = load_v3_model(v3_checkpoint, device)

        backtester = V3Backtester(
            model=model,
            config=config,
            normalizers=normalizers,
            buy_threshold=buy_threshold,
            sell_threshold=sell_threshold,
            cost_bps=cost_bps,
            device=device,
            chronos_wrapper=chronos_wrapper,
        )

        # Ensure start_idx is at least context_length
        start_idx = max(test_start, config.context_length)
        v3_result = backtester.run(df, start_idx)
        results["V3 Transformer"] = v3_result

    # V2 backtest (if available)
    if v2_checkpoint and v2_checkpoint.exists():
        try:
            from bagsv2.backtest import SimpleBacktester, load_v2_model
            logger.info(f"Loading V2 model from {v2_checkpoint}")
            v2_model, v2_normalizer, v2_config = load_v2_model(v2_checkpoint, device)
            v2_backtester = SimpleBacktester(
                model=v2_model,
                normalizer=v2_normalizer,
                context_bars=v2_config.get("context_bars", 32),
                buy_threshold=buy_threshold,
                sell_threshold=sell_threshold,
                cost_bps=cost_bps,
                device=device,
            )
            v2_result = v2_backtester.run(df, test_start)
            results["V2 LSTM"] = v2_result
        except Exception as e:
            logger.warning(f"Could not load V2 model: {e}")

    # V1 backtest (if available)
    if v1_checkpoint and v1_checkpoint.exists():
        try:
            from bagsfm.bags_model import BagsNeuralModel, FeatureNormalizer
            logger.info(f"Loading V1 model from {v1_checkpoint}")
            v1_ckpt = torch.load(v1_checkpoint, map_location="cpu", weights_only=False)
            # ... V1 backtest setup
        except Exception as e:
            logger.warning(f"Could not load V1 model: {e}")

    # Buy and hold baseline
    bh_result = buy_and_hold_backtest(df, test_start, cost_bps=cost_bps)
    results["Buy & Hold"] = bh_result

    return results


def print_results(results: Dict[str, BacktestResult]):
    """Print comparison table."""
    print("\n" + "=" * 80)
    print("BACKTEST RESULTS COMPARISON")
    print("=" * 80)

    headers = ["Model", "Return", "Trades", "Win Rate", "Avg Trade", "Max DD", "Sharpe"]
    widths = [20, 10, 8, 10, 10, 10, 10]

    # Header
    header_str = "".join(h.ljust(w) for h, w in zip(headers, widths))
    print(header_str)
    print("-" * 80)

    # Results
    for name, result in sorted(results.items(), key=lambda x: -x[1].total_return):
        row = [
            name[:18],
            f"{result.total_return*100:+.2f}%",
            str(result.num_trades),
            f"{result.win_rate*100:.1f}%",
            f"{result.avg_trade_return*100:+.2f}%",
            f"{result.max_drawdown*100:.1f}%",
            f"{result.sharpe_ratio:.2f}",
        ]
        row_str = "".join(str(r).ljust(w) for r, w in zip(row, widths))
        print(row_str)

    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description="Backtest BagsV3LLM")
    parser.add_argument("--ohlc", type=Path, default=Path("bagstraining/ohlc_data.csv"))
    parser.add_argument("--mint", type=str, required=True)
    parser.add_argument("--v3-checkpoint", type=Path, default=None)
    parser.add_argument("--v2-checkpoint", type=Path, default=None)
    parser.add_argument("--v1-checkpoint", type=Path, default=None)
    parser.add_argument("--test-split", type=float, default=0.2)
    parser.add_argument("--buy-threshold", type=float, default=0.5)
    parser.add_argument("--sell-threshold", type=float, default=0.45)
    parser.add_argument("--cost-bps", type=float, default=130.0)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--use-chronos", action="store_true", help="Use Chronos2 features")

    args = parser.parse_args()

    # Load Chronos2 if requested
    chronos_wrapper = None
    if args.use_chronos:
        try:
            from src.models.chronos2_wrapper import Chronos2OHLCWrapper
            logger.info("Loading Chronos2 for forecast features...")
            chronos_wrapper = Chronos2OHLCWrapper.from_pretrained(
                device_map=args.device if torch.cuda.is_available() else "cpu",
                default_context_length=512,
            )
            logger.info("Chronos2 loaded successfully")
        except Exception as e:
            logger.warning(f"Could not load Chronos2: {e}")

    results = compare_models(
        ohlc_path=args.ohlc,
        mint=args.mint,
        v3_checkpoint=args.v3_checkpoint,
        v2_checkpoint=args.v2_checkpoint,
        v1_checkpoint=args.v1_checkpoint,
        test_split=args.test_split,
        buy_threshold=args.buy_threshold,
        sell_threshold=args.sell_threshold,
        cost_bps=args.cost_bps,
        device=args.device,
        chronos_wrapper=chronos_wrapper,
    )

    print_results(results)


if __name__ == "__main__":
    main()
