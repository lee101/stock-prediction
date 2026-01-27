#!/usr/bin/env python3
"""Backtest v2 model and compare with v1."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Optional, Dict, Any
from dataclasses import dataclass

import numpy as np
import torch

from bagsv2.dataset import load_ohlc_dataframe, build_window_features_v2, FeatureNormalizerV2
from bagsv2.model import BagsNeuralModelV2, BagsNeuralModelV2Simple

# Try to import v1 for comparison
try:
    from bagsneural.dataset import build_window_features, FeatureNormalizer
    from bagsneural.model import BagsNeuralModel
    HAS_V1 = True
except ImportError:
    HAS_V1 = False

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


@dataclass
class BacktestResult:
    """Results from backtesting a model."""
    total_return_pct: float
    sharpe_ratio: float
    max_drawdown: float
    total_trades: int
    win_rate: float
    profit_factor: float
    avg_trade_return: float


class SimpleBacktester:
    """Simple backtester for neural trading models."""

    def __init__(
        self,
        initial_sol: float = 100.0,
        cost_bps: float = 130.0,
        max_position_sol: float = 10.0,
    ):
        self.initial_sol = initial_sol
        self.cost_bps = cost_bps
        self.cost_pct = cost_bps / 10000.0
        self.max_position_sol = max_position_sol

        # State
        self.sol_balance = initial_sol
        self.token_balance = 0.0
        self.holding = False
        self.entry_price = 0.0

        # History
        self.equity_history = [initial_sol]
        self.trades = []

    def reset(self):
        self.sol_balance = self.initial_sol
        self.token_balance = 0.0
        self.holding = False
        self.entry_price = 0.0
        self.equity_history = [self.initial_sol]
        self.trades = []

    def get_equity(self, price: float) -> float:
        return self.sol_balance + self.token_balance * price

    def buy(self, price: float, size_fraction: float = 1.0) -> bool:
        if self.holding:
            return False

        # Calculate trade size
        trade_sol = min(self.sol_balance * 0.95, self.max_position_sol) * size_fraction
        if trade_sol < 0.01:
            return False

        # Apply cost
        cost = trade_sol * self.cost_pct
        effective_sol = trade_sol - cost

        # Execute
        tokens = effective_sol / price
        self.sol_balance -= trade_sol
        self.token_balance = tokens
        self.holding = True
        self.entry_price = price

        self.trades.append({
            "type": "buy",
            "price": price,
            "sol": trade_sol,
            "tokens": tokens,
        })

        return True

    def sell(self, price: float) -> Optional[float]:
        if not self.holding:
            return None

        # Calculate return
        gross_sol = self.token_balance * price
        cost = gross_sol * self.cost_pct
        net_sol = gross_sol - cost

        trade_return = (price - self.entry_price) / self.entry_price

        self.sol_balance += net_sol
        self.token_balance = 0.0
        self.holding = False

        self.trades.append({
            "type": "sell",
            "price": price,
            "sol": net_sol,
            "return": trade_return,
        })

        self.entry_price = 0.0
        return trade_return

    def step(self, price: float):
        """Record equity at each step."""
        equity = self.get_equity(price)
        self.equity_history.append(equity)

    def compute_results(self) -> BacktestResult:
        equity = np.array(self.equity_history)

        # Total return
        total_return = (equity[-1] - equity[0]) / equity[0] * 100

        # Sharpe ratio (simplified, daily)
        returns = np.diff(equity) / equity[:-1]
        if len(returns) > 1 and returns.std() > 0:
            sharpe = returns.mean() / returns.std() * np.sqrt(252 * 6)  # Annualized, assuming 6 bars/day
        else:
            sharpe = 0.0

        # Max drawdown
        peak = np.maximum.accumulate(equity)
        drawdown = (peak - equity) / peak
        max_dd = drawdown.max()

        # Trade stats
        sell_trades = [t for t in self.trades if t["type"] == "sell"]
        if sell_trades:
            returns_list = [t["return"] for t in sell_trades]
            wins = sum(1 for r in returns_list if r > 0)
            win_rate = wins / len(sell_trades)
            avg_return = np.mean(returns_list)

            gains = sum(r for r in returns_list if r > 0)
            losses = abs(sum(r for r in returns_list if r < 0))
            profit_factor = gains / losses if losses > 0 else float("inf")
        else:
            win_rate = 0.0
            avg_return = 0.0
            profit_factor = 0.0

        return BacktestResult(
            total_return_pct=total_return,
            sharpe_ratio=sharpe,
            max_drawdown=max_dd,
            total_trades=len(sell_trades),
            win_rate=win_rate,
            profit_factor=profit_factor,
            avg_trade_return=avg_return,
        )


def load_v2_model(checkpoint_path: Path, device: torch.device):
    """Load v2 model from checkpoint."""
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = payload["config"]
    normalizer = FeatureNormalizerV2.from_dict(payload["normalizer"])

    model_type = config.get("model_type", "lstm")
    context_bars = config["context_bars"]
    features_per_bar = config.get("features_per_bar", 5)

    if model_type == "lstm":
        model = BagsNeuralModelV2(
            features_per_bar=features_per_bar,
            context_bars=context_bars,
            lstm_hidden=config.get("lstm_hidden", 64),
            lstm_layers=config.get("lstm_layers", 2),
            fc_hidden=config.get("fc_hidden", 64),
            dropout=config.get("dropout", 0.2),
        )
    else:
        model = BagsNeuralModelV2Simple(
            input_dim=context_bars * features_per_bar + 7,
            hidden_dims=tuple(config.get("mlp_hidden", [128, 64, 32])),
            dropout=config.get("dropout", 0.2),
        )

    model.load_state_dict(payload["model_state"])
    model.to(device)
    model.eval()

    return model, normalizer, config


def load_v1_model(checkpoint_path: Path, device: torch.device):
    """Load v1 model from checkpoint."""
    if not HAS_V1:
        raise ImportError("v1 model not available")

    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = payload["config"]
    normalizer = FeatureNormalizer.from_dict(payload["normalizer"])

    context_bars = config.get("context", 16)
    input_dim = context_bars * 3  # v1 has 3 features per bar

    model = BagsNeuralModel(
        input_dim=input_dim,
        hidden_dims=config.get("hidden", [32, 16]),
        dropout=config.get("dropout", 0.1),
    )
    model.load_state_dict(payload["model_state"])
    model.to(device)
    model.eval()

    return model, normalizer, config


def backtest_v2_model(
    model,
    normalizer: FeatureNormalizerV2,
    config: dict,
    df,
    buy_threshold: float = 0.5,
    sell_threshold: float = 0.4,
    initial_sol: float = 100.0,
    cost_bps: float = 130.0,
    test_start_pct: float = 0.8,  # Use last 20% for testing
    device: torch.device = None,
) -> BacktestResult:
    """Backtest a v2 model."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    context_bars = config["context_bars"]

    opens = df["open"].to_numpy(dtype=np.float32)
    highs = df["high"].to_numpy(dtype=np.float32)
    lows = df["low"].to_numpy(dtype=np.float32)
    closes = df["close"].to_numpy(dtype=np.float32)

    # Start from test period
    start_idx = int(len(df) * test_start_pct)
    start_idx = max(start_idx, context_bars)

    backtester = SimpleBacktester(
        initial_sol=initial_sol,
        cost_bps=cost_bps,
    )

    for idx in range(start_idx, len(df)):
        window_start = idx - context_bars

        # Build features
        features = build_window_features_v2(
            opens[window_start:idx],
            highs[window_start:idx],
            lows[window_start:idx],
            closes[window_start:idx],
        )

        normalized = normalizer.transform(features)[None, :]
        x = torch.tensor(normalized, dtype=torch.float32, device=device)

        with torch.no_grad():
            signal_logit, size_logit = model(x)
            prob = torch.sigmoid(signal_logit).item()
            size = torch.sigmoid(size_logit).item()

        current_price = float(closes[idx])

        # Trading logic
        if not backtester.holding and prob >= buy_threshold:
            backtester.buy(current_price, size)
        elif backtester.holding and prob <= sell_threshold:
            backtester.sell(current_price)

        backtester.step(current_price)

    # Close any open position
    if backtester.holding:
        backtester.sell(float(closes[-1]))

    return backtester.compute_results()


def backtest_v1_model(
    model,
    normalizer,
    config: dict,
    df,
    buy_threshold: float = 0.5,
    sell_threshold: float = 0.4,
    initial_sol: float = 100.0,
    cost_bps: float = 130.0,
    test_start_pct: float = 0.8,
    device: torch.device = None,
) -> BacktestResult:
    """Backtest a v1 model."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    context_bars = config.get("context", 16)

    opens = df["open"].to_numpy(dtype=np.float32)
    highs = df["high"].to_numpy(dtype=np.float32)
    lows = df["low"].to_numpy(dtype=np.float32)
    closes = df["close"].to_numpy(dtype=np.float32)

    start_idx = int(len(df) * test_start_pct)
    start_idx = max(start_idx, context_bars)

    backtester = SimpleBacktester(
        initial_sol=initial_sol,
        cost_bps=cost_bps,
    )

    for idx in range(start_idx, len(df)):
        window_start = idx - context_bars

        features = build_window_features(
            opens[window_start:idx],
            highs[window_start:idx],
            lows[window_start:idx],
            closes[window_start:idx],
        )

        normalized = normalizer.transform(features)[None, :]
        x = torch.tensor(normalized, dtype=torch.float32, device=device)

        with torch.no_grad():
            signal_logit, size_logit = model(x)
            prob = torch.sigmoid(signal_logit).item()
            size = torch.sigmoid(size_logit).item()

        current_price = float(closes[idx])

        if not backtester.holding and prob >= buy_threshold:
            backtester.buy(current_price, size)
        elif backtester.holding and prob <= sell_threshold:
            backtester.sell(current_price)

        backtester.step(current_price)

    if backtester.holding:
        backtester.sell(float(closes[-1]))

    return backtester.compute_results()


def compare_models(
    v1_checkpoint: Optional[Path],
    v2_checkpoint: Path,
    ohlc_path: Path,
    mint: str,
    buy_threshold: float = 0.5,
    sell_threshold: float = 0.4,
    test_start_pct: float = 0.8,
    device: str = "cuda",
) -> Dict[str, BacktestResult]:
    """Compare v1 and v2 models on same test data."""
    device = torch.device(device if torch.cuda.is_available() else "cpu")

    df = load_ohlc_dataframe(ohlc_path, mint)
    logger.info(f"Loaded {len(df)} bars, testing on last {(1-test_start_pct)*100:.0f}%")

    results = {}

    # V2 model
    logger.info("Backtesting v2 model...")
    model_v2, normalizer_v2, config_v2 = load_v2_model(v2_checkpoint, device)
    results["v2"] = backtest_v2_model(
        model_v2, normalizer_v2, config_v2, df,
        buy_threshold=buy_threshold,
        sell_threshold=sell_threshold,
        test_start_pct=test_start_pct,
        device=device,
    )

    # V1 model (if available)
    if v1_checkpoint and HAS_V1:
        logger.info("Backtesting v1 model...")
        model_v1, normalizer_v1, config_v1 = load_v1_model(v1_checkpoint, device)
        results["v1"] = backtest_v1_model(
            model_v1, normalizer_v1, config_v1, df,
            buy_threshold=buy_threshold,
            sell_threshold=sell_threshold,
            test_start_pct=test_start_pct,
            device=device,
        )

    # Buy and hold baseline
    logger.info("Computing buy-and-hold baseline...")
    start_idx = int(len(df) * test_start_pct)
    start_price = df.iloc[start_idx]["close"]
    end_price = df.iloc[-1]["close"]
    bh_return = (end_price - start_price) / start_price * 100
    results["buy_hold"] = BacktestResult(
        total_return_pct=bh_return,
        sharpe_ratio=0.0,
        max_drawdown=0.0,
        total_trades=1,
        win_rate=1.0 if bh_return > 0 else 0.0,
        profit_factor=0.0,
        avg_trade_return=bh_return / 100,
    )

    return results


def print_comparison(results: Dict[str, BacktestResult]):
    """Print comparison table."""
    print("\n" + "=" * 70)
    print("MODEL COMPARISON - BACKTEST RESULTS")
    print("=" * 70)

    headers = ["Metric", "V2 (LSTM)", "V1 (MLP)", "Buy&Hold"]
    if "v1" not in results:
        headers = ["Metric", "V2 (LSTM)", "Buy&Hold"]

    print(f"{headers[0]:<20} {' '.join(f'{h:>15}' for h in headers[1:])}")
    print("-" * 70)

    def get_val(key, attr):
        if key in results:
            return getattr(results[key], attr)
        return None

    metrics = [
        ("Return %", "total_return_pct", "{:.2f}"),
        ("Sharpe", "sharpe_ratio", "{:.2f}"),
        ("Max Drawdown", "max_drawdown", "{:.2%}"),
        ("Total Trades", "total_trades", "{:d}"),
        ("Win Rate", "win_rate", "{:.2%}"),
        ("Profit Factor", "profit_factor", "{:.2f}"),
        ("Avg Trade Ret", "avg_trade_return", "{:.4f}"),
    ]

    for label, attr, fmt in metrics:
        vals = []
        for key in ["v2", "v1", "buy_hold"]:
            if key in results:
                val = get_val(key, attr)
                if val is not None:
                    if isinstance(val, int):
                        vals.append(f"{val:>15d}")
                    else:
                        vals.append(f"{val:>15.4f}" if "." in fmt else f"{val:>15.2%}")
                else:
                    vals.append(f"{'N/A':>15}")
        print(f"{label:<20} {' '.join(vals)}")

    print("=" * 70)

    # Winner
    if "v1" in results:
        v2_better = results["v2"].total_return_pct > results["v1"].total_return_pct
        print(f"\n🏆 Winner: {'V2 (LSTM)' if v2_better else 'V1 (MLP)'}")
    else:
        print(f"\n📊 V2 Return: {results['v2'].total_return_pct:.2f}%")


def main():
    parser = argparse.ArgumentParser(description="Backtest and compare trading models")
    parser.add_argument("--v2-checkpoint", type=Path, required=True)
    parser.add_argument("--v1-checkpoint", type=Path, default=None)
    parser.add_argument("--ohlc", type=Path, default=Path("bagstraining/ohlc_data.csv"))
    parser.add_argument("--mint", type=str, required=True)
    parser.add_argument("--buy-threshold", type=float, default=0.5)
    parser.add_argument("--sell-threshold", type=float, default=0.4)
    parser.add_argument("--test-start-pct", type=float, default=0.8)
    parser.add_argument("--device", type=str, default="cuda")

    args = parser.parse_args()

    results = compare_models(
        v1_checkpoint=args.v1_checkpoint,
        v2_checkpoint=args.v2_checkpoint,
        ohlc_path=args.ohlc,
        mint=args.mint,
        buy_threshold=args.buy_threshold,
        sell_threshold=args.sell_threshold,
        test_start_pct=args.test_start_pct,
        device=args.device,
    )

    print_comparison(results)


if __name__ == "__main__":
    main()
