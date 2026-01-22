"""Neural market simulator for Bags.fm trading backtests.

Provides:
- Walk-forward backtesting with neural model predictions
- Realistic cost modeling (swap fees, slippage, network fees)
- Performance metrics: PnL, Sharpe, Sortino, max drawdown
- Threshold optimization for buy/sell signals
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch

logger = logging.getLogger(__name__)


@dataclass
class Trade:
    """Record of a completed trade."""
    timestamp: datetime
    side: str  # "buy" or "sell"
    price: float
    quantity: float
    notional_sol: float
    fee_sol: float
    pnl_sol: float = 0.0  # PnL for this trade (sell only)


@dataclass
class BacktestResult:
    """Complete backtest results with metrics."""

    # Basic metrics
    initial_sol: float
    final_sol: float
    total_return_sol: float
    total_return_pct: float

    # Risk-adjusted metrics
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    max_drawdown_pct: float

    # Trade statistics
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float

    # Cost breakdown
    total_fees_sol: float
    total_slippage_sol: float

    # Time series
    equity_curve: pd.Series
    trades: List[Trade]

    # Model info
    buy_threshold: float
    sell_threshold: float

    def summary(self) -> str:
        """Generate summary string."""
        return f"""
=== Backtest Results ===
Initial: {self.initial_sol:.4f} SOL
Final:   {self.final_sol:.4f} SOL
Return:  {self.total_return_pct:+.2f}%

Risk Metrics:
  Sharpe:   {self.sharpe_ratio:.3f}
  Sortino:  {self.sortino_ratio:.3f}
  Max DD:   {self.max_drawdown_pct:.2f}%

Trades: {self.total_trades}
  Wins:  {self.winning_trades} ({self.win_rate*100:.1f}%)
  Losses: {self.losing_trades}

Costs:
  Fees:     {self.total_fees_sol:.6f} SOL
  Slippage: {self.total_slippage_sol:.6f} SOL

Thresholds: buy={self.buy_threshold:.2f} sell={self.sell_threshold:.2f}
"""


class NeuralSimulator:
    """Simulates trading using a neural model for signals.

    Features:
    - Walk-forward backtesting on OHLC data
    - Neural model inference for buy/sell signals
    - Realistic swap cost modeling
    - Comprehensive performance metrics
    """

    def __init__(
        self,
        model: torch.nn.Module,
        normalizer,  # FeatureNormalizer
        context_bars: int = 64,
        initial_sol: float = 1.0,
        max_position_sol: float = 1.0,
        min_trade_sol: float = 0.01,
        swap_fee_bps: int = 30,
        slippage_bps: int = 100,
        network_fee_sol: float = 0.0001,
        device: str = "cuda",
    ):
        self.model = model
        self.normalizer = normalizer
        self.context_bars = context_bars
        self.initial_sol = initial_sol
        self.max_position_sol = max_position_sol
        self.min_trade_sol = min_trade_sol
        self.swap_fee_bps = swap_fee_bps
        self.slippage_bps = slippage_bps
        self.network_fee_sol = network_fee_sol
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

        self.model.to(self.device)
        self.model.eval()

    def _compute_swap_cost(self, notional_sol: float) -> Tuple[float, float]:
        """Compute swap costs.

        Returns:
            (fee_sol, slippage_sol)
        """
        fee_sol = notional_sol * (self.swap_fee_bps / 10000)
        slippage_sol = notional_sol * (self.slippage_bps / 10000)
        return fee_sol, slippage_sol

    def _build_features(
        self,
        opens: np.ndarray,
        highs: np.ndarray,
        lows: np.ndarray,
        closes: np.ndarray,
    ) -> np.ndarray:
        """Build features from OHLC window."""
        from bagsneural.dataset import build_window_features
        return build_window_features(opens, highs, lows, closes)

    def run_backtest(
        self,
        df: pd.DataFrame,
        buy_threshold: float = 0.50,
        sell_threshold: float = 0.45,
        test_split: float = 0.0,
    ) -> BacktestResult:
        """Run walk-forward backtest.

        Args:
            df: OHLC DataFrame with columns: timestamp, open, high, low, close
            buy_threshold: Probability threshold to open position
            sell_threshold: Probability threshold to close position
            test_split: Fraction of data to use for testing (0 = use all)

        Returns:
            BacktestResult with all metrics
        """
        # Prepare data
        if test_split > 0:
            start_idx = int(len(df) * (1 - test_split))
            start_idx = max(start_idx, self.context_bars)
            df = df.iloc[start_idx - self.context_bars:].reset_index(drop=True)

        opens = df["open"].to_numpy(dtype=np.float32)
        highs = df["high"].to_numpy(dtype=np.float32)
        lows = df["low"].to_numpy(dtype=np.float32)
        closes = df["close"].to_numpy(dtype=np.float32)
        timestamps = pd.to_datetime(df["timestamp"])

        # State
        sol_balance = self.initial_sol
        token_quantity = 0.0
        entry_price = 0.0
        holding = False

        trades: List[Trade] = []
        equity_history: List[Tuple[datetime, float]] = []
        total_fees = 0.0
        total_slippage = 0.0

        # Walk forward
        for idx in range(self.context_bars, len(df)):
            # Build features
            window_start = idx - self.context_bars
            features = self._build_features(
                opens[window_start:idx],
                highs[window_start:idx],
                lows[window_start:idx],
                closes[window_start:idx],
            )

            # Normalize and predict
            normalized = self.normalizer.transform(features)[None, :]
            x = torch.tensor(normalized, dtype=torch.float32, device=self.device)

            with torch.no_grad():
                signal_logit, size_logit = self.model(x)
                prob = torch.sigmoid(signal_logit).item()
                size = torch.sigmoid(size_logit).item()

            current_price = float(closes[idx])
            ts = timestamps.iloc[idx]

            # Trading logic
            if not holding and prob >= buy_threshold:
                # Buy
                trade_sol = min(size * self.max_position_sol, sol_balance * 0.99)
                if trade_sol >= self.min_trade_sol:
                    fee_sol, slip_sol = self._compute_swap_cost(trade_sol)
                    cost_sol = fee_sol + slip_sol + self.network_fee_sol

                    effective_sol = trade_sol - cost_sol
                    if effective_sol > 0:
                        token_quantity = effective_sol / current_price
                        sol_balance -= trade_sol
                        entry_price = current_price
                        holding = True

                        total_fees += fee_sol + self.network_fee_sol
                        total_slippage += slip_sol

                        trades.append(Trade(
                            timestamp=ts,
                            side="buy",
                            price=current_price,
                            quantity=token_quantity,
                            notional_sol=trade_sol,
                            fee_sol=fee_sol + self.network_fee_sol,
                        ))

            elif holding and prob <= sell_threshold:
                # Sell
                notional_sol = token_quantity * current_price
                fee_sol, slip_sol = self._compute_swap_cost(notional_sol)
                cost_sol = fee_sol + slip_sol + self.network_fee_sol

                received_sol = notional_sol - cost_sol
                pnl = received_sol - (token_quantity * entry_price)

                sol_balance += received_sol
                total_fees += fee_sol + self.network_fee_sol
                total_slippage += slip_sol

                trades.append(Trade(
                    timestamp=ts,
                    side="sell",
                    price=current_price,
                    quantity=token_quantity,
                    notional_sol=notional_sol,
                    fee_sol=fee_sol + self.network_fee_sol,
                    pnl_sol=pnl,
                ))

                token_quantity = 0.0
                entry_price = 0.0
                holding = False

            # Track equity
            portfolio_value = sol_balance + (token_quantity * current_price if holding else 0)
            equity_history.append((ts, portfolio_value))

        # Close any remaining position at end
        if holding and len(closes) > 0:
            final_price = float(closes[-1])
            notional_sol = token_quantity * final_price
            fee_sol, slip_sol = self._compute_swap_cost(notional_sol)
            received_sol = notional_sol - fee_sol - slip_sol - self.network_fee_sol
            sol_balance += received_sol
            total_fees += fee_sol + self.network_fee_sol
            total_slippage += slip_sol

        # Compute metrics
        return self._compute_results(
            initial_sol=self.initial_sol,
            final_sol=sol_balance,
            equity_history=equity_history,
            trades=trades,
            total_fees=total_fees,
            total_slippage=total_slippage,
            buy_threshold=buy_threshold,
            sell_threshold=sell_threshold,
        )

    def _compute_results(
        self,
        initial_sol: float,
        final_sol: float,
        equity_history: List[Tuple[datetime, float]],
        trades: List[Trade],
        total_fees: float,
        total_slippage: float,
        buy_threshold: float,
        sell_threshold: float,
    ) -> BacktestResult:
        """Compute all performance metrics."""

        # Basic returns
        total_return_sol = final_sol - initial_sol
        total_return_pct = (total_return_sol / initial_sol) * 100

        # Equity curve
        if equity_history:
            equity_curve = pd.Series(
                [e[1] for e in equity_history],
                index=[e[0] for e in equity_history],
            )
        else:
            equity_curve = pd.Series([initial_sol])

        # Daily returns for Sharpe/Sortino
        if len(equity_curve) > 1:
            returns = equity_curve.pct_change().dropna()

            # Sharpe ratio (annualized, assuming 365 days * 144 10-min bars per day)
            if len(returns) > 0 and returns.std() > 0:
                sharpe = (returns.mean() / returns.std()) * np.sqrt(365 * 144)
            else:
                sharpe = 0.0

            # Sortino ratio (downside deviation)
            downside_returns = returns[returns < 0]
            if len(downside_returns) > 0 and downside_returns.std() > 0:
                sortino = (returns.mean() / downside_returns.std()) * np.sqrt(365 * 144)
            else:
                sortino = sharpe  # If no downside, use Sharpe

            # Max drawdown
            rolling_max = equity_curve.expanding().max()
            drawdown = equity_curve - rolling_max
            max_dd = drawdown.min()
            max_dd_pct = (max_dd / rolling_max.iloc[-1]) * 100 if rolling_max.iloc[-1] > 0 else 0
        else:
            sharpe = 0.0
            sortino = 0.0
            max_dd = 0.0
            max_dd_pct = 0.0

        # Trade statistics
        sell_trades = [t for t in trades if t.side == "sell"]
        winning = [t for t in sell_trades if t.pnl_sol > 0]
        losing = [t for t in sell_trades if t.pnl_sol <= 0]
        win_rate = len(winning) / len(sell_trades) if sell_trades else 0.0

        return BacktestResult(
            initial_sol=initial_sol,
            final_sol=final_sol,
            total_return_sol=total_return_sol,
            total_return_pct=total_return_pct,
            sharpe_ratio=sharpe,
            sortino_ratio=sortino,
            max_drawdown=abs(max_dd),
            max_drawdown_pct=abs(max_dd_pct),
            total_trades=len(trades),
            winning_trades=len(winning),
            losing_trades=len(losing),
            win_rate=win_rate,
            total_fees_sol=total_fees,
            total_slippage_sol=total_slippage,
            equity_curve=equity_curve,
            trades=trades,
            buy_threshold=buy_threshold,
            sell_threshold=sell_threshold,
        )

    def sweep_thresholds(
        self,
        df: pd.DataFrame,
        buy_thresholds: List[float] = None,
        sell_thresholds: List[float] = None,
        test_split: float = 0.2,
    ) -> List[BacktestResult]:
        """Sweep threshold combinations to find optimal settings.

        Args:
            df: OHLC DataFrame
            buy_thresholds: List of buy thresholds to test
            sell_thresholds: List of sell thresholds to test
            test_split: Fraction for test set

        Returns:
            List of BacktestResults sorted by Sortino ratio
        """
        if buy_thresholds is None:
            buy_thresholds = [0.45, 0.48, 0.50, 0.52, 0.55]
        if sell_thresholds is None:
            sell_thresholds = [0.40, 0.43, 0.45, 0.48, 0.50]

        results = []
        for buy_t in buy_thresholds:
            for sell_t in sell_thresholds:
                if sell_t >= buy_t:
                    continue  # Skip invalid combinations

                result = self.run_backtest(
                    df=df,
                    buy_threshold=buy_t,
                    sell_threshold=sell_t,
                    test_split=test_split,
                )
                results.append(result)
                logger.info(
                    f"buy={buy_t:.2f} sell={sell_t:.2f}: "
                    f"return={result.total_return_pct:+.2f}% "
                    f"sortino={result.sortino_ratio:.3f} "
                    f"trades={result.total_trades}"
                )

        # Sort by Sortino ratio
        results.sort(key=lambda r: r.sortino_ratio, reverse=True)
        return results


def run_backtest(
    checkpoint_path: Path,
    ohlc_path: Path,
    mint: str,
    buy_threshold: float = 0.50,
    sell_threshold: float = 0.45,
    test_split: float = 0.2,
    initial_sol: float = 1.0,
    max_position_sol: float = 1.0,
    device: str = "cuda",
) -> BacktestResult:
    """Convenience function to run a backtest.

    Args:
        checkpoint_path: Path to model checkpoint
        ohlc_path: Path to OHLC CSV
        mint: Token mint address
        buy_threshold: Probability to buy
        sell_threshold: Probability to sell
        test_split: Fraction of data for testing
        initial_sol: Starting SOL balance
        max_position_sol: Max position size
        device: Torch device

    Returns:
        BacktestResult
    """
    from bagsneural.dataset import FeatureNormalizer, load_ohlc_dataframe
    from bagsneural.model import BagsNeuralModel

    # Load data
    df = load_ohlc_dataframe(ohlc_path, mint)

    # Load checkpoint
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    config = ckpt["config"]
    normalizer = FeatureNormalizer.from_dict(ckpt["normalizer"])
    context = config.get("context", 64)

    # Build model
    input_dim = context * 3  # 3 features per bar
    model = BagsNeuralModel(
        input_dim=input_dim,
        hidden_dims=config.get("hidden", [128, 64]),
        dropout=config.get("dropout", 0.1),
    )
    model.load_state_dict(ckpt["model_state"])

    # Create simulator
    simulator = NeuralSimulator(
        model=model,
        normalizer=normalizer,
        context_bars=context,
        initial_sol=initial_sol,
        max_position_sol=max_position_sol,
        device=device,
    )

    # Run backtest
    return simulator.run_backtest(
        df=df,
        buy_threshold=buy_threshold,
        sell_threshold=sell_threshold,
        test_split=test_split,
    )
