#!/usr/bin/env python3
"""Multi-Asset Walk-Forward Backtest Simulator.

Simulates hourly trading across crypto + stocks with realistic fills.
Walks forward hour by hour, using the aggregator to pick best opportunity.

Key features:
- Realistic fill simulation (buy fills if low <= price, sell fills if high >= price)
- Market hours filtering for stocks (9:30 AM - 4:00 PM ET)
- Aggregation across all symbols to pick highest expected return
- Position tracking with forced exit at position_length hours
- Fee-aware P&L calculation

Example:
    python backtest_multi_asset.py --days 7 --crypto-checkpoint path/to/crypto.pt --stock-checkpoint path/to/stock.pt
"""
from __future__ import annotations

import argparse
import logging
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch

from src.date_utils import is_nyse_trading_day_now

logger = logging.getLogger("backtest_multi")


@dataclass
class Position:
    """Active trading position."""
    symbol: str
    asset_class: str
    entry_timestamp: pd.Timestamp
    entry_price: float
    target_sell_price: float
    position_length: int  # Target hours to hold
    hours_held: int = 0
    quantity: float = 0.0
    entry_value: float = 0.0


@dataclass
class TradeRecord:
    """Completed trade record."""
    symbol: str
    asset_class: str
    entry_timestamp: pd.Timestamp
    entry_price: float
    exit_timestamp: pd.Timestamp
    exit_price: float
    target_sell_price: float
    position_length: int
    actual_hold_hours: int
    exit_type: str  # "tp", "forced", "end"
    quantity: float
    gross_pnl: float
    fees: float
    net_pnl: float
    return_pct: float


@dataclass
class BacktestResult:
    """Complete backtest result."""
    equity_curve: pd.Series
    trades: List[TradeRecord]
    metrics: Dict[str, float] = field(default_factory=dict)
    hourly_decisions: List[Dict] = field(default_factory=list)


class MultiAssetBacktester:
    """Walk-forward backtester for multi-asset hourly trading."""

    CRYPTO_MAKER_FEE = 0.0008  # 8 bps
    STOCK_MAKER_FEE = 0.0002   # 2 bps

    def __init__(
        self,
        crypto_checkpoint: Optional[str] = None,
        stock_checkpoint: Optional[str] = None,
        crypto_symbols: Optional[List[str]] = None,
        stock_symbols: Optional[List[str]] = None,
        initial_cash: float = 100000.0,
        device: str = "cuda",
    ):
        self.device = device
        self.initial_cash = initial_cash

        # Default symbols
        self.crypto_symbols = crypto_symbols or [
            "BTCUSD", "ETHUSD", "LINKUSD", "UNIUSD", "SOLUSD"
        ]
        self.stock_symbols = stock_symbols or self._get_stock_symbols()

        # Load models
        self.crypto_model = None
        self.crypto_normalizer = None
        self.crypto_features = None
        self.stock_model = None
        self.stock_normalizer = None
        self.stock_features = None

        if crypto_checkpoint and Path(crypto_checkpoint).exists():
            self._load_crypto_model(crypto_checkpoint)

        if stock_checkpoint and Path(stock_checkpoint).exists():
            self._load_stock_model(stock_checkpoint)

        # Load data for all symbols
        self.crypto_data: Dict[str, pd.DataFrame] = {}
        self.stock_data: Dict[str, pd.DataFrame] = {}
        self._load_all_data()

    def _get_stock_symbols(self) -> List[str]:
        """Get available stock symbols."""
        data_root = Path("trainingdatahourly/stocks")
        if not data_root.exists():
            return []
        return sorted([f.stem for f in data_root.glob("*.csv")])[:50]  # Limit for speed

    def _load_crypto_model(self, checkpoint_path: str) -> None:
        """Load crypto model from checkpoint."""
        from neuralhourlytradingv5.model import HourlyCryptoPolicyV5
        from neuralhourlytradingv5.data import FeatureNormalizer

        logger.info(f"Loading crypto model from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

        policy_config = checkpoint["config"]["policy"]
        model = HourlyCryptoPolicyV5(policy_config)

        state_dict = checkpoint["model_state_dict"]
        if any(k.startswith("_orig_mod.") for k in state_dict.keys()):
            state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}

        model.load_state_dict(state_dict)
        model = model.to(self.device)
        model.eval()

        self.crypto_model = model
        self.crypto_normalizer = FeatureNormalizer.from_dict(checkpoint["normalizer"])
        self.crypto_features = checkpoint["feature_columns"]

    def _load_stock_model(self, checkpoint_path: str) -> None:
        """Load stock model from checkpoint."""
        from neuralhourlystocksv5.model import HourlyStockPolicyV5
        from neuralhourlystocksv5.data import StockFeatureNormalizer

        logger.info(f"Loading stock model from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

        policy_config = checkpoint["config"]["policy"]
        model = HourlyStockPolicyV5(policy_config)

        state_dict = checkpoint["model_state_dict"]
        if any(k.startswith("_orig_mod.") for k in state_dict.keys()):
            state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}

        model.load_state_dict(state_dict)
        model = model.to(self.device)
        model.eval()

        self.stock_model = model
        self.stock_normalizer = StockFeatureNormalizer.from_dict(checkpoint["normalizer"])
        self.stock_features = checkpoint["feature_columns"]

    def _load_all_data(self) -> None:
        """Load OHLCV data for all symbols."""
        from neuralhourlytradingv5.data import HOURLY_FEATURES_V5
        from neuralhourlystocksv5.data import HOURLY_FEATURES_STOCKS_V5

        # Load crypto data
        for symbol in self.crypto_symbols:
            data_path = Path(f"trainingdatahourly/crypto/{symbol}.csv")
            if data_path.exists():
                df = pd.read_csv(data_path)
                df["timestamp"] = pd.to_datetime(df["timestamp"])
                df = df.sort_values("timestamp").reset_index(drop=True)
                for feat in HOURLY_FEATURES_V5:
                    if feat not in df.columns:
                        df[feat] = 0.0
                self.crypto_data[symbol] = df
                logger.info(f"Loaded {len(df)} bars for crypto {symbol}")

        # Load stock data
        for symbol in self.stock_symbols:
            data_path = Path(f"trainingdatahourly/stocks/{symbol}.csv")
            if data_path.exists():
                df = pd.read_csv(data_path)
                df["timestamp"] = pd.to_datetime(df["timestamp"])
                df = df.sort_values("timestamp").reset_index(drop=True)
                for feat in HOURLY_FEATURES_STOCKS_V5:
                    if feat not in df.columns:
                        df[feat] = 0.0
                self.stock_data[symbol] = df
                logger.debug(f"Loaded {len(df)} bars for stock {symbol}")

        logger.info(f"Loaded data: {len(self.crypto_data)} crypto, {len(self.stock_data)} stocks")

    def _get_bar_at_time(
        self, symbol: str, asset_class: str, timestamp: pd.Timestamp
    ) -> Optional[pd.Series]:
        """Get OHLCV bar at specific timestamp."""
        data = self.crypto_data if asset_class == "crypto" else self.stock_data
        if symbol not in data:
            return None

        df = data[symbol]
        # Find the bar at or just before this timestamp
        mask = df["timestamp"] <= timestamp
        if not mask.any():
            return None

        return df[mask].iloc[-1]

    def _get_features_at_time(
        self, symbol: str, asset_class: str, timestamp: pd.Timestamp, seq_len: int = 168
    ) -> Optional[np.ndarray]:
        """Get feature sequence ending at timestamp."""
        data = self.crypto_data if asset_class == "crypto" else self.stock_data
        if symbol not in data:
            return None

        df = data[symbol]
        features = self.crypto_features if asset_class == "crypto" else self.stock_features

        # Find index at timestamp
        mask = df["timestamp"] <= timestamp
        if not mask.any():
            return None

        end_idx = mask.sum()
        if end_idx < seq_len:
            return None

        feature_df = df[list(features)].iloc[end_idx - seq_len:end_idx]
        return feature_df.values

    def _get_model_prediction(
        self, symbol: str, asset_class: str, timestamp: pd.Timestamp
    ) -> Optional[Dict]:
        """Get model prediction for symbol at timestamp."""
        if asset_class == "crypto":
            if self.crypto_model is None:
                return None
            model = self.crypto_model
            normalizer = self.crypto_normalizer
        else:
            if self.stock_model is None:
                return None
            model = self.stock_model
            normalizer = self.stock_normalizer

        # Get features
        features = self._get_features_at_time(symbol, asset_class, timestamp)
        if features is None:
            return None

        # Get current bar
        bar = self._get_bar_at_time(symbol, asset_class, timestamp)
        if bar is None:
            return None

        current_close = float(bar["close"])

        # Normalize and infer
        features_norm = normalizer.transform(features)
        feature_tensor = (
            torch.from_numpy(features_norm)
            .unsqueeze(0)
            .float()
            .contiguous()
            .to(self.device)
        )
        ref_close_tensor = torch.tensor([current_close], device=self.device)

        with torch.no_grad():
            outputs = model(feature_tensor)
            actions = model.get_hard_actions(outputs, ref_close_tensor)

        return {
            "buy_price": float(actions["buy_price"].item()),
            "sell_price": float(actions["sell_price"].item()),
            "position_length": int(actions["position_length"].item()),
            "position_size": float(actions["position_size"].item()),
            "current_close": current_close,
        }

    def _calculate_expected_return(
        self, prediction: Dict, asset_class: str
    ) -> Tuple[float, float]:
        """Calculate expected return and risk-adjusted return."""
        fee = self.CRYPTO_MAKER_FEE if asset_class == "crypto" else self.STOCK_MAKER_FEE
        round_trip_fees = 2 * fee

        gross_return = (prediction["sell_price"] - prediction["buy_price"]) / prediction["buy_price"]
        net_return = gross_return - round_trip_fees
        risk_adjusted = net_return * prediction["position_size"]

        return net_return, risk_adjusted

    def _get_all_opportunities(
        self, timestamp: pd.Timestamp
    ) -> List[Dict]:
        """Get all trading opportunities at timestamp."""
        opportunities = []

        # Check if stock market is open
        is_stock_market_open = is_nyse_trading_day_now(timestamp)

        # Crypto opportunities (always available)
        for symbol in self.crypto_data.keys():
            pred = self._get_model_prediction(symbol, "crypto", timestamp)
            if pred and pred["position_length"] > 0:
                exp_ret, risk_adj = self._calculate_expected_return(pred, "crypto")
                opportunities.append({
                    "symbol": symbol,
                    "asset_class": "crypto",
                    "prediction": pred,
                    "expected_return": exp_ret,
                    "risk_adjusted_return": risk_adj,
                    "is_tradable": True,
                })

        # Stock opportunities (only during market hours)
        if is_stock_market_open:
            for symbol in list(self.stock_data.keys())[:30]:  # Limit for speed
                pred = self._get_model_prediction(symbol, "stock", timestamp)
                if pred and pred["position_length"] > 0:
                    exp_ret, risk_adj = self._calculate_expected_return(pred, "stock")
                    opportunities.append({
                        "symbol": symbol,
                        "asset_class": "stock",
                        "prediction": pred,
                        "expected_return": exp_ret,
                        "risk_adjusted_return": risk_adj,
                        "is_tradable": True,
                    })

        # Sort by risk-adjusted return
        opportunities.sort(key=lambda x: -x["risk_adjusted_return"])
        return opportunities

    def run(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        days: int = 7,
    ) -> BacktestResult:
        """Run walk-forward backtest.

        Args:
            start_date: Start date (default: end_date - days)
            end_date: End date (default: now)
            days: Number of days if start_date not specified

        Returns:
            BacktestResult with equity curve, trades, and metrics
        """
        # Determine date range
        if end_date is None:
            end_date = datetime.now(timezone.utc)
        if start_date is None:
            start_date = end_date - timedelta(days=days)

        # Find common time range across all data (crypto + stocks)
        all_timestamps = set()
        for df in self.crypto_data.values():
            ts_set = set(df["timestamp"])
            all_timestamps = all_timestamps.union(ts_set) if all_timestamps else ts_set
        for df in self.stock_data.values():
            ts_set = set(df["timestamp"])
            all_timestamps = all_timestamps.union(ts_set)

        # Filter to date range
        start_ts = pd.Timestamp(start_date).tz_localize("UTC") if start_date.tzinfo is None else pd.Timestamp(start_date)
        end_ts = pd.Timestamp(end_date).tz_localize("UTC") if end_date.tzinfo is None else pd.Timestamp(end_date)
        all_timestamps = sorted([
            ts for ts in all_timestamps
            if start_ts <= ts <= end_ts
        ])

        if not all_timestamps:
            logger.error("No timestamps in date range")
            return BacktestResult(
                equity_curve=pd.Series(),
                trades=[],
                metrics={},
            )

        logger.info(f"Running backtest from {all_timestamps[0]} to {all_timestamps[-1]}")
        logger.info(f"Total hours: {len(all_timestamps)}")

        # Initialize state
        cash = self.initial_cash
        current_position: Optional[Position] = None
        trades: List[TradeRecord] = []
        equity_values: List[float] = []
        timestamps: List[pd.Timestamp] = []
        hourly_decisions: List[Dict] = []

        # Walk forward hour by hour
        for i, timestamp in enumerate(all_timestamps):
            # Process existing position
            if current_position is not None:
                current_position.hours_held += 1
                bar = self._get_bar_at_time(
                    current_position.symbol,
                    current_position.asset_class,
                    timestamp
                )

                exit_trade = False
                exit_type = ""
                exit_price = 0.0

                if bar is not None:
                    bar_high = float(bar["high"])
                    bar_low = float(bar["low"])
                    bar_close = float(bar["close"])

                    # Check take-profit (sell fills if high >= target)
                    if bar_high >= current_position.target_sell_price:
                        exit_trade = True
                        exit_type = "tp"
                        exit_price = current_position.target_sell_price

                    # Check forced exit at position_length
                    elif current_position.hours_held >= current_position.position_length:
                        exit_trade = True
                        exit_type = "forced"
                        exit_price = bar_close * 0.9995  # 5bps slippage

                if exit_trade:
                    fee = (
                        self.CRYPTO_MAKER_FEE if current_position.asset_class == "crypto"
                        else self.STOCK_MAKER_FEE
                    )
                    exit_value = current_position.quantity * exit_price * (1 - fee)
                    gross_pnl = exit_value - current_position.entry_value
                    total_fees = fee * 2 * current_position.entry_value
                    net_pnl = gross_pnl
                    return_pct = net_pnl / current_position.entry_value

                    trades.append(TradeRecord(
                        symbol=current_position.symbol,
                        asset_class=current_position.asset_class,
                        entry_timestamp=current_position.entry_timestamp,
                        entry_price=current_position.entry_price,
                        exit_timestamp=timestamp,
                        exit_price=exit_price,
                        target_sell_price=current_position.target_sell_price,
                        position_length=current_position.position_length,
                        actual_hold_hours=current_position.hours_held,
                        exit_type=exit_type,
                        quantity=current_position.quantity,
                        gross_pnl=gross_pnl,
                        fees=total_fees,
                        net_pnl=net_pnl,
                        return_pct=return_pct,
                    ))

                    cash += exit_value
                    current_position = None

            # Get new opportunities if no position
            if current_position is None:
                opportunities = self._get_all_opportunities(timestamp)

                decision = {
                    "timestamp": timestamp,
                    "num_opportunities": len(opportunities),
                    "action": "skip",
                    "symbol": None,
                }

                if opportunities:
                    best = opportunities[0]
                    pred = best["prediction"]

                    # Validate spread
                    if pred["buy_price"] < pred["sell_price"]:
                        bar = self._get_bar_at_time(best["symbol"], best["asset_class"], timestamp)

                        if bar is not None:
                            bar_low = float(bar["low"])

                            # Check if entry fills (low <= buy_price)
                            if bar_low <= pred["buy_price"]:
                                fee = (
                                    self.CRYPTO_MAKER_FEE if best["asset_class"] == "crypto"
                                    else self.STOCK_MAKER_FEE
                                )

                                # Calculate position size
                                position_value = cash * pred["position_size"]
                                entry_cost = position_value
                                entry_value = entry_cost / (1 + fee)
                                quantity = entry_value / pred["buy_price"]

                                if entry_cost <= cash and position_value >= 100:
                                    cash -= entry_cost
                                    current_position = Position(
                                        symbol=best["symbol"],
                                        asset_class=best["asset_class"],
                                        entry_timestamp=timestamp,
                                        entry_price=pred["buy_price"],
                                        target_sell_price=pred["sell_price"],
                                        position_length=pred["position_length"],
                                        hours_held=0,
                                        quantity=quantity,
                                        entry_value=entry_value,
                                    )

                                    decision["action"] = "enter"
                                    decision["symbol"] = best["symbol"]
                                    decision["asset_class"] = best["asset_class"]
                                    decision["expected_return"] = best["expected_return"]

                hourly_decisions.append(decision)

            # Calculate equity
            equity = cash
            if current_position is not None:
                bar = self._get_bar_at_time(
                    current_position.symbol,
                    current_position.asset_class,
                    timestamp
                )
                if bar is not None:
                    equity += current_position.quantity * float(bar["close"])

            equity_values.append(equity)
            timestamps.append(timestamp)

            # Log progress
            if i % 24 == 0:
                logger.info(
                    f"Hour {i}/{len(all_timestamps)} | "
                    f"Equity: ${equity:,.2f} | "
                    f"Trades: {len(trades)} | "
                    f"Position: {current_position.symbol if current_position else 'None'}"
                )

        # Close any remaining position at end
        if current_position is not None:
            bar = self._get_bar_at_time(
                current_position.symbol,
                current_position.asset_class,
                all_timestamps[-1]
            )
            if bar is not None:
                exit_price = float(bar["close"])
                fee = (
                    self.CRYPTO_MAKER_FEE if current_position.asset_class == "crypto"
                    else self.STOCK_MAKER_FEE
                )
                exit_value = current_position.quantity * exit_price * (1 - fee)
                gross_pnl = exit_value - current_position.entry_value
                total_fees = fee * 2 * current_position.entry_value
                net_pnl = gross_pnl
                return_pct = net_pnl / current_position.entry_value

                trades.append(TradeRecord(
                    symbol=current_position.symbol,
                    asset_class=current_position.asset_class,
                    entry_timestamp=current_position.entry_timestamp,
                    entry_price=current_position.entry_price,
                    exit_timestamp=all_timestamps[-1],
                    exit_price=exit_price,
                    target_sell_price=current_position.target_sell_price,
                    position_length=current_position.position_length,
                    actual_hold_hours=current_position.hours_held,
                    exit_type="end",
                    quantity=current_position.quantity,
                    gross_pnl=gross_pnl,
                    fees=total_fees,
                    net_pnl=net_pnl,
                    return_pct=return_pct,
                ))

                cash += exit_value
                equity_values[-1] = cash

        # Create equity curve
        equity_curve = pd.Series(equity_values, index=timestamps)

        # Compute metrics
        metrics = self._compute_metrics(equity_curve, trades)

        return BacktestResult(
            equity_curve=equity_curve,
            trades=trades,
            metrics=metrics,
            hourly_decisions=hourly_decisions,
        )

    def _compute_metrics(
        self, equity_curve: pd.Series, trades: List[TradeRecord]
    ) -> Dict[str, float]:
        """Compute performance metrics."""
        if equity_curve.empty:
            return {}

        # Returns
        values = equity_curve.values
        returns = np.diff(values) / np.clip(values[:-1], 1e-8, None)
        mean_ret = returns.mean() if len(returns) else 0.0

        # Downside deviation
        downside = returns[returns < 0]
        downside_std = downside.std() if len(downside) else 0.0

        # Sortino ratio (annualized from hourly)
        sortino = (
            mean_ret / downside_std * np.sqrt(24 * 365) if downside_std > 0 else 0.0
        )

        # Sharpe ratio
        full_std = returns.std() if len(returns) else 0.0
        sharpe = mean_ret / full_std * np.sqrt(24 * 365) if full_std > 0 else 0.0

        # Total return
        total_return = (equity_curve.iloc[-1] - equity_curve.iloc[0]) / equity_curve.iloc[0]

        # Max drawdown
        peak = equity_curve.expanding().max()
        drawdown = (equity_curve - peak) / peak
        max_drawdown = drawdown.min()

        # Trade statistics
        if trades:
            winning_trades = [t for t in trades if t.net_pnl > 0]
            crypto_trades = [t for t in trades if t.asset_class == "crypto"]
            stock_trades = [t for t in trades if t.asset_class == "stock"]
            tp_trades = [t for t in trades if t.exit_type == "tp"]

            win_rate = len(winning_trades) / len(trades)
            tp_rate = len(tp_trades) / len(trades)
            avg_return = np.mean([t.return_pct for t in trades])
            avg_hold_hours = np.mean([t.actual_hold_hours for t in trades])
            total_pnl = sum(t.net_pnl for t in trades)
            total_fees = sum(t.fees for t in trades)
        else:
            win_rate = tp_rate = avg_return = avg_hold_hours = total_pnl = total_fees = 0.0
            crypto_trades = stock_trades = []

        return {
            "total_return": float(total_return),
            "total_return_pct": float(total_return * 100),
            "sortino": float(sortino),
            "sharpe": float(sharpe),
            "max_drawdown": float(max_drawdown),
            "max_drawdown_pct": float(max_drawdown * 100),
            "num_trades": len(trades),
            "num_crypto_trades": len(crypto_trades),
            "num_stock_trades": len(stock_trades),
            "win_rate": float(win_rate),
            "tp_rate": float(tp_rate),
            "avg_return_pct": float(avg_return * 100),
            "avg_hold_hours": float(avg_hold_hours),
            "total_pnl": float(total_pnl),
            "total_fees": float(total_fees),
            "final_equity": float(equity_curve.iloc[-1]),
        }


def print_backtest_summary(result: BacktestResult) -> None:
    """Print human-readable backtest summary."""
    m = result.metrics

    print("\n" + "=" * 60)
    print("MULTI-ASSET BACKTEST RESULTS")
    print("=" * 60)

    print(f"\nPerformance:")
    print(f"  Total Return:    {m.get('total_return_pct', 0):+.2f}%")
    print(f"  Final Equity:    ${m.get('final_equity', 0):,.2f}")
    print(f"  Sortino Ratio:   {m.get('sortino', 0):.2f}")
    print(f"  Sharpe Ratio:    {m.get('sharpe', 0):.2f}")
    print(f"  Max Drawdown:    {m.get('max_drawdown_pct', 0):.2f}%")

    print(f"\nTrades:")
    print(f"  Total Trades:    {m.get('num_trades', 0)}")
    print(f"  Crypto Trades:   {m.get('num_crypto_trades', 0)}")
    print(f"  Stock Trades:    {m.get('num_stock_trades', 0)}")
    print(f"  Win Rate:        {m.get('win_rate', 0) * 100:.1f}%")
    print(f"  TP Rate:         {m.get('tp_rate', 0) * 100:.1f}%")

    print(f"\nHolding:")
    print(f"  Avg Hold Hours:  {m.get('avg_hold_hours', 0):.1f}")
    print(f"  Avg Return:      {m.get('avg_return_pct', 0):+.2f}%")
    print(f"  Total P&L:       ${m.get('total_pnl', 0):+,.2f}")
    print(f"  Total Fees:      ${m.get('total_fees', 0):,.2f}")

    print("=" * 60)

    # Print trade breakdown by symbol
    if result.trades:
        print("\nTrade Breakdown by Symbol:")
        symbol_stats = defaultdict(lambda: {"count": 0, "pnl": 0.0, "wins": 0})
        for trade in result.trades:
            symbol_stats[trade.symbol]["count"] += 1
            symbol_stats[trade.symbol]["pnl"] += trade.net_pnl
            if trade.net_pnl > 0:
                symbol_stats[trade.symbol]["wins"] += 1

        for symbol, stats in sorted(symbol_stats.items(), key=lambda x: -x[1]["pnl"]):
            win_rate = stats["wins"] / stats["count"] * 100 if stats["count"] else 0
            print(f"  {symbol:10s}: {stats['count']:3d} trades, ${stats['pnl']:+8.2f} P&L, {win_rate:.0f}% win rate")

    print("\n")


def main():
    parser = argparse.ArgumentParser(
        description="Multi-asset walk-forward backtest",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--crypto-checkpoint",
        type=str,
        default=None,
        help="Path to crypto model checkpoint",
    )
    parser.add_argument(
        "--stock-checkpoint",
        type=str,
        default=None,
        help="Path to stock model checkpoint",
    )
    parser.add_argument(
        "--days",
        type=int,
        default=7,
        help="Number of days to backtest (default: 7)",
    )
    parser.add_argument(
        "--initial-cash",
        type=float,
        default=100000.0,
        help="Initial cash (default: $100,000)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run on",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        help="Logging level",
    )

    args = parser.parse_args()

    # Configure logging
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    # Find checkpoints if not specified
    crypto_checkpoint = args.crypto_checkpoint
    stock_checkpoint = args.stock_checkpoint

    if not crypto_checkpoint:
        crypto_dir = Path("neuralhourlytradingv5/checkpoints")
        if crypto_dir.exists():
            checkpoints = sorted(crypto_dir.glob("best_*.pt"))
            if checkpoints:
                crypto_checkpoint = str(checkpoints[-1])
                logger.info(f"Using crypto checkpoint: {crypto_checkpoint}")

    if not stock_checkpoint:
        stock_dir = Path("neuralhourlystocksv5/checkpoints")
        if stock_dir.exists():
            checkpoints = sorted(stock_dir.glob("best_*.pt"))
            if checkpoints:
                stock_checkpoint = str(checkpoints[-1])
                logger.info(f"Using stock checkpoint: {stock_checkpoint}")

    if not crypto_checkpoint and not stock_checkpoint:
        logger.error("No checkpoints found. Train models first.")
        return

    # Create backtester
    backtester = MultiAssetBacktester(
        crypto_checkpoint=crypto_checkpoint,
        stock_checkpoint=stock_checkpoint,
        initial_cash=args.initial_cash,
        device=args.device,
    )

    # Run backtest
    result = backtester.run(days=args.days)

    # Print results
    print_backtest_summary(result)

    # Save results
    output_dir = Path("backtest_results")
    output_dir.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result.equity_curve.to_csv(output_dir / f"equity_curve_{timestamp}.csv")

    trades_df = pd.DataFrame([
        {
            "symbol": t.symbol,
            "asset_class": t.asset_class,
            "entry_timestamp": t.entry_timestamp,
            "entry_price": t.entry_price,
            "exit_timestamp": t.exit_timestamp,
            "exit_price": t.exit_price,
            "exit_type": t.exit_type,
            "hold_hours": t.actual_hold_hours,
            "return_pct": t.return_pct * 100,
            "net_pnl": t.net_pnl,
        }
        for t in result.trades
    ])
    trades_df.to_csv(output_dir / f"trades_{timestamp}.csv", index=False)

    logger.info(f"Results saved to {output_dir}")


if __name__ == "__main__":
    main()
