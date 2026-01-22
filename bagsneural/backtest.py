#!/usr/bin/env python3
"""Backtest a trained Bags.fm neural model."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from bagsfm.config import SimulationConfig, CostConfig
from bagsfm.simulator import MarketSimulator
from bagsneural.dataset import (
    FeatureNormalizer,
    build_window_features,
    load_ohlc_dataframe,
)
from bagsneural.model import BagsNeuralModel

logger = logging.getLogger("bagsneural_backtest")


def _load_checkpoint(
    path: Path,
    input_dim: int,
    device: torch.device,
) -> tuple[BagsNeuralModel, FeatureNormalizer, dict]:
    payload = torch.load(path, map_location="cpu")
    config = payload["config"]
    normalizer = FeatureNormalizer.from_dict(payload["normalizer"])

    model = BagsNeuralModel(
        input_dim=input_dim,
        hidden_dims=config.get("hidden", [128, 64]),
        dropout=config.get("dropout", 0.1),
    )
    model.load_state_dict(payload["model_state"])
    model.to(device)
    model.eval()
    return model, normalizer, config


def main() -> None:
    parser = argparse.ArgumentParser(description="Backtest Bags.fm neural model")
    parser.add_argument("--ohlc", type=Path, default=Path("bagstraining") / "ohlc_data.csv")
    parser.add_argument("--mint", type=str, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--context", type=int, default=None)
    parser.add_argument("--horizon", type=int, default=None)
    parser.add_argument("--buy-threshold", type=float, default=0.55)
    parser.add_argument("--sell-threshold", type=float, default=0.45)
    parser.add_argument("--max-position-sol", type=float, default=1.0)
    parser.add_argument("--min-trade-sol", type=float, default=0.01)
    parser.add_argument(
        "--test-split",
        type=float,
        default=0.0,
        help="Evaluate only the last fraction of data (0-1)",
    )
    parser.add_argument(
        "--auto-thresholds",
        action="store_true",
        help="Derive thresholds from training probabilities",
    )
    parser.add_argument("--buy-quantile", type=float, default=0.8)
    parser.add_argument("--sell-quantile", type=float, default=0.2)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    df = load_ohlc_dataframe(args.ohlc, args.mint)
    start_idx = 0
    if args.test_split:
        if not 0.0 < args.test_split < 1.0:
            raise ValueError("test-split must be between 0 and 1")
        start_idx = int(len(df) * (1 - args.test_split))
        start_idx = max(start_idx, 0)
    opens = df["open"].to_numpy(dtype=np.float32)
    highs = df["high"].to_numpy(dtype=np.float32)
    lows = df["low"].to_numpy(dtype=np.float32)
    closes = df["close"].to_numpy(dtype=np.float32)
    timestamps = df["timestamp"].to_numpy()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    context = args.context
    horizon = args.horizon

    if context is None or horizon is None:
        preview_context = 64 if context is None else context
        preview_features = build_window_features(
            opens[:preview_context],
            highs[:preview_context],
            lows[:preview_context],
            closes[:preview_context],
        )
        model, normalizer, config = _load_checkpoint(
            args.checkpoint,
            input_dim=preview_features.shape[0],
            device=device,
        )
        if context is None:
            context = int(config.get("context", preview_context))
        if horizon is None:
            horizon = int(config.get("horizon", 3))
    else:
        preview_features = build_window_features(
            opens[:context],
            highs[:context],
            lows[:context],
            closes[:context],
        )
        model, normalizer, config = _load_checkpoint(
            args.checkpoint,
            input_dim=preview_features.shape[0],
            device=device,
        )

    sim_config = SimulationConfig(
        initial_sol=1.0,
        max_position_pct=1.0,
        max_position_sol=args.max_position_sol,
        min_trade_value_sol=args.min_trade_sol,
        costs=CostConfig(),
    )

    simulator = MarketSimulator(sim_config)
    simulator.reset()

    holding = False

    if args.auto_thresholds and start_idx > 0:
        train_df = df.iloc[:start_idx].reset_index(drop=True)
        if len(train_df) > context:
            train_probs = []
            train_opens = train_df["open"].to_numpy(dtype=np.float32)
            train_highs = train_df["high"].to_numpy(dtype=np.float32)
            train_lows = train_df["low"].to_numpy(dtype=np.float32)
            train_closes = train_df["close"].to_numpy(dtype=np.float32)
            for idx in range(context, len(train_df)):
                window_start = idx - context
                feats = build_window_features(
                    train_opens[window_start:idx],
                    train_highs[window_start:idx],
                    train_lows[window_start:idx],
                    train_closes[window_start:idx],
                )
                feats = normalizer.transform(feats)[None, :]
                x = torch.tensor(feats, dtype=torch.float32, device=device)
                with torch.no_grad():
                    signal_logit, _ = model(x)
                    train_probs.append(torch.sigmoid(signal_logit).item())
            if train_probs:
                args.buy_threshold = float(np.quantile(train_probs, args.buy_quantile))
                args.sell_threshold = float(np.quantile(train_probs, args.sell_quantile))

    if start_idx > 0:
        df = df.iloc[max(start_idx - context, 0) :].reset_index(drop=True)

    for idx in range(context, len(df)):
        window_start = idx - context
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
        simulator._prices[args.mint] = current_price
        simulator.state.timestamp = timestamps[idx]

        if not holding and prob >= args.buy_threshold:
            trade_amount = size * args.max_position_sol
            if trade_amount >= args.min_trade_sol:
                trade = simulator.open_position(
                    token=simulator_token(args.mint, df),
                    sol_amount=trade_amount,
                    price_sol=current_price,
                    timestamp=timestamps[idx].astype("datetime64[ns]").astype(object),
                )
                holding = trade is not None

        elif holding and prob <= args.sell_threshold:
            trade = simulator.close_position(
                token=simulator_token(args.mint, df),
                price_sol=current_price,
                timestamp=timestamps[idx].astype("datetime64[ns]").astype(object),
            )
            holding = False if trade is not None else holding

        portfolio_value = simulator.state.get_portfolio_value(simulator._prices)
        simulator.state.equity_history.append(
            (
                timestamps[idx].astype("datetime64[ns]").astype(object),
                portfolio_value,
            )
        )

    result = simulator._compute_results()
    print("\n=== Neural Backtest Summary ===")
    print(f"Token mint: {args.mint}")
    print(f"Total return: {result.total_return_pct:.2f}%")
    print(f"Sharpe: {result.sharpe_ratio:.2f} | Sortino: {result.sortino_ratio:.2f}")
    print(f"Max drawdown: {result.max_drawdown:.2%}")
    print(f"Trades: {result.total_trades} | Win rate: {result.win_rate:.2%}")


def simulator_token(mint: str, df) -> object:
    symbol = str(df.iloc[-1]["token_symbol"]) if "token_symbol" in df.columns else mint[:6]
    from bagsfm.config import TokenConfig

    return TokenConfig(symbol=symbol, mint=mint, decimals=9)


if __name__ == "__main__":
    main()
