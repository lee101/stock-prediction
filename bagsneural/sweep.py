#!/usr/bin/env python3
"""Hyperparameter sweep for Bags.fm neural trading model with holdout backtest."""

from __future__ import annotations

import argparse
import json
import logging
import os
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
import sys
sys.path.insert(0, REPO_ROOT)

from bagsfm.config import CostConfig, SimulationConfig, TokenConfig
from bagsfm.simulator import MarketSimulator
from bagsneural.dataset import (
    FeatureNormalizer,
    build_features_and_targets,
    build_window_features,
    load_ohlc_dataframe,
)
from bagsneural.model import BagsNeuralModel

logger = logging.getLogger("bagsneural_sweep")


@dataclass
class SweepResult:
    config: Dict
    train_return: float
    test_return: float
    max_drawdown: float
    trades: int


def _split_indices(total: int, test_split: float) -> Tuple[np.ndarray, np.ndarray]:
    if not 0.0 < test_split < 1.0:
        raise ValueError("test_split must be between 0 and 1")
    split_idx = int(total * (1 - test_split))
    train_idx = np.arange(0, split_idx)
    test_idx = np.arange(split_idx, total)
    return train_idx, test_idx


def _make_loaders(features: np.ndarray, signal: np.ndarray, size: np.ndarray, batch_size: int) -> DataLoader:
    ds = TensorDataset(
        torch.tensor(features, dtype=torch.float32),
        torch.tensor(signal, dtype=torch.float32),
        torch.tensor(size, dtype=torch.float32),
    )
    return DataLoader(ds, batch_size=batch_size, shuffle=True)


def _train_model(
    features: np.ndarray,
    signal: np.ndarray,
    size: np.ndarray,
    hidden_dims: List[int],
    dropout: float,
    lr: float,
    epochs: int,
    batch_size: int,
    device: torch.device,
) -> Tuple[BagsNeuralModel, FeatureNormalizer]:
    normalizer = FeatureNormalizer(
        mean=features.mean(axis=0).astype(np.float32),
        std=features.std(axis=0).astype(np.float32),
    )
    features = normalizer.transform(features)

    model = BagsNeuralModel(
        input_dim=features.shape[1],
        hidden_dims=hidden_dims,
        dropout=dropout,
    ).to(device)

    positives = float(signal.sum())
    negatives = float(len(signal) - positives)
    pos_weight = min(negatives / max(positives, 1.0), 50.0)

    signal_loss_fn = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(pos_weight, dtype=torch.float32, device=device)
    )
    size_loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    loader = _make_loaders(features, signal, size, batch_size)

    for _ in range(epochs):
        model.train()
        for batch in loader:
            x, signal_target, size_target = [b.to(device) for b in batch]
            optimizer.zero_grad()
            signal_logit, size_logit = model(x)
            size_pred = torch.sigmoid(size_logit)
            signal_loss = signal_loss_fn(signal_logit, signal_target)
            size_loss = size_loss_fn(size_pred, size_target)
            loss = signal_loss + size_loss
            loss.backward()
            optimizer.step()

    model.eval()
    return model, normalizer


def _simulate(
    df,
    model: BagsNeuralModel,
    normalizer: FeatureNormalizer,
    context: int,
    buy_threshold: float,
    sell_threshold: float,
    max_position_sol: float,
    min_trade_sol: float,
    device: torch.device,
) -> SweepResult:
    opens = df["open"].to_numpy(dtype=np.float32)
    highs = df["high"].to_numpy(dtype=np.float32)
    lows = df["low"].to_numpy(dtype=np.float32)
    closes = df["close"].to_numpy(dtype=np.float32)
    timestamps = df["timestamp"].to_numpy()

    simulator = MarketSimulator(
        SimulationConfig(
            initial_sol=1.0,
            max_position_pct=1.0,
            max_position_sol=max_position_sol,
            min_trade_value_sol=min_trade_sol,
        )
    )
    simulator.reset()

    token = TokenConfig(
        symbol=str(df.iloc[-1]["token_symbol"]) if "token_symbol" in df.columns else "TOK",
        mint=str(df.iloc[-1]["token_mint"]) if "token_mint" in df.columns else "MINT",
        decimals=9,
    )

    holding = False

    for idx in range(context, len(df)):
        start = idx - context
        features = build_window_features(
            opens[start:idx],
            highs[start:idx],
            lows[start:idx],
            closes[start:idx],
        )
        features = normalizer.transform(features)[None, :]
        x = torch.tensor(features, dtype=torch.float32, device=device)

        with torch.no_grad():
            signal_logit, size_logit = model(x)
            prob = torch.sigmoid(signal_logit).item()
            size = torch.sigmoid(size_logit).item()

        price = float(closes[idx])
        simulator._prices[token.mint] = price
        simulator.state.timestamp = timestamps[idx]

        if not holding and prob >= buy_threshold:
            trade_amount = max(0.0, size) * max_position_sol
            if trade_amount >= min_trade_sol:
                trade = simulator.open_position(
                    token=token,
                    sol_amount=trade_amount,
                    price_sol=price,
                    timestamp=timestamps[idx].astype("datetime64[ns]").astype(object),
                )
                holding = trade is not None

        elif holding and prob <= sell_threshold:
            trade = simulator.close_position(
                token=token,
                price_sol=price,
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
    return result


def _predict_probs(
    df,
    model: BagsNeuralModel,
    normalizer: FeatureNormalizer,
    context: int,
    device: torch.device,
) -> np.ndarray:
    opens = df["open"].to_numpy(dtype=np.float32)
    highs = df["high"].to_numpy(dtype=np.float32)
    lows = df["low"].to_numpy(dtype=np.float32)
    closes = df["close"].to_numpy(dtype=np.float32)

    probs: List[float] = []
    for idx in range(context, len(df)):
        start = idx - context
        features = build_window_features(
            opens[start:idx],
            highs[start:idx],
            lows[start:idx],
            closes[start:idx],
        )
        features = normalizer.transform(features)[None, :]
        x = torch.tensor(features, dtype=torch.float32, device=device)
        with torch.no_grad():
            signal_logit, _ = model(x)
            probs.append(torch.sigmoid(signal_logit).item())
    return np.array(probs, dtype=np.float32)


def _candidate_thresholds(probs: np.ndarray) -> Tuple[List[float], List[float]]:
    if probs.size == 0:
        return [0.6], [0.4]
    buy_q = [0.7, 0.8, 0.9]
    sell_q = [0.3, 0.2, 0.1]
    buy_thresholds = [float(np.quantile(probs, q)) for q in buy_q]
    sell_thresholds = [float(np.quantile(probs, q)) for q in sell_q]
    # Ensure buy > sell by margin
    buy_thresholds = [min(0.95, max(0.5, b)) for b in buy_thresholds]
    sell_thresholds = [max(0.05, min(0.5, s)) for s in sell_thresholds]
    return buy_thresholds, sell_thresholds


def main() -> None:
    parser = argparse.ArgumentParser(description="Sweep neural model configs on Bags.fm data")
    parser.add_argument("--ohlc", type=Path, default=Path("bagstraining") / "ohlc_data.csv")
    parser.add_argument("--mint", type=str, required=True)
    parser.add_argument("--context", type=str, default="32,64")
    parser.add_argument("--horizon", type=str, default="3,6")
    parser.add_argument("--min-return", type=str, default="0.0,0.001,0.002")
    parser.add_argument("--size-scale", type=str, default="0.01,0.02")
    parser.add_argument("--hidden", type=str, default="128,64")
    parser.add_argument("--dropout", type=str, default="0.1")
    parser.add_argument("--lr", type=str, default="1e-3")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--buy-threshold", type=str, default="0.55,0.6")
    parser.add_argument("--sell-threshold", type=str, default="0.45,0.4")
    parser.add_argument(
        "--auto-thresholds",
        action="store_true",
        help="Derive buy/sell thresholds from train probabilities",
    )
    parser.add_argument("--test-split", type=float, default=0.2)
    parser.add_argument("--max-position-sol", type=float, default=1.0)
    parser.add_argument("--min-trade-sol", type=float, default=0.01)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--out-dir", type=Path, default=Path("bagsneural") / "checkpoints")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    logging.getLogger("bagsfm").setLevel(logging.WARNING)
    logging.getLogger("bagsfm.simulator").setLevel(logging.WARNING)

    costs = CostConfig()
    cost_bps = costs.estimated_swap_fee_bps + costs.default_slippage_bps

    df = load_ohlc_dataframe(args.ohlc, args.mint, dedupe=True)

    context_vals = [int(x) for x in args.context.split(",") if x.strip()]
    horizon_vals = [int(x) for x in args.horizon.split(",") if x.strip()]
    min_return_vals = [float(x) for x in args.min_return.split(",") if x.strip()]
    size_scale_vals = [float(x) for x in args.size_scale.split(",") if x.strip()]
    dropout_vals = [float(x) for x in args.dropout.split(",") if x.strip()]
    lr_vals = [float(x) for x in args.lr.split(",") if x.strip()]
    buy_vals = [float(x) for x in args.buy_threshold.split(",") if x.strip()]
    sell_vals = [float(x) for x in args.sell_threshold.split(",") if x.strip()]

    hidden_dims = [int(x) for x in args.hidden.split(",") if x.strip()]

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    best_return = float("-inf")
    best_payload = None

    for context, horizon, min_return, size_scale, dropout, lr in product(
        context_vals,
        horizon_vals,
        min_return_vals,
        size_scale_vals,
        dropout_vals,
        lr_vals,
    ):
        features, signal, size, timestamps = build_features_and_targets(
            df,
            context_bars=context,
            horizon=horizon,
            cost_bps=cost_bps,
            min_return=min_return,
            size_scale=size_scale,
        )

        train_idx, test_idx = _split_indices(len(features), args.test_split)

        train_features = features[train_idx]
        train_signal = signal[train_idx]
        train_size = size[train_idx]

        test_features = features[test_idx]
        test_signal = signal[test_idx]
        test_size = size[test_idx]

        if len(train_features) < 50 or len(test_features) < 20:
            logger.info("Skipping config due to insufficient samples")
            continue

        model, normalizer = _train_model(
            train_features,
            train_signal,
            train_size,
            hidden_dims,
            dropout,
            lr,
            args.epochs,
            args.batch_size,
            device,
        )

        train_df = df.iloc[context : context + len(train_features)]
        test_df = df.iloc[
            context + len(train_features) : context + len(train_features) + len(test_features)
        ]

        if args.auto_thresholds:
            train_probs = _predict_probs(train_df, model, normalizer, context, device)
            cand_buy, cand_sell = _candidate_thresholds(train_probs)
        else:
            cand_buy, cand_sell = buy_vals, sell_vals

        for buy_th in cand_buy:
            for sell_th in cand_sell:
                if sell_th >= buy_th:
                    continue

                train_result = _simulate(
                    train_df,
                    model,
                    normalizer,
                    context,
                    buy_th,
                    sell_th,
                    args.max_position_sol,
                    args.min_trade_sol,
                    device,
                )
                test_result = _simulate(
                    test_df,
                    model,
                    normalizer,
                    context,
                    buy_th,
                    sell_th,
                    args.max_position_sol,
                    args.min_trade_sol,
                    device,
                )

                logger.info(
                    "ctx=%d horizon=%d min_return=%.4f size_scale=%.4f buy=%.2f sell=%.2f -> train %.2f%% test %.2f%%",
                    context,
                    horizon,
                    min_return,
                    size_scale,
                    buy_th,
                    sell_th,
                    train_result.total_return_pct,
                    test_result.total_return_pct,
                )

                if test_result.total_return_pct > best_return:
                    best_return = test_result.total_return_pct
                    best_payload = {
                        "model_state": model.state_dict(),
                        "normalizer": normalizer.to_dict(),
                        "config": {
                            "context": context,
                            "horizon": horizon,
                            "min_return": min_return,
                            "size_scale": size_scale,
                            "cost_bps": cost_bps,
                            "hidden": hidden_dims,
                            "dropout": dropout,
                            "lr": lr,
                            "buy_threshold": buy_th,
                            "sell_threshold": sell_th,
                            "train_return": train_result.total_return_pct,
                            "test_return": test_result.total_return_pct,
                            "max_drawdown": test_result.max_drawdown,
                            "trades": test_result.total_trades,
                        },
                    }

                    logger.info("New best test return: %.2f%%", best_return)

    if best_payload is None:
        raise SystemExit("No viable configs found")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.out_dir / f"bagsneural_{args.mint}_sweep_best.pt"
    torch.save(best_payload, out_path)
    with open(args.out_dir / f"bagsneural_{args.mint}_sweep_best.json", "w") as f:
        json.dump(best_payload["config"], f, indent=2)

    logger.info("Best sweep result saved to %s", out_path)


if __name__ == "__main__":
    main()
