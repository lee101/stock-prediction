#!/usr/bin/env python3
"""Optimize a simple linear allocation model on full-length sequences.

Instead of training a transformer on 72-bar windows (which fails at lag=2),
directly optimize allocation = tanh(w @ features + bias) on the full val
period using LBFGS or Adam. The EMA crossover analysis showed that slow
trend-following works at lag=2, so a linear model on long-horizon features
should capture this.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import json
import numpy as np
import torch
import torch.nn.functional as F
from datetime import datetime

from binanceneural.config import DatasetConfig
from binanceneural.data import BinanceHourlyDataModule


def compute_sortino(returns: torch.Tensor, target: float = 0.0) -> torch.Tensor:
    excess = returns - target / returns.shape[-1]
    mean_ret = excess.mean(dim=-1)
    downside = torch.clamp(excess, max=0.0)
    downside_var = (downside ** 2).mean(dim=-1)
    downside_std = torch.sqrt(downside_var + 1e-8)
    return mean_ret / downside_std


def simulate_rebalance_vec(
    closes: torch.Tensor,
    allocation: torch.Tensor,
    maker_fee: float = 0.001,
    decision_lag_bars: int = 0,
    max_leverage: float = 1.0,
    margin_annual_rate: float = 0.0,
) -> dict:
    """Vectorized rebalance sim supporting long-short."""
    if decision_lag_bars > 0:
        lag = decision_lag_bars
        closes = closes[lag:]
        allocation = allocation[:-lag]

    alloc = torch.clamp(allocation, min=-max_leverage, max=max_leverage)
    asset_returns = (closes[1:] - closes[:-1]) / torch.clamp(closes[:-1], min=1e-8)
    alloc_t = alloc[:-1]
    alloc_delta = torch.abs(alloc[1:] - alloc[:-1])
    initial_fee = torch.abs(alloc[:1]) * maker_fee
    turnover_fee = alloc_delta * maker_fee
    margin_per_step = margin_annual_rate / 8766.0
    margin_cost = torch.abs(alloc_t) * margin_per_step
    portfolio_returns = alloc_t * asset_returns - turnover_fee - margin_cost
    returns = torch.cat([-initial_fee, portfolio_returns], dim=-1)
    cum_log = torch.cumsum(torch.log1p(torch.clamp(returns, min=-0.99)), dim=-1)
    values = torch.exp(cum_log)
    final_ret = values[-1].item() - 1.0
    max_dd = (1.0 - values / torch.cummax(values, dim=0)[0]).max().item()
    sortino = compute_sortino(returns.unsqueeze(0)).item()
    turnover = alloc_delta.mean().item()
    return {
        "return": final_ret,
        "sortino": sortino,
        "max_dd": max_dd,
        "turnover": turnover,
        "mean_alloc": alloc.mean().item(),
        "std_alloc": alloc.std().item(),
    }


def extract_full_sequence(data_module, split="val"):
    """Extract full-length features and prices from data module."""
    ds = data_module.val_dataset if split == "val" else data_module.train_dataset
    n = len(ds.closes)
    features = torch.from_numpy(ds.features[:n]).float()
    closes = torch.from_numpy(ds.closes[:n]).float()
    return features, closes


def optimize_linear(
    features: torch.Tensor,
    closes: torch.Tensor,
    allow_short: bool = True,
    maker_fee: float = 0.001,
    decision_lag: int = 2,
    max_leverage: float = 1.0,
    margin_rate: float = 0.0625,
    smooth_weight: float = 0.0,
    l2_weight: float = 0.01,
    optimizer_name: str = "adam",
    lr: float = 0.01,
    steps: int = 2000,
    init_bias: float = 0.0,
):
    T, D = features.shape
    w = torch.zeros(D, requires_grad=True)
    b = torch.tensor([init_bias], requires_grad=True)
    params = [w, b]

    if optimizer_name == "lbfgs":
        optimizer = torch.optim.LBFGS(params, lr=lr, max_iter=20, line_search_fn="strong_wolfe")
    else:
        optimizer = torch.optim.Adam(params, lr=lr)

    best_sortino = -999.0
    best_w = None
    best_b = None

    for step in range(steps):
        def closure():
            optimizer.zero_grad()
            logits = features @ w + b
            if allow_short:
                alloc = torch.tanh(logits)
            else:
                alloc = torch.sigmoid(logits)
            asset_returns = (closes[1:] - closes[:-1]) / torch.clamp(closes[:-1], min=1e-8)
            if decision_lag > 0:
                ar = asset_returns[decision_lag:]
                al = alloc[:-(decision_lag + 1)]
            else:
                al = alloc[:-1]
                ar = asset_returns
            alloc_full = alloc[:-1] if decision_lag == 0 else alloc[:-(decision_lag)]
            alloc_delta = torch.abs(alloc_full[1:] - alloc_full[:-1])
            turnover_cost = alloc_delta * maker_fee
            margin_cost = torch.abs(al) * (margin_rate / 8766.0)
            port_returns = al * ar
            min_len = min(port_returns.shape[0], turnover_cost.shape[0], margin_cost.shape[0])
            port_returns = port_returns[:min_len]
            turnover_cost = turnover_cost[:min_len]
            margin_cost = margin_cost[:min_len]
            net_returns = port_returns - turnover_cost - margin_cost
            excess = net_returns
            mean_ret = excess.mean()
            downside = torch.clamp(excess, max=0.0)
            downside_var = (downside ** 2).mean()
            downside_std = torch.sqrt(downside_var + 1e-8)
            sortino = mean_ret / downside_std
            loss = -sortino
            if smooth_weight > 0:
                loss = loss + smooth_weight * alloc_delta.mean()
            if l2_weight > 0:
                loss = loss + l2_weight * (w ** 2).sum()
            loss.backward()
            return loss

        if optimizer_name == "lbfgs":
            loss = optimizer.step(closure)
        else:
            loss = closure()
            optimizer.step()

        if step % 100 == 0 or step == steps - 1:
            with torch.no_grad():
                logits = features @ w + b
                alloc = torch.tanh(logits) if allow_short else torch.sigmoid(logits)
                result = simulate_rebalance_vec(
                    closes, alloc, maker_fee=maker_fee,
                    decision_lag_bars=decision_lag, max_leverage=max_leverage,
                    margin_annual_rate=margin_rate,
                )
                s = result["sortino"]
                if s > best_sortino:
                    best_sortino = s
                    best_w = w.detach().clone()
                    best_b = b.detach().clone()
                print(f"  step {step:4d}: sort={s:+.3f} ret={result['return']:+.2%} "
                      f"dd={result['max_dd']:.2%} turn={result['turnover']:.4f} "
                      f"alloc={result['mean_alloc']:+.3f}+/-{result['std_alloc']:.3f} "
                      f"loss={loss.item() if isinstance(loss, torch.Tensor) else loss:.4f}")

    return best_w, best_b, best_sortino


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", default="BTCUSD")
    parser.add_argument("--decision-lag", type=int, default=2)
    parser.add_argument("--allow-short", action="store_true", default=True)
    parser.add_argument("--long-only", action="store_true")
    parser.add_argument("--fee", type=float, default=0.001)
    parser.add_argument("--margin-rate", type=float, default=0.0625)
    parser.add_argument("--max-leverage", type=float, default=1.0)
    parser.add_argument("--smooth", type=float, default=0.0)
    parser.add_argument("--l2", type=float, default=0.01)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--optimizer", default="adam", choices=["adam", "lbfgs"])
    parser.add_argument("--init-bias", type=float, default=0.0)
    parser.add_argument("--val-days", type=int, default=70)
    parser.add_argument("--seq-len", type=int, default=72)
    args = parser.parse_args()

    allow_short = not args.long_only

    dataset_cfg = DatasetConfig(
        symbol=args.symbol,
        data_root=Path("trainingdatahourly/crypto"),
        forecast_cache_root=Path("binanceneural/forecast_cache"),
        sequence_length=args.seq_len,
        validation_days=args.val_days,
        forecast_horizons=(1, 24),
        cache_only=True,
    )
    data = BinanceHourlyDataModule(dataset_cfg)
    features, closes = extract_full_sequence(data, "val")
    print(f"{args.symbol}: {features.shape[0]} val bars, {features.shape[1]} features")
    print(f"Features: {list(data.feature_columns)}")
    print(f"Price range: {closes.min():.0f} - {closes.max():.0f}")
    total_ret = (closes[-1] / closes[0] - 1).item()
    print(f"Buy-and-hold return: {total_ret:+.2%}")
    print()

    # Grid search over initial biases
    biases_to_try = [0.0]
    if allow_short:
        biases_to_try = [-1.0, -0.5, 0.0, 0.5, 1.0]

    all_results = []
    for init_bias in biases_to_try:
        print(f"=== init_bias={init_bias:+.1f}, short={'yes' if allow_short else 'no'}, lag={args.decision_lag} ===")
        w, b, sort = optimize_linear(
            features, closes,
            allow_short=allow_short,
            maker_fee=args.fee,
            decision_lag=args.decision_lag,
            max_leverage=args.max_leverage,
            margin_rate=args.margin_rate,
            smooth_weight=args.smooth,
            l2_weight=args.l2,
            optimizer_name=args.optimizer,
            lr=args.lr,
            steps=args.steps,
            init_bias=init_bias,
        )
        with torch.no_grad():
            logits = features @ w + b
            alloc = torch.tanh(logits) if allow_short else torch.sigmoid(logits)
            for lag in [0, 1, 2, 3]:
                r = simulate_rebalance_vec(
                    closes, alloc, maker_fee=args.fee,
                    decision_lag_bars=lag, max_leverage=args.max_leverage,
                    margin_annual_rate=args.margin_rate,
                )
                print(f"  lag={lag}: ret={r['return']:+.2%} sort={r['sortino']:+.3f} "
                      f"dd={r['max_dd']:.2%} turn={r['turnover']:.4f}")

        # Show top feature weights
        if w is not None:
            feature_names = list(data.feature_columns)
            w_abs = w.abs()
            top_idx = w_abs.argsort(descending=True)[:10]
            print("  Top features:")
            for i in top_idx:
                print(f"    {feature_names[i]:30s} w={w[i].item():+.4f}")

        all_results.append({
            "init_bias": init_bias,
            "allow_short": allow_short,
            "best_sortino_lag2": sort,
            "weights": w.tolist() if w is not None else None,
            "bias": b.item() if b is not None else None,
        })
        print()

    # Also test on train data for comparison
    print("=== Train data evaluation (best model) ===")
    best_result = max(all_results, key=lambda x: x["best_sortino_lag2"])
    w = torch.tensor(best_result["weights"])
    b = torch.tensor([best_result["bias"]])
    train_feats, train_closes = extract_full_sequence(data, "train")
    with torch.no_grad():
        logits = train_feats @ w + b
        alloc = torch.tanh(logits) if allow_short else torch.sigmoid(logits)
        for lag in [0, 1, 2]:
            r = simulate_rebalance_vec(
                train_closes, alloc, maker_fee=args.fee,
                decision_lag_bars=lag, max_leverage=args.max_leverage,
                margin_annual_rate=args.margin_rate,
            )
            print(f"  train lag={lag}: ret={r['return']:+.2%} sort={r['sortino']:+.3f} "
                  f"dd={r['max_dd']:.2%} turn={r['turnover']:.4f}")


def extract_full_sequence(data_module, split="val"):
    ds = data_module.val_dataset if split == "val" else data_module.train_dataset
    n = len(ds.closes)
    features = torch.from_numpy(ds.features[:n]).float()
    closes = torch.from_numpy(ds.closes[:n]).float()
    return features, closes


if __name__ == "__main__":
    main()
