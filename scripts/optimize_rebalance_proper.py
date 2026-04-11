#!/usr/bin/env python3
"""Proper train->val linear rebalance optimization.

Previous linear optimization was on val data (overfit). This one:
1. Optimizes on TRAIN data
2. Tests on VAL data (true out-of-sample)
3. Uses EMA features computed in-sample
4. Tests walk-forward robustness
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
from binanceneural.config import DatasetConfig
from binanceneural.data import BinanceHourlyDataModule


def compute_sortino_torch(returns, eps=1e-8):
    mean_ret = returns.mean()
    downside = torch.clamp(returns, max=0.0)
    downside_std = torch.sqrt((downside ** 2).mean() + eps)
    return mean_ret / downside_std


def sim_rebalance_vec(closes, alloc, fee=0.001, lag=0, margin_rate=0.0625):
    if lag > 0:
        closes = closes[lag:]
        alloc = alloc[:-lag]
    asset_ret = (closes[1:] - closes[:-1]) / torch.clamp(closes[:-1], min=1e-8)
    al = alloc[:-1]
    al_delta = torch.abs(alloc[1:] - alloc[:-1])
    margin_per_step = margin_rate / 8766.0
    port_ret = al * asset_ret - al_delta * fee - torch.abs(al) * margin_per_step
    init_fee = torch.abs(alloc[:1]) * fee
    port_ret[0] = port_ret[0] - init_fee.squeeze()
    return port_ret


def sim_rebalance_np(closes, alloc, fee=0.001, lag=0, margin_rate=0.0625):
    if lag > 0:
        closes = closes[lag:]
        alloc = alloc[:-lag]
    asset_ret = np.diff(closes) / np.maximum(closes[:-1], 1e-8)
    al = alloc[:-1]
    al_delta = np.abs(np.diff(alloc))
    margin_per_step = margin_rate / 8766.0
    port_ret = al * asset_ret - al_delta[:len(al)] * fee - np.abs(al) * margin_per_step
    port_ret[0] -= np.abs(alloc[0]) * fee
    cum_log = np.cumsum(np.log1p(np.clip(port_ret, -0.99, None)))
    values = np.exp(cum_log)
    final_ret = values[-1] - 1.0
    running_max = np.maximum.accumulate(values)
    max_dd = np.max(1.0 - values / running_max)
    mean_ret = port_ret.mean()
    downside = np.minimum(port_ret, 0.0)
    downside_std = np.sqrt((downside ** 2).mean() + 1e-10)
    sortino = mean_ret / downside_std
    turnover = np.abs(np.diff(alloc[:len(al)+1])).mean()
    return {"return": final_ret, "sortino": sortino, "max_dd": max_dd, "turnover": turnover,
            "mean_alloc": al.mean(), "n_bars": len(port_ret)}


def ema_np(data, span):
    alpha = 2.0 / (span + 1)
    out = np.empty_like(data)
    out[0] = data[0]
    for i in range(1, len(data)):
        out[i] = alpha * data[i] + (1 - alpha) * out[i - 1]
    return out


def add_ema_features(closes, features_np):
    """Add EMA crossover features to existing features."""
    for fast, slow in [(48, 336), (168, 720), (24, 168)]:
        ef = ema_np(closes, fast)
        es = ema_np(closes, slow)
        cross = (ef - es) / np.maximum(es, 1e-8)
        features_np = np.column_stack([features_np, cross])
    # Rolling vol
    ret = np.diff(closes, prepend=closes[0])
    vol = np.zeros(len(closes))
    for i in range(168, len(closes)):
        vol[i] = np.std(ret[i-168:i])
    vol[:168] = vol[168] if len(closes) > 168 else 0.01
    features_np = np.column_stack([features_np, vol])
    return features_np


def train_linear(train_features, train_closes, lag=2, fee=0.001, margin_rate=0.0625,
                 allow_short=True, lr=0.005, steps=2000, l2=0.01, smooth_w=0.0):
    T, D = train_features.shape
    features_t = torch.from_numpy(train_features).float()
    closes_t = torch.from_numpy(train_closes).float()
    w = torch.zeros(D, requires_grad=True)
    b = torch.zeros(1, requires_grad=True)
    optimizer = torch.optim.Adam([w, b], lr=lr)
    best_sort = -999
    best_w, best_b = None, None
    for step in range(steps):
        optimizer.zero_grad()
        logits = features_t @ w + b
        alloc = torch.tanh(logits) if allow_short else torch.sigmoid(logits)
        port_ret = sim_rebalance_vec(closes_t, alloc, fee=fee, lag=lag, margin_rate=margin_rate)
        sortino = compute_sortino_torch(port_ret)
        loss = -sortino + l2 * (w ** 2).sum()
        if smooth_w > 0:
            loss = loss + smooth_w * torch.abs(alloc[1:] - alloc[:-1]).mean()
        loss.backward()
        optimizer.step()
        if step % 500 == 0:
            s = sortino.item()
            if s > best_sort:
                best_sort = s
                best_w = w.detach().clone()
                best_b = b.detach().clone()
            with torch.no_grad():
                a = alloc.detach().numpy()
                print(f"  step {step:4d}: sort={s:+.4f} alloc={a.mean():+.3f}+/-{a.std():.3f}")
    # Final check
    with torch.no_grad():
        s = compute_sortino_torch(sim_rebalance_vec(closes_t, torch.tanh(features_t @ w + b) if allow_short else torch.sigmoid(features_t @ w + b), fee=fee, lag=lag, margin_rate=margin_rate)).item()
        if s > best_sort:
            best_sort = s
            best_w = w.detach().clone()
            best_b = b.detach().clone()
    return best_w.numpy(), best_b.numpy(), best_sort


def main():
    symbols = ["BTCUSD", "ETHUSD", "SOLUSD", "DOGEUSD", "LINKUSD"]

    for symbol in symbols:
        try:
            cfg = DatasetConfig(
                symbol=symbol,
                data_root=Path("trainingdatahourly/crypto"),
                forecast_cache_root=Path("binanceneural/forecast_cache"),
                sequence_length=72, validation_days=180,
                forecast_horizons=(1, 24), cache_only=True,
            )
            data = BinanceHourlyDataModule(cfg)
        except Exception as e:
            print(f"{symbol}: FAIL {e}")
            continue

        train_ds = data.train_dataset
        val_ds = data.val_dataset
        train_closes = train_ds.closes.copy()
        val_closes = val_ds.closes.copy()
        train_feats = train_ds.features.copy()
        val_feats = val_ds.features.copy()

        # Add EMA features (computed from closes, in-sample)
        train_feats = add_ema_features(train_closes, train_feats)
        val_feats = add_ema_features(val_closes, val_feats)
        # Normalize EMA features using train stats
        for col in range(len(data.feature_columns), train_feats.shape[1]):
            mu = train_feats[:, col].mean()
            sd = train_feats[:, col].std() + 1e-8
            train_feats[:, col] = (train_feats[:, col] - mu) / sd
            val_feats[:, col] = (val_feats[:, col] - mu) / sd

        train_ret = (train_closes[-1]/train_closes[0]-1)*100
        val_ret = (val_closes[-1]/val_closes[0]-1)*100

        print(f"\n{'='*60}")
        print(f"{symbol}: train={len(train_closes)} bars ({train_ret:+.1f}%), val={len(val_closes)} bars ({val_ret:+.1f}%)")
        print(f"Features: {train_feats.shape[1]} (base={len(data.feature_columns)} + EMA/vol)")
        print(f"{'='*60}")

        for allow_short in [True, False]:
            for l2 in [0.01, 0.1]:
                mode = "long-short" if allow_short else "long-only"
                print(f"\n--- {mode}, l2={l2} ---")
                print("Training on TRAIN data:")
                w, b, train_sort = train_linear(
                    train_feats, train_closes, lag=2,
                    allow_short=allow_short, lr=0.005, steps=3000, l2=l2,
                )

                # Evaluate on train
                alloc_fn = np.tanh if allow_short else lambda x: 1/(1+np.exp(-x))
                train_alloc = alloc_fn(train_feats @ w + b)
                r_train = sim_rebalance_np(train_closes, train_alloc.flatten(), lag=2)
                print(f"  TRAIN lag=2: ret={r_train['return']:+.2%} sort={r_train['sortino']:+.4f} dd={r_train['max_dd']:.2%}")

                # Evaluate on val (OUT-OF-SAMPLE)
                val_alloc = alloc_fn(val_feats @ w + b)
                print("  VAL (out-of-sample):")
                for lag in [0, 1, 2, 3]:
                    r_val = sim_rebalance_np(val_closes, val_alloc.flatten(), lag=lag)
                    print(f"    lag={lag}: ret={r_val['return']:+.2%} sort={r_val['sortino']:+.4f} "
                          f"dd={r_val['max_dd']:.2%} turn={r_val['turnover']:.4f} alloc={r_val['mean_alloc']:+.3f}")

                # Top feature weights
                feat_names = list(data.feature_columns) + [
                    "ema_48_336", "ema_168_720", "ema_24_168", "rolling_vol"
                ]
                w_flat = w.flatten()
                top_idx = np.argsort(np.abs(w_flat))[::-1][:8]
                print("  Top features:")
                for i in top_idx:
                    name = feat_names[i] if i < len(feat_names) else f"feat_{i}"
                    print(f"    {name:30s} w={w_flat[i]:+.4f}")

        # EMA baseline for comparison
        print(f"\n--- EMA(48,336) baseline ---")
        for lag in [0, 2]:
            ef = ema_np(val_closes, 48)
            es = ema_np(val_closes, 336)
            signal = (ef - es) / np.maximum(es, 1e-8)
            alloc_ema = np.tanh(signal * 1000)  # binary-ish
            r = sim_rebalance_np(val_closes, alloc_ema, lag=lag)
            print(f"  val lag={lag}: ret={r['return']:+.2%} sort={r['sortino']:+.4f} dd={r['max_dd']:.2%}")

            ef = ema_np(train_closes, 48)
            es = ema_np(train_closes, 336)
            signal = (ef - es) / np.maximum(es, 1e-8)
            alloc_ema = np.tanh(signal * 1000)
            r = sim_rebalance_np(train_closes, alloc_ema, lag=lag)
            print(f"  train lag={lag}: ret={r['return']:+.2%} sort={r['sortino']:+.4f} dd={r['max_dd']:.2%}")


if __name__ == "__main__":
    main()
