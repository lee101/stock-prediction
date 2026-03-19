#!/usr/bin/env python3
"""Learned meta-selector: trains a model to choose which per-symbol policy to follow.

Uses frozen per-symbol checkpoints to generate actions, then trains a selector
network that decides which symbol to trade at each hour. Much more effective than
softmax allocation for 2-asset crypto trading.
"""
from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from binancechronossolexperiment.data import ChronosSolDataModule, SplitConfig
from binancechronossolexperiment.inference import load_policy_checkpoint
from binanceneural.data import FeatureNormalizer
from binanceneural.inference import generate_actions_from_frame
from binanceleveragesui.run_leverage_sweep import LeverageConfig, simulate_with_margin_cost
from src.forecast_horizon_utils import resolve_required_forecast_horizons

REPO = Path(__file__).resolve().parents[1]
MARGIN_RATE = 0.0000025457


class MetaSelectorNet(nn.Module):
    """Small transformer that sees both symbols' features and picks which to trade."""

    def __init__(self, feature_dim: int, num_symbols: int = 2, hidden: int = 128,
                 num_heads: int = 4, num_layers: int = 2, seq_len: int = 24, dropout: float = 0.1):
        super().__init__()
        self.num_symbols = num_symbols
        self.num_choices = num_symbols + 1  # +1 for cash
        total_dim = feature_dim * num_symbols

        self.input_proj = nn.Linear(total_dim, hidden)
        self.pos_enc = nn.Embedding(seq_len, hidden)

        layer = nn.TransformerEncoderLayer(
            d_model=hidden, nhead=num_heads,
            dim_feedforward=hidden * 2, dropout=dropout, batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.head = nn.Linear(hidden, self.num_choices)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, seq_len, feature_dim * num_symbols) -> (batch, num_choices) logits"""
        b, s, _ = x.shape
        pos = torch.arange(s, device=x.device).unsqueeze(0).expand(b, -1)
        h = self.input_proj(x) + self.pos_enc(pos)
        h = self.encoder(h)
        return self.head(h[:, -1])


class MetaSelectorDataset(Dataset):
    """Dataset for meta-selector training.

    At each timestep, provides concatenated features from all symbols
    and per-symbol hourly returns for computing reward.
    """

    def __init__(
        self,
        features: Dict[str, np.ndarray],
        per_symbol_returns: Dict[str, np.ndarray],
        symbols: List[str],
        seq_len: int = 24,
    ):
        self.symbols = symbols
        self.seq_len = seq_len
        n = min(len(v) for v in features.values())
        self.n = n
        self.num_samples = max(0, n - seq_len)

        # stack features: (n, total_feat_dim)
        self.concat_features = np.concatenate(
            [features[sym] for sym in symbols], axis=1
        ).astype(np.float32)

        # stack returns: (n, num_symbols)
        self.returns = np.column_stack(
            [per_symbol_returns[sym] for sym in symbols]
        ).astype(np.float32)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        feat = torch.from_numpy(self.concat_features[idx : idx + self.seq_len])
        ret = torch.from_numpy(self.returns[idx + self.seq_len - 1])  # return at decision time
        return feat, ret


def load_symbol_data(
    symbols: List[str],
    checkpoints: Dict[str, str],
    data_root: Path,
    forecast_cache: Path,
    seq_len: int,
    val_days: int = 30,
    test_days: int = 30,
    max_history_days: int = 365,
    device: str = "cuda",
):
    """Load features and compute per-symbol returns using frozen per-symbol policies."""
    forecast_horizons = resolve_required_forecast_horizons((1,), fallback_horizons=(1,))
    all_features = {}
    all_returns = {}
    feat_cols = None
    normalizer = None

    for sym in symbols:
        print(f"loading {sym}...")
        ckpt_path = checkpoints[sym]
        model, norm, fcols, meta = load_policy_checkpoint(ckpt_path, device=device)
        model_seq = int(meta.get("sequence_length", 72))

        dm = ChronosSolDataModule(
            symbol=sym,
            data_root=data_root,
            forecast_cache_root=forecast_cache,
            forecast_horizons=forecast_horizons,
            context_hours=512,
            quantile_levels=(0.1, 0.5, 0.9),
            batch_size=32,
            model_id="amazon/chronos-t5-small",
            sequence_length=model_seq,
            split_config=SplitConfig(val_days=val_days, test_days=test_days),
            cache_only=True,
            max_history_days=max_history_days,
        )

        frame = dm.full_frame.copy()
        close = frame["close"].to_numpy(dtype=np.float32)
        hourly_ret = np.zeros_like(close)
        hourly_ret[1:] = (close[1:] - close[:-1]) / (close[:-1] + 1e-10)

        # extract features for selector (use raw feature columns from data, not per-symbol)
        if feat_cols is None:
            feat_cols = [c for c in frame.columns
                         if c not in ("timestamp", "symbol", "open", "high", "low", "close",
                                      "volume", "reference_close") and not c.startswith("predicted_")]
        available = [c for c in feat_cols if c in frame.columns]
        raw_feats = frame[available].to_numpy(dtype=np.float32)

        if normalizer is None:
            split_idx = int(len(raw_feats) * 0.85)
            normalizer = FeatureNormalizer.fit(raw_feats[:split_idx])

        all_features[sym] = normalizer.transform(raw_feats)
        all_returns[sym] = hourly_ret

        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # align timestamps
    frames_dict = {}
    for sym in symbols:
        dm = ChronosSolDataModule(
            symbol=sym,
            data_root=data_root,
            forecast_cache_root=forecast_cache,
            forecast_horizons=forecast_horizons,
            context_hours=512,
            quantile_levels=(0.1, 0.5, 0.9),
            batch_size=32,
            model_id="amazon/chronos-t5-small",
            sequence_length=72,
            split_config=SplitConfig(val_days=val_days, test_days=test_days),
            cache_only=True,
            max_history_days=max_history_days,
        )
        frames_dict[sym] = dm.full_frame

    ts_sets = [set(f["timestamp"].tolist()) for f in frames_dict.values()]
    common_ts = sorted(set.intersection(*ts_sets))
    print(f"common timestamps: {len(common_ts)}")

    # re-index to common timestamps
    aligned_features = {}
    aligned_returns = {}
    for sym in symbols:
        f = frames_dict[sym].set_index("timestamp")
        idx = [f.index.get_loc(t) for t in common_ts if t in f.index]
        aligned_features[sym] = all_features[sym][idx]
        aligned_returns[sym] = all_returns[sym][idx]

    return aligned_features, aligned_returns, feat_cols, normalizer, len(common_ts)


def compute_selector_loss(
    logits: torch.Tensor,  # (batch, num_choices)
    returns: torch.Tensor,  # (batch, num_symbols)
    temperature: float = 1.0,
) -> Tuple[torch.Tensor, dict]:
    """RL-style loss: maximize expected return of selected symbol."""
    num_symbols = returns.shape[1]
    # add cash return (0)
    cash_ret = torch.zeros(returns.shape[0], 1, device=returns.device)
    all_returns = torch.cat([returns, cash_ret], dim=1)  # (batch, num_choices)

    # soft selection (Gumbel-softmax for differentiability)
    probs = F.softmax(logits / temperature, dim=-1)
    expected_return = (probs * all_returns).sum(dim=-1)

    # reward: mean return - downside penalty
    mean_ret = expected_return.mean()
    downside = torch.clamp(-expected_return, min=0)
    downside_risk = (downside ** 2).mean().sqrt() + 1e-8
    sortino = mean_ret / downside_risk

    # entropy bonus
    entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=-1).mean()

    loss = -sortino - 0.05 * entropy

    # track which symbol is selected most
    choices = probs.argmax(dim=-1)
    choice_counts = [(choices == i).float().mean().item() for i in range(logits.shape[1])]

    return loss, {
        "loss": loss.item(),
        "sortino": sortino.item(),
        "mean_return": mean_ret.item() * 100,
        "entropy": entropy.item(),
        "choice_dist": choice_counts,
    }


def train_meta_selector(
    symbols: List[str],
    checkpoints: Dict[str, str],
    data_root: Path,
    forecast_cache: Path,
    checkpoint_root: Path,
    hidden: int = 128,
    num_layers: int = 2,
    num_heads: int = 4,
    seq_len: int = 24,
    batch_size: int = 128,
    lr: float = 3e-4,
    wd: float = 0.01,
    epochs: int = 50,
    val_days: int = 30,
    test_days: int = 30,
    device: str = "cuda",
):
    features, returns, feat_cols, normalizer, n_common = load_symbol_data(
        symbols, checkpoints, data_root, forecast_cache, seq_len,
        val_days=val_days, test_days=test_days, device=device,
    )

    # split
    split_idx = int(n_common * 0.85)
    train_feats = {s: features[s][:split_idx] for s in symbols}
    train_rets = {s: returns[s][:split_idx] for s in symbols}
    val_feats = {s: features[s][split_idx:] for s in symbols}
    val_rets = {s: returns[s][split_idx:] for s in symbols}

    train_ds = MetaSelectorDataset(train_feats, train_rets, symbols, seq_len)
    val_ds = MetaSelectorDataset(val_feats, val_rets, symbols, seq_len)
    print(f"train: {len(train_ds)}, val: {len(val_ds)}")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    feat_dim = len(feat_cols)
    model = MetaSelectorNet(
        feature_dim=feat_dim, num_symbols=len(symbols),
        hidden=hidden, num_heads=num_heads, num_layers=num_layers,
        seq_len=seq_len, dropout=0.1,
    ).to(device)

    param_count = sum(p.numel() for p in model.parameters())
    print(f"selector params: {param_count:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    tag = "_".join(s.replace("USD", "").lower() for s in symbols)
    ts = time.strftime("%Y%m%d_%H%M%S")
    ckpt_dir = checkpoint_root / f"meta_selector_{tag}_h{hidden}_{ts}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    best_val_sortino = float("-inf")
    best_ckpt = None

    for epoch in range(1, epochs + 1):
        model.train()
        tm = {"loss": 0, "sortino": 0, "mean_return": 0, "entropy": 0}
        nb = 0
        total_choices = [0.0] * (len(symbols) + 1)
        for feat, ret in train_loader:
            feat, ret = feat.to(device), ret.to(device)
            optimizer.zero_grad()
            logits = model(feat)
            loss, metrics = compute_selector_loss(logits, ret, temperature=max(1.0, 2.0 - epoch * 0.05))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            for k in tm:
                tm[k] += metrics[k]
            for i, c in enumerate(metrics["choice_dist"]):
                total_choices[i] += c
            nb += 1
        for k in tm:
            tm[k] /= max(1, nb)
        for i in range(len(total_choices)):
            total_choices[i] /= max(1, nb)

        model.eval()
        vm = {"loss": 0, "sortino": 0, "mean_return": 0, "entropy": 0}
        nb = 0
        val_choices = [0.0] * (len(symbols) + 1)
        with torch.no_grad():
            for feat, ret in val_loader:
                feat, ret = feat.to(device), ret.to(device)
                logits = model(feat)
                _, metrics = compute_selector_loss(logits, ret, temperature=1.0)
                for k in vm:
                    vm[k] += metrics[k]
                for i, c in enumerate(metrics["choice_dist"]):
                    val_choices[i] += c
                nb += 1
        for k in vm:
            vm[k] /= max(1, nb)
        for i in range(len(val_choices)):
            val_choices[i] /= max(1, nb)

        scheduler.step()

        ckpt_path = ckpt_dir / f"epoch_{epoch:03d}.pt"
        torch.save({
            "state_dict": model.state_dict(),
            "symbols": symbols,
            "feature_columns": feat_cols,
            "normalizer": normalizer.to_dict(),
            "hidden": hidden,
            "num_layers": num_layers,
            "num_heads": num_heads,
            "seq_len": seq_len,
            "per_symbol_checkpoints": {s: str(checkpoints[s]) for s in symbols},
        }, ckpt_path)

        is_best = vm["sortino"] > best_val_sortino
        if is_best:
            best_val_sortino = vm["sortino"]
            best_ckpt = ckpt_path

        choice_labels = symbols + ["cash"]
        train_dist = " ".join(f"{choice_labels[i]}={total_choices[i]:.0%}" for i in range(len(choice_labels)))
        val_dist = " ".join(f"{choice_labels[i]}={val_choices[i]:.0%}" for i in range(len(choice_labels)))

        print(
            f"ep{epoch:02d} train sort={tm['sortino']:.3f} ret={tm['mean_return']:.3f}% [{train_dist}] | "
            f"val sort={vm['sortino']:.3f} ret={vm['mean_return']:.3f}% [{val_dist}] "
            f"{'*' if is_best else ''}"
        )

    (ckpt_dir / "config.json").write_text(json.dumps({
        "symbols": symbols,
        "feature_columns": feat_cols,
        "hidden": hidden,
        "num_layers": num_layers,
        "num_heads": num_heads,
        "seq_len": seq_len,
        "per_symbol_checkpoints": {s: str(checkpoints[s]) for s in symbols},
        "best_epoch": str(best_ckpt.name) if best_ckpt else None,
        "best_val_sortino": best_val_sortino,
    }, indent=2))
    print(f"done. best val sortino={best_val_sortino:.4f} at {best_ckpt}")
    return best_ckpt


def main():
    parser = argparse.ArgumentParser(description="Train learned meta-selector for crypto")
    parser.add_argument("--symbols", default="DOGEUSD,AAVEUSD")
    parser.add_argument("--doge-checkpoint", required=True)
    parser.add_argument("--aave-checkpoint", required=True)
    parser.add_argument("--data-root", type=Path, default=REPO / "trainingdatahourlybinance")
    parser.add_argument("--forecast-cache", type=Path, default=REPO / "binanceneural/forecast_cache")
    parser.add_argument("--checkpoint-root", type=Path, default=REPO / "binanceneural/checkpoints")
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--seq-len", type=int, default=24)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--wd", type=float, default=0.01)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--val-days", type=int, default=30)
    parser.add_argument("--test-days", type=int, default=30)
    args = parser.parse_args()

    symbols = [s.strip().upper() for s in args.symbols.split(",")]
    checkpoints = {}
    if "DOGEUSD" in symbols:
        checkpoints["DOGEUSD"] = args.doge_checkpoint
    if "AAVEUSD" in symbols:
        checkpoints["AAVEUSD"] = args.aave_checkpoint

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"training meta-selector: {symbols} on {device}")

    train_meta_selector(
        symbols=symbols,
        checkpoints=checkpoints,
        data_root=args.data_root,
        forecast_cache=args.forecast_cache,
        checkpoint_root=args.checkpoint_root,
        hidden=args.hidden,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        seq_len=args.seq_len,
        batch_size=args.batch_size,
        lr=args.lr,
        wd=args.wd,
        epochs=args.epochs,
        val_days=args.val_days,
        test_days=args.test_days,
        device=device,
    )


if __name__ == "__main__":
    main()
