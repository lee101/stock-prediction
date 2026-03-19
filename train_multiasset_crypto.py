#!/usr/bin/env python3
"""Multi-asset crypto RL policy: learns cross-asset allocation for DOGE+AAVE (extensible)."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from binancechronossolexperiment.data import ChronosSolDataModule, SplitConfig
from binanceneural.data import FeatureNormalizer
from src.forecast_horizon_utils import resolve_required_forecast_horizons
from unified_hourly_experiment.multiasset_policy import (
    MultiAssetConfig,
    MultiAssetPolicy,
    DifferentiablePortfolioSim,
)

REPO = Path(__file__).resolve().parents[1]


class MultiAssetCryptoDataset(Dataset):
    def __init__(
        self,
        features: Dict[str, np.ndarray],
        returns: Dict[str, np.ndarray],
        symbols: List[str],
        sequence_length: int = 48,
        horizon: int = 24,
    ):
        self.symbols = symbols
        self.sequence_length = sequence_length
        self.horizon = horizon
        self.features = features
        self.returns = returns
        n = min(len(v) for v in features.values())
        self.num_samples = max(0, n - sequence_length - horizon)

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        feats = []
        rets = []
        for sym in self.symbols:
            f = self.features[sym][idx : idx + self.sequence_length]
            r = self.returns[sym][idx + self.sequence_length : idx + self.sequence_length + self.horizon]
            feats.append(torch.from_numpy(f))
            rets.append(torch.from_numpy(r))
        return torch.stack(feats), torch.stack(rets).T  # (num_assets, seq, feat), (horizon, num_assets)


def load_crypto_data(
    symbols: List[str],
    data_root: Path,
    forecast_cache_root: Path,
    sequence_length: int,
    val_days: int = 30,
    test_days: int = 30,
    max_history_days: int = 365,
) -> Tuple[Dict, Dict, List[str], FeatureNormalizer]:
    """Load multiple crypto symbols, align timestamps, return train/val splits."""
    forecast_horizons = (1,)
    frames = {}

    for sym in symbols:
        try:
            fc = resolve_required_forecast_horizons(forecast_horizons, fallback_horizons=(1,))
            dm = ChronosSolDataModule(
                symbol=sym,
                data_root=data_root,
                forecast_cache_root=forecast_cache_root,
                forecast_horizons=fc,
                context_hours=512,
                quantile_levels=(0.1, 0.5, 0.9),
                batch_size=32,
                model_id="amazon/chronos-t5-small",
                sequence_length=sequence_length,
                split_config=SplitConfig(val_days=val_days, test_days=test_days),
                cache_only=True,
                max_history_days=max_history_days,
            )
            frames[sym] = dm.full_frame
            print(f"loaded {sym}: {len(dm.full_frame)} rows")
        except Exception as e:
            print(f"SKIP {sym}: {e}")

    if len(frames) < 2:
        raise ValueError(f"Need at least 2 symbols, got {len(frames)}")

    # align on common timestamps
    ts_sets = [set(f["timestamp"].tolist()) for f in frames.values()]
    common_ts = sorted(set.intersection(*ts_sets))
    print(f"common timestamps: {len(common_ts)}")

    # compute returns and features on aligned data
    all_symbols = list(frames.keys())
    # fit normalizer on first 85% of common timestamps
    split_idx = int(len(common_ts) * 0.85)

    # collect feature columns from first symbol
    first_sym = all_symbols[0]
    first_frame = frames[first_sym]
    feat_cols = [c for c in first_frame.columns
                 if c not in ("timestamp", "symbol", "open", "high", "low", "close",
                              "volume", "reference_close") and not c.startswith("predicted_")]
    print(f"feature columns ({len(feat_cols)}): {feat_cols[:5]}...")

    # align and extract
    train_features = {}
    train_returns = {}
    val_features = {}
    val_returns = {}

    # fit normalizer on training data from all symbols combined
    train_stack = []
    for sym in all_symbols:
        f = frames[sym].set_index("timestamp").loc[common_ts[:split_idx]]
        available = [c for c in feat_cols if c in f.columns]
        train_stack.append(f[available].to_numpy(dtype=np.float32))
    normalizer = FeatureNormalizer.fit(np.concatenate(train_stack, axis=0))

    for sym in all_symbols:
        f = frames[sym].set_index("timestamp")
        available = [c for c in feat_cols if c in f.columns]

        # train split
        train_f = f.loc[common_ts[:split_idx]]
        train_feats = normalizer.transform(train_f[available].to_numpy(dtype=np.float32))
        train_close = train_f["close"].to_numpy(dtype=np.float32)
        train_ret = np.zeros_like(train_close)
        train_ret[1:] = (train_close[1:] - train_close[:-1]) / (train_close[:-1] + 1e-10)
        train_features[sym] = train_feats
        train_returns[sym] = train_ret

        # val split
        val_f = f.loc[common_ts[split_idx:]]
        val_feats = normalizer.transform(val_f[available].to_numpy(dtype=np.float32))
        val_close = val_f["close"].to_numpy(dtype=np.float32)
        val_ret = np.zeros_like(val_close)
        val_ret[1:] = (val_close[1:] - val_close[:-1]) / (val_close[:-1] + 1e-10)
        val_features[sym] = val_feats
        val_returns[sym] = val_ret

    return (
        {"features": train_features, "returns": train_returns},
        {"features": val_features, "returns": val_returns},
        all_symbols,
        normalizer,
        feat_cols,
    )


class MultiAssetCryptoTrainer:
    def __init__(
        self,
        symbols: List[str],
        data_root: Path,
        forecast_cache_root: Path,
        checkpoint_root: Path,
        hidden_dim: int = 384,
        num_layers: int = 4,
        num_heads: int = 8,
        sequence_length: int = 48,
        horizon: int = 24,
        batch_size: int = 32,
        lr: float = 1e-4,
        wd: float = 0.03,
        epochs: int = 50,
        maker_fee: float = 0.001,
        margin_hourly_rate: float = 0.0000025457,
        max_history_days: int = 365,
        val_days: int = 30,
        test_days: int = 30,
    ):
        self.symbols = symbols
        self.data_root = data_root
        self.forecast_cache_root = forecast_cache_root
        self.checkpoint_root = checkpoint_root
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.sequence_length = sequence_length
        self.horizon = horizon
        self.batch_size = batch_size
        self.lr = lr
        self.wd = wd
        self.epochs = epochs
        self.maker_fee = maker_fee
        self.margin_hourly_rate = margin_hourly_rate
        self.max_history_days = max_history_days
        self.val_days = val_days
        self.test_days = test_days
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        tag = "_".join(s.replace("USD", "").lower() for s in symbols)
        ts = time.strftime("%Y%m%d_%H%M%S")
        self.checkpoint_dir = checkpoint_root / f"multiasset_{tag}_h{hidden_dim}_{ts}"
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def compute_loss(
        self,
        model: MultiAssetPolicy,
        sim: DifferentiablePortfolioSim,
        features: torch.Tensor,
        returns: torch.Tensor,
        entropy_coef: float = 0.1,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        batch_size, horizon, num_assets = returns.shape
        include_cash = getattr(model, "include_cash", False)

        allocations_full = []
        portfolio_state = torch.ones(batch_size, num_assets, device=self.device) / num_assets

        for t in range(horizon):
            alloc_raw, _ = model(features, portfolio_state)
            allocations_full.append(alloc_raw)
            portfolio_state = portfolio_state * (1 + returns[:, t])
            portfolio_state = portfolio_state / (portfolio_state.sum(dim=-1, keepdim=True) + 1e-8)

        allocations_full = torch.stack(allocations_full, dim=1)  # (batch, horizon, num_assets+cash)
        # extract asset allocations for sim (cash earns 0)
        asset_allocs = allocations_full[:, :, :num_assets]

        equity_curve, portfolio_returns = sim(asset_allocs, returns)
        sortino = sim.compute_sortino(portfolio_returns)
        entropy = -(allocations_full * torch.log(allocations_full + 1e-8)).sum(dim=-1).mean()

        margin_penalty = self.margin_hourly_rate * asset_allocs.sum(dim=-1).mean() * horizon

        loss = -sortino.mean() - entropy_coef * entropy + margin_penalty

        total_return = (equity_curve[:, -1] - 1).mean()
        turnover = torch.abs(asset_allocs[:, 1:] - asset_allocs[:, :-1]).sum(dim=-1).mean()
        cash_frac = allocations_full[:, :, -1].mean().item() if include_cash else 0.0

        return loss, {
            "loss": loss.item(),
            "sortino": sortino.mean().item(),
            "return": total_return.item() * 100,
            "turnover": turnover.item(),
            "entropy": entropy.item(),
            "cash_pct": cash_frac * 100,
        }

    def train(self) -> Path:
        print(f"loading data for {self.symbols}...")
        train_data, val_data, symbols, normalizer, feat_cols = load_crypto_data(
            self.symbols,
            self.data_root,
            self.forecast_cache_root,
            self.sequence_length,
            val_days=self.val_days,
            test_days=self.test_days,
            max_history_days=self.max_history_days,
        )

        train_ds = MultiAssetCryptoDataset(
            train_data["features"], train_data["returns"],
            symbols, self.sequence_length, self.horizon,
        )
        val_ds = MultiAssetCryptoDataset(
            val_data["features"], val_data["returns"],
            symbols, self.sequence_length, self.horizon,
        )
        print(f"train: {len(train_ds)} samples, val: {len(val_ds)} samples")

        train_loader = DataLoader(train_ds, batch_size=self.batch_size, shuffle=True, num_workers=0)
        val_loader = DataLoader(val_ds, batch_size=self.batch_size, shuffle=False, num_workers=0)

        feature_dim = train_ds.features[symbols[0]].shape[1]
        config = MultiAssetConfig(
            num_assets=len(symbols),
            feature_dim=feature_dim,
            hidden_dim=self.hidden_dim,
            num_heads=self.num_heads,
            num_layers=self.num_layers,
            max_len=self.sequence_length,
            dropout=0.2,
        )

        model = MultiAssetPolicy(config, include_cash=True).to(self.device)
        sim = DifferentiablePortfolioSim(len(symbols), transaction_cost=self.maker_fee).to(self.device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=self.lr, weight_decay=self.wd)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.epochs)

        param_count = sum(p.numel() for p in model.parameters())
        print(f"model params: {param_count:,} | device: {self.device}")

        best_sortino = float("-inf")
        best_ckpt = None

        for epoch in range(1, self.epochs + 1):
            model.train()
            tm = {"loss": 0, "sortino": 0, "return": 0, "turnover": 0, "entropy": 0, "cash_pct": 0}
            nb = 0
            for features, returns in train_loader:
                features, returns = features.to(self.device), returns.to(self.device)
                optimizer.zero_grad()
                loss, metrics = self.compute_loss(model, sim, features, returns)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                for k, v in metrics.items():
                    tm[k] += v
                nb += 1
            for k in tm:
                tm[k] /= max(1, nb)

            model.eval()
            vm = {"loss": 0, "sortino": 0, "return": 0, "turnover": 0, "entropy": 0, "cash_pct": 0}
            nb = 0
            with torch.no_grad():
                for features, returns in val_loader:
                    features, returns = features.to(self.device), returns.to(self.device)
                    _, metrics = self.compute_loss(model, sim, features, returns)
                    for k, v in metrics.items():
                        vm[k] += v
                    nb += 1
            for k in vm:
                vm[k] /= max(1, nb)

            scheduler.step()

            ckpt_path = self.checkpoint_dir / f"epoch_{epoch:03d}.pt"
            torch.save({
                "state_dict": model.state_dict(),
                "config": {
                    "num_assets": config.num_assets,
                    "feature_dim": config.feature_dim,
                    "hidden_dim": config.hidden_dim,
                    "num_heads": config.num_heads,
                    "num_layers": config.num_layers,
                    "max_len": config.max_len,
                    "dropout": config.dropout,
                    "include_cash": True,
                },
                "symbols": symbols,
                "feature_columns": feat_cols,
                "normalizer": normalizer.to_dict(),
                "sequence_length": self.sequence_length,
                "horizon": self.horizon,
                "maker_fee": self.maker_fee,
                "margin_hourly_rate": self.margin_hourly_rate,
            }, ckpt_path)

            if vm["sortino"] > best_sortino:
                best_sortino = vm["sortino"]
                best_ckpt = ckpt_path

            print(
                f"ep{epoch:02d} train sort={tm['sortino']:.3f} ret={tm['return']:.2f}% "
                f"turn={tm['turnover']:.3f} cash={tm.get('cash_pct',0):.0f}% | "
                f"val sort={vm['sortino']:.3f} ret={vm['return']:.2f}% "
                f"turn={vm['turnover']:.3f} cash={vm.get('cash_pct',0):.0f}% {'*' if ckpt_path == best_ckpt else ''}"
            )

        config_out = {
            "symbols": symbols,
            "feature_columns": feat_cols,
            "hidden_dim": self.hidden_dim,
            "num_heads": self.num_heads,
            "num_layers": self.num_layers,
            "sequence_length": self.sequence_length,
            "horizon": self.horizon,
            "maker_fee": self.maker_fee,
            "margin_hourly_rate": self.margin_hourly_rate,
            "best_epoch": str(best_ckpt.name) if best_ckpt else None,
            "best_sortino": best_sortino,
        }
        (self.checkpoint_dir / "config.json").write_text(json.dumps(config_out, indent=2))
        print(f"done. best val sortino={best_sortino:.4f} at {best_ckpt}")
        return best_ckpt


def main():
    parser = argparse.ArgumentParser(description="Multi-asset crypto policy trainer")
    parser.add_argument("--symbols", default="DOGEUSD,AAVEUSD")
    parser.add_argument("--data-root", type=Path, default=REPO / "trainingdatahourlybinance")
    parser.add_argument("--forecast-cache", type=Path, default=REPO / "binanceneural/forecast_cache")
    parser.add_argument("--checkpoint-root", type=Path, default=REPO / "binanceneural/checkpoints")
    parser.add_argument("--hidden-dim", type=int, default=384)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--sequence-length", type=int, default=48)
    parser.add_argument("--horizon", type=int, default=24)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--wd", type=float, default=0.03)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--maker-fee", type=float, default=0.001)
    parser.add_argument("--margin-rate", type=float, default=0.0000025457)
    parser.add_argument("--max-history-days", type=int, default=365)
    parser.add_argument("--val-days", type=int, default=30)
    parser.add_argument("--test-days", type=int, default=30)
    args = parser.parse_args()

    symbols = [s.strip().upper() for s in args.symbols.split(",")]
    print(f"training multi-asset policy: {symbols}")

    trainer = MultiAssetCryptoTrainer(
        symbols=symbols,
        data_root=args.data_root,
        forecast_cache_root=args.forecast_cache,
        checkpoint_root=args.checkpoint_root,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        sequence_length=args.sequence_length,
        horizon=args.horizon,
        batch_size=args.batch_size,
        lr=args.lr,
        wd=args.wd,
        epochs=args.epochs,
        maker_fee=args.maker_fee,
        margin_hourly_rate=args.margin_rate,
        max_history_days=args.max_history_days,
        val_days=args.val_days,
        test_days=args.test_days,
    )
    trainer.train()


if __name__ == "__main__":
    main()
