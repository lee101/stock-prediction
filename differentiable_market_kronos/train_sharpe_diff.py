from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

from .config import ExperimentConfig
from .kronos_embedder import KronosEmbedder, KronosFeatureSpec, precompute_feature_table


def differentiable_pnl(position: torch.Tensor, returns: torch.Tensor, transaction_cost: float, slippage: float, hold_penalty: float) -> torch.Tensor:
    turnover = torch.cat([torch.zeros_like(position[:1]), position[1:] - position[:-1]], dim=0).abs()
    costs = turnover * (transaction_cost + slippage) + hold_penalty * (position**2)
    return position.squeeze(-1) * returns - costs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ohlcv", type=str, required=True)
    parser.add_argument("--timestamp-col", type=str, default="timestamp")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=3e-4)
    args = parser.parse_args()

    cfg = ExperimentConfig()

    path = Path(args.ohlcv)
    if path.suffix == ".parquet":
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path)
    df[cfg.data.timestamp_col] = pd.to_datetime(df[cfg.data.timestamp_col])
    df = df.dropna().sort_values(cfg.data.timestamp_col).reset_index(drop=True)

    embedder = KronosEmbedder(
        model_id=cfg.kronos.model_id,
        tokenizer_id=cfg.kronos.tokenizer_id,
        device=cfg.kronos.device,
        max_context=cfg.kronos.max_context,
        temperature=cfg.kronos.temperature,
        top_p=cfg.kronos.top_p,
        sample_count=cfg.kronos.sample_count,
        bf16=cfg.train.bf16,
        feature_spec=KronosFeatureSpec(horizons=(1, 12, cfg.env.pred_horizon)),
    )

    cols = [cfg.data.open_col, cfg.data.high_col, cfg.data.low_col, cfg.data.price_col]
    if cfg.data.volume_col in df.columns:
        cols.append(cfg.data.volume_col)
    if cfg.data.amount_col in df.columns:
        cols.append(cfg.data.amount_col)
    x_df = df[cols].rename(
        columns={
            cfg.data.open_col: "open",
            cfg.data.high_col: "high",
            cfg.data.low_col: "low",
            cfg.data.price_col: "close",
            cfg.data.volume_col: "volume" if cfg.data.volume_col in df.columns else cfg.data.volume_col,
            cfg.data.amount_col: "amount" if cfg.data.amount_col in df.columns else cfg.data.amount_col,
        }
    )
    ts = df[cfg.data.timestamp_col]

    features_df = precompute_feature_table(
        df=x_df,
        ts=ts,
        lookback=cfg.env.lookback,
        horizon_main=cfg.env.pred_horizon,
        embedder=embedder,
    ).astype("float32")
    features = torch.from_numpy(features_df.to_numpy(dtype=np.float32))

    returns = torch.from_numpy(df.set_index(cfg.data.timestamp_col)[cfg.data.price_col].pct_change().loc[features_df.index].to_numpy(dtype=np.float32))
    returns = returns.unsqueeze(-1)

    model = nn.Sequential(nn.Linear(features.shape[1], 64), nn.Tanh(), nn.Linear(64, 1))
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    transaction_cost = cfg.env.transaction_cost_bps / 1e4
    slippage = cfg.env.slippage_bps / 1e4

    for epoch in range(args.epochs):
        optimizer.zero_grad()
        pos = torch.tanh(model(features))
        pnl = differentiable_pnl(pos, returns.squeeze(-1), transaction_cost, slippage, cfg.env.hold_penalty)
        sharpe = pnl.mean() / (pnl.std(unbiased=False) + 1e-8)
        loss = -sharpe
        loss.backward()
        optimizer.step()
        print(f"epoch={epoch} sharpe={sharpe.item():.4f}")

    torch.save(model.state_dict(), "sharpe_model.pt")


if __name__ == "__main__":
    main()
