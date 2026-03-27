from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import pandas as pd
import torch

from differentiable_market.config import DataConfig
from differentiable_market.data import load_aligned_ohlc

from .config import E2EDataConfig
from .universe import load_stock_universe


@dataclass(slots=True)
class StockDataset:
    symbols: list[str]
    index: pd.DatetimeIndex
    ohlc: torch.Tensor
    close: torch.Tensor
    close_returns: torch.Tensor


def load_stock_dataset(cfg: E2EDataConfig) -> StockDataset:
    symbols = load_stock_universe(
        data_root=cfg.data_root,
        universe_file=cfg.universe_file,
        include_symbols=cfg.include_symbols,
        exclude_symbols=cfg.exclude_symbols,
        max_assets=cfg.max_assets,
    )
    dm_cfg = DataConfig(
        root=cfg.data_root,
        include_symbols=tuple(symbols),
        max_assets=len(symbols),
        cache_dir=cfg.cache_dir,
        min_timesteps=cfg.min_timesteps,
        include_cash=False,
    )
    ohlc, aligned_symbols, index = load_aligned_ohlc(dm_cfg)
    close = ohlc[..., 3].contiguous()
    close_returns = (close[1:] / close[:-1].clamp_min(1e-8)) - 1.0
    return StockDataset(
        symbols=aligned_symbols,
        index=index,
        ohlc=ohlc.contiguous(),
        close=close,
        close_returns=close_returns.contiguous(),
    )


def split_dataset(dataset: StockDataset, train_ratio: float) -> tuple[StockDataset, StockDataset]:
    if not 0.5 <= float(train_ratio) < 1.0:
        raise ValueError("train_ratio must be in [0.5, 1.0)")
    split_idx = int(len(dataset.index) * float(train_ratio))
    if split_idx <= 2 or len(dataset.index) - split_idx <= 2:
        raise ValueError("Insufficient data for the requested split")

    def _slice(start: int, end: int) -> StockDataset:
        ohlc = dataset.ohlc[start:end].clone()
        close = ohlc[..., 3].contiguous()
        close_returns = (close[1:] / close[:-1].clamp_min(1e-8)) - 1.0
        return StockDataset(
            symbols=list(dataset.symbols),
            index=dataset.index[start:end],
            ohlc=ohlc,
            close=close,
            close_returns=close_returns.contiguous(),
        )

    return _slice(0, split_idx), _slice(split_idx, len(dataset.index))


def sample_start_indices(
    *,
    total_steps: int,
    context_length: int,
    rollout_length: int,
    batch_size: int,
    generator: torch.Generator,
) -> torch.Tensor:
    max_start = total_steps - context_length - rollout_length - 1
    if max_start < 0:
        raise ValueError("Dataset is shorter than context_length + rollout_length + 1")
    if max_start == 0:
        return torch.zeros(batch_size, dtype=torch.long)
    return torch.randint(0, max_start + 1, (batch_size,), generator=generator, dtype=torch.long)


def realized_volatility(context_close: torch.Tensor, window: int = 20) -> torch.Tensor:
    if context_close.ndim != 2:
        raise ValueError("context_close must have shape [assets, context]")
    window = min(int(window), max(1, context_close.shape[-1] - 1))
    returns = torch.log(context_close[:, -window:] / context_close[:, -window - 1 : -1].clamp_min(1e-8))
    return returns.std(dim=-1, correction=0)


def momentum_feature(context_close: torch.Tensor, window: int = 5) -> torch.Tensor:
    if context_close.ndim != 2:
        raise ValueError("context_close must have shape [assets, context]")
    window = min(int(window), max(1, context_close.shape[-1] - 1))
    return (context_close[:, -1] / context_close[:, -window - 1].clamp_min(1e-8)) - 1.0
