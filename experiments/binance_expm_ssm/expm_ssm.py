#!/usr/bin/env python3
"""Differentiable matrix-exponential SSM for crypto allocation.

The model learns a continuous-time generator Q_t from lagged market features,
computes exact transition probabilities with torch.matrix_exp(Q_t * dt), and
maps the state distribution to long-only portfolio weights. This is a research
prototype for exact differentiable transition dynamics, not a production trader.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from experiments.binance_transition_graph.graph_policy import (
    DEFAULT_SYMBOLS,
    HOURS_PER_MONTH,
    HOURS_PER_YEAR,
    close_matrix,
    load_hourly_panel,
    train_eval_split,
)

FeatureStats = tuple[pd.Series, pd.Series]


@dataclass(frozen=True)
class ExpmSSMConfig:
    num_states: int = 8
    hidden_dim: int = 64
    feature_lookback_hours: int = 72
    target_leverage: float = 1.0
    fee_bps: float = 10.0
    slippage_bps: float = 5.0
    turnover_penalty: float = 0.15
    drawdown_penalty: float = 0.25
    entropy_penalty: float = 0.001
    lr: float = 3e-4
    epochs: int = 20
    decision_lag_hours: int = 2


@dataclass(frozen=True)
class ExpmMetrics:
    total_return: float
    monthly_return: float
    sortino: float
    max_drawdown: float
    turnover: float
    final_equity: float


def _make_features(
    close: pd.DataFrame,
    lookback: int,
    *,
    feature_stats: FeatureStats | None = None,
    return_feature_stats: bool = False,
) -> pd.DataFrame | tuple[pd.DataFrame, FeatureStats]:
    returns = close.pct_change().replace([np.inf, -np.inf], np.nan)
    parts: list[pd.DataFrame] = []
    for horizon in (1, 6, 24, int(lookback)):
        parts.append(close.pct_change(horizon).add_suffix(f"_ret{horizon}"))
    vol = returns.rolling(24, min_periods=8).std().add_suffix("_vol24")
    drawdown = (close / close.rolling(int(lookback), min_periods=12).max() - 1.0).add_suffix("_dd")
    parts.extend([vol, drawdown])
    feat = pd.concat(parts, axis=1)
    feat = feat.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    if feature_stats is None:
        mean = feat.mean(axis=0)
        std = feat.std(axis=0).replace(0.0, 1.0)
    else:
        mean, std = feature_stats
        mean = mean.reindex(feat.columns).fillna(0.0)
        std = std.reindex(feat.columns).fillna(1.0).replace(0.0, 1.0)
    normalized = ((feat - mean) / std).clip(-8.0, 8.0)
    if return_feature_stats:
        return normalized, (mean, std)
    return normalized


def _tensorize(
    close: pd.DataFrame,
    cfg: ExpmSSMConfig,
    *,
    feature_stats: FeatureStats | None = None,
    return_feature_stats: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, list[str]] | tuple[torch.Tensor, torch.Tensor, list[str], FeatureStats]:
    feature_df, fitted_stats = _make_features(
        close,
        cfg.feature_lookback_hours,
        feature_stats=feature_stats,
        return_feature_stats=True,
    )
    next_returns = close.pct_change().shift(-1)
    start = max(cfg.feature_lookback_hours + cfg.decision_lag_hours + 1, 4)
    end = len(close) - 1
    if end <= start:
        raise ValueError("Not enough rows for expm SSM tensorization")
    lagged_features = feature_df.iloc[start - cfg.decision_lag_hours : end - cfg.decision_lag_hours]
    returns = next_returns.iloc[start:end]
    x = torch.tensor(lagged_features.to_numpy(dtype=np.float32), dtype=torch.float32)
    y = torch.tensor(returns.to_numpy(dtype=np.float32), dtype=torch.float32)
    if return_feature_stats:
        return x, y, list(close.columns), fitted_stats
    return x, y, list(close.columns)


class ExpmTransitionPolicy(nn.Module):
    def __init__(self, input_dim: int, num_assets: int, cfg: ExpmSSMConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.num_assets = int(num_assets)
        self.num_states = max(int(cfg.num_states), int(num_assets) + 1)
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, int(cfg.hidden_dim)),
            nn.SiLU(),
            nn.Linear(int(cfg.hidden_dim), int(cfg.hidden_dim)),
            nn.SiLU(),
        )
        self.q_head = nn.Linear(int(cfg.hidden_dim), self.num_states * self.num_states)
        self.weight_head = nn.Linear(self.num_states, int(num_assets))
        self.initial_logits = nn.Parameter(torch.zeros(self.num_states))
        self.base_q = nn.Parameter(torch.randn(self.num_states, self.num_states) * 0.01)

    def generator(self, h: torch.Tensor) -> torch.Tensor:
        raw = self.q_head(h).view(-1, self.num_states, self.num_states)
        raw = raw + self.base_q.unsqueeze(0)
        offdiag = F.softplus(raw) * (1.0 - torch.eye(self.num_states, device=raw.device).unsqueeze(0))
        diagonal = -offdiag.sum(dim=-1)
        return offdiag + torch.diag_embed(diagonal)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.encoder(x)
        q = self.generator(h)
        probs = torch.softmax(self.initial_logits, dim=-1)
        states: list[torch.Tensor] = []
        weights: list[torch.Tensor] = []
        for q_t in q:
            transition = torch.matrix_exp(q_t)
            probs = torch.matmul(probs, transition)
            probs = probs / probs.sum().clamp_min(1e-8)
            raw_weights = self.weight_head(probs)
            weights_t = torch.softmax(raw_weights, dim=-1) * float(self.cfg.target_leverage)
            states.append(probs)
            weights.append(weights_t)
        return torch.stack(weights, dim=0), torch.stack(states, dim=0)


def _equity_from_weights(
    weights: torch.Tensor,
    returns: torch.Tensor,
    cfg: ExpmSSMConfig,
) -> tuple[torch.Tensor, torch.Tensor]:
    returns = torch.nan_to_num(returns, nan=0.0, posinf=0.0, neginf=0.0)
    gross = (weights * returns).sum(dim=-1)
    fee_rate = (float(cfg.fee_bps) + float(cfg.slippage_bps)) / 10_000.0
    turnover = torch.zeros_like(gross)
    if len(weights) > 1:
        turnover = torch.cat(
            [torch.zeros(1, device=weights.device), (weights[1:] - weights[:-1]).abs().sum(dim=-1)]
        )
    net = gross - fee_rate * turnover
    equity = torch.cumprod((1.0 + net).clamp_min(1e-6), dim=0)
    return equity, turnover


def _loss(weights: torch.Tensor, states: torch.Tensor, returns: torch.Tensor, cfg: ExpmSSMConfig) -> torch.Tensor:
    equity, turnover = _equity_from_weights(weights, returns, cfg)
    log_growth = torch.log((equity[-1]).clamp_min(1e-8))
    peak = torch.cummax(equity, dim=0).values
    drawdown = (equity / peak - 1.0).abs().max()
    entropy = -(states * torch.log(states.clamp_min(1e-8))).sum(dim=-1).mean()
    return (
        -log_growth
        + float(cfg.turnover_penalty) * turnover.mean()
        + float(cfg.drawdown_penalty) * drawdown
        - float(cfg.entropy_penalty) * entropy
    )


def _metrics_from_equity(equity: np.ndarray, turnover: np.ndarray) -> ExpmMetrics:
    total_return = float(equity[-1] / max(equity[0], 1e-12) - 1.0)
    hours = max(1, len(equity) - 1)
    monthly = float(max(equity[-1] / max(equity[0], 1e-12), 1e-12) ** (HOURS_PER_MONTH / hours) - 1.0)
    returns = pd.Series(equity).pct_change().replace([np.inf, -np.inf], np.nan).dropna()
    downside = returns[returns < 0.0]
    if len(downside) == 0:
        sortino = 0.0 if returns.mean() <= 0 else 999.0
    else:
        sortino = float(returns.mean() / max(downside.std(ddof=0), 1e-12) * math.sqrt(HOURS_PER_YEAR))
    peak = np.maximum.accumulate(equity)
    max_dd = float(abs(np.min(equity / np.maximum(peak, 1e-12) - 1.0)))
    return ExpmMetrics(
        total_return=total_return,
        monthly_return=monthly,
        sortino=sortino,
        max_drawdown=max_dd,
        turnover=float(np.mean(turnover)),
        final_equity=float(equity[-1]),
    )


def train_model(
    train_close: pd.DataFrame,
    eval_close: pd.DataFrame,
    cfg: ExpmSSMConfig,
    *,
    device: str = "cpu",
) -> tuple[ExpmTransitionPolicy, dict[str, object]]:
    x_train, y_train, symbols, feature_stats = _tensorize(train_close, cfg, return_feature_stats=True)
    x_eval, y_eval, _ = _tensorize(eval_close, cfg, feature_stats=feature_stats)
    dev = torch.device(device)
    model = ExpmTransitionPolicy(x_train.shape[1], len(symbols), cfg).to(dev)
    x_train = x_train.to(dev)
    y_train = y_train.to(dev)
    x_eval = x_eval.to(dev)
    y_eval = y_eval.to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=float(cfg.lr), weight_decay=1e-4)
    history: list[dict[str, float]] = []
    for epoch in range(int(cfg.epochs)):
        model.train()
        opt.zero_grad(set_to_none=True)
        weights, states = model(x_train)
        loss = _loss(weights, states, y_train, cfg)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if epoch == 0 or epoch == int(cfg.epochs) - 1 or (epoch + 1) % 5 == 0:
            with torch.no_grad():
                equity, turnover = _equity_from_weights(weights, y_train, cfg)
                metric = _metrics_from_equity(
                    equity.detach().cpu().numpy(),
                    turnover.detach().cpu().numpy(),
                )
            history.append({"epoch": float(epoch + 1), "loss": float(loss.item()), **asdict(metric)})

    model.eval()
    with torch.no_grad():
        train_weights, train_states = model(x_train)
        train_equity, train_turnover = _equity_from_weights(train_weights, y_train, cfg)
        eval_weights, eval_states = model(x_eval)
        eval_equity, eval_turnover = _equity_from_weights(eval_weights, y_eval, cfg)
    report = {
        "config": asdict(cfg),
        "symbols": symbols,
        "train_metrics": asdict(
            _metrics_from_equity(train_equity.detach().cpu().numpy(), train_turnover.detach().cpu().numpy())
        ),
        "eval_metrics": asdict(
            _metrics_from_equity(eval_equity.detach().cpu().numpy(), eval_turnover.detach().cpu().numpy())
        ),
        "history": history,
        "eval_final_weights": eval_weights[-1].detach().cpu().numpy().tolist(),
        "eval_final_state_probs": eval_states[-1].detach().cpu().numpy().tolist(),
    }
    return model, report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default="trainingdatahourlybinance")
    parser.add_argument("--symbols", nargs="+", default=list(DEFAULT_SYMBOLS))
    parser.add_argument("--train-hours", type=int, default=24 * 30)
    parser.add_argument("--eval-hours", type=int, default=24 * 7)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--num-states", type=int, default=8)
    parser.add_argument("--target-leverage", type=float, default=1.0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output-dir", default="experiments/binance_expm_ssm/results")
    args = parser.parse_args(argv)

    cfg = ExpmSSMConfig(
        epochs=int(args.epochs),
        hidden_dim=int(args.hidden_dim),
        num_states=int(args.num_states),
        target_leverage=float(args.target_leverage),
    )
    frames = load_hourly_panel(
        args.data_root,
        args.symbols,
        tail_hours=int(args.train_hours) + int(args.eval_hours) + cfg.feature_lookback_hours + 240,
    )
    train_frames, eval_frames = train_eval_split(
        frames,
        train_hours=int(args.train_hours),
        eval_hours=int(args.eval_hours),
    )
    train_close = close_matrix(train_frames)
    eval_close = close_matrix(eval_frames)
    _model, report = train_model(train_close, eval_close, cfg, device=str(args.device))

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"expm_ssm_{time.strftime('%Y%m%d_%H%M%S')}.json"
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    eval_metrics = report["eval_metrics"]
    print(
        "eval "
        f"monthly={eval_metrics['monthly_return'] * 100:+.2f}% "
        f"dd={eval_metrics['max_drawdown'] * 100:.2f}% "
        f"sortino={eval_metrics['sortino']:.2f}"
    )
    print(f"wrote {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
