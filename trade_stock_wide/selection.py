from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Literal, Mapping, Sequence

import math

import numpy as np
import torch

from .types import WideCandidate


SelectionObjective = Literal["pnl", "sortino", "hybrid", "tiny_net", "torch_mlp"]


@dataclass(frozen=True)
class WideSelectionConfig:
    objective: SelectionObjective = "hybrid"
    lookback_days: int = 20
    tiny_net_hidden_dim: int = 8
    tiny_net_epochs: int = 120
    tiny_net_learning_rate: float = 0.03
    tiny_net_l2: float = 1e-4
    tiny_net_augment_copies: int = 3
    tiny_net_noise_scale: float = 0.04
    tiny_net_min_train_samples: int = 12
    seed: int = 1337
    rl_prior_weight: float = 0.0
    rl_prior_scale: float = 2.0
    rl_prior_by_symbol: Mapping[str, float] | None = None
    torch_device: str = "auto"
    torch_batch_size: int = 256


def estimate_candidate_daily_return(
    candidate: WideCandidate,
    *,
    fee_bps: float = 10.0,
    fill_buffer_bps: float = 5.0,
) -> float:
    if candidate.realized_low > candidate.entry_price:
        return 0.0
    fee_rate = float(fee_bps) / 10_000.0
    fill_buffer = float(fill_buffer_bps) / 10_000.0
    entry_price = candidate.entry_price * (1.0 + fill_buffer)
    if entry_price <= 0.0:
        return 0.0
    exit_price = candidate.take_profit_price if candidate.realized_high >= candidate.take_profit_price else candidate.realized_close
    return ((exit_price - entry_price) / entry_price) - (2.0 * fee_rate)


def _candidate_feature_vector(candidate: WideCandidate) -> np.ndarray:
    dollar_vol = 0.0 if candidate.dollar_vol_20d is None else max(float(candidate.dollar_vol_20d), 0.0)
    spread_bps = 0.0 if candidate.spread_bps_estimate is None else max(float(candidate.spread_bps_estimate), 0.0)
    return np.asarray(
        [
            float(candidate.forecasted_pnl),
            float(candidate.avg_return),
            float(candidate.expected_return_pct),
            float(candidate.entry_gap_pct),
            float((candidate.predicted_high - candidate.predicted_low) / max(candidate.last_close, 1e-9)),
            math.log1p(dollar_vol),
            spread_bps / 100.0,
            1.0 if candidate.strategy == "highlow" else 0.0,
            1.0 if candidate.strategy == "maxdiff" else 0.0,
            1.0 if candidate.strategy == "takeprofit" else 0.0,
        ],
        dtype=np.float64,
    )


def _safe_sortino(returns: Sequence[float]) -> float:
    if not returns:
        return 0.0
    values = np.asarray(list(returns), dtype=np.float64)
    if values.size == 0:
        return 0.0
    downside = np.minimum(values, 0.0)
    downside_rms = float(np.sqrt(np.mean(np.square(downside))))
    if downside_rms <= 1e-9:
        positive_mean = float(np.mean(values))
        return max(positive_mean * 50.0, 0.0)
    return float(np.mean(values) / downside_rms)


def _history_returns_for_candidate(
    candidate: WideCandidate,
    history_days: Sequence[Sequence[WideCandidate]],
    *,
    lookback_days: int,
    fee_bps: float,
    fill_buffer_bps: float,
) -> list[float]:
    symbol_and_strategy_matches: list[float] = []
    symbol_matches: list[float] = []
    for day in history_days[-max(int(lookback_days), 0) :]:
        for prior in day:
            if prior.symbol != candidate.symbol:
                continue
            realized = estimate_candidate_daily_return(
                prior,
                fee_bps=fee_bps,
                fill_buffer_bps=fill_buffer_bps,
            )
            symbol_matches.append(realized)
            if prior.strategy == candidate.strategy:
                symbol_and_strategy_matches.append(realized)
    return symbol_and_strategy_matches or symbol_matches


def normalize_rl_prior_score(raw_score: float, *, scale: float) -> float:
    normalized_scale = max(float(scale), 1e-6)
    positive = max(float(raw_score), 0.0)
    return float(np.tanh(positive / normalized_scale))


def build_symbol_rl_prior(
    rows,
    *,
    score_column: str = "robust_score",
    symbols_column: str = "symbols",
    aggregation: str = "max",
    min_score: float = 0.0,
    scale: float = 2.0,
) -> dict[str, float]:
    if rows is None:
        return {}
    out: dict[str, list[float]] = {}
    if hasattr(rows, "iterrows"):
        iterable = (record for _, record in rows.iterrows())
    else:
        iterable = rows
    for row in iterable:
        raw_score = row.get(score_column) if hasattr(row, "get") else None
        try:
            numeric_score = float(raw_score)
        except (TypeError, ValueError):
            continue
        if not math.isfinite(numeric_score) or numeric_score < float(min_score):
            continue
        raw_symbols = row.get(symbols_column) if hasattr(row, "get") else None
        if raw_symbols is None:
            continue
        symbols = [item.strip().upper() for item in str(raw_symbols).split(",") if item.strip()]
        if not symbols:
            continue
        normalized = normalize_rl_prior_score(numeric_score, scale=scale)
        for symbol in symbols:
            out.setdefault(symbol, []).append(normalized)
    reduced: dict[str, float] = {}
    for symbol, values in out.items():
        if aggregation == "mean":
            reduced[symbol] = float(np.mean(np.asarray(values, dtype=np.float64)))
        else:
            reduced[symbol] = float(max(values))
    return reduced


def resolve_torch_device(requested: str = "auto") -> str:
    normalized = str(requested or "auto").strip().lower()
    if normalized not in {"auto", "cpu", "cuda"}:
        raise ValueError(f"unsupported torch device: {requested!r}")
    if normalized == "cpu":
        return "cpu"
    if normalized == "cuda":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return "cuda" if torch.cuda.is_available() else "cpu"


@dataclass(frozen=True)
class _TinyNetModel:
    mean: np.ndarray
    scale: np.ndarray
    w1: np.ndarray
    b1: np.ndarray
    w2: np.ndarray
    b2: np.ndarray


@dataclass(frozen=True)
class _TorchMLPModel:
    mean: np.ndarray
    scale: np.ndarray
    network: torch.nn.Module
    device: str


def _fit_tiny_net(
    features: np.ndarray,
    targets: np.ndarray,
    *,
    config: WideSelectionConfig,
    rng: np.random.Generator,
) -> _TinyNetModel | None:
    if features.ndim != 2 or targets.ndim != 1 or len(features) != len(targets):
        return None
    if len(features) < max(int(config.tiny_net_min_train_samples), 1):
        return None

    mean = features.mean(axis=0)
    scale = features.std(axis=0)
    scale = np.where(scale > 1e-6, scale, 1.0)
    x = (features - mean) / scale
    y = targets.astype(np.float64)

    if config.tiny_net_augment_copies > 0 and config.tiny_net_noise_scale > 0.0:
        feature_scale = np.where(features.std(axis=0) > 1e-6, features.std(axis=0), 1.0)
        aug_x = [features]
        aug_y = [y]
        for _ in range(int(config.tiny_net_augment_copies)):
            noisy = features + (
                rng.normal(loc=0.0, scale=float(config.tiny_net_noise_scale), size=features.shape) * feature_scale
            )
            aug_x.append(noisy)
            aug_y.append(y)
        features = np.concatenate(aug_x, axis=0)
        y = np.concatenate(aug_y, axis=0)
        mean = features.mean(axis=0)
        scale = features.std(axis=0)
        scale = np.where(scale > 1e-6, scale, 1.0)
        x = (features - mean) / scale

    hidden_dim = max(int(config.tiny_net_hidden_dim), 2)
    input_dim = int(x.shape[1])
    w1 = rng.normal(loc=0.0, scale=0.15, size=(input_dim, hidden_dim))
    b1 = np.zeros(hidden_dim, dtype=np.float64)
    w2 = rng.normal(loc=0.0, scale=0.15, size=(hidden_dim, 1))
    b2 = np.zeros(1, dtype=np.float64)
    learning_rate = float(config.tiny_net_learning_rate)
    l2 = float(config.tiny_net_l2)
    sample_scale = 1.0 / max(len(x), 1)

    y_col = y.reshape(-1, 1)
    for _ in range(max(int(config.tiny_net_epochs), 1)):
        hidden = np.tanh((x @ w1) + b1)
        pred = (hidden @ w2) + b2
        err = pred - y_col

        d_pred = 2.0 * err * sample_scale
        grad_w2 = hidden.T @ d_pred + (l2 * w2)
        grad_b2 = d_pred.sum(axis=0)
        d_hidden = (d_pred @ w2.T) * (1.0 - np.square(hidden))
        grad_w1 = x.T @ d_hidden + (l2 * w1)
        grad_b1 = d_hidden.sum(axis=0)

        w2 -= learning_rate * grad_w2
        b2 -= learning_rate * grad_b2
        w1 -= learning_rate * grad_w1
        b1 -= learning_rate * grad_b1

    return _TinyNetModel(mean=mean, scale=scale, w1=w1, b1=b1, w2=w2, b2=b2)


def _predict_tiny_net(model: _TinyNetModel, candidate: WideCandidate) -> float:
    x = (_candidate_feature_vector(candidate) - model.mean) / model.scale
    hidden = np.tanh((x @ model.w1) + model.b1)
    pred = float((hidden @ model.w2).reshape(-1)[0] + model.b2.reshape(-1)[0])
    return pred


def _fit_torch_mlp(
    features: np.ndarray,
    targets: np.ndarray,
    *,
    config: WideSelectionConfig,
) -> _TorchMLPModel | None:
    if features.ndim != 2 or targets.ndim != 1 or len(features) != len(targets):
        return None
    if len(features) < max(int(config.tiny_net_min_train_samples), 1):
        return None

    rng = np.random.default_rng(int(config.seed))
    working_features = np.asarray(features, dtype=np.float32)
    working_targets = np.asarray(targets, dtype=np.float32)
    if config.tiny_net_augment_copies > 0 and config.tiny_net_noise_scale > 0.0:
        feature_scale = np.where(working_features.std(axis=0) > 1e-6, working_features.std(axis=0), 1.0)
        aug_x = [working_features]
        aug_y = [working_targets]
        for _ in range(int(config.tiny_net_augment_copies)):
            noisy = working_features + (
                rng.normal(loc=0.0, scale=float(config.tiny_net_noise_scale), size=working_features.shape).astype(np.float32)
                * feature_scale.astype(np.float32)
            )
            aug_x.append(noisy)
            aug_y.append(working_targets)
        working_features = np.concatenate(aug_x, axis=0)
        working_targets = np.concatenate(aug_y, axis=0)

    mean = working_features.mean(axis=0, dtype=np.float32)
    scale = working_features.std(axis=0, dtype=np.float32)
    scale = np.where(scale > 1e-6, scale, 1.0).astype(np.float32)
    x = ((working_features - mean) / scale).astype(np.float32)
    y = working_targets.astype(np.float32).reshape(-1, 1)

    device = resolve_torch_device(config.torch_device)
    torch.manual_seed(int(config.seed))
    if device == "cuda":
        torch.cuda.manual_seed_all(int(config.seed))

    network = torch.nn.Sequential(
        torch.nn.Linear(int(x.shape[1]), max(int(config.tiny_net_hidden_dim), 4)),
        torch.nn.Tanh(),
        torch.nn.Linear(max(int(config.tiny_net_hidden_dim), 4), 1),
    ).to(device)
    optimizer = torch.optim.AdamW(
        network.parameters(),
        lr=float(config.tiny_net_learning_rate),
        weight_decay=float(config.tiny_net_l2),
    )
    loss_fn = torch.nn.MSELoss()
    x_tensor = torch.as_tensor(x, device=device)
    y_tensor = torch.as_tensor(y, device=device)
    batch_size = max(1, min(int(config.torch_batch_size), int(x_tensor.shape[0])))
    permutation_generator = torch.Generator(device="cpu")
    permutation_generator.manual_seed(int(config.seed))

    network.train()
    for _ in range(max(int(config.tiny_net_epochs), 1)):
        permutation = torch.randperm(int(x_tensor.shape[0]), generator=permutation_generator, device="cpu")
        for start in range(0, int(x_tensor.shape[0]), batch_size):
            batch_indices = permutation[start : start + batch_size].to(device)
            batch_x = x_tensor.index_select(0, batch_indices)
            batch_y = y_tensor.index_select(0, batch_indices)
            optimizer.zero_grad(set_to_none=True)
            pred = network(batch_x)
            loss = loss_fn(pred, batch_y)
            loss.backward()
            optimizer.step()

    network.eval()
    return _TorchMLPModel(
        mean=mean.astype(np.float64),
        scale=scale.astype(np.float64),
        network=network,
        device=device,
    )


def _predict_torch_mlp(model: _TorchMLPModel, candidate: WideCandidate) -> float:
    x = ((_candidate_feature_vector(candidate) - model.mean) / model.scale).astype(np.float32)
    with torch.no_grad():
        x_tensor = torch.as_tensor(x.reshape(1, -1), device=model.device)
        pred = model.network(x_tensor).reshape(-1)[0]
    return float(pred.detach().cpu().item())


def _fit_global_and_symbol_tiny_nets(
    history_days: Sequence[Sequence[WideCandidate]],
    *,
    config: WideSelectionConfig,
    fee_bps: float,
    fill_buffer_bps: float,
) -> tuple[_TinyNetModel | None, dict[str, _TinyNetModel]]:
    flat = [candidate for day in history_days[-max(int(config.lookback_days), 0) :] for candidate in day]
    if not flat:
        return None, {}
    rng = np.random.default_rng(int(config.seed))
    x = np.vstack([_candidate_feature_vector(candidate) for candidate in flat])
    y = np.asarray(
        [
            estimate_candidate_daily_return(candidate, fee_bps=fee_bps, fill_buffer_bps=fill_buffer_bps)
            for candidate in flat
        ],
        dtype=np.float64,
    )
    global_model = _fit_tiny_net(x, y, config=config, rng=rng)
    by_symbol: dict[str, list[WideCandidate]] = {}
    for candidate in flat:
        by_symbol.setdefault(candidate.symbol, []).append(candidate)
    symbol_models: dict[str, _TinyNetModel] = {}
    for symbol, symbol_history in by_symbol.items():
        if len(symbol_history) < max(int(config.tiny_net_min_train_samples), 1):
            continue
        symbol_x = np.vstack([_candidate_feature_vector(candidate) for candidate in symbol_history])
        symbol_y = np.asarray(
            [
                estimate_candidate_daily_return(candidate, fee_bps=fee_bps, fill_buffer_bps=fill_buffer_bps)
                for candidate in symbol_history
            ],
            dtype=np.float64,
        )
        model = _fit_tiny_net(symbol_x, symbol_y, config=config, rng=np.random.default_rng(int(config.seed) + len(symbol)))
        if model is not None:
            symbol_models[symbol] = model
    return global_model, symbol_models


def _fit_global_and_symbol_torch_mlps(
    history_days: Sequence[Sequence[WideCandidate]],
    *,
    config: WideSelectionConfig,
    fee_bps: float,
    fill_buffer_bps: float,
) -> tuple[_TorchMLPModel | None, dict[str, _TorchMLPModel]]:
    flat = [candidate for day in history_days[-max(int(config.lookback_days), 0) :] for candidate in day]
    if not flat:
        return None, {}
    x = np.vstack([_candidate_feature_vector(candidate) for candidate in flat]).astype(np.float32)
    y = np.asarray(
        [
            estimate_candidate_daily_return(candidate, fee_bps=fee_bps, fill_buffer_bps=fill_buffer_bps)
            for candidate in flat
        ],
        dtype=np.float32,
    )
    global_model = _fit_torch_mlp(x, y, config=config)
    by_symbol: dict[str, list[WideCandidate]] = {}
    for candidate in flat:
        by_symbol.setdefault(candidate.symbol, []).append(candidate)
    symbol_models: dict[str, _TorchMLPModel] = {}
    for symbol, symbol_history in by_symbol.items():
        if len(symbol_history) < max(int(config.tiny_net_min_train_samples), 1):
            continue
        symbol_x = np.vstack([_candidate_feature_vector(candidate) for candidate in symbol_history]).astype(np.float32)
        symbol_y = np.asarray(
            [
                estimate_candidate_daily_return(candidate, fee_bps=fee_bps, fill_buffer_bps=fill_buffer_bps)
                for candidate in symbol_history
            ],
            dtype=np.float32,
        )
        symbol_seed = replace(config, seed=int(config.seed) + len(symbol))
        model = _fit_torch_mlp(symbol_x, symbol_y, config=symbol_seed)
        if model is not None:
            symbol_models[symbol] = model
    return global_model, symbol_models


def rank_candidates(
    candidates: Sequence[WideCandidate],
    *,
    history_days: Sequence[Sequence[WideCandidate]],
    config: WideSelectionConfig | None = None,
    fee_bps: float = 10.0,
    fill_buffer_bps: float = 5.0,
) -> list[WideCandidate]:
    selection = config or WideSelectionConfig()
    if not candidates:
        return []

    tiny_net_global = None
    tiny_net_by_symbol: dict[str, _TinyNetModel] = {}
    torch_mlp_global = None
    torch_mlp_by_symbol: dict[str, _TorchMLPModel] = {}
    if selection.objective == "tiny_net":
        tiny_net_global, tiny_net_by_symbol = _fit_global_and_symbol_tiny_nets(
            history_days,
            config=selection,
            fee_bps=fee_bps,
            fill_buffer_bps=fill_buffer_bps,
        )
    elif selection.objective == "torch_mlp":
        torch_mlp_global, torch_mlp_by_symbol = _fit_global_and_symbol_torch_mlps(
            history_days,
            config=selection,
            fee_bps=fee_bps,
            fill_buffer_bps=fill_buffer_bps,
        )

    ranked: list[WideCandidate] = []
    for candidate in candidates:
        history_returns = _history_returns_for_candidate(
            candidate,
            history_days,
            lookback_days=selection.lookback_days,
            fee_bps=fee_bps,
            fill_buffer_bps=fill_buffer_bps,
        )
        hit_rate = 0.0 if not history_returns else float(np.mean(np.asarray(history_returns, dtype=np.float64) > 0.0))
        sortino = _safe_sortino(history_returns)
        base_score = float(candidate.score)
        expected_edge = float(candidate.expected_return_pct)
        rl_prior_score = 0.0
        if selection.rl_prior_by_symbol:
            rl_prior_score = float(selection.rl_prior_by_symbol.get(candidate.symbol, 0.0))
        if selection.objective == "pnl":
            score = base_score
        elif selection.objective == "sortino":
            score = (0.65 * sortino) + (0.20 * hit_rate) + (0.10 * candidate.forecasted_pnl) + (0.05 * expected_edge)
        elif selection.objective == "hybrid":
            score = (
                (0.40 * candidate.forecasted_pnl)
                + (0.20 * candidate.avg_return)
                + (0.15 * expected_edge)
                + (0.20 * sortino)
                + (0.05 * hit_rate)
            )
        elif selection.objective == "tiny_net":
            model = tiny_net_by_symbol.get(candidate.symbol) or tiny_net_global
            predicted_return = base_score if model is None else _predict_tiny_net(model, candidate)
            score = (0.65 * predicted_return) + (0.20 * candidate.forecasted_pnl) + (0.10 * sortino) + (0.05 * hit_rate)
        elif selection.objective == "torch_mlp":
            model = torch_mlp_by_symbol.get(candidate.symbol) or torch_mlp_global
            predicted_return = base_score if model is None else _predict_torch_mlp(model, candidate)
            score = (0.70 * predicted_return) + (0.15 * candidate.forecasted_pnl) + (0.10 * sortino) + (0.05 * hit_rate)
        else:
            raise ValueError(f"unsupported selection objective: {selection.objective!r}")
        score += float(selection.rl_prior_weight) * float(rl_prior_score)
        ranked.append(replace(candidate, score=float(score), rl_prior_score=float(rl_prior_score)))

    return sorted(
        ranked,
        key=lambda item: (item.score, item.forecasted_pnl, item.avg_return),
        reverse=True,
    )


def rerank_candidate_days(
    candidate_days: Sequence[Sequence[WideCandidate]],
    *,
    config: WideSelectionConfig | None = None,
    fee_bps: float = 10.0,
    fill_buffer_bps: float = 5.0,
) -> list[list[WideCandidate]]:
    selection = config or WideSelectionConfig()
    history: list[Sequence[WideCandidate]] = []
    reranked: list[list[WideCandidate]] = []
    for day in candidate_days:
        reranked_day = rank_candidates(
            day,
            history_days=history,
            config=selection,
            fee_bps=fee_bps,
            fill_buffer_bps=fill_buffer_bps,
        )
        reranked.append(reranked_day)
        history.append(tuple(day))
    return reranked


__all__ = [
    "WideSelectionConfig",
    "build_symbol_rl_prior",
    "estimate_candidate_daily_return",
    "normalize_rl_prior_score",
    "rank_candidates",
    "resolve_torch_device",
    "rerank_candidate_days",
]
