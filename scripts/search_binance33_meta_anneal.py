#!/usr/bin/env python3
"""Anneal blended Binance33 portfolio strategies before exact validation.

This is an inference-time learning / strategy-combination research harness:

1. Build multiple daily cross-sectional score channels from linear rules,
   handcrafted momentum features, and optionally XGBoost experiments.
2. Anneal channel weights plus threshold/gate/leverage/top-K/long-short knobs on
   training history with a fast vectorized simulator.
3. Re-score the best candidates on untouched validation.
4. For single-position candidates, replay the best blends through the existing
   exact action simulator so they are comparable with the current leaderboard.

The fast simulator is for search efficiency. Exact simulator rows are the only
rows that should be treated as serious promotion candidates.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from pufferlib_market.hourly_replay import MktdData, P_CLOSE, P_HIGH, P_LOW, read_mktd, simulate_daily_policy
from scripts.search_binance33_linear_rules import FEATURE_NAMES, LinearRule, _score_vector
from scripts.sweep_binance33_rules import FEATURES, _candidate_starts, _slice_window
from scripts.sweep_binance33_xgb import (
    _build_dataset,
    _experiments,
    _monthly_equivalent_return,
    _passes_production_target,
    _precompute_scores,
    _train_xgb,
)
from src.robust_trading_metrics import compute_pnl_smoothness_from_equity, compute_ulcer_index


@dataclass(frozen=True)
class ScoreBank:
    names: list[str]
    scores: np.ndarray  # float64 [C, T, S], lower means stronger short, higher means stronger long


@dataclass(frozen=True)
class Candidate:
    candidate_id: str
    logits: np.ndarray
    threshold: float
    max_gross: float
    max_weight: float
    top_k: int
    book_mode: str
    score_temp: float
    btc_gate: float
    market_gate: float
    rebalance_days: int
    long_fraction: float = 0.5
    short_risk_mult: float = 1.0
    always_trade: bool = False


def _parse_float_list(value: str) -> list[float]:
    out = [float(part.strip()) for part in str(value).split(",") if part.strip()]
    if not out:
        raise ValueError(f"expected at least one float in {value!r}")
    return out


def _parse_int_list(value: str) -> list[int]:
    out = [int(part.strip()) for part in str(value).split(",") if part.strip()]
    if not out:
        raise ValueError(f"expected at least one int in {value!r}")
    return out


def _parse_str_list(value: str) -> list[str]:
    return [part.strip() for part in str(value).split(",") if part.strip()]


def _file_cache_tag(path: Path) -> str:
    resolved = Path(path).resolve()
    stat = resolved.stat()
    raw = f"{resolved}:{stat.st_size}:{int(stat.st_mtime)}"
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:12]


def _softmax(values: np.ndarray, *, temp: float = 1.0) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64) / max(1e-9, float(temp))
    arr = arr - float(np.max(arr))
    exp = np.exp(arr)
    total = float(exp.sum())
    if total <= 0.0 or not np.isfinite(total):
        return np.ones_like(arr, dtype=np.float64) / max(1, arr.size)
    return exp / total


def _normalize_score_matrix(scores: np.ndarray, valid: np.ndarray | None = None) -> np.ndarray:
    values = np.asarray(scores, dtype=np.float64).copy()
    if valid is not None:
        values = np.where(np.asarray(valid, dtype=bool), values, np.nan)
    for t in range(values.shape[0]):
        row = values[t]
        finite = np.isfinite(row)
        if int(finite.sum()) <= 1:
            values[t] = np.nan
            continue
        mean = float(row[finite].mean())
        std = float(row[finite].std())
        values[t, finite] = (row[finite] - mean) / max(std, 1e-6)
        values[t, ~finite] = np.nan
    return values


def _load_rule_rows(path: Path, *, configs: Sequence[str], max_rules: int) -> list[LinearRule]:
    rows: list[dict[str, str]] = []
    with path.open(newline="") as fh:
        rows = list(csv.DictReader(fh))
    if configs:
        wanted = list(dict.fromkeys(configs))
        selected = [row for name in wanted for row in rows if row.get("config") == name]
    else:
        def key(row: dict[str, str]) -> float:
            dd = float(row.get("p90_dd_pct") or 100.0)
            median = float(row.get("median_monthly_pct") or -999.0)
            p10 = float(row.get("p10_monthly_pct") or -999.0)
            neg = int(float(row.get("neg_windows") or 99))
            return median + 0.5 * p10 - 1.5 * max(0.0, dd - 20.0) - 20.0 * neg

        by_config: dict[str, dict[str, str]] = {}
        for row in sorted(rows, key=key, reverse=True):
            by_config.setdefault(str(row.get("config", "")), row)
        selected = list(by_config.values())
    out: list[LinearRule] = []
    seen: set[str] = set()
    for row in selected:
        name = str(row.get("config", ""))
        if not name or name in seen:
            continue
        coeff_map = json.loads(str(row["coeffs_json"]))
        coeffs = np.asarray([float(coeff_map[feature]) for feature in FEATURE_NAMES], dtype=np.float64)
        out.append(
            LinearRule(
                name=name,
                coeffs=coeffs,
                rebalance_days=int(float(row["rebalance_days"])),
                min_abs_score=float(row["min_abs_score"]),
                btc_min_return_20d=float(row["btc_min_return_20d"]),
            )
        )
        seen.add(name)
        if len(out) >= int(max_rules):
            break
    if not out:
        raise ValueError(f"no linear rules loaded from {path}")
    return out


def _valid_mask(data: MktdData) -> np.ndarray:
    close = np.asarray(data.prices[:, :, P_CLOSE], dtype=np.float64)
    mask = np.isfinite(close) & (close > 0.0)
    if data.tradable is not None:
        mask &= np.asarray(data.tradable, dtype=bool)
    return mask


def _slice_data(data: MktdData, start: int, end: int) -> MktdData:
    start = max(0, int(start))
    end = min(int(end), data.num_timesteps)
    if end <= start:
        raise ValueError(f"empty data slice start={start} end={end}")
    return MktdData(
        version=int(data.version),
        symbols=list(data.symbols),
        features=data.features[start:end].copy(),
        prices=data.prices[start:end].copy(),
        tradable=data.tradable[start:end].copy() if data.tradable is not None else None,
    )


def _handcrafted_channels(data: MktdData) -> list[tuple[str, np.ndarray]]:
    feat = np.asarray(data.features, dtype=np.float64)
    valid = _valid_mask(data)
    channels = [
        ("hand_winner_1d", -feat[:, :, FEATURES["return_1d"]]),
        ("hand_winner_5d", -feat[:, :, FEATURES["return_5d"]]),
        ("hand_winner_20d", -feat[:, :, FEATURES["return_20d"]]),
        ("hand_low_rsi_winner5", -feat[:, :, FEATURES["return_5d"]] + 0.25 * feat[:, :, FEATURES["rsi_14"]]),
        ("hand_drawdown_rebound", feat[:, :, FEATURES["drawdown_20d"]] - 0.5 * feat[:, :, FEATURES["return_5d"]]),
        ("hand_lowvol_winner5", -feat[:, :, FEATURES["return_5d"]] + 0.5 * feat[:, :, FEATURES["volatility_20d"]]),
    ]
    return [(name, _normalize_score_matrix(scores, valid)) for name, scores in channels]


def _linear_channels(data: MktdData, rules: Sequence[LinearRule]) -> list[tuple[str, np.ndarray]]:
    out: list[tuple[str, np.ndarray]] = []
    for rule in rules:
        rows: list[np.ndarray] = []
        valid_rows: list[np.ndarray] = []
        for t in range(data.num_timesteps):
            scores, valid = _score_vector(data, rule, t)
            rows.append(scores)
            valid_rows.append(valid)
        out.append((f"linear:{rule.name}", _normalize_score_matrix(np.vstack(rows), np.vstack(valid_rows))))
    return out


def _train_xgb_channel_models(
    train_data: MktdData,
    *,
    experiment_names: Sequence[str],
    rounds: int,
    device: str,
    model_dir: Path | None = None,
    cache_tag: str = "",
) -> list[tuple[str, object]]:
    if not experiment_names:
        return []
    by_name = {exp.name: exp for exp in _experiments()}
    missing = sorted(set(experiment_names) - set(by_name))
    if missing:
        raise ValueError(f"unknown XGB experiments: {', '.join(missing)}")
    out: list[tuple[str, object]] = []
    if model_dir is not None:
        Path(model_dir).mkdir(parents=True, exist_ok=True)
    for idx, name in enumerate(experiment_names, start=1):
        exp = by_name[name]
        cache_path = (
            Path(model_dir) / f"{name}_rounds{int(rounds)}_{str(device)}_{cache_tag}.json"
            if model_dir is not None
            else None
        )
        if cache_path is not None and cache_path.exists():
            print(f"xgb load {idx}/{len(experiment_names)} {name} {cache_path}", flush=True)
            import xgboost as xgb

            model = xgb.Booster()
            model.load_model(str(cache_path))
            out.append((name, model))
            continue
        print(f"xgb train {idx}/{len(experiment_names)} {name} horizon={exp.horizon} label={exp.label}", flush=True)
        x_train, y_train = _build_dataset(train_data, horizon=exp.horizon, label=exp.label)
        model = _train_xgb(x_train, y_train, exp, rounds=int(rounds), device=str(device))
        if cache_path is not None:
            model.save_model(str(cache_path))
        out.append((name, model))
    return out


def _xgb_channels(data: MktdData, models: Sequence[tuple[str, object]]) -> list[tuple[str, np.ndarray]]:
    valid = _valid_mask(data)
    out: list[tuple[str, np.ndarray]] = []
    for idx, (name, model) in enumerate(models, start=1):
        print(f"xgb score {idx}/{len(models)} {name} T={data.num_timesteps}", flush=True)
        out.append((f"xgb_in_sample:{name}", _normalize_score_matrix(_precompute_scores(data, model), valid)))
    return out


def _build_bank(
    data: MktdData,
    *,
    rules: Sequence[LinearRule],
    xgb_models: Sequence[tuple[str, object]],
    include_handcrafted: bool,
) -> ScoreBank:
    channels: list[tuple[str, np.ndarray]] = []
    channels.extend(_linear_channels(data, rules))
    if include_handcrafted:
        channels.extend(_handcrafted_channels(data))
    if xgb_models:
        channels.extend(_xgb_channels(data, xgb_models))
    names = [name for name, _scores in channels]
    scores = np.stack([scores for _name, scores in channels], axis=0).astype(np.float64, copy=False)
    return ScoreBank(names=names, scores=scores)


def _combine_scores(bank: ScoreBank, candidate: Candidate) -> np.ndarray:
    weights = _softmax(candidate.logits)
    return np.tensordot(weights, bank.scores, axes=(0, 0)).astype(np.float64, copy=False)


def _btc_idx(data: MktdData) -> int:
    return next((idx for idx, symbol in enumerate(data.symbols) if symbol.upper().startswith("BTC")), 0)


def _gate_allows(data: MktdData, *, t: int, btc_idx: int, btc_gate: float, market_gate: float) -> bool:
    t = max(0, min(int(t), data.num_timesteps - 1))
    if float(btc_gate) > -9.0:
        btc_ret20 = float(data.features[t, btc_idx, FEATURES["return_20d"]])
        if not np.isfinite(btc_ret20) or btc_ret20 < float(btc_gate):
            return False
    if float(market_gate) > -9.0:
        values = np.asarray(data.features[t, :, FEATURES["return_20d"]], dtype=np.float64)
        valid = np.isfinite(values)
        if data.tradable is not None:
            valid &= np.asarray(data.tradable[t], dtype=bool)
        if int(valid.sum()) <= 0 or float(np.median(values[valid])) < float(market_gate):
            return False
    return True


def _normalise_alloc(raw: np.ndarray, *, gross: float, max_weight: float) -> np.ndarray:
    raw = np.asarray(raw, dtype=np.float64)
    raw = np.where(np.isfinite(raw) & (raw > 0.0), raw, 0.0)
    gross = max(0.0, float(gross))
    if gross <= 0.0 or float(raw.sum()) <= 0.0:
        return np.zeros_like(raw, dtype=np.float64)
    alloc = raw / float(raw.sum()) * gross
    cap = max(0.0, float(max_weight))
    if cap <= 0.0:
        return alloc
    for _ in range(raw.size + 1):
        over = alloc > cap
        if not over.any():
            break
        alloc[over] = cap
        remaining = ~over & (raw > 0.0)
        rem_gross = gross - float(alloc[over].sum())
        if rem_gross <= 0.0 or not remaining.any():
            alloc[remaining] = 0.0
            break
        alloc[remaining] = raw[remaining] / float(raw[remaining].sum()) * rem_gross
    total = float(alloc.sum())
    if total > gross and total > 0.0:
        alloc *= gross / total
    return alloc


def _ranked_side(
    scores: np.ndarray,
    valid: np.ndarray,
    *,
    side: str,
    threshold: float,
    top_k: int,
    always_trade: bool,
) -> np.ndarray:
    scores = np.asarray(scores, dtype=np.float64)
    valid = np.asarray(valid, dtype=bool) & np.isfinite(scores)
    if str(side) == "long":
        mask = valid & (scores >= float(threshold))
        if not mask.any() and bool(always_trade):
            mask = valid
        candidates = np.flatnonzero(mask)
        if candidates.size == 0:
            return candidates
        ordered = candidates[np.argsort(-scores[candidates])]
    elif str(side) == "short":
        mask = valid & (-scores >= float(threshold))
        if not mask.any() and bool(always_trade):
            mask = valid
        candidates = np.flatnonzero(mask)
        if candidates.size == 0:
            return candidates
        ordered = candidates[np.argsort(scores[candidates])]
    else:
        raise ValueError(f"unknown side: {side}")
    return ordered[: max(1, int(top_k))]


def _side_alloc(scores: np.ndarray, symbols: np.ndarray, *, side: str, gross: float, max_weight: float, temp: float) -> np.ndarray:
    if symbols.size == 0:
        return np.asarray([], dtype=np.float64)
    if str(side) == "long":
        edge = np.maximum(np.asarray(scores[symbols], dtype=np.float64), 1e-9)
    elif str(side) == "short":
        edge = np.maximum(-np.asarray(scores[symbols], dtype=np.float64), 1e-9)
    else:
        raise ValueError(f"unknown side: {side}")
    temp = max(1e-6, float(temp))
    logits = edge / temp
    logits = logits - float(np.max(logits))
    raw = np.exp(logits)
    return _normalise_alloc(raw, gross=float(gross), max_weight=float(max_weight))


def _desired_weights(
    data: MktdData,
    scores_by_t: np.ndarray,
    candidate: Candidate,
    *,
    t: int,
    btc_idx: int,
) -> np.ndarray:
    weights = np.zeros(data.num_symbols, dtype=np.float64)
    if not _gate_allows(
        data,
        t=t,
        btc_idx=btc_idx,
        btc_gate=float(candidate.btc_gate),
        market_gate=float(candidate.market_gate),
    ):
        return weights
    t = max(0, min(int(t), data.num_timesteps - 1))
    scores = np.asarray(scores_by_t[t], dtype=np.float64)
    valid = np.isfinite(scores)
    if data.tradable is not None:
        valid &= np.asarray(data.tradable[t], dtype=bool)
    mode = str(candidate.book_mode)
    always_trade = bool(candidate.always_trade)
    threshold = float(candidate.threshold)
    top_k = max(1, int(candidate.top_k))
    if mode == "single":
        top = _ranked_side(
            scores,
            valid,
            side="short",
            threshold=threshold,
            top_k=top_k,
            always_trade=always_trade,
        )
        if top.size == 0:
            return weights
        weights[int(top[0])] = -float(candidate.max_gross) / max(1e-9, float(candidate.short_risk_mult))
        return weights
    if mode == "long_single":
        top = _ranked_side(
            scores,
            valid,
            side="long",
            threshold=threshold,
            top_k=top_k,
            always_trade=always_trade,
        )
        if top.size == 0:
            return weights
        weights[int(top[0])] = float(candidate.max_gross)
        return weights
    if mode == "portfolio":
        top = _ranked_side(
            scores,
            valid,
            side="short",
            threshold=threshold,
            top_k=top_k,
            always_trade=always_trade,
        )
        alloc = _side_alloc(
            scores,
            top,
            side="short",
            gross=float(candidate.max_gross) / max(1e-9, float(candidate.short_risk_mult)),
            max_weight=float(candidate.max_weight),
            temp=float(candidate.score_temp),
        )
        weights[top] = -alloc
        return weights
    if mode == "long_portfolio":
        top = _ranked_side(
            scores,
            valid,
            side="long",
            threshold=threshold,
            top_k=top_k,
            always_trade=always_trade,
        )
        alloc = _side_alloc(
            scores,
            top,
            side="long",
            gross=float(candidate.max_gross),
            max_weight=float(candidate.max_weight),
            temp=float(candidate.score_temp),
        )
        weights[top] = alloc
        return weights
    if mode == "longshort_portfolio":
        long_fraction = float(np.clip(candidate.long_fraction, 0.0, 1.0))
        if long_fraction <= 0.0:
            long_k, short_k = 0, top_k
        elif long_fraction >= 1.0:
            long_k, short_k = top_k, 0
        elif top_k <= 1:
            long_k, short_k = (1, 0) if long_fraction >= 0.5 else (0, 1)
        else:
            long_k = max(1, min(top_k - 1, int(round(top_k * long_fraction))))
            short_k = max(1, top_k - long_k)
        long_top = _ranked_side(
            scores,
            valid,
            side="long",
            threshold=threshold,
            top_k=long_k,
            always_trade=always_trade,
        ) if long_k > 0 else np.asarray([], dtype=np.int64)
        short_valid = valid.copy()
        if long_top.size:
            short_valid[long_top] = False
        short_top = _ranked_side(
            scores,
            short_valid,
            side="short",
            threshold=threshold,
            top_k=short_k,
            always_trade=always_trade,
        ) if short_k > 0 else np.asarray([], dtype=np.int64)
        long_alloc = _side_alloc(
            scores,
            long_top,
            side="long",
            gross=float(candidate.max_gross) * long_fraction,
            max_weight=float(candidate.max_weight),
            temp=float(candidate.score_temp),
        )
        short_alloc = _side_alloc(
            scores,
            short_top,
            side="short",
            gross=float(candidate.max_gross) * (1.0 - long_fraction) / max(1e-9, float(candidate.short_risk_mult)),
            max_weight=float(candidate.max_weight),
            temp=float(candidate.score_temp),
        )
        weights[long_top] = long_alloc
        weights[short_top] = -short_alloc
        return weights
    raise ValueError(f"unknown book_mode: {candidate.book_mode}")
    return weights


def _tradable_at(data: MktdData, t: int) -> np.ndarray:
    t = max(0, min(int(t), data.num_timesteps - 1))
    close = np.asarray(data.prices[t, :, P_CLOSE], dtype=np.float64)
    tradable = np.isfinite(close) & (close > 0.0)
    if data.tradable is not None:
        tradable &= np.asarray(data.tradable[t], dtype=bool)
    return tradable


def _apply_binary_fills(
    data: MktdData,
    current: np.ndarray,
    desired: np.ndarray,
    *,
    t: int,
    fill_buffer_bps: float,
) -> np.ndarray:
    t = max(0, min(int(t), data.num_timesteps - 1))
    current = np.asarray(current, dtype=np.float64)
    desired = np.asarray(desired, dtype=np.float64)
    out = current.copy()
    close = np.asarray(data.prices[t, :, P_CLOSE], dtype=np.float64)
    low = np.asarray(data.prices[t, :, P_LOW], dtype=np.float64)
    high = np.asarray(data.prices[t, :, P_HIGH], dtype=np.float64)
    tradable = _tradable_at(data, t) & np.isfinite(low) & np.isfinite(high) & (low > 0.0) & (high > 0.0)
    long_fill = low <= close * (1.0 - max(0.0, float(fill_buffer_bps)) / 10_000.0)
    short_fill = high >= close * (1.0 + max(0.0, float(fill_buffer_bps)) / 10_000.0)
    touched = np.flatnonzero((np.abs(current) > 1e-12) | (np.abs(desired) > 1e-12))
    for sym in touched:
        if not bool(tradable[int(sym)]):
            continue
        old = float(current[int(sym)])
        want = float(desired[int(sym)])
        if abs(want - old) <= 1e-12:
            out[int(sym)] = old
            continue
        if old >= 0.0 and want >= 0.0:
            # Selling/reducing a long is allowed immediately; opening or adding
            # long exposure must be reachable by the binary limit-fill rule.
            out[int(sym)] = want if want <= old or bool(long_fill[int(sym)]) else old
        elif old <= 0.0 and want <= 0.0:
            # Covering/reducing a short is allowed immediately; opening or
            # adding short exposure must be reachable by the binary fill rule.
            out[int(sym)] = want if want >= old or bool(short_fill[int(sym)]) else old
        elif old < 0.0 < want:
            out[int(sym)] = want if bool(long_fill[int(sym)]) else 0.0
        elif old > 0.0 > want:
            out[int(sym)] = want if bool(short_fill[int(sym)]) else 0.0
    return out


def _apply_short_binary_fills(
    data: MktdData,
    current: np.ndarray,
    desired: np.ndarray,
    *,
    t: int,
    fill_buffer_bps: float,
) -> np.ndarray:
    current_short = np.minimum(np.asarray(current, dtype=np.float64), 0.0)
    desired_short = np.minimum(np.asarray(desired, dtype=np.float64), 0.0)
    return _apply_binary_fills(
        data,
        current_short,
        desired_short,
        t=int(t),
        fill_buffer_bps=float(fill_buffer_bps),
    )


def _max_drawdown(equity: np.ndarray) -> float:
    if equity.size == 0:
        return 0.0
    peak = np.maximum.accumulate(equity)
    with np.errstate(divide="ignore", invalid="ignore"):
        dd = np.where(peak > 0.0, equity / peak - 1.0, 0.0)
    return float(max(0.0, -np.nanmin(dd)))


def _sortino_from_equity(equity: np.ndarray, *, periods_per_year: float = 365.0) -> float:
    if equity.size <= 1:
        return 0.0
    prev = equity[:-1]
    cur = equity[1:]
    valid = np.isfinite(prev) & np.isfinite(cur) & (prev > 0.0)
    if not valid.any():
        return 0.0
    returns = cur[valid] / prev[valid] - 1.0
    downside = returns[returns < 0.0]
    if downside.size == 0:
        return float("inf") if float(returns.mean()) > 0.0 else 0.0
    denom = float(downside.std())
    if denom <= 1e-12:
        return 0.0
    return float(returns.mean() / denom * np.sqrt(float(periods_per_year)))


def _evolve_weights_after_return(target: np.ndarray, gross_return: np.ndarray, growth: float) -> np.ndarray:
    if not np.isfinite(growth) or float(growth) <= 1e-8:
        return np.zeros_like(np.asarray(target, dtype=np.float64), dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        weights = np.where(np.abs(target) > 1e-12, np.asarray(target, dtype=np.float64) * gross_return / growth, 0.0)
    weights = np.where(np.isfinite(weights), weights, 0.0)
    gross = float(np.abs(weights).sum())
    if gross > 10.0:
        weights *= 10.0 / gross
    return weights


def _simulate_vector_window(
    data: MktdData,
    scores_by_t: np.ndarray,
    candidate: Candidate,
    *,
    start: int,
    eval_days: int,
    fee_rate: float,
    slippage_bps: float,
    fill_buffer_bps: float,
    decision_lag: int,
    short_borrow_apr: float,
    periods_per_year: float,
    btc_idx: int,
) -> dict[str, float | int | np.ndarray]:
    close = np.asarray(data.prices[:, :, P_CLOSE], dtype=np.float64)
    weights = np.zeros(data.num_symbols, dtype=np.float64)
    equity = 1.0
    curve = [equity]
    trades = 0
    cost_per_turnover = float(fee_rate) + float(slippage_bps) / 10_000.0
    borrow_per_period = max(0.0, float(short_borrow_apr)) / max(1.0, float(periods_per_year))
    for step in range(int(eval_days)):
        global_t = int(start) + int(step)
        target = weights
        local_signal_t = int(step) - int(decision_lag)
        signal_t = int(start) + local_signal_t
        if local_signal_t >= 0 and (step == 0 or step % max(1, int(candidate.rebalance_days)) == 0):
            desired = _desired_weights(data, scores_by_t, candidate, t=signal_t, btc_idx=btc_idx)
            old_idx = np.flatnonzero(np.abs(weights) > 1e-12)
            new_idx = np.flatnonzero(np.abs(desired) > 1e-12)
            same_book = np.array_equal(old_idx, new_idx) and np.all(np.sign(weights[old_idx]) == np.sign(desired[old_idx]))
            target = weights if same_book and not bool(candidate.always_trade) else desired
        target = _apply_binary_fills(
            data,
            weights,
            target,
            t=global_t,
            fill_buffer_bps=float(fill_buffer_bps),
        )
        turnover = float(np.abs(target - weights).sum())
        if turnover > 1e-9:
            trades += int(np.count_nonzero(np.abs(target - weights) > 1e-9))
        p0 = close[global_t]
        p1 = close[global_t + 1]
        valid = np.isfinite(p0) & np.isfinite(p1) & (p0 > 0.0) & (p1 > 0.0)
        day_ret = np.zeros(data.num_symbols, dtype=np.float64)
        day_ret[valid] = p1[valid] / p0[valid] - 1.0
        pnl = float(np.dot(target, day_ret))
        borrow = float(np.abs(target[target < 0.0]).sum()) * borrow_per_period
        cost = turnover * cost_per_turnover
        growth = max(1e-9, 1.0 + pnl - cost - borrow)
        equity = max(1e-9, equity * growth)
        gross_return = np.where(valid, 1.0 + day_ret, 1.0)
        weights = _evolve_weights_after_return(target, gross_return, growth)
        curve.append(equity)
    final_turnover = float(np.abs(weights).sum())
    if final_turnover > 1e-9:
        trades += int(np.count_nonzero(np.abs(weights) > 1e-9))
        equity = max(1e-9, equity * (1.0 - final_turnover * cost_per_turnover))
        curve[-1] = equity
    equity_curve = np.asarray(curve, dtype=np.float64)
    return {
        "total_return": float(equity_curve[-1] - 1.0),
        "max_drawdown": _max_drawdown(equity_curve),
        "sortino": _sortino_from_equity(equity_curve, periods_per_year=float(periods_per_year)),
        "trades": int(trades),
        "equity_curve": equity_curve,
    }


def _summarise_results(
    results: Sequence[dict[str, float | int | np.ndarray]],
    *,
    candidate: Candidate,
    phase: str,
    eval_days: int,
    slippage_bps: float,
    fill_buffer_bps: float,
    target_monthly_pct: float,
    target_max_dd_pct: float,
    channel_weights: dict[str, float],
) -> dict[str, float | int | str]:
    returns = np.asarray([float(result["total_return"]) for result in results], dtype=np.float64)
    maxdds = np.asarray([float(result["max_drawdown"]) for result in results], dtype=np.float64)
    sortinos = np.asarray([float(result["sortino"]) for result in results], dtype=np.float64)
    trades = np.asarray([int(result["trades"]) for result in results], dtype=np.float64)
    monthly = np.asarray([_monthly_equivalent_return(float(ret), int(eval_days)) for ret in returns], dtype=np.float64)
    smooths = []
    ulcers = []
    for result in results:
        curve = np.asarray(result.get("equity_curve", []), dtype=np.float64)
        smooths.append(float(compute_pnl_smoothness_from_equity(curve)))
        ulcers.append(float(compute_ulcer_index(curve)))
    row: dict[str, float | int | str] = {
        "phase": phase,
        "candidate_id": candidate.candidate_id,
        "book_mode": candidate.book_mode,
        "top_k": int(candidate.top_k),
        "max_gross": float(candidate.max_gross),
        "max_weight": float(candidate.max_weight),
        "threshold": float(candidate.threshold),
        "score_temp": float(candidate.score_temp),
        "btc_gate": float(candidate.btc_gate),
        "market_gate": float(candidate.market_gate),
        "rebalance_days": int(candidate.rebalance_days),
        "long_fraction": float(candidate.long_fraction),
        "short_risk_mult": float(candidate.short_risk_mult),
        "always_trade": int(bool(candidate.always_trade)),
        "channel_weights_json": json.dumps(channel_weights, sort_keys=True),
        "eval_days": int(eval_days),
        "slip_bps": float(slippage_bps),
        "fill_buffer_bps": float(fill_buffer_bps),
        "median_pct": float(100.0 * np.percentile(returns, 50)),
        "p10_pct": float(100.0 * np.percentile(returns, 10)),
        "p90_pct": float(100.0 * np.percentile(returns, 90)),
        "median_monthly_pct": float(100.0 * np.percentile(monthly, 50)),
        "p10_monthly_pct": float(100.0 * np.percentile(monthly, 10)),
        "worst_monthly_pct": float(100.0 * np.min(monthly)),
        "best_monthly_pct": float(100.0 * np.max(monthly)),
        "neg_windows": int(np.sum(returns < 0.0)),
        "windows": int(returns.size),
        "p90_dd_pct": float(100.0 * np.percentile(maxdds, 90)),
        "median_smooth": float(np.percentile(np.asarray(smooths, dtype=np.float64), 50)),
        "median_ulcer": float(np.percentile(np.asarray(ulcers, dtype=np.float64), 50)),
        "median_sortino": float(np.percentile(sortinos, 50)),
        "median_trades": float(np.percentile(trades, 50)),
        "target_monthly_pct": float(target_monthly_pct),
        "target_max_dd_pct": float(target_max_dd_pct),
    }
    row["passes_target"] = int(
        _passes_production_target(
            row,
            target_monthly_pct=float(target_monthly_pct),
            max_dd_pct=float(target_max_dd_pct),
        )
    )
    row["objective"] = float(
        row["median_monthly_pct"]
        + 0.75 * row["p10_monthly_pct"]
        - 1.25 * row["p90_dd_pct"]
        - 18.0 * row["neg_windows"]
        - max(0.0, 10.0 - float(row["median_trades"]))
    )
    return row


def _channel_weights(bank: ScoreBank, candidate: Candidate, *, max_items: int = 12) -> dict[str, float]:
    weights = _softmax(candidate.logits)
    order = np.argsort(weights)[::-1][: int(max_items)]
    return {bank.names[int(idx)]: round(float(weights[int(idx)]), 6) for idx in order if weights[int(idx)] > 1e-4}


def _candidate_json(bank: ScoreBank, candidate: Candidate) -> str:
    return json.dumps(
        {
            "candidate_id": candidate.candidate_id,
            "channel_names": list(bank.names),
            "channel_weights": _softmax(candidate.logits).round(10).tolist(),
            "threshold": float(candidate.threshold),
            "max_gross": float(candidate.max_gross),
            "max_weight": float(candidate.max_weight),
            "top_k": int(candidate.top_k),
            "book_mode": str(candidate.book_mode),
            "score_temp": float(candidate.score_temp),
            "btc_gate": float(candidate.btc_gate),
            "market_gate": float(candidate.market_gate),
            "rebalance_days": int(candidate.rebalance_days),
            "long_fraction": float(candidate.long_fraction),
            "short_risk_mult": float(candidate.short_risk_mult),
            "always_trade": bool(candidate.always_trade),
        },
        sort_keys=True,
        separators=(",", ":"),
    )


def _eval_vector_candidate(
    data: MktdData,
    bank: ScoreBank,
    candidate: Candidate,
    *,
    phase: str,
    eval_days: int,
    stride: int,
    fee_rate: float,
    slippage_bps: float,
    fill_buffer_bps: float,
    decision_lag: int,
    short_borrow_apr: float,
    periods_per_year: float,
    target_monthly_pct: float,
    target_max_dd_pct: float,
) -> dict[str, float | int | str]:
    scores = _combine_scores(bank, candidate)
    btc_idx = _btc_idx(data)
    results = [
        _simulate_vector_window(
            data,
            scores,
            candidate,
            start=int(start),
            eval_days=int(eval_days),
            fee_rate=float(fee_rate),
            slippage_bps=float(slippage_bps),
            fill_buffer_bps=float(fill_buffer_bps),
            decision_lag=int(decision_lag),
            short_borrow_apr=float(short_borrow_apr),
            periods_per_year=float(periods_per_year),
            btc_idx=int(btc_idx),
        )
        for start in _candidate_starts(data, eval_days, stride=int(stride))
    ]
    row = _summarise_results(
        results,
        candidate=candidate,
        phase=phase,
        eval_days=int(eval_days),
        slippage_bps=float(slippage_bps),
        fill_buffer_bps=float(fill_buffer_bps),
        target_monthly_pct=float(target_monthly_pct),
        target_max_dd_pct=float(target_max_dd_pct),
        channel_weights=_channel_weights(bank, candidate),
    )
    row["candidate_json"] = _candidate_json(bank, candidate)
    return row


def _resolve_trace_start(data: MktdData, eval_days: int, requested: str) -> int:
    max_start = int(data.num_timesteps) - int(eval_days) - 1
    if max_start < 0:
        raise ValueError(f"data has {data.num_timesteps} bars; cannot make {eval_days}d + 1-bar trace")
    value = str(requested).strip().lower()
    if value in {"latest", "last", "auto"}:
        return max_start
    start = int(value)
    if start < 0 or start > max_start:
        raise ValueError(f"trace_window_start must be in [0, {max_start}], got {start}")
    return start


def _position_rows_for_weights(
    *,
    symbols: Sequence[str],
    weights: np.ndarray,
    entry_prices: np.ndarray,
    close: np.ndarray,
    equity: float,
    bar_index: int,
    initial_cash: float,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    weights = np.asarray(weights, dtype=np.float64)
    close = np.asarray(close, dtype=np.float64)
    entry_prices = np.asarray(entry_prices, dtype=np.float64)
    for sym in np.flatnonzero(np.abs(weights) > 1e-8):
        price = float(close[int(sym)])
        if not np.isfinite(price) or price <= 0.0:
            continue
        weight = float(weights[int(sym)])
        entry = float(entry_prices[int(sym)])
        if not np.isfinite(entry) or entry <= 0.0:
            entry = price
        notional = abs(weight) * max(1e-9, float(equity)) * float(initial_cash)
        rows.append(
            {
                "bar_index": int(bar_index),
                "symbol": str(symbols[int(sym)]).upper(),
                "side": "short" if weight < 0.0 else "long",
                "qty": float(notional / price),
                "entry_price": float(entry),
                "weight": float(weight),
            }
        )
    rows.sort(key=lambda row: str(row["symbol"]))
    return rows


def _trade_event_rows(
    *,
    symbols: Sequence[str],
    old_weights: np.ndarray,
    new_weights: np.ndarray,
    close: np.ndarray,
    equity: float,
    bar_index: int,
    label: str,
    initial_cash: float,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    old_weights = np.asarray(old_weights, dtype=np.float64)
    new_weights = np.asarray(new_weights, dtype=np.float64)
    close = np.asarray(close, dtype=np.float64)

    def add(sym: int, side: str, weight_delta: float) -> None:
        price = float(close[int(sym)])
        if not np.isfinite(price) or price <= 0.0 or abs(float(weight_delta)) <= 1e-10:
            return
        qty = abs(float(weight_delta)) * max(1e-9, float(equity)) * float(initial_cash) / price
        rows.append(
            {
                "step": int(bar_index),
                "bar_index": int(bar_index),
                "x": str(label),
                "symbol": str(symbols[int(sym)]).upper(),
                "symbol_index": int(sym),
                "side": str(side),
                "price": float(price),
                "qty": float(qty),
                "action": 0,
                "reason": "vector_rebalance",
                "is_short": bool(side == "short_sell"),
            }
        )

    for sym in np.flatnonzero(np.abs(new_weights - old_weights) > 1e-9):
        old = float(old_weights[int(sym)])
        new = float(new_weights[int(sym)])
        if old > 1e-9 and new <= 0.0:
            add(int(sym), "sell", old)
        if old < -1e-9 and new >= 0.0:
            add(int(sym), "buy_cover", old)
        if new > 1e-9 and old <= 0.0:
            add(int(sym), "buy", new)
        if new < -1e-9 and old >= 0.0:
            add(int(sym), "short_sell", new)
        if old > 1e-9 and new > 1e-9:
            add(int(sym), "buy" if new > old else "sell", new - old)
        if old < -1e-9 and new < -1e-9:
            add(int(sym), "short_sell" if new < old else "buy_cover", new - old)
    return rows


def _action_summary_row(weights: np.ndarray, *, labels: Sequence[str], symbols: Sequence[str], bar_index: int) -> dict[str, Any]:
    weights = np.asarray(weights, dtype=np.float64)
    if weights.size == 0 or not np.any(np.abs(weights) > 1e-12):
        return {"bar_index": int(bar_index), "x": labels[int(bar_index)], "action": 0, "symbol": "", "side": "flat", "symbol_index": -1}
    sym = int(np.argmax(np.abs(weights)))
    side = "short" if float(weights[sym]) < 0.0 else "long"
    action = 1 + sym if side == "long" else 1 + len(symbols) + sym
    return {
        "bar_index": int(bar_index),
        "x": labels[int(bar_index)],
        "action": int(action),
        "symbol": str(symbols[sym]).upper(),
        "side": side,
        "symbol_index": int(sym),
    }


def _simulate_vector_trace(
    data: MktdData,
    scores_by_t: np.ndarray,
    candidate: Candidate,
    *,
    start: int,
    eval_days: int,
    labels: Sequence[str],
    fee_rate: float,
    slippage_bps: float,
    fill_buffer_bps: float,
    decision_lag: int,
    short_borrow_apr: float,
    periods_per_year: float,
    initial_cash: float = 10000.0,
) -> dict[str, Any]:
    close = np.asarray(data.prices[:, :, P_CLOSE], dtype=np.float64)
    weights = np.zeros(data.num_symbols, dtype=np.float64)
    entry_prices = np.full(data.num_symbols, np.nan, dtype=np.float64)
    equity = 1.0
    curve = [float(initial_cash)]
    positions_by_bar: list[list[dict[str, Any]]] = [[]]
    trades: list[dict[str, Any]] = []
    actions: list[dict[str, Any]] = [
        {"bar_index": 0, "x": labels[0], "action": 0, "symbol": "", "side": "flat", "symbol_index": -1}
    ]
    cost_per_turnover = float(fee_rate) + float(slippage_bps) / 10_000.0
    borrow_per_period = max(0.0, float(short_borrow_apr)) / max(1.0, float(periods_per_year))
    btc_idx = _btc_idx(data)
    for step in range(int(eval_days)):
        global_t = int(start) + int(step)
        target = weights
        local_signal_t = int(step) - int(decision_lag)
        signal_t = int(start) + local_signal_t
        if local_signal_t >= 0 and (step == 0 or step % max(1, int(candidate.rebalance_days)) == 0):
            desired = _desired_weights(data, scores_by_t, candidate, t=signal_t, btc_idx=btc_idx)
            old_idx = np.flatnonzero(np.abs(weights) > 1e-12)
            new_idx = np.flatnonzero(np.abs(desired) > 1e-12)
            same_book = np.array_equal(old_idx, new_idx) and np.all(np.sign(weights[old_idx]) == np.sign(desired[old_idx]))
            target = weights if same_book and not bool(candidate.always_trade) else desired
        target = _apply_binary_fills(
            data,
            weights,
            target,
            t=global_t,
            fill_buffer_bps=float(fill_buffer_bps),
        )
        trades.extend(
            _trade_event_rows(
                symbols=data.symbols,
                old_weights=weights,
                new_weights=target,
                close=close[global_t],
                equity=equity,
                bar_index=step,
                label=labels[step],
                initial_cash=float(initial_cash),
            )
        )
        for sym in np.flatnonzero(np.abs(target - weights) > 1e-9):
            old = float(weights[int(sym)])
            new = float(target[int(sym)])
            if abs(new) <= 1e-9:
                entry_prices[int(sym)] = np.nan
            elif abs(old) <= 1e-9 or np.sign(old) != np.sign(new):
                entry_prices[int(sym)] = close[global_t, int(sym)]
        turnover = float(np.abs(target - weights).sum())
        p0 = close[global_t]
        p1 = close[global_t + 1]
        valid = np.isfinite(p0) & np.isfinite(p1) & (p0 > 0.0) & (p1 > 0.0)
        day_ret = np.zeros(data.num_symbols, dtype=np.float64)
        day_ret[valid] = p1[valid] / p0[valid] - 1.0
        pnl = float(np.dot(target, day_ret))
        borrow = float(np.abs(target[target < 0.0]).sum()) * borrow_per_period
        cost = turnover * cost_per_turnover
        growth = max(1e-9, 1.0 + pnl - cost - borrow)
        equity = max(1e-9, equity * growth)
        gross_return = np.where(valid, 1.0 + day_ret, 1.0)
        weights = _evolve_weights_after_return(target, gross_return, growth)
        curve.append(float(equity * initial_cash))
        positions_by_bar.append(
            _position_rows_for_weights(
                symbols=data.symbols,
                weights=weights,
                entry_prices=entry_prices,
                close=p1,
                equity=equity,
                bar_index=step + 1,
                initial_cash=float(initial_cash),
            )
        )
        actions.append(_action_summary_row(target, labels=labels, symbols=data.symbols, bar_index=step + 1))
    final_turnover = float(np.abs(weights).sum())
    if final_turnover > 1e-9:
        trades.extend(
            _trade_event_rows(
                symbols=data.symbols,
                old_weights=weights,
                new_weights=np.zeros_like(weights),
                close=close[int(start) + int(eval_days)],
                equity=equity,
                bar_index=int(eval_days),
                label=labels[int(eval_days)],
                initial_cash=float(initial_cash),
            )
        )
        equity = max(1e-9, equity * (1.0 - final_turnover * cost_per_turnover))
        curve[-1] = float(equity * initial_cash)
        positions_by_bar[-1] = []
    equity_curve = np.asarray(curve, dtype=np.float64)
    return {
        "positions_by_bar": positions_by_bar,
        "trades": trades,
        "actions": actions,
        "equity_curve": equity_curve,
        "total_return": float(equity_curve[-1] / float(initial_cash) - 1.0),
        "max_drawdown": _max_drawdown(equity_curve),
        "sortino": _sortino_from_equity(equity_curve, periods_per_year=float(periods_per_year)),
        "num_trades": int(len(trades)),
    }


def _write_trace_outputs(
    *,
    data: MktdData,
    bank: ScoreBank,
    candidate: Candidate,
    row: dict[str, Any],
    eval_days: int,
    trace_window_start: str,
    date_start: str,
    date_end: str,
    fee_rate: float,
    slippage_bps: float,
    fill_buffer_bps: float,
    decision_lag: int,
    short_borrow_apr: float,
    periods_per_year: float,
    trace_output: Path,
    html_output: Path,
) -> None:
    from scripts.plotly_backtest_space import _bars_payload, _date_labels, _html_document

    start = _resolve_trace_start(data, int(eval_days), str(trace_window_start))
    labels = _date_labels(
        data=data,
        window_start=int(start),
        window_len=int(eval_days) + 1,
        date_start=str(date_start),
        date_end=str(date_end),
    )
    scores = _combine_scores(bank, candidate)
    trace = _simulate_vector_trace(
        data,
        scores,
        candidate,
        start=int(start),
        eval_days=int(eval_days),
        labels=labels,
        fee_rate=float(fee_rate),
        slippage_bps=float(slippage_bps),
        fill_buffer_bps=float(fill_buffer_bps),
        decision_lag=int(decision_lag),
        short_borrow_apr=float(short_borrow_apr),
        periods_per_year=float(periods_per_year),
    )
    window = _slice_window(data, int(start), int(eval_days))
    symbols = [str(symbol).upper() for symbol in window.symbols]
    positions_by_bar = trace["positions_by_bar"]
    current_positions = positions_by_bar[-1] if positions_by_bar else []
    default_symbol = str(current_positions[0]["symbol"]).upper() if current_positions else symbols[0]
    payload = {
        "meta": {
            "data_path": str("<meta_anneal_eval_data>"),
            "rule_config": f"meta_anneal:{candidate.candidate_id}:{candidate.book_mode}",
            "window_start": int(start),
            "eval_days": int(eval_days),
            "decision_lag": int(decision_lag),
            "fee_rate": float(fee_rate),
            "slippage_bps": float(slippage_bps),
            "fill_buffer_bps": float(fill_buffer_bps),
            "max_leverage": float(candidate.max_gross),
            "periods_per_year": float(periods_per_year),
            "total_return_pct": float(trace["total_return"] * 100.0),
            "sortino": float(trace["sortino"]),
            "max_drawdown_pct": float(trace["max_drawdown"] * 100.0),
            "num_trades": int(trace["num_trades"]),
            "win_rate_pct": 0.0,
            "default_symbol": default_symbol,
            "book_mode": str(candidate.book_mode),
            "top_k": int(candidate.top_k),
            "long_fraction": float(candidate.long_fraction),
            "short_risk_mult": float(candidate.short_risk_mult),
            "always_trade": int(bool(candidate.always_trade)),
            "validation_objective": float(row.get("objective", 0.0)),
            "validation_median_monthly_pct": float(row.get("median_monthly_pct", 0.0)),
            "validation_p10_monthly_pct": float(row.get("p10_monthly_pct", 0.0)),
            "validation_p90_dd_pct": float(row.get("p90_dd_pct", 0.0)),
        },
        "symbols": symbols,
        "labels": labels,
        "bars": _bars_payload(window, labels),
        "trades": trace["trades"],
        "positions": [],
        "positions_by_bar": positions_by_bar,
        "actions": trace["actions"],
        "equity": {"x": labels[: len(trace["equity_curve"])], "value": [float(x) for x in trace["equity_curve"]]},
    }
    trace_output.parent.mkdir(parents=True, exist_ok=True)
    trace_output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    html_output.parent.mkdir(parents=True, exist_ok=True)
    html_output.write_text(_html_document(payload), encoding="utf-8")
    print(
        f"trace_html={html_output} trace_json={trace_output} "
        f"latest_return={payload['meta']['total_return_pct']:+.2f}% "
        f"dd={payload['meta']['max_drawdown_pct']:.2f}% trades={payload['meta']['num_trades']}",
        flush=True,
    )


def _make_exact_policy(data: MktdData, scores_by_t: np.ndarray, candidate: Candidate, *, decision_lag: int):
    btc_idx = _btc_idx(data)
    pending: list[int] = []
    state = {"step": 0, "last_action": 0}

    def raw_policy(_obs: np.ndarray) -> int:
        step = int(state["step"])
        state["step"] = step + 1
        if step > 0 and step % max(1, int(candidate.rebalance_days)) != 0:
            return int(state["last_action"])
        feat_idx = max(0, min(step - 1, data.num_timesteps - 1))
        if not _gate_allows(
            data,
            t=feat_idx,
            btc_idx=btc_idx,
            btc_gate=float(candidate.btc_gate),
            market_gate=float(candidate.market_gate),
        ):
            state["last_action"] = 0
            return 0
        tradable_idx = max(0, min(step, data.num_timesteps - 1))
        scores = np.asarray(scores_by_t[tradable_idx], dtype=np.float64)
        valid = np.isfinite(scores)
        if data.tradable is not None:
            valid &= np.asarray(data.tradable[tradable_idx], dtype=bool)
        candidates = np.flatnonzero(valid & (-scores >= float(candidate.threshold)))
        if candidates.size == 0:
            state["last_action"] = 0
            return 0
        chosen = int(candidates[np.argmin(scores[candidates])])
        action = 1 + data.num_symbols + chosen
        state["last_action"] = int(action)
        return int(action)

    if int(decision_lag) <= 0:
        return raw_policy

    def lagged_policy(obs: np.ndarray) -> int:
        pending.append(int(raw_policy(obs)))
        if len(pending) <= int(decision_lag):
            return 0
        return int(pending.pop(0))

    return lagged_policy


def _eval_exact_single_candidate(
    data: MktdData,
    bank: ScoreBank,
    candidate: Candidate,
    *,
    phase: str,
    eval_days: int,
    stride: int,
    fee_rate: float,
    slippage_bps: float,
    fill_buffer_bps: float,
    decision_lag: int,
    short_borrow_apr: float,
    periods_per_year: float,
    target_monthly_pct: float,
    target_max_dd_pct: float,
) -> dict[str, float | int | str]:
    exact_candidate = Candidate(
        candidate_id=candidate.candidate_id,
        logits=candidate.logits,
        threshold=candidate.threshold,
        max_gross=candidate.max_gross,
        max_weight=candidate.max_weight,
        top_k=1,
        book_mode="single",
        score_temp=candidate.score_temp,
        btc_gate=candidate.btc_gate,
        market_gate=candidate.market_gate,
        rebalance_days=candidate.rebalance_days,
        long_fraction=candidate.long_fraction,
        short_risk_mult=candidate.short_risk_mult,
        always_trade=candidate.always_trade,
    )
    scores = _combine_scores(bank, exact_candidate)
    results: list[dict[str, float | int | np.ndarray]] = []
    for start in _candidate_starts(data, eval_days, stride=int(stride)):
        window = _slice_window(data, start, eval_days)
        window_scores = np.asarray(scores[start : start + int(eval_days) + 1], dtype=np.float64)
        result = simulate_daily_policy(
            window,
            _make_exact_policy(window, window_scores, exact_candidate, decision_lag=int(decision_lag)),
            max_steps=int(eval_days),
            fee_rate=float(fee_rate),
            slippage_bps=float(slippage_bps),
            fill_buffer_bps=float(fill_buffer_bps),
            max_leverage=float(exact_candidate.max_gross),
            periods_per_year=float(periods_per_year),
            short_borrow_apr=float(short_borrow_apr),
            enable_drawdown_profit_early_exit=False,
            enable_metric_threshold_early_exit=False,
        )
        curve = np.asarray(result.equity_curve if result.equity_curve is not None else [], dtype=np.float64)
        results.append(
            {
                "total_return": float(result.total_return),
                "max_drawdown": float(result.max_drawdown),
                "sortino": float(result.sortino),
                "trades": int(result.num_trades),
                "equity_curve": curve,
            }
        )
    return _summarise_results(
        results,
        candidate=exact_candidate,
        phase=phase,
        eval_days=int(eval_days),
        slippage_bps=float(slippage_bps),
        fill_buffer_bps=float(fill_buffer_bps),
        target_monthly_pct=float(target_monthly_pct),
        target_max_dd_pct=float(target_max_dd_pct),
        channel_weights=_channel_weights(bank, exact_candidate),
    )


def _random_candidate(
    rng: np.random.Generator,
    *,
    idx: int,
    n_channels: int,
    gross_grid: Sequence[float],
    max_weight_grid: Sequence[float],
    top_k_grid: Sequence[int],
    book_modes: Sequence[str],
    btc_gates: Sequence[float],
    market_gates: Sequence[float],
    rebalance_grid: Sequence[int],
    long_fraction_grid: Sequence[float],
    short_risk_mult_grid: Sequence[float],
    always_trade_grid: Sequence[int],
) -> Candidate:
    return Candidate(
        candidate_id=f"cand{idx:05d}",
        logits=rng.normal(0.0, 1.0, size=int(n_channels)),
        threshold=float(rng.uniform(0.0, 1.5)),
        max_gross=float(rng.choice(gross_grid)),
        max_weight=float(rng.choice(max_weight_grid)),
        top_k=int(rng.choice(top_k_grid)),
        book_mode=str(rng.choice(book_modes)),
        score_temp=float(np.exp(rng.uniform(math.log(0.2), math.log(2.0)))),
        btc_gate=float(rng.choice(btc_gates)),
        market_gate=float(rng.choice(market_gates)),
        rebalance_days=int(rng.choice(rebalance_grid)),
        long_fraction=float(rng.choice(long_fraction_grid)),
        short_risk_mult=float(rng.choice(short_risk_mult_grid)),
        always_trade=bool(int(rng.choice(always_trade_grid))),
    )


def _seed_candidates(
    *,
    n_channels: int,
    gross_grid: Sequence[float],
    max_weight_grid: Sequence[float],
    top_k_grid: Sequence[int],
    book_modes: Sequence[str],
    btc_gates: Sequence[float],
    market_gates: Sequence[float],
    rebalance_grid: Sequence[int],
    long_fraction_grid: Sequence[float],
    short_risk_mult_grid: Sequence[float],
    always_trade_grid: Sequence[int],
) -> list[Candidate]:
    out: list[Candidate] = []
    for ch in range(int(n_channels)):
        logits = np.full(int(n_channels), -4.0, dtype=np.float64)
        logits[ch] = 4.0
        out.append(
            Candidate(
                candidate_id=f"seed_ch{ch:03d}",
                logits=logits,
                threshold=0.5,
                max_gross=float(gross_grid[0]),
                max_weight=float(max_weight_grid[0]),
                top_k=int(top_k_grid[0]),
                book_mode=str(book_modes[0]),
                score_temp=1.0,
                btc_gate=float(btc_gates[0]),
                market_gate=float(market_gates[0]),
                rebalance_days=int(rebalance_grid[0]),
                long_fraction=float(long_fraction_grid[0]),
                short_risk_mult=float(short_risk_mult_grid[0]),
                always_trade=bool(int(always_trade_grid[0])),
            )
        )
    return out


def _mutate_candidate(
    rng: np.random.Generator,
    parent: Candidate,
    *,
    idx: int,
    temp: float,
    gross_grid: Sequence[float],
    max_weight_grid: Sequence[float],
    top_k_grid: Sequence[int],
    book_modes: Sequence[str],
    btc_gates: Sequence[float],
    market_gates: Sequence[float],
    rebalance_grid: Sequence[int],
    long_fraction_grid: Sequence[float],
    short_risk_mult_grid: Sequence[float],
    always_trade_grid: Sequence[int],
) -> Candidate:
    logits = np.asarray(parent.logits, dtype=np.float64) + rng.normal(0.0, 0.75 * temp, size=parent.logits.shape)
    if rng.random() < 0.20:
        logits[int(rng.integers(0, logits.size))] += rng.normal(1.5 * temp, 0.5)
    threshold = float(np.clip(parent.threshold + rng.normal(0.0, 0.25 * temp), 0.0, 2.5))
    score_temp = float(np.clip(parent.score_temp * np.exp(rng.normal(0.0, 0.35 * temp)), 0.05, 5.0))
    return Candidate(
        candidate_id=f"cand{idx:05d}",
        logits=logits,
        threshold=threshold,
        max_gross=float(rng.choice(gross_grid) if rng.random() < 0.20 * temp else parent.max_gross),
        max_weight=float(rng.choice(max_weight_grid) if rng.random() < 0.20 * temp else parent.max_weight),
        top_k=int(rng.choice(top_k_grid) if rng.random() < 0.20 * temp else parent.top_k),
        book_mode=str(rng.choice(book_modes) if rng.random() < 0.10 * temp else parent.book_mode),
        score_temp=score_temp,
        btc_gate=float(rng.choice(btc_gates) if rng.random() < 0.15 * temp else parent.btc_gate),
        market_gate=float(rng.choice(market_gates) if rng.random() < 0.15 * temp else parent.market_gate),
        rebalance_days=int(rng.choice(rebalance_grid) if rng.random() < 0.15 * temp else parent.rebalance_days),
        long_fraction=float(
            rng.choice(long_fraction_grid) if rng.random() < 0.20 * temp else parent.long_fraction
        ),
        short_risk_mult=float(
            rng.choice(short_risk_mult_grid) if rng.random() < 0.15 * temp else parent.short_risk_mult
        ),
        always_trade=bool(
            int(rng.choice(always_trade_grid)) if rng.random() < 0.15 * temp else parent.always_trade
        ),
    )


def _fieldnames() -> list[str]:
    return [
        "phase",
        "generation",
        "rank",
        "candidate_id",
        "book_mode",
        "top_k",
        "max_gross",
        "max_weight",
        "threshold",
        "score_temp",
        "btc_gate",
        "market_gate",
        "rebalance_days",
        "long_fraction",
        "short_risk_mult",
        "always_trade",
        "channel_weights_json",
        "candidate_json",
        "eval_days",
        "slip_bps",
        "fill_buffer_bps",
        "median_pct",
        "p10_pct",
        "p90_pct",
        "median_monthly_pct",
        "p10_monthly_pct",
        "worst_monthly_pct",
        "best_monthly_pct",
        "neg_windows",
        "windows",
        "p90_dd_pct",
        "median_smooth",
        "median_ulcer",
        "median_sortino",
        "median_trades",
        "target_monthly_pct",
        "target_max_dd_pct",
        "passes_target",
        "objective",
    ]


def _write_row(writer: csv.DictWriter, row: dict[str, Any], *, generation: int, rank: int) -> None:
    out = dict(row)
    out["generation"] = int(generation)
    out["rank"] = int(rank)
    writer.writerow(out)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rule-csv", type=Path, default=Path("analysis/binance33_linear_rule_search_20260430.csv"))
    parser.add_argument("--linear-configs", default="rand0045_rb1_thr0.5_btcmom20gt-0.05,rand0124_rb1_thr0_btcmom20gt-0.02,rand0038_rb1_thr0.2_btcmom20gt-0.05")
    parser.add_argument("--max-linear-rules", type=int, default=8)
    parser.add_argument("--train-data", type=Path, default=Path("pufferlib_market/data/binance33_daily_train.bin"))
    parser.add_argument("--eval-data", type=Path, default=Path("pufferlib_market/data/binance33_daily_val.bin"))
    parser.add_argument("--train-tail-days", type=int, default=0)
    parser.add_argument("--xgb-experiment-names", default="")
    parser.add_argument("--xgb-rounds", type=int, default=80)
    parser.add_argument("--xgb-device", default="cuda")
    parser.add_argument("--xgb-model-dir", type=Path, default=None)
    parser.add_argument("--no-handcrafted", action="store_true")
    parser.add_argument("--out", type=Path, default=Path("analysis/binance33_meta_anneal.csv"))
    parser.add_argument("--eval-days", type=int, default=120)
    parser.add_argument("--train-stride", type=int, default=20)
    parser.add_argument("--val-stride", type=int, default=1)
    parser.add_argument("--generations", type=int, default=6)
    parser.add_argument("--population", type=int, default=96)
    parser.add_argument("--elite", type=int, default=16)
    parser.add_argument("--validate-top", type=int, default=24)
    parser.add_argument("--exact-top", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260501)
    parser.add_argument("--gross-grid", default="1.0,1.2,1.305")
    parser.add_argument("--max-weight-grid", default="1.0,0.67,0.5")
    parser.add_argument("--top-k-grid", default="1,2,3")
    parser.add_argument("--book-modes", default="single,portfolio,long_single,long_portfolio,longshort_portfolio")
    parser.add_argument("--long-fraction-grid", default="0.35,0.5,0.65")
    parser.add_argument("--short-risk-mult-grid", default="1.0,1.1,1.25")
    parser.add_argument("--always-trade-grid", default="0,1")
    parser.add_argument("--btc-gates", default="-99,-0.08,-0.05,-0.02,0")
    parser.add_argument("--market-gates", default="-99,-0.10,-0.05,-0.02,0")
    parser.add_argument("--rebalance-grid", default="1,3")
    parser.add_argument("--slippage-bps", type=float, default=20.0)
    parser.add_argument("--fee-rate", type=float, default=0.001)
    parser.add_argument("--fill-buffer-bps", type=float, default=5.0)
    parser.add_argument("--decision-lag", type=int, default=2)
    parser.add_argument("--short-borrow-apr", type=float, default=0.08)
    parser.add_argument("--periods-per-year", type=float, default=365.0)
    parser.add_argument("--target-monthly-pct", type=float, default=30.0)
    parser.add_argument("--target-max-dd-pct", type=float, default=20.0)
    parser.add_argument("--trace-output", type=Path, default=None)
    parser.add_argument("--html-output", type=Path, default=None)
    parser.add_argument("--trace-window-start", default="latest")
    parser.add_argument("--date-start", default="2025-10-01")
    parser.add_argument("--date-end", default="2026-03-14")
    args = parser.parse_args()

    rng = np.random.default_rng(int(args.seed))
    full_train_data = read_mktd(args.train_data)
    train_data = full_train_data
    if int(args.train_tail_days) > 0:
        tail = min(int(args.train_tail_days), full_train_data.num_timesteps)
        train_data = _slice_data(full_train_data, full_train_data.num_timesteps - tail, full_train_data.num_timesteps)
    val_data = read_mktd(args.eval_data)
    rules = _load_rule_rows(
        args.rule_csv,
        configs=_parse_str_list(args.linear_configs),
        max_rules=int(args.max_linear_rules),
    )
    xgb_names = _parse_str_list(args.xgb_experiment_names)
    xgb_models = _train_xgb_channel_models(
        full_train_data,
        experiment_names=xgb_names,
        rounds=int(args.xgb_rounds),
        device=str(args.xgb_device),
        model_dir=args.xgb_model_dir,
        cache_tag=_file_cache_tag(args.train_data) if xgb_names else "",
    )
    train_bank = _build_bank(
        train_data,
        rules=rules,
        xgb_models=xgb_models,
        include_handcrafted=not bool(args.no_handcrafted),
    )
    val_bank = _build_bank(
        val_data,
        rules=rules,
        xgb_models=xgb_models,
        include_handcrafted=not bool(args.no_handcrafted),
    )
    if train_bank.names != val_bank.names:
        raise RuntimeError("train/validation score banks have different channels")

    gross_grid = _parse_float_list(args.gross_grid)
    max_weight_grid = _parse_float_list(args.max_weight_grid)
    top_k_grid = _parse_int_list(args.top_k_grid)
    book_modes = _parse_str_list(args.book_modes)
    long_fraction_grid = _parse_float_list(args.long_fraction_grid)
    short_risk_mult_grid = _parse_float_list(args.short_risk_mult_grid)
    always_trade_grid = _parse_int_list(args.always_trade_grid)
    btc_gates = _parse_float_list(args.btc_gates)
    market_gates = _parse_float_list(args.market_gates)
    rebalance_grid = _parse_int_list(args.rebalance_grid)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    all_train: list[tuple[Candidate, dict[str, float | int | str]]] = []
    candidate_counter = 0
    population = _seed_candidates(
        n_channels=len(train_bank.names),
        gross_grid=gross_grid,
        max_weight_grid=max_weight_grid,
        top_k_grid=top_k_grid,
        book_modes=book_modes,
        btc_gates=btc_gates,
        market_gates=market_gates,
        rebalance_grid=rebalance_grid,
        long_fraction_grid=long_fraction_grid,
        short_risk_mult_grid=short_risk_mult_grid,
        always_trade_grid=always_trade_grid,
    )
    while len(population) < int(args.population):
        candidate_counter += 1
        population.append(
            _random_candidate(
                rng,
                idx=candidate_counter,
                n_channels=len(train_bank.names),
                gross_grid=gross_grid,
                max_weight_grid=max_weight_grid,
                top_k_grid=top_k_grid,
                book_modes=book_modes,
                btc_gates=btc_gates,
                market_gates=market_gates,
                rebalance_grid=rebalance_grid,
                long_fraction_grid=long_fraction_grid,
                short_risk_mult_grid=short_risk_mult_grid,
                always_trade_grid=always_trade_grid,
            )
        )

    with args.out.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=_fieldnames())
        writer.writeheader()
        for gen in range(int(args.generations)):
            temp = max(0.10, 1.0 - gen / max(1, int(args.generations) - 1))
            scored: list[tuple[Candidate, dict[str, float | int | str]]] = []
            for candidate in population:
                row = _eval_vector_candidate(
                    train_data,
                    train_bank,
                    candidate,
                    phase="train_fast",
                    eval_days=int(args.eval_days),
                    stride=int(args.train_stride),
                    fee_rate=float(args.fee_rate),
                    slippage_bps=float(args.slippage_bps),
                    fill_buffer_bps=float(args.fill_buffer_bps),
                    decision_lag=int(args.decision_lag),
                    short_borrow_apr=float(args.short_borrow_apr),
                    periods_per_year=float(args.periods_per_year),
                    target_monthly_pct=float(args.target_monthly_pct),
                    target_max_dd_pct=float(args.target_max_dd_pct),
                )
                scored.append((candidate, row))
            scored.sort(key=lambda item: float(item[1]["objective"]), reverse=True)
            all_train.extend(scored)
            for rank, (_candidate, row) in enumerate(scored[: int(args.elite)], start=1):
                _write_row(writer, row, generation=gen, rank=rank)
            fh.flush()
            best = scored[0][1]
            print(
                f"gen {gen}: obj={best['objective']:+.2f} mo={best['median_monthly_pct']:+.2f}% "
                f"p10={best['p10_monthly_pct']:+.2f}% dd90={best['p90_dd_pct']:.2f}% "
                f"neg={best['neg_windows']}/{best['windows']} mode={best['book_mode']} "
                f"top={best['top_k']} gross={best['max_gross']} "
                f"long={best['long_fraction']} always={best['always_trade']}",
                flush=True,
            )
            elites = [candidate for candidate, _row in scored[: int(args.elite)]]
            population = list(elites)
            while len(population) < int(args.population):
                parent = elites[int(rng.integers(0, len(elites)))]
                candidate_counter += 1
                population.append(
                    _mutate_candidate(
                        rng,
                        parent,
                        idx=candidate_counter,
                        temp=float(temp),
                        gross_grid=gross_grid,
                        max_weight_grid=max_weight_grid,
                        top_k_grid=top_k_grid,
                        book_modes=book_modes,
                        btc_gates=btc_gates,
                        market_gates=market_gates,
                        rebalance_grid=rebalance_grid,
                        long_fraction_grid=long_fraction_grid,
                        short_risk_mult_grid=short_risk_mult_grid,
                        always_trade_grid=always_trade_grid,
                    )
                )

        best_by_id: dict[str, tuple[Candidate, dict[str, float | int | str]]] = {}
        for candidate, row in all_train:
            key = json.dumps(
                {
                    "w": np.round(_softmax(candidate.logits), 6).tolist(),
                    "thr": round(candidate.threshold, 4),
                    "gross": candidate.max_gross,
                    "maxw": candidate.max_weight,
                    "top": candidate.top_k,
                    "mode": candidate.book_mode,
                    "st": round(candidate.score_temp, 4),
                    "btc": candidate.btc_gate,
                    "mkt": candidate.market_gate,
                    "rb": candidate.rebalance_days,
                    "lf": round(candidate.long_fraction, 4),
                    "sr": round(candidate.short_risk_mult, 4),
                    "at": int(bool(candidate.always_trade)),
                },
                sort_keys=True,
            )
            if key not in best_by_id or float(row["objective"]) > float(best_by_id[key][1]["objective"]):
                best_by_id[key] = (candidate, row)
        finalists = sorted(best_by_id.values(), key=lambda item: float(item[1]["objective"]), reverse=True)[
            : int(args.validate_top)
        ]
        val_rows: list[tuple[Candidate, dict[str, float | int | str]]] = []
        for rank, (candidate, _train_row) in enumerate(finalists, start=1):
            row = _eval_vector_candidate(
                val_data,
                val_bank,
                candidate,
                phase="val_fast",
                eval_days=int(args.eval_days),
                stride=int(args.val_stride),
                fee_rate=float(args.fee_rate),
                slippage_bps=float(args.slippage_bps),
                fill_buffer_bps=float(args.fill_buffer_bps),
                decision_lag=int(args.decision_lag),
                short_borrow_apr=float(args.short_borrow_apr),
                periods_per_year=float(args.periods_per_year),
                target_monthly_pct=float(args.target_monthly_pct),
                target_max_dd_pct=float(args.target_max_dd_pct),
            )
            val_rows.append((candidate, row))
            _write_row(writer, row, generation=int(args.generations), rank=rank)
            fh.flush()
        val_rows.sort(key=lambda item: float(item[1]["objective"]), reverse=True)
        exact_candidates = [item for item in val_rows if item[0].book_mode == "single"][: int(args.exact_top)]
        for rank, (candidate, _val_row) in enumerate(exact_candidates, start=1):
            row = _eval_exact_single_candidate(
                val_data,
                val_bank,
                candidate,
                phase="val_exact_single",
                eval_days=int(args.eval_days),
                stride=int(args.val_stride),
                fee_rate=float(args.fee_rate),
                slippage_bps=float(args.slippage_bps),
                fill_buffer_bps=float(args.fill_buffer_bps),
                decision_lag=int(args.decision_lag),
                short_borrow_apr=float(args.short_borrow_apr),
                periods_per_year=float(args.periods_per_year),
                target_monthly_pct=float(args.target_monthly_pct),
                target_max_dd_pct=float(args.target_max_dd_pct),
            )
            _write_row(writer, row, generation=int(args.generations) + 1, rank=rank)
            fh.flush()

        if val_rows and args.trace_output is not None and args.html_output is not None:
            best_candidate, best_row = val_rows[0]
            _write_trace_outputs(
                data=val_data,
                bank=val_bank,
                candidate=best_candidate,
                row=best_row,
                eval_days=int(args.eval_days),
                trace_window_start=str(args.trace_window_start),
                date_start=str(args.date_start),
                date_end=str(args.date_end),
                fee_rate=float(args.fee_rate),
                slippage_bps=float(args.slippage_bps),
                fill_buffer_bps=float(args.fill_buffer_bps),
                decision_lag=int(args.decision_lag),
                short_borrow_apr=float(args.short_borrow_apr),
                periods_per_year=float(args.periods_per_year),
                trace_output=args.trace_output,
                html_output=args.html_output,
            )

    print("\n=== Best validation candidates ===")
    for rank, (_candidate, row) in enumerate(val_rows[:15], start=1):
        print(
            f"{rank:02d} obj={row['objective']:+7.2f} {row['median_monthly_pct']:+7.2f}%/mo "
            f"p10={row['p10_monthly_pct']:+7.2f}% dd90={row['p90_dd_pct']:.2f}% "
            f"neg={row['neg_windows']}/{row['windows']} pass={row['passes_target']} "
            f"{row['phase']} {row['book_mode']} top={row['top_k']} gross={row['max_gross']} "
            f"long={row['long_fraction']} always={row['always_trade']}"
        )
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
