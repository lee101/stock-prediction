#!/usr/bin/env python3
"""Sweep small hedged Binance33 long/short CVaR portfolio packs.

This is the long/short companion to ``sweep_binance33_linear_cvar_pack.py``.
It packs a small number of high-score longs and low-score shorts, sizes them by
tail-risk-adjusted edge, and evaluates the resulting vector book with binary
limit-fill semantics. The goal is a smoother, hedged 2-3 position book rather
than the recent high-gross short-only reversion edge.
"""

from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import dataclass
from itertools import product
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from pufferlib_market.hourly_replay import MktdData, P_CLOSE, read_mktd
from scripts.search_binance33_linear_rules import LinearRule, _score_vector
from scripts.search_binance33_meta_anneal import _apply_binary_fills, _evolve_weights_after_return
from scripts.search_binance33_regime_gates import (
    GateSpec,
    _gate_allows,
    _gate_label,
    _gate_universe,
    _load_rule_from_csv,
)
from scripts.sweep_binance33_rules import FEATURES, _candidate_starts
from scripts.sweep_binance33_xgb import _monthly_equivalent_return, _passes_production_target
from src.robust_trading_metrics import compute_pnl_smoothness_from_equity, compute_ulcer_index


Gate = GateSpec | tuple[GateSpec, GateSpec] | None


@dataclass(frozen=True)
class HedgedConfig:
    n_longs: int
    n_shorts: int
    long_signal: str
    allocation_mode: str
    max_gross: float
    max_weight: float
    long_fraction: float
    max_net: float
    lookback_days: int
    cvar_alpha: float
    edge_power: float
    risk_power: float
    min_abs_score: float
    rebalance_days: int
    rebalance_style: str


@dataclass(frozen=True)
class SignalCache:
    scores_by_t: np.ndarray
    valid_by_t: np.ndarray
    daily_returns: np.ndarray
    btc_idx: int


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


def _parse_gate_options(spec: str) -> list[Gate]:
    labels = _parse_str_list(spec) or ["none"]
    universe = _gate_universe()
    by_label = {_gate_label(gate): gate for gate in universe}
    out: list[Gate] = []
    for label in labels:
        if label.lower() in {"none", "nogate"}:
            out.append(None)
            continue
        if label not in by_label:
            examples = ", ".join(["none", *sorted(by_label)[:10]])
            raise ValueError(f"unknown gate {label!r}; examples: {examples}")
        out.append(by_label[label])
    return out


def _gate_name(gate: Gate) -> str:
    return "none" if gate is None else _gate_label(gate)


def _btc_idx(data: MktdData) -> int:
    symbols = [str(symbol).upper() for symbol in data.symbols]
    return next((idx for idx, symbol in enumerate(symbols) if symbol.startswith("BTC")), 0)


def _build_cache(data: MktdData, rule: LinearRule, exclude_symbols: set[str]) -> SignalCache:
    scores: list[np.ndarray] = []
    valid: list[np.ndarray] = []
    excluded_idx = {idx for idx, symbol in enumerate(data.symbols) if str(symbol).upper() in exclude_symbols}
    for t in range(data.num_timesteps):
        score_t, valid_t = _score_vector(data, rule, t)
        valid_t = np.asarray(valid_t, dtype=bool).copy()
        for idx in excluded_idx:
            valid_t[int(idx)] = False
            score_t[int(idx)] = np.nan
        scores.append(np.asarray(score_t, dtype=np.float64))
        valid.append(valid_t)

    close = np.asarray(data.prices[:, :, P_CLOSE], dtype=np.float64)
    p0 = close[:-1]
    p1 = close[1:]
    ret_valid = np.isfinite(p0) & np.isfinite(p1) & (p0 > 0.0) & (p1 > 0.0)
    daily_returns = np.full_like(p0, np.nan, dtype=np.float64)
    daily_returns[ret_valid] = p1[ret_valid] / p0[ret_valid] - 1.0
    return SignalCache(
        scores_by_t=np.stack(scores, axis=0),
        valid_by_t=np.stack(valid, axis=0),
        daily_returns=daily_returns,
        btc_idx=_btc_idx(data),
    )


def _zscore_valid(values: np.ndarray, valid: np.ndarray) -> np.ndarray:
    out = np.full(values.shape, np.nan, dtype=np.float64)
    mask = np.asarray(valid, dtype=bool) & np.isfinite(values)
    if int(mask.sum()) <= 1:
        return out
    series = np.asarray(values, dtype=np.float64)
    mean = float(series[mask].mean())
    std = float(series[mask].std())
    out[mask] = (series[mask] - mean) / max(std, 1e-6)
    return out


def _long_score_vector(data: MktdData, cache: SignalCache, cfg: HedgedConfig, *, t: int) -> np.ndarray:
    signal = str(cfg.long_signal)
    if signal == "rule":
        return np.asarray(cache.scores_by_t[t], dtype=np.float64)

    valid = np.asarray(cache.valid_by_t[t], dtype=bool)
    if signal == "trend60":
        raw = np.asarray(data.features[t, :, FEATURES["trend_60d"]], dtype=np.float64)
    elif signal == "ma20":
        raw = np.asarray(data.features[t, :, FEATURES["ma_delta_20d"]], dtype=np.float64)
    elif signal == "rsi_high":
        raw = np.asarray(data.features[t, :, FEATURES["rsi_14"]], dtype=np.float64)
    elif signal in {"rsi", "rsi_low"}:
        raw = -np.asarray(data.features[t, :, FEATURES["rsi_14"]], dtype=np.float64)
    else:
        raise ValueError(f"unknown long_signal: {signal}")
    return _zscore_valid(raw, valid)


def _rule_allows(data: MktdData, cache: SignalCache, rule: LinearRule, gate: Gate, t: int) -> bool:
    t = max(0, min(int(t), data.num_timesteps - 1))
    if float(rule.btc_min_return_20d) > -9.0:
        btc_mom = float(data.features[t, cache.btc_idx, FEATURES["return_20d"]])
        if not np.isfinite(btc_mom) or btc_mom < float(rule.btc_min_return_20d):
            return False
    if gate is not None and not _gate_allows(gate, data, t):
        return False
    return True


def _capped_alloc(raw: np.ndarray, *, gross: float, max_weight: float) -> np.ndarray:
    raw = np.asarray(raw, dtype=np.float64)
    raw = np.where(np.isfinite(raw) & (raw > 0.0), raw, 0.0)
    gross = max(0.0, float(gross))
    if gross <= 0.0 or float(raw.sum()) <= 0.0:
        return np.zeros_like(raw, dtype=np.float64)
    cap = max(0.0, float(max_weight))
    if cap <= 0.0:
        return raw / float(raw.sum()) * gross

    alloc = np.zeros_like(raw, dtype=np.float64)
    remaining = raw > 0.0
    for _ in range(raw.size + 1):
        remaining_gross = gross - float(alloc.sum())
        if remaining_gross <= 1e-12 or not remaining.any():
            break
        idx = np.flatnonzero(remaining)
        proposed = raw[idx] / float(raw[idx].sum()) * remaining_gross
        capped = proposed > cap
        if not capped.any():
            alloc[idx] = proposed
            break
        cap_idx = idx[capped]
        alloc[cap_idx] = cap
        remaining[cap_idx] = False
    return alloc


def _tail_stats(
    cache: SignalCache,
    idx: np.ndarray,
    signs: np.ndarray,
    *,
    t: int,
    lookback_days: int,
    cvar_alpha: float,
    fallback_vol: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    means = np.zeros(idx.size, dtype=np.float64)
    risks = np.zeros(idx.size, dtype=np.float64)
    end = max(0, min(int(t), cache.daily_returns.shape[0]))
    start = max(0, end - max(1, int(lookback_days)))
    alpha = min(0.995, max(0.50, float(cvar_alpha)))
    for out_idx, sym in enumerate(idx):
        series = cache.daily_returns[start:end, int(sym)] * float(signs[out_idx])
        series = series[np.isfinite(series)]
        if series.size:
            means[out_idx] = float(series.mean())
            losses = -series
            tail_n = max(1, int(np.ceil((1.0 - alpha) * float(losses.size))))
            worst = np.partition(losses, max(0, losses.size - tail_n))[-tail_n:]
            risks[out_idx] = max(float(np.mean(worst)), 0.5 * float(series.std()))
    fallback = np.where(np.isfinite(fallback_vol) & (fallback_vol > 1e-6), fallback_vol, 0.02)
    risks = np.maximum(risks, 0.25 * fallback)
    return means, np.maximum(risks, 1e-4)


def _raw_alloc(
    *,
    mode: str,
    edge: np.ndarray,
    mean: np.ndarray,
    risk: np.ndarray,
    vol: np.ndarray,
    edge_power: float,
    risk_power: float,
) -> np.ndarray:
    edge = np.maximum(np.asarray(edge, dtype=np.float64), 1e-12)
    mean = np.asarray(mean, dtype=np.float64)
    risk = np.maximum(np.asarray(risk, dtype=np.float64), 1e-6)
    vol = np.where(np.isfinite(vol) & (vol > 1e-6), vol, np.inf)
    base = np.power(edge, max(0.0, float(edge_power)))
    if mode == "equal":
        raw = np.ones_like(edge)
    elif mode == "score":
        raw = base
    elif mode == "inv_vol_score":
        raw = base / vol
    elif mode == "cvar":
        raw = base / np.power(risk, max(0.0, float(risk_power)))
    elif mode == "mean_cvar":
        raw = base * np.maximum(mean, 0.0) / np.power(risk, max(0.0, float(risk_power)))
    elif mode == "soft_mean_cvar":
        raw = base * np.maximum(mean + 0.0025 * edge, 0.0) / np.power(risk, max(0.0, float(risk_power)))
    else:
        raise ValueError(f"unknown allocation_mode: {mode}")
    return np.where(np.isfinite(raw) & (raw > 0.0), raw, 0.0)


def _select_side(
    cache: SignalCache,
    cfg: HedgedConfig,
    *,
    t: int,
    side: str,
    scores: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    scores = np.asarray(cache.scores_by_t[t] if scores is None else scores, dtype=np.float64)
    valid = np.asarray(cache.valid_by_t[t], dtype=bool) & np.isfinite(scores)
    if side == "long":
        candidates = np.flatnonzero(valid & (scores >= float(cfg.min_abs_score)))
        if candidates.size == 0:
            return np.asarray([], dtype=np.int64), np.asarray([], dtype=np.float64), np.asarray([], dtype=np.float64)
        ordered = candidates[np.argsort(scores[candidates])[::-1]]
        idx = ordered[: max(0, int(cfg.n_longs))].astype(np.int64, copy=False)
        return idx, np.ones(idx.size, dtype=np.float64), scores[idx]
    if side == "short":
        candidates = np.flatnonzero(valid & (-scores >= float(cfg.min_abs_score)))
        if candidates.size == 0:
            return np.asarray([], dtype=np.int64), np.asarray([], dtype=np.float64), np.asarray([], dtype=np.float64)
        ordered = candidates[np.argsort(scores[candidates])]
        idx = ordered[: max(0, int(cfg.n_shorts))].astype(np.int64, copy=False)
        return idx, -np.ones(idx.size, dtype=np.float64), -scores[idx]
    raise ValueError(f"unknown side: {side}")


def _side_weights(
    data: MktdData,
    cache: SignalCache,
    cfg: HedgedConfig,
    *,
    t: int,
    idx: np.ndarray,
    signs: np.ndarray,
    edge: np.ndarray,
    gross: float,
) -> tuple[np.ndarray, np.ndarray]:
    if idx.size == 0 or gross <= 0.0:
        return idx, np.zeros(idx.size, dtype=np.float64)
    vol = np.asarray(data.features[t, idx, FEATURES["volatility_20d"]], dtype=np.float64)
    mean, risk = _tail_stats(
        cache,
        idx,
        signs,
        t=int(t),
        lookback_days=int(cfg.lookback_days),
        cvar_alpha=float(cfg.cvar_alpha),
        fallback_vol=vol,
    )
    raw = _raw_alloc(
        mode=str(cfg.allocation_mode),
        edge=edge,
        mean=mean,
        risk=risk,
        vol=vol,
        edge_power=float(cfg.edge_power),
        risk_power=float(cfg.risk_power),
    )
    alloc = _capped_alloc(raw, gross=float(gross), max_weight=float(cfg.max_weight))
    return idx, signs * alloc


def _target_weights(data: MktdData, cache: SignalCache, rule: LinearRule, gate: Gate, cfg: HedgedConfig, *, t: int) -> np.ndarray:
    t = max(0, min(int(t), data.num_timesteps - 1))
    weights = np.zeros(data.num_symbols, dtype=np.float64)
    if not _rule_allows(data, cache, rule, gate, t):
        return weights

    long_scores = _long_score_vector(data, cache, cfg, t=t)
    long_idx, long_sign, long_edge = _select_side(cache, cfg, t=t, side="long", scores=long_scores)
    short_idx, short_sign, short_edge = _select_side(cache, cfg, t=t, side="short")
    if long_idx.size == 0 and short_idx.size == 0:
        return weights

    gross = max(0.0, float(cfg.max_gross))
    long_fraction = min(1.0, max(0.0, float(cfg.long_fraction)))
    long_gross = gross * long_fraction if long_idx.size else 0.0
    short_gross = gross * (1.0 - long_fraction) if short_idx.size else 0.0
    if long_idx.size and not short_idx.size:
        long_gross = gross
    elif short_idx.size and not long_idx.size:
        short_gross = gross

    idx, signed = _side_weights(data, cache, cfg, t=t, idx=long_idx, signs=long_sign, edge=long_edge, gross=long_gross)
    weights[idx] = signed
    idx, signed = _side_weights(data, cache, cfg, t=t, idx=short_idx, signs=short_sign, edge=short_edge, gross=short_gross)
    weights[idx] = signed

    gross_now = float(np.abs(weights).sum())
    if gross_now > gross and gross_now > 0.0:
        weights *= gross / gross_now
    net = float(weights.sum())
    max_net = max(0.0, float(cfg.max_net))
    if abs(net) > max_net and gross_now > 0.0:
        if net > 0.0:
            pos = weights > 0.0
            reducible = float(weights[pos].sum())
            if reducible > 0.0:
                weights[pos] *= max(0.0, (reducible - (net - max_net)) / reducible)
        else:
            neg = weights < 0.0
            reducible = float(-weights[neg].sum())
            if reducible > 0.0:
                weights[neg] *= max(0.0, (reducible - (-net - max_net)) / reducible)
    return weights


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


def _simulate_window(
    data: MktdData,
    cache: SignalCache,
    rule: LinearRule,
    gate: Gate,
    cfg: HedgedConfig,
    *,
    start: int,
    eval_days: int,
    fee_rate: float,
    slippage_bps: float,
    fill_buffer_bps: float,
    decision_lag: int,
    short_borrow_apr: float,
    periods_per_year: float,
) -> dict[str, float | int]:
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
        if local_signal_t >= 0 and (step == 0 or step % max(1, int(cfg.rebalance_days)) == 0):
            desired = _target_weights(data, cache, rule, gate, cfg, t=int(start) + local_signal_t)
            if str(cfg.rebalance_style) == "hold_same":
                old_idx = np.flatnonzero(np.abs(weights) > 1e-12)
                new_idx = np.flatnonzero(np.abs(desired) > 1e-12)
                same_book = np.array_equal(old_idx, new_idx) and np.all(np.sign(weights[old_idx]) == np.sign(desired[old_idx]))
                target = weights if same_book else desired
            elif str(cfg.rebalance_style) == "full":
                target = desired
            else:
                raise ValueError(f"unknown rebalance_style: {cfg.rebalance_style}")
        target = _apply_binary_fills(data, weights, target, t=global_t, fill_buffer_bps=float(fill_buffer_bps))
        turnover = float(np.abs(target - weights).sum())
        if turnover > 1e-9:
            trades += int(np.count_nonzero(np.abs(target - weights) > 1e-9))
        day_ret = np.asarray(cache.daily_returns[global_t], dtype=np.float64)
        pnl = float(np.nansum(target * day_ret))
        borrow = float(np.abs(target[target < 0.0]).sum()) * borrow_per_period
        cost = turnover * cost_per_turnover
        growth = max(1e-9, 1.0 + pnl - cost - borrow)
        equity = max(1e-9, equity * growth)
        gross_return = np.where(np.isfinite(day_ret), 1.0 + day_ret, 1.0)
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
        "smooth": float(compute_pnl_smoothness_from_equity(equity_curve)),
        "ulcer": float(compute_ulcer_index(equity_curve)),
        "trades": int(trades),
    }


def _eval_config(
    data: MktdData,
    cache: SignalCache,
    rule: LinearRule,
    gate: Gate,
    cfg: HedgedConfig,
    *,
    scope: str,
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
    results = [
        _simulate_window(
            data,
            cache,
            rule,
            gate,
            cfg,
            start=int(start),
            eval_days=int(eval_days),
            fee_rate=float(fee_rate),
            slippage_bps=float(slippage_bps),
            fill_buffer_bps=float(fill_buffer_bps),
            decision_lag=int(decision_lag),
            short_borrow_apr=float(short_borrow_apr),
            periods_per_year=float(periods_per_year),
        )
        for start in _candidate_starts(data, eval_days, stride=int(stride))
    ]
    returns = np.asarray([float(result["total_return"]) for result in results], dtype=np.float64)
    maxdds = np.asarray([float(result["max_drawdown"]) for result in results], dtype=np.float64)
    sortinos = np.asarray([float(result["sortino"]) for result in results], dtype=np.float64)
    smooths = np.asarray([float(result["smooth"]) for result in results], dtype=np.float64)
    ulcers = np.asarray([float(result["ulcer"]) for result in results], dtype=np.float64)
    trades = np.asarray([int(result["trades"]) for result in results], dtype=np.float64)
    monthly = np.asarray([_monthly_equivalent_return(float(ret), int(eval_days)) for ret in returns], dtype=np.float64)
    row: dict[str, float | int | str] = {
        "scope": scope,
        "config": rule.name,
        "gate": _gate_name(gate),
        "n_longs": int(cfg.n_longs),
        "n_shorts": int(cfg.n_shorts),
        "positions": int(cfg.n_longs) + int(cfg.n_shorts),
        "long_signal": cfg.long_signal,
        "allocation_mode": cfg.allocation_mode,
        "max_gross": float(cfg.max_gross),
        "max_weight": float(cfg.max_weight),
        "long_fraction": float(cfg.long_fraction),
        "max_net": float(cfg.max_net),
        "lookback_days": int(cfg.lookback_days),
        "cvar_alpha": float(cfg.cvar_alpha),
        "edge_power": float(cfg.edge_power),
        "risk_power": float(cfg.risk_power),
        "min_abs_score": float(cfg.min_abs_score),
        "rebalance_days": int(cfg.rebalance_days),
        "rebalance_style": cfg.rebalance_style,
        "eval_days": int(eval_days),
        "slip_bps": float(slippage_bps),
        "fill_buffer_bps": float(fill_buffer_bps),
        "fee_rate": float(fee_rate),
        "short_borrow_apr": float(short_borrow_apr),
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
        "worst_dd_pct": float(100.0 * np.max(maxdds)) if maxdds.size else 0.0,
        "median_sortino": float(np.percentile(sortinos, 50)),
        "median_smooth": float(np.percentile(smooths, 50)),
        "median_ulcer": float(np.percentile(ulcers, 50)),
        "median_trades": float(np.percentile(trades, 50)),
        "target_monthly_pct": float(target_monthly_pct),
        "target_max_dd_pct": float(target_max_dd_pct),
    }
    row["passes_target"] = int(
        _passes_production_target(row, target_monthly_pct=float(target_monthly_pct), max_dd_pct=float(target_max_dd_pct))
    )
    row["goodness_score"] = float(
        row["median_monthly_pct"]
        + 0.75 * row["p10_monthly_pct"]
        - 1.35 * row["p90_dd_pct"]
        - 0.35 * row["worst_dd_pct"]
        - 8.0 * row["median_ulcer"]
        - 18.0 * row["neg_windows"]
        - max(0.0, 8.0 - float(row["median_trades"]))
    )
    return row


def _fieldnames() -> list[str]:
    return [
        "scope",
        "config",
        "gate",
        "n_longs",
        "n_shorts",
        "positions",
        "long_signal",
        "allocation_mode",
        "max_gross",
        "max_weight",
        "long_fraction",
        "max_net",
        "lookback_days",
        "cvar_alpha",
        "edge_power",
        "risk_power",
        "min_abs_score",
        "rebalance_days",
        "rebalance_style",
        "eval_days",
        "slip_bps",
        "fill_buffer_bps",
        "fee_rate",
        "short_borrow_apr",
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
        "worst_dd_pct",
        "median_sortino",
        "median_smooth",
        "median_ulcer",
        "median_trades",
        "target_monthly_pct",
        "target_max_dd_pct",
        "passes_target",
        "goodness_score",
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rule-csv", type=Path, default=Path("analysis/binance33_linear_rule_search_20260430.csv"))
    parser.add_argument("--configs", default="rand0045_rb1_thr0.5_btcmom20gt-0.05,rand0124_rb1_thr0_btcmom20gt-0.02,rand0038_rb1_thr0.2_btcmom20gt-0.05")
    parser.add_argument("--eval-data", type=Path, default=Path("analysis/binance33_refresh_20260502/binance33_daily_val.bin"))
    parser.add_argument("--train-data", type=Path, default=Path("analysis/binance33_refresh_20260502/binance33_daily_train.bin"))
    parser.add_argument("--out", type=Path, default=Path("analysis/binance33_hedged_cvar_pack.csv"))
    parser.add_argument("--scopes", default="train,val")
    parser.add_argument("--eval-days", type=int, default=120)
    parser.add_argument("--train-stride", type=int, default=20)
    parser.add_argument("--val-stride", type=int, default=1)
    parser.add_argument("--slippage-bps", default="20")
    parser.add_argument("--fee-rate", type=float, default=0.001)
    parser.add_argument("--fill-buffer-bps", type=float, default=5.0)
    parser.add_argument("--decision-lag", type=int, default=2)
    parser.add_argument("--short-borrow-apr", type=float, default=0.08)
    parser.add_argument("--periods-per-year", type=float, default=365.0)
    parser.add_argument("--gates", default="none,mkt_ret20gt-0.1,btc_vol20lt0.035")
    parser.add_argument("--exclude-symbols", default="PEPEUSD,ICPUSD,TAOUSD,ATOMUSD,FILUSD,TIAUSD,ALGOUSD,BNBUSD,AAVEUSD")
    parser.add_argument("--n-longs-grid", default="1,2")
    parser.add_argument("--n-shorts-grid", default="1,2")
    parser.add_argument("--max-positions", type=int, default=3)
    parser.add_argument("--long-signals", default="rule")
    parser.add_argument("--allocation-modes", default="equal,inv_vol_score,cvar,mean_cvar,soft_mean_cvar")
    parser.add_argument("--max-gross-grid", default="1.0,1.5,2.0,2.5")
    parser.add_argument("--max-weight-grid", default="0.34,0.5,0.67")
    parser.add_argument("--long-fraction-grid", default="0.4,0.5,0.6")
    parser.add_argument("--max-net-grid", default="0.0,0.25,0.5")
    parser.add_argument("--lookback-days-grid", default="20,40,80")
    parser.add_argument("--cvar-alpha-grid", default="0.85,0.9,0.95")
    parser.add_argument("--edge-power-grid", default="1.0")
    parser.add_argument("--risk-power-grid", default="0.5,1.0")
    parser.add_argument("--min-abs-score-grid", default="0.0,0.5,1.0")
    parser.add_argument("--rebalance-days-grid", default="2,3,5")
    parser.add_argument("--rebalance-styles", default="hold_same,full")
    parser.add_argument("--target-monthly-pct", type=float, default=30.0)
    parser.add_argument("--target-max-dd-pct", type=float, default=20.0)
    args = parser.parse_args()

    scope_names = _parse_str_list(args.scopes)
    scope_data: dict[str, tuple[MktdData, int]] = {}
    if "train" in scope_names:
        scope_data["train"] = (read_mktd(args.train_data), int(args.train_stride))
    if "val" in scope_names:
        scope_data["val"] = (read_mktd(args.eval_data), int(args.val_stride))
    unknown = sorted(set(scope_names) - set(scope_data))
    if unknown:
        raise ValueError(f"unknown scopes: {', '.join(unknown)}")

    rules = [_load_rule_from_csv(args.rule_csv, config) for config in _parse_str_list(args.configs)]
    gates = _parse_gate_options(args.gates)
    exclude_symbols = {symbol.upper() for symbol in _parse_str_list(args.exclude_symbols)}
    configs = [
        HedgedConfig(
            n_longs=n_longs,
            n_shorts=n_shorts,
            long_signal=long_signal,
            allocation_mode=allocation_mode,
            max_gross=max_gross,
            max_weight=max_weight,
            long_fraction=long_fraction,
            max_net=max_net,
            lookback_days=lookback,
            cvar_alpha=alpha,
            edge_power=edge_power,
            risk_power=risk_power,
            min_abs_score=min_abs_score,
            rebalance_days=rebalance_days,
            rebalance_style=rebalance_style,
        )
        for n_longs, n_shorts, long_signal, allocation_mode, max_gross, max_weight, long_fraction, max_net, lookback, alpha, edge_power, risk_power, min_abs_score, rebalance_days, rebalance_style in product(
            _parse_int_list(args.n_longs_grid),
            _parse_int_list(args.n_shorts_grid),
            _parse_str_list(args.long_signals),
            _parse_str_list(args.allocation_modes),
            _parse_float_list(args.max_gross_grid),
            _parse_float_list(args.max_weight_grid),
            _parse_float_list(args.long_fraction_grid),
            _parse_float_list(args.max_net_grid),
            _parse_int_list(args.lookback_days_grid),
            _parse_float_list(args.cvar_alpha_grid),
            _parse_float_list(args.edge_power_grid),
            _parse_float_list(args.risk_power_grid),
            _parse_float_list(args.min_abs_score_grid),
            _parse_int_list(args.rebalance_days_grid),
            _parse_str_list(args.rebalance_styles),
        )
        if n_longs + n_shorts <= int(args.max_positions)
    ]
    slippages = _parse_float_list(args.slippage_bps)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, float | int | str]] = []
    caches: dict[tuple[str, str], SignalCache] = {}
    with args.out.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=_fieldnames())
        writer.writeheader()
        for scope in scope_names:
            data, stride = scope_data[scope]
            for rule in rules:
                caches[(scope, rule.name)] = _build_cache(data, rule, exclude_symbols)
                cache = caches[(scope, rule.name)]
                for gate in gates:
                    for cfg in configs:
                        for slip in slippages:
                            row = _eval_config(
                                data,
                                cache,
                                rule,
                                gate,
                                cfg,
                                scope=scope,
                                eval_days=int(args.eval_days),
                                stride=int(stride),
                                fee_rate=float(args.fee_rate),
                                slippage_bps=float(slip),
                                fill_buffer_bps=float(args.fill_buffer_bps),
                                decision_lag=int(args.decision_lag),
                                short_borrow_apr=float(args.short_borrow_apr),
                                periods_per_year=float(args.periods_per_year),
                                target_monthly_pct=float(args.target_monthly_pct),
                                target_max_dd_pct=float(args.target_max_dd_pct),
                            )
                            rows.append(row)
                            writer.writerow(row)
                        fh.flush()

    val_rows = [row for row in rows if row["scope"] == "val"]
    val_rows.sort(key=lambda row: float(row["goodness_score"]), reverse=True)
    print(f"\n=== Best hedged CVaR packs val {int(args.eval_days)}d ===")
    for row in val_rows[:25]:
        print(
            f"good={row['goodness_score']:+7.2f} {row['median_monthly_pct']:+7.2f}%/mo "
            f"p10={row['p10_monthly_pct']:+7.2f}% dd90={row['p90_dd_pct']:.2f}% "
            f"worstdd={row['worst_dd_pct']:.2f}% neg={row['neg_windows']}/{row['windows']} "
            f"trades={row['median_trades']:.0f} {row['config']} gate={row['gate']} "
            f"L/S={row['n_longs']}/{row['n_shorts']} long={row['long_signal']} {row['allocation_mode']} "
            f"gross={row['max_gross']} maxw={row['max_weight']} lf={row['long_fraction']} "
            f"net={row['max_net']} lb={row['lookback_days']} rb={row['rebalance_days']} "
            f"reb={row['rebalance_style']} pass={row['passes_target']}"
        )
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
