#!/usr/bin/env python3
"""CVaR-style portfolio packing for Binance33 linear short rules.

The NVIDIA portfolio-optimization blueprint is useful here as an idea, not as
the execution engine: generate candidate alphas, pack them with explicit tail
risk, then evaluate the resulting weights under our Binance simulator charges.

This script takes a previously searched linear rule, shorts the strongest
bottom-ranked symbols, and allocates across them using rolling historical CVaR
statistics. It is intentionally CPU/numpy-fast so we can sweep many variants
before deciding whether a heavier cuOpt/CVX layer is worth wiring in.
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

from pufferlib_market.hourly_replay import MktdData, P_CLOSE, P_HIGH, read_mktd
from scripts.search_binance33_linear_rules import LinearRule, _score_vector
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
class PackConfig:
    top_n: int
    allocation_mode: str
    max_gross: float
    max_weight: float
    lookback_days: int
    cvar_alpha: float
    edge_power: float
    risk_power: float
    edge_return_scale: float
    score_temp: float
    rebalance_style: str


@dataclass(frozen=True)
class SignalCache:
    scores_by_t: np.ndarray
    valid_by_t: np.ndarray
    daily_returns: np.ndarray
    daily_return_valid: np.ndarray
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
    out = [part.strip() for part in str(value).split(",") if part.strip()]
    if not out:
        raise ValueError(f"expected at least one value in {value!r}")
    return out


def _btc_idx(data: MktdData) -> int:
    symbols = [symbol.upper() for symbol in data.symbols]
    return next((idx for idx, symbol in enumerate(symbols) if symbol.startswith("BTC")), 0)


def _build_cache(data: MktdData, rule: LinearRule) -> SignalCache:
    scores: list[np.ndarray] = []
    valid: list[np.ndarray] = []
    for t in range(data.num_timesteps):
        score_t, valid_t = _score_vector(data, rule, t)
        scores.append(np.asarray(score_t, dtype=np.float64))
        valid.append(np.asarray(valid_t, dtype=bool))

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
        daily_return_valid=ret_valid,
        btc_idx=_btc_idx(data),
    )


def _gate_options(spec: str) -> list[Gate]:
    requested = [part.strip() for part in str(spec).split(",") if part.strip()]
    if not requested:
        requested = ["none"]
    universe = _gate_universe()
    by_label = {_gate_label(gate): gate for gate in universe}
    out: list[Gate] = []
    for label in requested:
        if label.lower() in {"none", "nogate", "cashless"}:
            out.append(None)
            continue
        if label not in by_label:
            known = ", ".join(sorted(by_label)[:12])
            raise ValueError(f"unknown gate {label!r}; examples: none, {known}")
        out.append(by_label[label])
    return out


def _gate_name(gate: Gate) -> str:
    return "none" if gate is None else _gate_label(gate)


def _rule_allows(data: MktdData, cache: SignalCache, rule: LinearRule, gate: Gate, t: int) -> bool:
    t = max(0, min(int(t), data.num_timesteps - 1))
    if float(rule.btc_min_return_20d) > -9.0:
        btc_mom = float(data.features[t, cache.btc_idx, FEATURES["return_20d"]])
        if not np.isfinite(btc_mom) or btc_mom < float(rule.btc_min_return_20d):
            return False
    if gate is not None and not _gate_allows(gate, data, t):
        return False
    return True


def _normalise_alloc(raw: np.ndarray, *, gross: float, max_weight: float) -> np.ndarray:
    raw = np.asarray(raw, dtype=np.float64)
    raw = np.where(np.isfinite(raw) & (raw > 0.0), raw, 0.0)
    gross = max(0.0, float(gross))
    if gross <= 0.0 or float(raw.sum()) <= 0.0:
        return np.zeros_like(raw, dtype=np.float64)
    if max_weight <= 0.0:
        return raw / float(raw.sum()) * gross

    cap = max(0.0, float(max_weight))
    alloc = np.zeros_like(raw, dtype=np.float64)
    remaining = raw > 0.0
    remaining_gross = gross
    for _ in range(raw.size + 1):
        if remaining_gross <= 1e-12 or not remaining.any():
            break
        weights = raw[remaining] / float(raw[remaining].sum()) * remaining_gross
        capped = weights > cap
        idx = np.flatnonzero(remaining)
        if not capped.any():
            alloc[idx] = weights
            break
        cap_idx = idx[capped]
        alloc[cap_idx] = cap
        remaining[cap_idx] = False
        remaining_gross = gross - float(alloc.sum())
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
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = int(idx.size)
    mean = np.zeros(n, dtype=np.float64)
    risk = np.zeros(n, dtype=np.float64)
    hit_rate = np.zeros(n, dtype=np.float64)
    end = max(0, min(int(t), cache.daily_returns.shape[0]))
    start = max(0, end - max(1, int(lookback_days)))
    alpha = min(0.995, max(0.50, float(cvar_alpha)))
    for j, sym in enumerate(idx):
        series = cache.daily_returns[start:end, int(sym)] * float(signs[j])
        series = series[np.isfinite(series)]
        if series.size:
            mean[j] = float(series.mean())
            losses = -series
            tail_n = max(1, int(np.ceil((1.0 - alpha) * float(losses.size))))
            worst = np.partition(losses, max(0, losses.size - tail_n))[-tail_n:]
            tail = float(np.mean(worst))
            std = float(np.std(series))
            hit_rate[j] = float(np.mean(series > 0.0))
            risk[j] = max(tail, 0.50 * std)
        else:
            risk[j] = 0.0
    fallback = np.where(np.isfinite(fallback_vol) & (fallback_vol > 1e-6), fallback_vol, 0.02)
    risk = np.maximum(risk, 0.25 * fallback)
    risk = np.maximum(risk, 1e-4)
    return mean, risk, hit_rate


def _risk_adjusted_raw(
    *,
    mode: str,
    edge: np.ndarray,
    risk: np.ndarray,
    mean: np.ndarray,
    vol: np.ndarray,
    edge_power: float,
    risk_power: float,
    edge_return_scale: float,
    score_temp: float,
) -> np.ndarray:
    edge = np.maximum(np.asarray(edge, dtype=np.float64), 1e-12)
    risk = np.maximum(np.asarray(risk, dtype=np.float64), 1e-6)
    mean = np.asarray(mean, dtype=np.float64)
    vol = np.where(np.isfinite(vol) & (vol > 1e-6), vol, np.inf)
    base_edge = np.power(edge, max(0.0, float(edge_power)))
    if mode == "equal":
        raw = np.ones_like(edge)
    elif mode == "score":
        raw = base_edge
    elif mode == "inv_vol_score":
        raw = base_edge / vol
    elif mode == "cvar":
        raw = base_edge / np.power(risk, max(0.0, float(risk_power)))
    elif mode == "mean_cvar":
        raw = base_edge * np.maximum(mean, 0.0) / np.power(risk, max(0.0, float(risk_power)))
    elif mode == "kelly_cvar":
        alpha_mean = np.maximum(mean, 0.0) + float(edge_return_scale) * edge
        raw = alpha_mean / np.power(risk, max(0.0, float(risk_power)))
    elif mode == "softmax_cvar":
        temp = max(1e-6, float(score_temp))
        logits = base_edge / risk
        logits = logits / temp
        logits = logits - float(np.max(logits))
        raw = np.exp(logits)
    else:
        raise ValueError(f"unknown allocation_mode: {mode}")
    return np.where(np.isfinite(raw) & (raw > 0.0), raw, 0.0)


def _select_short_candidates(
    data: MktdData,
    cache: SignalCache,
    rule: LinearRule,
    gate: Gate,
    *,
    t: int,
    top_n: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    t = max(0, min(int(t), data.num_timesteps - 1))
    if not _rule_allows(data, cache, rule, gate, t):
        empty_i = np.asarray([], dtype=np.int64)
        empty_f = np.asarray([], dtype=np.float64)
        return empty_i, empty_f, empty_f
    scores = np.asarray(cache.scores_by_t[t], dtype=np.float64)
    valid = np.asarray(cache.valid_by_t[t], dtype=bool) & np.isfinite(scores)
    candidates = np.flatnonzero(valid & (-scores >= float(rule.min_abs_score)))
    if candidates.size == 0:
        empty_i = np.asarray([], dtype=np.int64)
        empty_f = np.asarray([], dtype=np.float64)
        return empty_i, empty_f, empty_f
    ordered = candidates[np.argsort(scores[candidates])]
    idx = ordered[: max(0, int(top_n))].astype(np.int64, copy=False)
    signs = -np.ones(idx.size, dtype=np.float64)
    edge = -scores[idx]
    return idx, signs, edge


def _target_weights(
    data: MktdData,
    cache: SignalCache,
    rule: LinearRule,
    gate: Gate,
    cfg: PackConfig,
    *,
    t: int,
) -> np.ndarray:
    idx, signs, edge = _select_short_candidates(data, cache, rule, gate, t=t, top_n=int(cfg.top_n))
    weights = np.zeros(data.num_symbols, dtype=np.float64)
    if idx.size == 0:
        return weights

    vol = np.asarray(data.features[max(0, min(int(t), data.num_timesteps - 1)), idx, FEATURES["volatility_20d"]])
    vol = vol.astype(np.float64, copy=False)
    mean, risk, _hit_rate = _tail_stats(
        cache,
        idx,
        signs,
        t=int(t),
        lookback_days=int(cfg.lookback_days),
        cvar_alpha=float(cfg.cvar_alpha),
        fallback_vol=vol,
    )
    raw = _risk_adjusted_raw(
        mode=str(cfg.allocation_mode),
        edge=edge,
        risk=risk,
        mean=mean,
        vol=vol,
        edge_power=float(cfg.edge_power),
        risk_power=float(cfg.risk_power),
        edge_return_scale=float(cfg.edge_return_scale),
        score_temp=float(cfg.score_temp),
    )
    alloc = _normalise_alloc(raw, gross=float(cfg.max_gross), max_weight=float(cfg.max_weight))
    weights[idx] = signs * alloc
    gross_now = float(np.abs(weights).sum())
    max_gross = max(0.0, float(cfg.max_gross))
    if gross_now > max_gross and gross_now > 0.0:
        weights *= max_gross / gross_now
    return weights


def _tradable_at(data: MktdData, t: int) -> np.ndarray:
    t = max(0, min(int(t), data.num_timesteps - 1))
    close = np.asarray(data.prices[t, :, P_CLOSE], dtype=np.float64)
    tradable = np.isfinite(close) & (close > 0.0)
    if data.tradable is not None:
        tradable &= np.asarray(data.tradable[t], dtype=bool)
    return tradable


def _apply_short_binary_fills(
    data: MktdData,
    current: np.ndarray,
    desired: np.ndarray,
    *,
    t: int,
    fill_buffer_bps: float,
) -> np.ndarray:
    """Apply action-sim binary fill semantics for short-only target weights.

    Reducing or covering an existing short fills at the current close when the
    symbol is tradable. Opening or increasing a short requires the current bar's
    high to cross close * (1 + fill_buffer), matching `_open_short_limit`.
    """
    t = max(0, min(int(t), data.num_timesteps - 1))
    current = np.asarray(current, dtype=np.float64)
    desired = np.asarray(desired, dtype=np.float64)
    out = current.copy()
    close = np.asarray(data.prices[t, :, P_CLOSE], dtype=np.float64)
    high = np.asarray(data.prices[t, :, P_HIGH], dtype=np.float64)
    tradable = _tradable_at(data, t) & np.isfinite(high) & (high > 0.0)
    short_fill = high >= close * (1.0 + max(0.0, float(fill_buffer_bps)) / 10_000.0)

    for sym in np.flatnonzero((np.abs(current) > 1e-12) | (np.abs(desired) > 1e-12)):
        old = min(0.0, float(current[int(sym)]))
        want = min(0.0, float(desired[int(sym)]))
        if not bool(tradable[int(sym)]):
            out[int(sym)] = old
            continue
        if want >= old:
            # Covering/reducing a short fills immediately at the close.
            out[int(sym)] = want
        elif bool(short_fill[int(sym)]):
            # More negative means opening or increasing the short.
            out[int(sym)] = want
        else:
            out[int(sym)] = old
    return out


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
    cfg: PackConfig,
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
        signal_t = int(start) + local_signal_t
        if local_signal_t >= 0 and (step == 0 or step % max(1, int(rule.rebalance_days)) == 0):
            desired = _target_weights(data, cache, rule, gate, cfg, t=signal_t)
            if str(cfg.rebalance_style) == "hold_same":
                old_sign = np.sign(weights)
                new_sign = np.sign(desired)
                old_idx = np.flatnonzero(np.abs(old_sign) > 0.0)
                new_idx = np.flatnonzero(np.abs(new_sign) > 0.0)
                same_book = np.array_equal(old_idx, new_idx) and np.all(old_sign[old_idx] == new_sign[old_idx])
                target = weights if same_book else desired
            elif str(cfg.rebalance_style) == "full":
                target = desired
            else:
                raise ValueError(f"unknown rebalance_style: {cfg.rebalance_style}")

        target = _apply_short_binary_fills(
            data,
            weights,
            target,
            t=global_t,
            fill_buffer_bps=float(fill_buffer_bps),
        )
        turnover = float(np.abs(target - weights).sum())
        if turnover > 1e-9:
            trades += int(np.count_nonzero(np.abs(target - weights) > 1e-9))
        day_ret = np.asarray(cache.daily_returns[global_t], dtype=np.float64)
        pnl = float(np.nansum(target * day_ret))
        borrow = float(np.abs(target[target < 0.0]).sum()) * borrow_per_period
        cost = turnover * cost_per_turnover
        period_growth = max(1e-9, 1.0 + pnl - cost - borrow)
        equity = max(1e-9, equity * period_growth)
        gross_return = np.where(np.isfinite(day_ret), 1.0 + day_ret, 1.0)
        weights = np.where(np.abs(target) > 1e-12, target * gross_return / period_growth, 0.0)
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
        "sortino": _sortino_from_equity(equity_curve),
        "smooth": float(compute_pnl_smoothness_from_equity(equity_curve)),
        "ulcer": float(compute_ulcer_index(equity_curve)),
        "trades": int(trades),
    }


def _eval_config(
    data: MktdData,
    cache: SignalCache,
    rule: LinearRule,
    gate: Gate,
    cfg: PackConfig,
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
    returns: list[float] = []
    maxdds: list[float] = []
    sortinos: list[float] = []
    smooths: list[float] = []
    ulcers: list[float] = []
    trades: list[int] = []
    for start in _candidate_starts(data, eval_days, stride=int(stride)):
        result = _simulate_window(
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
        returns.append(float(result["total_return"]))
        maxdds.append(float(result["max_drawdown"]))
        sortinos.append(float(result["sortino"]))
        smooths.append(float(result["smooth"]))
        ulcers.append(float(result["ulcer"]))
        trades.append(int(result["trades"]))

    arr = np.asarray(returns, dtype=np.float64)
    monthly = np.asarray([_monthly_equivalent_return(float(ret), int(eval_days)) for ret in arr], dtype=np.float64)
    row: dict[str, float | int | str] = {
        "scope": scope,
        "config": rule.name,
        "gate": _gate_name(gate),
        "top_n": int(cfg.top_n),
        "allocation_mode": cfg.allocation_mode,
        "max_gross": float(cfg.max_gross),
        "max_weight": float(cfg.max_weight),
        "lookback_days": int(cfg.lookback_days),
        "cvar_alpha": float(cfg.cvar_alpha),
        "edge_power": float(cfg.edge_power),
        "risk_power": float(cfg.risk_power),
        "edge_return_scale": float(cfg.edge_return_scale),
        "score_temp": float(cfg.score_temp),
        "rebalance_style": cfg.rebalance_style,
        "rebalance_days": int(rule.rebalance_days),
        "min_abs_score": float(rule.min_abs_score),
        "btc_min_return_20d": float(rule.btc_min_return_20d),
        "eval_days": int(eval_days),
        "slip_bps": float(slippage_bps),
        "fill_buffer_bps": float(fill_buffer_bps),
        "fee_rate": float(fee_rate),
        "short_borrow_apr": float(short_borrow_apr),
        "median_pct": float(100.0 * np.percentile(arr, 50)),
        "p10_pct": float(100.0 * np.percentile(arr, 10)),
        "p90_pct": float(100.0 * np.percentile(arr, 90)),
        "median_monthly_pct": float(100.0 * np.percentile(monthly, 50)),
        "p10_monthly_pct": float(100.0 * np.percentile(monthly, 10)),
        "worst_monthly_pct": float(100.0 * np.min(monthly)),
        "best_monthly_pct": float(100.0 * np.max(monthly)),
        "neg_windows": int(np.sum(arr < 0.0)),
        "windows": int(arr.size),
        "p90_dd_pct": float(100.0 * np.percentile(np.asarray(maxdds, dtype=np.float64), 90)),
        "median_sortino": float(np.percentile(np.asarray(sortinos, dtype=np.float64), 50)),
        "median_smooth": float(np.percentile(np.asarray(smooths, dtype=np.float64), 50)),
        "median_ulcer": float(np.percentile(np.asarray(ulcers, dtype=np.float64), 50)),
        "median_trades": float(np.percentile(np.asarray(trades, dtype=np.float64), 50)),
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
    row["goodness_score"] = float(
        row["median_monthly_pct"]
        + 0.75 * row["p10_monthly_pct"]
        - 1.25 * row["p90_dd_pct"]
        - 18.0 * row["neg_windows"]
    )
    return row


def _fieldnames() -> list[str]:
    return [
        "scope",
        "config",
        "gate",
        "top_n",
        "allocation_mode",
        "max_gross",
        "max_weight",
        "lookback_days",
        "cvar_alpha",
        "edge_power",
        "risk_power",
        "edge_return_scale",
        "score_temp",
        "rebalance_style",
        "rebalance_days",
        "min_abs_score",
        "btc_min_return_20d",
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
    parser = argparse.ArgumentParser(description="Sweep CVaR-packed Binance33 linear short rules.")
    parser.add_argument("--rule-csv", type=Path, default=Path("analysis/binance33_linear_rule_search_20260430.csv"))
    parser.add_argument("--config", default="rand0045_rb1_thr0.5_btcmom20gt-0.05")
    parser.add_argument("--eval-data", type=Path, default=Path("pufferlib_market/data/binance33_daily_val.bin"))
    parser.add_argument("--train-data", type=Path, default=Path("pufferlib_market/data/binance33_daily_train.bin"))
    parser.add_argument("--out", type=Path, default=Path("analysis/binance33_linear_cvar_pack.csv"))
    parser.add_argument("--scopes", default="val")
    parser.add_argument("--eval-days", type=int, default=120)
    parser.add_argument("--val-stride", type=int, default=1)
    parser.add_argument("--train-stride", type=int, default=20)
    parser.add_argument("--slippage-bps", default="20")
    parser.add_argument("--fee-rate", type=float, default=0.001)
    parser.add_argument("--fill-buffer-bps", type=float, default=5.0)
    parser.add_argument("--decision-lag", type=int, default=2)
    parser.add_argument("--short-borrow-apr", type=float, default=0.08)
    parser.add_argument("--periods-per-year", type=float, default=365.0)
    parser.add_argument("--gates", default="none,mkt_ret20gt-0.1")
    parser.add_argument("--top-n-grid", default="1,2,3,5")
    parser.add_argument("--allocation-modes", default="equal,score,inv_vol_score,cvar,mean_cvar,kelly_cvar")
    parser.add_argument("--max-gross-grid", default="1.0,1.2,1.305,1.4")
    parser.add_argument("--max-weight-grid", default="1.0,0.67,0.5,0.33")
    parser.add_argument("--lookback-days-grid", default="20,40,60")
    parser.add_argument("--cvar-alpha-grid", default="0.85,0.9,0.95")
    parser.add_argument("--edge-power-grid", default="1.0")
    parser.add_argument("--risk-power-grid", default="0.5,1.0")
    parser.add_argument("--edge-return-scale-grid", default="0.0025,0.005")
    parser.add_argument("--score-temp-grid", default="1.0")
    parser.add_argument("--rebalance-styles", default="hold_same,full")
    parser.add_argument("--target-monthly-pct", type=float, default=30.0)
    parser.add_argument("--target-max-dd-pct", type=float, default=20.0)
    parser.add_argument("--require-production-target", action="store_true")
    args = parser.parse_args()

    rule = _load_rule_from_csv(args.rule_csv, args.config)
    scope_names = _parse_str_list(args.scopes)
    scope_data: dict[str, tuple[MktdData, int]] = {}
    if "val" in scope_names:
        scope_data["val"] = (read_mktd(args.eval_data), int(args.val_stride))
    if "train" in scope_names:
        scope_data["train"] = (read_mktd(args.train_data), int(args.train_stride))
    unknown_scopes = sorted(set(scope_names) - set(scope_data))
    if unknown_scopes:
        raise ValueError(f"unknown scopes: {', '.join(unknown_scopes)}")

    gates = _gate_options(args.gates)
    pack_configs = [
        PackConfig(
            top_n=top_n,
            allocation_mode=mode,
            max_gross=max_gross,
            max_weight=max_weight,
            lookback_days=lookback,
            cvar_alpha=alpha,
            edge_power=edge_power,
            risk_power=risk_power,
            edge_return_scale=edge_return_scale,
            score_temp=score_temp,
            rebalance_style=rebalance_style,
        )
        for top_n, mode, max_gross, max_weight, lookback, alpha, edge_power, risk_power, edge_return_scale, score_temp, rebalance_style in product(
            _parse_int_list(args.top_n_grid),
            _parse_str_list(args.allocation_modes),
            _parse_float_list(args.max_gross_grid),
            _parse_float_list(args.max_weight_grid),
            _parse_int_list(args.lookback_days_grid),
            _parse_float_list(args.cvar_alpha_grid),
            _parse_float_list(args.edge_power_grid),
            _parse_float_list(args.risk_power_grid),
            _parse_float_list(args.edge_return_scale_grid),
            _parse_float_list(args.score_temp_grid),
            _parse_str_list(args.rebalance_styles),
        )
    ]
    slippages = _parse_float_list(args.slippage_bps)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, float | int | str]] = []
    caches: dict[str, SignalCache] = {}
    with args.out.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=_fieldnames())
        writer.writeheader()
        for scope in scope_names:
            data, stride = scope_data[scope]
            caches[scope] = _build_cache(data, rule)
            for gate in gates:
                for cfg in pack_configs:
                    for slip in slippages:
                        row = _eval_config(
                            data,
                            caches[scope],
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
    print(f"\n=== Best CVaR-packed linear rules val {int(args.eval_days)}d ===")
    for row in val_rows[:25]:
        print(
            f"good={row['goodness_score']:+7.2f} {row['median_monthly_pct']:+7.2f}%/mo "
            f"p10={row['p10_monthly_pct']:+7.2f}% dd90={row['p90_dd_pct']:.2f}% "
            f"neg={row['neg_windows']}/{row['windows']} trades={row['median_trades']:.0f} "
            f"pass={row['passes_target']} gate={row['gate']} top={row['top_n']} "
            f"{row['allocation_mode']} gross={row['max_gross']} maxw={row['max_weight']} "
            f"lb={row['lookback_days']} a={row['cvar_alpha']} rp={row['risk_power']} "
            f"reb={row['rebalance_style']}"
        )

    promoted = [row for row in val_rows if int(row["passes_target"]) == 1]
    print(f"\n=== Target candidates val {int(args.eval_days)}d ===")
    if promoted:
        for row in promoted[:10]:
            print(
                f"{row['median_monthly_pct']:+7.2f}%/mo p10={row['p10_monthly_pct']:+7.2f}% "
                f"dd90={row['p90_dd_pct']:.2f}% gate={row['gate']} top={row['top_n']} "
                f"{row['allocation_mode']} gross={row['max_gross']} reb={row['rebalance_style']}"
            )
    else:
        print("none")
    print(f"\nwrote {args.out}")
    if args.require_production_target and not promoted:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
