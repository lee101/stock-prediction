#!/usr/bin/env python3
"""Sweep causal risk overlays around Binance33 meta-anneal candidates.

The meta annealer can find recent-regime books with strong upside but ugly
old-history drawdowns. This harness keeps the candidate score book fixed and
tests realistic overlays:

* shadow-PnL gate: only trade when the always-on strategy has recently worked
* book-level circuit breaker with cooldown
* drawdown throttle

All decisions use state available before the next order and the execution path
uses the same lag-2 binary fills as the vector meta simulator.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Any, Sequence

import numpy as np

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from pufferlib_market.hourly_replay import MktdData, P_CLOSE, read_mktd
from scripts.search_binance33_meta_anneal import (
    Candidate,
    ScoreBank,
    _apply_binary_fills,
    _apply_excluded_symbols,
    _build_bank,
    _combine_scores,
    _desired_weights,
    _evolve_weights_after_return,
    _load_rule_rows,
    _softmax,
)
from scripts.sweep_binance33_rules import _candidate_starts
from scripts.sweep_binance33_xgb import _monthly_equivalent_return, _passes_production_target
from src.robust_trading_metrics import compute_pnl_smoothness_from_equity, compute_ulcer_index


@dataclass(frozen=True)
class OverlayConfig:
    gross: float
    max_weight: float
    threshold: float | None
    rebalance_days: int | None
    short_risk_mult: float | None
    lookback_days: int
    min_shadow_return: float
    max_shadow_dd: float
    fail_scale: float
    circuit_dd: float
    cooldown_days: int
    throttle_dd: float
    throttle_scale: float


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


def _parse_optional_float_list(value: str) -> list[float | None]:
    out: list[float | None] = []
    for part in str(value).split(","):
        token = part.strip().lower()
        if not token:
            continue
        out.append(None if token in {"payload", "base", "none"} else float(token))
    if not out:
        raise ValueError(f"expected at least one float/base token in {value!r}")
    return out


def _parse_optional_int_list(value: str) -> list[int | None]:
    out: list[int | None] = []
    for part in str(value).split(","):
        token = part.strip().lower()
        if not token:
            continue
        out.append(None if token in {"payload", "base", "none"} else int(token))
    if not out:
        raise ValueError(f"expected at least one int/base token in {value!r}")
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


def _candidate_from_payload(payload: dict[str, Any], *, bank: ScoreBank, cfg: OverlayConfig) -> Candidate:
    names = [str(name) for name in payload["channel_names"]]
    weights = np.asarray(payload["channel_weights"], dtype=np.float64)
    if weights.size != len(names):
        raise ValueError("candidate channel names/weights length mismatch")
    by_name = {name: float(weight) for name, weight in zip(names, weights)}
    missing = [name for name in bank.names if name not in by_name]
    if missing:
        raise ValueError(f"candidate is missing channels present in bank: {missing[:5]}")
    ordered = np.asarray([max(1e-12, by_name[name]) for name in bank.names], dtype=np.float64)
    logits = np.log(ordered)
    return Candidate(
        candidate_id=str(payload.get("candidate_id", "candidate")),
        logits=logits,
        threshold=float(payload["threshold"] if cfg.threshold is None else cfg.threshold),
        max_gross=float(cfg.gross),
        max_weight=float(cfg.max_weight),
        top_k=int(payload["top_k"]),
        book_mode=str(payload["book_mode"]),
        score_temp=float(payload["score_temp"]),
        btc_gate=float(payload["btc_gate"]),
        market_gate=float(payload["market_gate"]),
        rebalance_days=int(payload["rebalance_days"] if cfg.rebalance_days is None else cfg.rebalance_days),
        long_fraction=float(payload.get("long_fraction", 0.5)),
        short_risk_mult=float(payload.get("short_risk_mult", 1.0) if cfg.short_risk_mult is None else cfg.short_risk_mult),
        always_trade=bool(payload.get("always_trade", False)),
    )


def _load_candidate_payloads(path: Path, ids: Sequence[str], phase: str) -> list[dict[str, Any]]:
    wanted = set(ids)
    payloads: list[dict[str, Any]] = []
    seen: set[str] = set()
    with path.open(newline="") as fh:
        for row in csv.DictReader(fh):
            candidate_id = str(row.get("candidate_id", ""))
            if row.get("phase") != phase or (wanted and candidate_id not in wanted) or candidate_id in seen:
                continue
            raw = str(row.get("candidate_json", "")).strip()
            if not raw:
                continue
            payloads.append(json.loads(raw))
            seen.add(candidate_id)
    if not payloads:
        raise ValueError(f"no candidate payloads found in {path} for phase={phase!r} ids={sorted(wanted)}")
    return payloads


def _linear_configs_for_payloads(payloads: Sequence[dict[str, Any]]) -> list[str]:
    out: list[str] = []
    for payload in payloads:
        for name in payload["channel_names"]:
            name = str(name)
            if name.startswith("linear:"):
                out.append(name.split(":", 1)[1])
    return list(dict.fromkeys(out))


def _daily_returns(data: MktdData) -> np.ndarray:
    close = np.asarray(data.prices[:, :, P_CLOSE], dtype=np.float64)
    p0 = close[:-1]
    p1 = close[1:]
    valid = np.isfinite(p0) & np.isfinite(p1) & (p0 > 0.0) & (p1 > 0.0)
    out = np.zeros_like(p0, dtype=np.float64)
    out[valid] = p1[valid] / p0[valid] - 1.0
    return out


def _shadow_gate_state(curve: list[float], cfg: OverlayConfig) -> tuple[bool, float, float]:
    lookback = max(1, int(cfg.lookback_days))
    if len(curve) <= lookback:
        return True, 0.0, 0.0
    recent = np.asarray(curve[-lookback - 1 :], dtype=np.float64)
    shadow_return = float(recent[-1] / max(1e-9, recent[0]) - 1.0)
    shadow_dd = _max_drawdown(recent)
    allowed = shadow_return >= float(cfg.min_shadow_return) and shadow_dd <= float(cfg.max_shadow_dd)
    return bool(allowed), shadow_return, shadow_dd


def _simulate_window(
    data: MktdData,
    daily_returns: np.ndarray,
    scores_by_t: np.ndarray,
    candidate: Candidate,
    cfg: OverlayConfig,
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
    actual_weights = np.zeros(data.num_symbols, dtype=np.float64)
    shadow_weights = np.zeros(data.num_symbols, dtype=np.float64)
    actual_equity = 1.0
    shadow_equity = 1.0
    actual_curve = [actual_equity]
    shadow_curve = [shadow_equity]
    peak = 1.0
    cooldown = 0
    trades = 0
    gated_days = 0
    circuit_days = 0
    cost_per_turnover = float(fee_rate) + float(slippage_bps) / 10_000.0
    borrow_per_period = max(0.0, float(short_borrow_apr)) / max(1.0, float(periods_per_year))

    for step in range(int(eval_days)):
        global_t = int(start) + int(step)
        signal_t = int(start) + int(step) - int(decision_lag)
        day_ret = np.asarray(daily_returns[global_t], dtype=np.float64)

        shadow_target = shadow_weights
        base_desired = actual_weights
        if signal_t >= int(start) and (step == 0 or step % max(1, int(candidate.rebalance_days)) == 0):
            desired = _desired_weights(data, scores_by_t, candidate, t=signal_t, btc_idx=btc_idx)
            shadow_target = desired
            old_idx = np.flatnonzero(np.abs(actual_weights) > 1e-12)
            new_idx = np.flatnonzero(np.abs(desired) > 1e-12)
            same_book = np.array_equal(old_idx, new_idx) and np.all(
                np.sign(actual_weights[old_idx]) == np.sign(desired[old_idx])
            )
            base_desired = actual_weights if same_book and not bool(candidate.always_trade) else desired

        shadow_target = _apply_binary_fills(
            data,
            shadow_weights,
            shadow_target,
            t=global_t,
            fill_buffer_bps=float(fill_buffer_bps),
        )

        allow_shadow, _shadow_ret, _shadow_dd = _shadow_gate_state(shadow_curve, cfg)
        scale = 1.0 if allow_shadow else float(cfg.fail_scale)
        if scale < 1.0:
            gated_days += 1
        current_dd = max(0.0, (peak - actual_equity) / max(1e-9, peak))
        if float(cfg.throttle_dd) > 0.0 and current_dd >= float(cfg.throttle_dd):
            scale = min(scale, float(cfg.throttle_scale))
        if cooldown > 0:
            scale = 0.0
            cooldown -= 1
            circuit_days += 1
        elif float(cfg.circuit_dd) > 0.0 and current_dd >= float(cfg.circuit_dd):
            scale = 0.0
            cooldown = max(0, int(cfg.cooldown_days))
            circuit_days += 1
        actual_target = np.asarray(base_desired, dtype=np.float64) * float(scale)
        actual_target = _apply_binary_fills(
            data,
            actual_weights,
            actual_target,
            t=global_t,
            fill_buffer_bps=float(fill_buffer_bps),
        )

        actual_turnover = float(np.abs(actual_target - actual_weights).sum())
        if actual_turnover > 1e-9:
            trades += int(np.count_nonzero(np.abs(actual_target - actual_weights) > 1e-9))
        actual_pnl = float(np.dot(actual_target, day_ret))
        actual_borrow = float(np.abs(actual_target[actual_target < 0.0]).sum()) * borrow_per_period
        actual_growth = max(1e-9, 1.0 + actual_pnl - actual_turnover * cost_per_turnover - actual_borrow)
        actual_equity = max(1e-9, actual_equity * actual_growth)
        peak = max(peak, actual_equity)
        gross_return = np.where(np.isfinite(day_ret), 1.0 + day_ret, 1.0)
        actual_weights = _evolve_weights_after_return(actual_target, gross_return, actual_growth)
        actual_curve.append(actual_equity)

        shadow_turnover = float(np.abs(shadow_target - shadow_weights).sum())
        shadow_pnl = float(np.dot(shadow_target, day_ret))
        shadow_borrow = float(np.abs(shadow_target[shadow_target < 0.0]).sum()) * borrow_per_period
        shadow_growth = max(1e-9, 1.0 + shadow_pnl - shadow_turnover * cost_per_turnover - shadow_borrow)
        shadow_equity = max(1e-9, shadow_equity * shadow_growth)
        shadow_weights = _evolve_weights_after_return(shadow_target, gross_return, shadow_growth)
        shadow_curve.append(shadow_equity)

    final_turnover = float(np.abs(actual_weights).sum())
    if final_turnover > 1e-9:
        trades += int(np.count_nonzero(np.abs(actual_weights) > 1e-9))
        actual_equity = max(1e-9, actual_equity * (1.0 - final_turnover * cost_per_turnover))
        actual_curve[-1] = actual_equity

    curve = np.asarray(actual_curve, dtype=np.float64)
    return {
        "total_return": float(curve[-1] - 1.0),
        "max_drawdown": _max_drawdown(curve),
        "sortino": _sortino_from_equity(curve, periods_per_year=float(periods_per_year)),
        "smooth": float(compute_pnl_smoothness_from_equity(curve)),
        "ulcer": float(compute_ulcer_index(curve)),
        "trades": int(trades),
        "gated_days": int(gated_days),
        "circuit_days": int(circuit_days),
    }


def _eval_overlay(
    data: MktdData,
    bank: ScoreBank,
    payload: dict[str, Any],
    cfg: OverlayConfig,
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
    candidate = _candidate_from_payload(payload, bank=bank, cfg=cfg)
    scores = _combine_scores(bank, candidate)
    daily_returns = _daily_returns(data)
    symbols = [str(symbol).upper() for symbol in data.symbols]
    btc_idx = next((idx for idx, symbol in enumerate(symbols) if symbol.startswith("BTC")), 0)
    results = [
        _simulate_window(
            data,
            daily_returns,
            scores,
            candidate,
            cfg,
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
    returns = np.asarray([float(row["total_return"]) for row in results], dtype=np.float64)
    maxdds = np.asarray([float(row["max_drawdown"]) for row in results], dtype=np.float64)
    sortinos = np.asarray([float(row["sortino"]) for row in results], dtype=np.float64)
    smooths = np.asarray([float(row["smooth"]) for row in results], dtype=np.float64)
    ulcers = np.asarray([float(row["ulcer"]) for row in results], dtype=np.float64)
    trades = np.asarray([int(row["trades"]) for row in results], dtype=np.float64)
    gated = np.asarray([int(row["gated_days"]) for row in results], dtype=np.float64)
    circuit = np.asarray([int(row["circuit_days"]) for row in results], dtype=np.float64)
    monthly = np.asarray([_monthly_equivalent_return(float(ret), int(eval_days)) for ret in returns], dtype=np.float64)
    channel_weights = {
        name: round(float(weight), 6)
        for name, weight in zip(bank.names, _softmax(candidate.logits))
        if float(weight) > 1e-4
    }
    out: dict[str, float | int | str] = {
        "scope": scope,
        "candidate_id": str(payload.get("candidate_id", "")),
        "book_mode": str(payload.get("book_mode", "")),
        "top_k": int(payload.get("top_k", 0)),
        "gross": float(cfg.gross),
        "max_weight": float(cfg.max_weight),
        "threshold": float(payload["threshold"] if cfg.threshold is None else cfg.threshold),
        "rebalance_days": int(payload["rebalance_days"] if cfg.rebalance_days is None else cfg.rebalance_days),
        "short_risk_mult": float(
            payload.get("short_risk_mult", 1.0) if cfg.short_risk_mult is None else cfg.short_risk_mult
        ),
        "lookback_days": int(cfg.lookback_days),
        "min_shadow_return": float(cfg.min_shadow_return),
        "max_shadow_dd": float(cfg.max_shadow_dd),
        "fail_scale": float(cfg.fail_scale),
        "circuit_dd": float(cfg.circuit_dd),
        "cooldown_days": int(cfg.cooldown_days),
        "throttle_dd": float(cfg.throttle_dd),
        "throttle_scale": float(cfg.throttle_scale),
        "channel_weights_json": json.dumps(channel_weights, sort_keys=True),
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
        "worst_dd_pct": float(100.0 * np.max(maxdds)),
        "median_sortino": float(np.percentile(sortinos, 50)),
        "median_smooth": float(np.percentile(smooths, 50)),
        "median_ulcer": float(np.percentile(ulcers, 50)),
        "median_trades": float(np.percentile(trades, 50)),
        "median_gated_days": float(np.percentile(gated, 50)),
        "median_circuit_days": float(np.percentile(circuit, 50)),
        "target_monthly_pct": float(target_monthly_pct),
        "target_max_dd_pct": float(target_max_dd_pct),
    }
    out["passes_target"] = int(
        _passes_production_target(
            out,
            target_monthly_pct=float(target_monthly_pct),
            max_dd_pct=float(target_max_dd_pct),
        )
    )
    out["goodness_score"] = float(
        out["median_monthly_pct"]
        + 0.75 * out["p10_monthly_pct"]
        - 1.35 * out["p90_dd_pct"]
        - 0.35 * out["worst_dd_pct"]
        - 12.0 * (float(out["neg_windows"]) / max(1.0, float(out["windows"])))
        - 0.05 * out["median_gated_days"]
        - max(0.0, 8.0 - float(out["median_trades"]))
    )
    return out


def _fieldnames() -> list[str]:
    return [
        "scope",
        "candidate_id",
        "book_mode",
        "top_k",
        "gross",
        "max_weight",
        "threshold",
        "rebalance_days",
        "short_risk_mult",
        "lookback_days",
        "min_shadow_return",
        "max_shadow_dd",
        "fail_scale",
        "circuit_dd",
        "cooldown_days",
        "throttle_dd",
        "throttle_scale",
        "channel_weights_json",
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
        "median_gated_days",
        "median_circuit_days",
        "target_monthly_pct",
        "target_max_dd_pct",
        "passes_target",
        "goodness_score",
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-csv", type=Path, default=Path("analysis/binance33_meta_anneal_linear_hand_aug_active_20260501.csv"))
    parser.add_argument("--candidate-ids", default="cand00426,cand00406,cand00469")
    parser.add_argument("--candidate-phase", default="val_fast")
    parser.add_argument("--rule-csv", type=Path, default=Path("analysis/binance33_linear_rule_search_20260430.csv"))
    parser.add_argument("--train-data", type=Path, default=Path("analysis/binance33_refresh_20260502/binance33_daily_train.bin"))
    parser.add_argument("--eval-data", type=Path, default=Path("analysis/binance33_refresh_20260502/binance33_daily_val.bin"))
    parser.add_argument("--out", type=Path, default=Path("analysis/binance33_meta_risk_overlay.csv"))
    parser.add_argument("--scopes", default="train,val")
    parser.add_argument("--eval-days", type=int, default=120)
    parser.add_argument("--train-stride", type=int, default=40)
    parser.add_argument("--val-stride", type=int, default=2)
    parser.add_argument("--exclude-symbols", default="PEPEUSD,ICPUSD,TAOUSD,ATOMUSD,FILUSD,TIAUSD,ALGOUSD,BNBUSD,AAVEUSD")
    parser.add_argument("--gross-grid", default="2.5,4.6")
    parser.add_argument("--max-weight-grid", default="0.34,0.5")
    parser.add_argument("--threshold-grid", default="payload")
    parser.add_argument("--rebalance-days-grid", default="payload")
    parser.add_argument("--short-risk-mult-grid", default="payload")
    parser.add_argument("--lookback-days-grid", default="20,80")
    parser.add_argument("--min-shadow-return-grid", default="-0.05,0.05,0.20")
    parser.add_argument("--max-shadow-dd-grid", default="0.10,0.20")
    parser.add_argument("--fail-scale-grid", default="0.0,0.25")
    parser.add_argument("--circuit-dd-grid", default="0.0,0.15")
    parser.add_argument("--cooldown-days-grid", default="0,20")
    parser.add_argument("--throttle-dd-grid", default="0.0,0.10")
    parser.add_argument("--throttle-scale-grid", default="0.5")
    parser.add_argument("--slippage-bps", type=float, default=20.0)
    parser.add_argument("--fee-rate", type=float, default=0.001)
    parser.add_argument("--fill-buffer-bps", type=float, default=5.0)
    parser.add_argument("--decision-lag", type=int, default=2)
    parser.add_argument("--short-borrow-apr", type=float, default=0.08)
    parser.add_argument("--periods-per-year", type=float, default=365.0)
    parser.add_argument("--target-monthly-pct", type=float, default=30.0)
    parser.add_argument("--target-max-dd-pct", type=float, default=20.0)
    args = parser.parse_args()

    payloads = _load_candidate_payloads(args.candidate_csv, _parse_str_list(args.candidate_ids), str(args.candidate_phase))
    rules = _load_rule_rows(
        args.rule_csv,
        configs=_linear_configs_for_payloads(payloads),
        max_rules=64,
    )
    scope_names = _parse_str_list(args.scopes)
    scope_data: dict[str, tuple[MktdData, int]] = {}
    if "train" in scope_names:
        scope_data["train"] = (read_mktd(args.train_data), int(args.train_stride))
    if "val" in scope_names:
        scope_data["val"] = (read_mktd(args.eval_data), int(args.val_stride))
    unknown = sorted(set(scope_names) - set(scope_data))
    if unknown:
        raise ValueError(f"unknown scopes: {', '.join(unknown)}")
    excluded_symbols = _parse_str_list(args.exclude_symbols)
    banks: dict[str, ScoreBank] = {}
    for scope, (data, _stride) in scope_data.items():
        bank = _build_bank(data, rules=rules, xgb_models=[], include_handcrafted=True)
        if excluded_symbols:
            bank = _apply_excluded_symbols(bank, data, excluded_symbols)
        banks[scope] = bank

    configs = [
        OverlayConfig(
            gross=gross,
            max_weight=max_weight,
            threshold=threshold,
            rebalance_days=rebalance_days,
            short_risk_mult=short_risk_mult,
            lookback_days=lookback,
            min_shadow_return=min_shadow_return,
            max_shadow_dd=max_shadow_dd,
            fail_scale=fail_scale,
            circuit_dd=circuit_dd,
            cooldown_days=cooldown_days,
            throttle_dd=throttle_dd,
            throttle_scale=throttle_scale,
        )
        for gross, max_weight, threshold, rebalance_days, short_risk_mult, lookback, min_shadow_return, max_shadow_dd, fail_scale, circuit_dd, cooldown_days, throttle_dd, throttle_scale in product(
            _parse_float_list(args.gross_grid),
            _parse_float_list(args.max_weight_grid),
            _parse_optional_float_list(args.threshold_grid),
            _parse_optional_int_list(args.rebalance_days_grid),
            _parse_optional_float_list(args.short_risk_mult_grid),
            _parse_int_list(args.lookback_days_grid),
            _parse_float_list(args.min_shadow_return_grid),
            _parse_float_list(args.max_shadow_dd_grid),
            _parse_float_list(args.fail_scale_grid),
            _parse_float_list(args.circuit_dd_grid),
            _parse_int_list(args.cooldown_days_grid),
            _parse_float_list(args.throttle_dd_grid),
            _parse_float_list(args.throttle_scale_grid),
        )
    ]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, float | int | str]] = []
    with args.out.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=_fieldnames())
        writer.writeheader()
        for payload in payloads:
            for cfg in configs:
                for scope in scope_names:
                    data, stride = scope_data[scope]
                    row = _eval_overlay(
                        data,
                        banks[scope],
                        payload,
                        cfg,
                        scope=scope,
                        eval_days=int(args.eval_days),
                        stride=int(stride),
                        fee_rate=float(args.fee_rate),
                        slippage_bps=float(args.slippage_bps),
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
    print("\n=== Best risk-overlay validation rows ===")
    for row in val_rows[:25]:
        print(
            f"good={row['goodness_score']:+7.2f} {row['median_monthly_pct']:+7.2f}%/mo "
            f"p10={row['p10_monthly_pct']:+7.2f}% dd90={row['p90_dd_pct']:.2f}% "
            f"worstdd={row['worst_dd_pct']:.2f}% neg={row['neg_windows']}/{row['windows']} "
            f"trades={row['median_trades']:.0f} gated={row['median_gated_days']:.0f} "
            f"circ={row['median_circuit_days']:.0f} id={row['candidate_id']} gross={row['gross']} "
            f"maxw={row['max_weight']} thr={row['threshold']} rb={row['rebalance_days']} "
            f"sr={row['short_risk_mult']} lb={row['lookback_days']} minret={row['min_shadow_return']} "
            f"maxshdd={row['max_shadow_dd']} fail={row['fail_scale']} cdd={row['circuit_dd']} "
            f"cd={row['cooldown_days']} tdd={row['throttle_dd']} ts={row['throttle_scale']} "
            f"pass={row['passes_target']}"
        )
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
