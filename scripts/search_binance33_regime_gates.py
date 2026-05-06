#!/usr/bin/env python3
"""Search simple regime/cash gates around a linear Binance33 rule."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from pufferlib_market.hourly_replay import MktdData, simulate_daily_policy, read_mktd
from scripts.search_binance33_linear_rules import FEATURE_NAMES, LinearRule, _make_policy as _make_linear_policy
from scripts.sweep_binance33_rules import FEATURES, _candidate_starts, _slice_window
from scripts.sweep_binance33_xgb import _monthly_equivalent_return, _passes_production_target
from src.robust_trading_metrics import compute_pnl_smoothness_from_equity, compute_ulcer_index


@dataclass(frozen=True)
class GateSpec:
    name: str
    fn: Callable[[MktdData, int], float]
    op: str
    threshold: float

    @property
    def label(self) -> str:
        return f"{self.name}{self.op}{self.threshold:g}"

    def allows(self, data: MktdData, t: int) -> bool:
        value = float(self.fn(data, t))
        if not np.isfinite(value):
            return False
        if self.op == "gt":
            return value > float(self.threshold)
        if self.op == "lt":
            return value < float(self.threshold)
        raise ValueError(f"unknown op: {self.op}")


def _btc_idx(data: MktdData) -> int:
    symbols = [symbol.upper() for symbol in data.symbols]
    return next((idx for idx, symbol in enumerate(symbols) if symbol.startswith("BTC")), 0)


def _valid_feature(data: MktdData, t: int, feature_name: str) -> np.ndarray:
    values = np.asarray(data.features[t, :, FEATURES[feature_name]], dtype=np.float64)
    mask = np.isfinite(values)
    if data.tradable is not None:
        mask &= np.asarray(data.tradable[t], dtype=bool)
    return values[mask]


def _median_feature(data: MktdData, t: int, feature_name: str) -> float:
    values = _valid_feature(data, t, feature_name)
    return float(np.median(values)) if values.size else float("nan")


def _breadth_gt_zero(data: MktdData, t: int, feature_name: str) -> float:
    values = _valid_feature(data, t, feature_name)
    return float(np.mean(values > 0.0)) if values.size else float("nan")


def _iqr_feature(data: MktdData, t: int, feature_name: str) -> float:
    values = _valid_feature(data, t, feature_name)
    if values.size <= 1:
        return float("nan")
    return float(np.percentile(values, 75) - np.percentile(values, 25))


def _skew_feature(data: MktdData, t: int, feature_name: str) -> float:
    values = _valid_feature(data, t, feature_name)
    if values.size <= 2:
        return float("nan")
    centered = values - float(np.mean(values))
    std = float(np.std(centered))
    if std <= 1e-9:
        return 0.0
    return float(np.mean((centered / std) ** 3))


def _median_abs_feature(data: MktdData, t: int, feature_name: str) -> float:
    values = _valid_feature(data, t, feature_name)
    return float(np.median(np.abs(values))) if values.size else float("nan")


def _gate_universe() -> list[GateSpec | tuple[GateSpec, GateSpec]]:
    single: list[GateSpec] = []

    def btc(name: str) -> Callable[[MktdData, int], float]:
        return lambda data, t: float(data.features[t, _btc_idx(data), FEATURES[name]])

    def median(name: str) -> Callable[[MktdData, int], float]:
        return lambda data, t: _median_feature(data, t, name)

    def breadth(name: str) -> Callable[[MktdData, int], float]:
        return lambda data, t: _breadth_gt_zero(data, t, name)

    def iqr(name: str) -> Callable[[MktdData, int], float]:
        return lambda data, t: _iqr_feature(data, t, name)

    def skew(name: str) -> Callable[[MktdData, int], float]:
        return lambda data, t: _skew_feature(data, t, name)

    def median_abs(name: str) -> Callable[[MktdData, int], float]:
        return lambda data, t: _median_abs_feature(data, t, name)

    specs = [
        ("btc_ret20", btc("return_20d"), "gt", [-0.10, -0.05, -0.02, 0.0, 0.03, 0.06]),
        ("btc_ret5", btc("return_5d"), "gt", [-0.08, -0.04, 0.0, 0.04, 0.08]),
        ("btc_vol20", btc("volatility_20d"), "lt", [0.025, 0.035, 0.045, 0.060, 0.080]),
        ("btc_dd60", btc("drawdown_60d"), "gt", [-0.35, -0.25, -0.18, -0.12, -0.08]),
        ("btc_rsi", btc("rsi_14"), "gt", [-0.5, -0.25, 0.0, 0.25]),
        ("breadth20", breadth("return_20d"), "gt", [0.25, 0.35, 0.45, 0.55, 0.65]),
        ("breadth5", breadth("return_5d"), "gt", [0.25, 0.35, 0.45, 0.55, 0.65]),
        ("mkt_ret20", median("return_20d"), "gt", [-0.10, -0.05, -0.02, 0.0, 0.03]),
        ("mkt_vol20", median("volatility_20d"), "lt", [0.030, 0.045, 0.060, 0.080]),
        ("mkt_dd60", median("drawdown_60d"), "gt", [-0.35, -0.25, -0.18, -0.12, -0.08]),
        ("iqr_ret5", iqr("return_5d"), "lt", [0.04, 0.07, 0.10, 0.15, 0.22]),
        ("iqr_ret20", iqr("return_20d"), "lt", [0.08, 0.15, 0.25, 0.40, 0.60]),
        ("skew_ret5", skew("return_5d"), "gt", [-1.0, -0.5, 0.0, 0.5, 1.0]),
        ("skew_ret20", skew("return_20d"), "gt", [-1.0, -0.5, 0.0, 0.5, 1.0]),
        ("mkt_absret5", median_abs("return_5d"), "lt", [0.03, 0.05, 0.08, 0.12, 0.18]),
        ("mkt_range1", median("range_pct_1d"), "lt", [0.03, 0.05, 0.08, 0.12]),
    ]
    for name, fn, op, thresholds in specs:
        single.extend(GateSpec(name=name, fn=fn, op=op, threshold=float(threshold)) for threshold in thresholds)

    pairs: list[tuple[GateSpec, GateSpec]] = []
    lead = [
        gate
        for gate in single
        if gate.name in {"btc_ret20", "breadth20", "btc_vol20", "btc_dd60", "iqr_ret5", "iqr_ret20"}
    ]
    aux = [
        gate
        for gate in single
        if gate.name in {"breadth5", "mkt_ret20", "mkt_vol20", "btc_rsi", "skew_ret5", "mkt_absret5", "mkt_range1"}
    ]
    for left in lead:
        for right in aux:
            if left.name != right.name:
                pairs.append((left, right))
    return [*single, *pairs]


def _gate_label(gate: GateSpec | tuple[GateSpec, GateSpec]) -> str:
    if isinstance(gate, tuple):
        return "__".join(part.label for part in gate)
    return gate.label


def _gate_allows(gate: GateSpec | tuple[GateSpec, GateSpec], data: MktdData, t: int) -> bool:
    if isinstance(gate, tuple):
        return all(part.allows(data, t) for part in gate)
    return gate.allows(data, t)


def _load_rule_from_csv(path: Path, config: str) -> LinearRule:
    with path.open() as fh:
        for row in csv.DictReader(fh):
            if row.get("config") != config:
                continue
            coeff_map = json.loads(str(row["coeffs_json"]))
            coeffs = np.asarray([float(coeff_map[name]) for name in FEATURE_NAMES], dtype=np.float64)
            return LinearRule(
                name=config,
                coeffs=coeffs,
                rebalance_days=int(float(row["rebalance_days"])),
                min_abs_score=float(row["min_abs_score"]),
                btc_min_return_20d=float(row["btc_min_return_20d"]),
            )
    raise ValueError(f"config {config!r} not found in {path}")


def _make_gated_policy(data: MktdData, rule: LinearRule, gate: GateSpec | tuple[GateSpec, GateSpec], *, decision_lag: int):
    base = _make_linear_policy(data, rule, decision_lag=int(decision_lag))
    step_state = {"step": 0}

    def policy(obs: np.ndarray) -> int:
        step = int(step_state["step"])
        step_state["step"] = step + 1
        feat_idx = max(0, min(step - 1, data.num_timesteps - 1))
        if not _gate_allows(gate, data, feat_idx):
            # Still advance the base policy state so lag/rebalance semantics stay aligned.
            _ = base(obs)
            return 0
        return int(base(obs))

    return policy


def _eval_gate(
    data: MktdData,
    rule: LinearRule,
    gate: GateSpec | tuple[GateSpec, GateSpec],
    *,
    eval_days: int,
    stride: int,
    max_leverage: float,
    slippage_bps: float,
    fee_rate: float,
    fill_buffer_bps: float,
    decision_lag: int,
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
        window = _slice_window(data, start, eval_days)
        result = simulate_daily_policy(
            window,
            _make_gated_policy(window, rule, gate, decision_lag=int(decision_lag)),
            max_steps=int(eval_days),
            fee_rate=float(fee_rate),
            slippage_bps=float(slippage_bps),
            fill_buffer_bps=float(fill_buffer_bps),
            max_leverage=float(max_leverage),
            periods_per_year=365.0,
            short_borrow_apr=0.08,
            enable_drawdown_profit_early_exit=False,
            enable_metric_threshold_early_exit=False,
        )
        curve = np.asarray(result.equity_curve if result.equity_curve is not None else [], dtype=np.float64)
        returns.append(float(result.total_return))
        maxdds.append(float(result.max_drawdown))
        sortinos.append(float(result.sortino))
        smooths.append(float(compute_pnl_smoothness_from_equity(curve)))
        ulcers.append(float(compute_ulcer_index(curve)))
        trades.append(int(result.num_trades))
    arr = np.asarray(returns, dtype=np.float64)
    monthly = np.asarray([_monthly_equivalent_return(float(ret), int(eval_days)) for ret in arr], dtype=np.float64)
    row: dict[str, float | int | str] = {
        "config": rule.name,
        "gate": _gate_label(gate),
        "max_leverage": float(max_leverage),
        "eval_days": int(eval_days),
        "slip_bps": float(slippage_bps),
        "fill_buffer_bps": float(fill_buffer_bps),
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
        "median_smooth": float(np.percentile(np.asarray(smooths, dtype=np.float64), 50)),
        "median_ulcer": float(np.percentile(np.asarray(ulcers, dtype=np.float64), 50)),
        "median_sortino": float(np.percentile(np.asarray(sortinos, dtype=np.float64), 50)),
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
    return row


def main() -> int:
    parser = argparse.ArgumentParser(description="Search regime gates around a Binance33 linear rule.")
    parser.add_argument("--rule-csv", type=Path, default=Path("analysis/binance33_linear_rule_search_20260430.csv"))
    parser.add_argument("--config", default="rand0045_rb1_thr0.5_btcmom20gt-0.05")
    parser.add_argument("--eval-data", type=Path, default=Path("pufferlib_market/data/binance33_daily_val.bin"))
    parser.add_argument("--train-data", type=Path, default=Path("pufferlib_market/data/binance33_daily_train.bin"))
    parser.add_argument("--out", type=Path, default=Path("analysis/binance33_regime_gate_search.csv"))
    parser.add_argument("--eval-days", type=int, default=120)
    parser.add_argument("--val-stride", type=int, default=1)
    parser.add_argument("--train-stride", type=int, default=20)
    parser.add_argument("--max-leverage", type=float, default=1.305)
    parser.add_argument("--slippage-bps", type=float, default=20.0)
    parser.add_argument("--fee-rate", type=float, default=0.001)
    parser.add_argument("--fill-buffer-bps", type=float, default=5.0)
    parser.add_argument("--decision-lag", type=int, default=2)
    parser.add_argument("--target-monthly-pct", type=float, default=30.0)
    parser.add_argument("--target-max-dd-pct", type=float, default=20.0)
    args = parser.parse_args()

    rule = _load_rule_from_csv(args.rule_csv, args.config)
    val_data = read_mktd(args.eval_data)
    train_data = read_mktd(args.train_data)
    gates = _gate_universe()
    fieldnames = [
        "scope",
        "config",
        "gate",
        "max_leverage",
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
    ]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, float | int | str]] = []
    with args.out.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for idx, gate in enumerate(gates, start=1):
            val = _eval_gate(
                val_data,
                rule,
                gate,
                eval_days=int(args.eval_days),
                stride=int(args.val_stride),
                max_leverage=float(args.max_leverage),
                slippage_bps=float(args.slippage_bps),
                fee_rate=float(args.fee_rate),
                fill_buffer_bps=float(args.fill_buffer_bps),
                decision_lag=int(args.decision_lag),
                target_monthly_pct=float(args.target_monthly_pct),
                target_max_dd_pct=float(args.target_max_dd_pct),
            )
            val["scope"] = "val"
            writer.writerow(val)
            rows.append(val)
            train = _eval_gate(
                train_data,
                rule,
                gate,
                eval_days=int(args.eval_days),
                stride=int(args.train_stride),
                max_leverage=float(args.max_leverage),
                slippage_bps=float(args.slippage_bps),
                fee_rate=float(args.fee_rate),
                fill_buffer_bps=float(args.fill_buffer_bps),
                decision_lag=int(args.decision_lag),
                target_monthly_pct=float(args.target_monthly_pct),
                target_max_dd_pct=float(args.target_max_dd_pct),
            )
            train["scope"] = "trainhistory"
            writer.writerow(train)
            rows.append(train)
            if idx % 50 == 0:
                print(f"evaluated {idx}/{len(gates)} gates")

    val_rows = [row for row in rows if row["scope"] == "val"]
    train_by_gate = {str(row["gate"]): row for row in rows if row["scope"] == "trainhistory"}
    best = sorted(
        val_rows,
        key=lambda row: (
            int(row["passes_target"]),
            min(float(row["median_monthly_pct"]), float(train_by_gate[str(row["gate"])]["median_monthly_pct"])),
            -int(train_by_gate[str(row["gate"])]["neg_windows"]),
        ),
        reverse=True,
    )[:20]
    print("\n=== Best regime gates ===")
    for row in best:
        train = train_by_gate[str(row["gate"])]
        print(
            f"val {row['median_monthly_pct']:+7.2f}%/mo dd={row['p90_dd_pct']:.2f}% "
            f"neg={row['neg_windows']}/{row['windows']} trades={row['median_trades']:.0f} pass={row['passes_target']} | "
            f"train {train['median_monthly_pct']:+7.2f}%/mo dd={train['p90_dd_pct']:.2f}% "
            f"neg={train['neg_windows']}/{train['windows']} trades={train['median_trades']:.0f} | {row['gate']}"
        )
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
