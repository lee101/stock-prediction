#!/usr/bin/env python3
"""Train/evaluate Binance33 XGBoost cross-sectional policies.

The policy trains on daily MKTD features, predicts forward returns per
symbol, then sends the ranked symbol into the same lag-2 daily simulator used
for PPO checkpoints. It is intentionally compact so we can quickly try many
tree variants before doing any leverage sweep.
"""
from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from pufferlib_market.hourly_replay import MktdData, P_CLOSE, read_mktd, simulate_daily_policy
from src.robust_trading_metrics import compute_pnl_smoothness_from_equity, compute_ulcer_index


FEATURES = {
    "return_1d": 0,
    "return_5d": 1,
    "return_20d": 2,
    "volatility_20d": 4,
}


@dataclass(frozen=True)
class Experiment:
    name: str
    horizon: int
    label: str
    mode: str
    rebalance_days: int
    max_depth: int
    eta: float
    subsample: float
    colsample: float
    min_child_weight: float
    reg_lambda: float
    min_abs_score: float
    btc_gate: float


def _slice_window(data: MktdData, start: int, steps: int) -> MktdData:
    end = int(start) + int(steps) + 1
    return MktdData(
        version=data.version,
        symbols=list(data.symbols),
        features=data.features[start:end].copy(),
        prices=data.prices[start:end].copy(),
        tradable=data.tradable[start:end].copy() if data.tradable is not None else None,
    )


def _candidate_starts(data: MktdData, steps: int, *, stride: int) -> list[int]:
    window_len = int(steps) + 1
    if window_len > data.num_timesteps:
        raise ValueError(f"data too short for {steps}d windows: {data.num_timesteps} timesteps")
    return list(range(0, data.num_timesteps - window_len + 1, max(1, int(stride))))


def _feature_matrix(data: MktdData, t: int) -> np.ndarray:
    raw = np.asarray(data.features[t], dtype=np.float32)
    close = np.asarray(data.prices[t, :, P_CLOSE], dtype=np.float32)
    valid = np.isfinite(raw).all(axis=1) & np.isfinite(close) & (close > 0.0)
    if data.tradable is not None:
        valid &= np.asarray(data.tradable[t], dtype=bool)
    raw_valid = raw.copy()
    raw = np.where(valid[:, None], raw, 0.0).astype(np.float32, copy=False)

    cs_parts = []
    for idx in range(raw.shape[1]):
        values = raw_valid[:, idx].astype(np.float64, copy=False)
        finite = valid & np.isfinite(values)
        if finite.sum() <= 1:
            z = np.zeros_like(values, dtype=np.float32)
            rank = np.zeros_like(values, dtype=np.float32)
        else:
            mean = float(values[finite].mean())
            std = float(values[finite].std())
            z = np.zeros_like(values, dtype=np.float32)
            z[finite] = ((values[finite] - mean) / max(std, 1e-6)).astype(np.float32)
            order = np.flatnonzero(finite)[np.argsort(values[finite])]
            rank = np.empty_like(values, dtype=np.float32)
            rank.fill(0.0)
            rank[order] = np.linspace(-1.0, 1.0, int(finite.sum()), dtype=np.float32)
        cs_parts.extend([z[:, None], rank[:, None]])

    sym_idx = np.linspace(-1.0, 1.0, raw.shape[0], dtype=np.float32)[:, None]
    valid_col = valid.astype(np.float32)[:, None]
    return np.concatenate([raw, *cs_parts, sym_idx, valid_col], axis=1).astype(np.float32)


def _build_dataset(data: MktdData, *, horizon: int, label: str) -> tuple[np.ndarray, np.ndarray]:
    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    close = np.asarray(data.prices[:, :, P_CLOSE], dtype=np.float64)
    for t in range(0, data.num_timesteps - int(horizon)):
        x_t = _feature_matrix(data, t)
        p0 = close[t]
        p1 = close[t + int(horizon)]
        valid = np.isfinite(p0) & np.isfinite(p1) & (p0 > 0.0) & (p1 > 0.0)
        if data.tradable is not None:
            valid &= np.asarray(data.tradable[t], dtype=bool)
            valid &= np.asarray(data.tradable[t + int(horizon)], dtype=bool)
        ret = np.zeros(data.num_symbols, dtype=np.float64)
        ret[valid] = p1[valid] / p0[valid] - 1.0
        if label == "voladj":
            vol = np.maximum(np.asarray(data.features[t, :, FEATURES["volatility_20d"]], dtype=np.float64), 0.01)
            target = ret / vol
        elif label == "rank":
            target = np.zeros_like(ret)
            if valid.sum() > 1:
                order = np.flatnonzero(valid)[np.argsort(ret[valid])]
                target[order] = np.linspace(-1.0, 1.0, int(valid.sum()), dtype=np.float64)
        else:
            target = ret
        xs.append(x_t[valid])
        ys.append(target[valid].astype(np.float32))
    if not xs:
        raise ValueError("No training rows built")
    return np.concatenate(xs, axis=0), np.concatenate(ys, axis=0)


def _train_xgb(x_train: np.ndarray, y_train: np.ndarray, exp: Experiment, *, rounds: int, device: str):
    import xgboost as xgb

    dtrain = xgb.DMatrix(x_train, label=y_train)
    params = {
        "objective": "reg:squarederror",
        "eval_metric": "rmse",
        "max_depth": int(exp.max_depth),
        "eta": float(exp.eta),
        "subsample": float(exp.subsample),
        "colsample_bytree": float(exp.colsample),
        "min_child_weight": float(exp.min_child_weight),
        "lambda": float(exp.reg_lambda),
        "tree_method": "hist",
        "device": str(device),
        "seed": 1337,
    }
    try:
        return xgb.train(params, dtrain, num_boost_round=int(rounds), verbose_eval=False)
    except Exception:
        if str(device) == "cpu":
            raise
        params["device"] = "cpu"
        return xgb.train(params, dtrain, num_boost_round=int(rounds), verbose_eval=False)


def _predict_scores(model, features: np.ndarray) -> np.ndarray:
    import xgboost as xgb

    scores = model.predict(xgb.DMatrix(features))
    return np.asarray(scores, dtype=np.float64)


def _precompute_scores(data: MktdData, model) -> np.ndarray:
    return np.vstack([
        _predict_scores(model, _feature_matrix(data, t))
        for t in range(data.num_timesteps)
    ]).astype(np.float64, copy=False)


def _make_policy(
    data: MktdData,
    model,
    exp: Experiment,
    *,
    decision_lag: int,
    scores_by_t: np.ndarray | None = None,
):
    symbols = [symbol.upper() for symbol in data.symbols]
    btc_idx = next((idx for idx, symbol in enumerate(symbols) if symbol.startswith("BTC")), 0)
    pending: list[int] = []
    state = {"step": 0, "last_action": 0}

    def raw_policy(_obs: np.ndarray) -> int:
        step = int(state["step"])
        state["step"] = step + 1
        if step > 0 and exp.rebalance_days > 1 and step % exp.rebalance_days != 0:
            return int(state["last_action"])
        feat_idx = max(0, min(step - 1, data.num_timesteps - 1))
        if scores_by_t is not None:
            scores = np.asarray(scores_by_t[feat_idx], dtype=np.float64)
        else:
            features = _feature_matrix(data, feat_idx)
            scores = _predict_scores(model, features)
        if data.tradable is not None:
            tradable_idx = max(0, min(step, data.num_timesteps - 1))
            scores = np.where(np.asarray(data.tradable[tradable_idx], dtype=bool), scores, np.nan)
        btc_mom = float(data.features[feat_idx, btc_idx, FEATURES["return_20d"]])
        if exp.btc_gate > -9.0 and (not np.isfinite(btc_mom) or btc_mom < exp.btc_gate):
            if exp.mode == "long_top":
                state["last_action"] = 0
                return 0

        if not np.isfinite(scores).any():
            state["last_action"] = 0
            return 0

        best_idx = int(np.nanargmax(scores))
        worst_idx = int(np.nanargmin(scores))
        best = float(scores[best_idx])
        worst = float(scores[worst_idx])
        action = 0
        if exp.mode == "long_top":
            if best >= exp.min_abs_score:
                action = 1 + best_idx
        elif exp.mode == "short_bottom":
            if -worst >= exp.min_abs_score:
                action = 1 + data.num_symbols + worst_idx
        elif exp.mode == "long_or_short":
            if best >= exp.min_abs_score and best >= -worst:
                action = 1 + best_idx
            elif -worst >= exp.min_abs_score:
                action = 1 + data.num_symbols + worst_idx
        elif exp.mode == "regime":
            if np.isfinite(btc_mom) and btc_mom >= max(exp.btc_gate, 0.0):
                if best >= exp.min_abs_score:
                    action = 1 + best_idx
            elif -worst >= exp.min_abs_score:
                action = 1 + data.num_symbols + worst_idx
        else:
            raise ValueError(f"unknown mode: {exp.mode}")
        state["last_action"] = int(action)
        return int(action)

    if decision_lag <= 0:
        return raw_policy

    def lagged_policy(obs: np.ndarray) -> int:
        pending.append(int(raw_policy(obs)))
        if len(pending) <= int(decision_lag):
            return 0
        return int(pending.pop(0))

    return lagged_policy


def _eval_model(
    data: MktdData,
    model,
    exp: Experiment,
    *,
    eval_days: int,
    slippage_bps: float,
    stride: int,
    max_leverage: float,
    fee_rate: float,
    fill_buffer_bps: float,
    decision_lag: int,
    scores_by_t: np.ndarray | None = None,
) -> dict[str, float | int | str]:
    returns: list[float] = []
    sortinos: list[float] = []
    maxdds: list[float] = []
    smooths: list[float] = []
    ulcers: list[float] = []
    trades: list[int] = []
    for start in _candidate_starts(data, eval_days, stride=stride):
        window = _slice_window(data, start, eval_days)
        window_scores = None
        if scores_by_t is not None:
            window_scores = np.asarray(scores_by_t[start : start + int(eval_days) + 1], dtype=np.float64)
        policy = _make_policy(window, model, exp, decision_lag=decision_lag, scores_by_t=window_scores)
        result = simulate_daily_policy(
            window,
            policy,
            max_steps=int(eval_days),
            fee_rate=float(fee_rate),
            slippage_bps=float(slippage_bps),
            fill_buffer_bps=float(fill_buffer_bps),
            max_leverage=float(max_leverage),
            periods_per_year=365.0,
            enable_drawdown_profit_early_exit=False,
        )
        curve = np.asarray(result.equity_curve if result.equity_curve is not None else [], dtype=np.float64)
        returns.append(float(result.total_return))
        sortinos.append(float(result.sortino))
        maxdds.append(float(result.max_drawdown))
        smooths.append(float(compute_pnl_smoothness_from_equity(curve)))
        ulcers.append(float(compute_ulcer_index(curve)))
        trades.append(int(result.num_trades))

    arr = np.asarray(returns, dtype=np.float64)
    return {
        "experiment": exp.name,
        "horizon": exp.horizon,
        "label": exp.label,
        "mode": exp.mode,
        "rebalance_days": exp.rebalance_days,
        "max_leverage": float(max_leverage),
        "eval_days": int(eval_days),
        "slip_bps": float(slippage_bps),
        "median_pct": float(100.0 * np.percentile(arr, 50)),
        "p10_pct": float(100.0 * np.percentile(arr, 10)),
        "p90_pct": float(100.0 * np.percentile(arr, 90)),
        "neg_windows": int(np.sum(arr < 0.0)),
        "windows": int(arr.size),
        "p90_dd_pct": float(100.0 * np.percentile(np.asarray(maxdds), 90)),
        "median_smooth": float(np.percentile(np.asarray(smooths), 50)),
        "median_ulcer": float(np.percentile(np.asarray(ulcers), 50)),
        "median_sortino": float(np.percentile(np.asarray(sortinos), 50)),
        "median_trades": float(np.percentile(np.asarray(trades), 50)),
    }


def _experiments() -> list[Experiment]:
    configs: list[Experiment] = []
    seeds = [
        (1, "raw", "long_top", 3, 3, 0.05, 0.9, 0.9, 10.0, 2.0, 0.0, -99.0),
        (1, "raw", "short_bottom", 3, 3, 0.05, 0.9, 0.9, 10.0, 2.0, 0.0, -99.0),
        (1, "rank", "long_or_short", 3, 4, 0.04, 0.8, 0.8, 20.0, 5.0, 0.0, -99.0),
        (3, "raw", "long_top", 3, 3, 0.05, 0.9, 0.9, 10.0, 2.0, 0.0, -99.0),
        (3, "raw", "short_bottom", 3, 3, 0.05, 0.9, 0.9, 10.0, 2.0, 0.0, -99.0),
        (3, "voladj", "long_or_short", 3, 3, 0.05, 0.8, 0.9, 10.0, 3.0, 0.0, -0.05),
        (5, "raw", "long_top", 7, 3, 0.04, 0.9, 0.8, 15.0, 3.0, 0.0, -99.0),
        (5, "raw", "short_bottom", 7, 3, 0.04, 0.9, 0.8, 15.0, 3.0, 0.0, -99.0),
        (5, "rank", "regime", 7, 4, 0.03, 0.8, 0.8, 20.0, 5.0, 0.0, 0.0),
        (10, "raw", "long_top", 7, 3, 0.04, 0.9, 0.9, 15.0, 3.0, 0.0, -99.0),
        (10, "raw", "short_bottom", 7, 3, 0.04, 0.9, 0.9, 15.0, 3.0, 0.0, -99.0),
        (10, "voladj", "regime", 7, 4, 0.03, 0.8, 0.8, 25.0, 5.0, 0.0, -0.05),
        (20, "raw", "long_top", 14, 3, 0.03, 0.9, 0.9, 20.0, 5.0, 0.0, -99.0),
        (20, "raw", "short_bottom", 14, 3, 0.03, 0.9, 0.9, 20.0, 5.0, 0.0, -99.0),
        (20, "rank", "regime", 14, 4, 0.025, 0.8, 0.8, 30.0, 8.0, 0.0, 0.0),
        (5, "raw", "long_top", 1, 2, 0.08, 1.0, 1.0, 5.0, 1.0, 0.0, 0.0),
        (5, "raw", "short_bottom", 1, 2, 0.08, 1.0, 1.0, 5.0, 1.0, 0.0, -0.05),
        (10, "rank", "long_or_short", 1, 2, 0.08, 1.0, 1.0, 5.0, 1.0, 0.0, -99.0),
        (3, "rank", "regime", 1, 2, 0.08, 1.0, 1.0, 5.0, 1.0, 0.0, 0.0),
        (1, "voladj", "long_or_short", 1, 2, 0.08, 1.0, 1.0, 5.0, 1.0, 0.0, -99.0),
    ]
    for idx, cfg in enumerate(seeds, start=1):
        configs.append(Experiment(f"xgb{idx:02d}", *cfg))
    return configs


def _parse_float_list(value: str) -> list[float]:
    parsed = [float(part.strip()) for part in str(value).split(",") if part.strip()]
    if not parsed:
        raise ValueError(f"expected at least one float in {value!r}")
    return parsed


def main() -> int:
    parser = argparse.ArgumentParser(description="Sweep Binance33 XGBoost cross-sectional policies.")
    parser.add_argument("--train-data", type=Path, default=Path("pufferlib_market/data/binance33_daily_train.bin"))
    parser.add_argument("--eval-data", type=Path, default=Path("pufferlib_market/data/binance33_daily_val.bin"))
    parser.add_argument("--out", type=Path, default=Path("analysis/binance33_xgb_sweep.csv"))
    parser.add_argument("--rounds", type=int, default=160)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-experiments", type=int, default=0)
    parser.add_argument("--eval-days", default="30,120")
    parser.add_argument("--slippage-bps", default="20")
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--max-leverage", default="1.0", help="Single leverage or comma list, e.g. 1,1.5,2")
    parser.add_argument("--experiment-names", default="", help="Comma list of experiment names to run, e.g. xgb08,xgb14")
    parser.add_argument("--fee-rate", type=float, default=0.001)
    parser.add_argument("--fill-buffer-bps", type=float, default=5.0)
    parser.add_argument("--decision-lag", type=int, default=2)
    args = parser.parse_args()

    train_data = read_mktd(args.train_data)
    eval_data = read_mktd(args.eval_data)
    eval_days = [int(part.strip()) for part in str(args.eval_days).split(",") if part.strip()]
    slippages = [float(part.strip()) for part in str(args.slippage_bps).split(",") if part.strip()]
    leverages = _parse_float_list(str(args.max_leverage))
    experiments = _experiments()
    if args.experiment_names:
        requested = [part.strip() for part in str(args.experiment_names).split(",") if part.strip()]
        by_name = {exp.name: exp for exp in experiments}
        missing = sorted(set(requested) - set(by_name))
        if missing:
            raise ValueError(f"unknown experiment names: {', '.join(missing)}")
        experiments = [by_name[name] for name in requested]
    elif args.max_experiments > 0:
        experiments = experiments[: int(args.max_experiments)]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "experiment",
        "horizon",
        "label",
        "mode",
        "rebalance_days",
        "max_leverage",
        "eval_days",
        "slip_bps",
        "median_pct",
        "p10_pct",
        "p90_pct",
        "neg_windows",
        "windows",
        "p90_dd_pct",
        "median_smooth",
        "median_ulcer",
        "median_sortino",
        "median_trades",
    ]
    rows: list[dict[str, float | int | str]] = []
    with args.out.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for idx, exp in enumerate(experiments, start=1):
            x_train, y_train = _build_dataset(train_data, horizon=exp.horizon, label=exp.label)
            model = _train_xgb(x_train, y_train, exp, rounds=int(args.rounds), device=str(args.device))
            scores_by_t = _precompute_scores(eval_data, model)
            for leverage in leverages:
                for days in eval_days:
                    for slip in slippages:
                        row = _eval_model(
                            eval_data,
                            model,
                            exp,
                            eval_days=int(days),
                            slippage_bps=float(slip),
                            stride=int(args.stride),
                            max_leverage=float(leverage),
                            fee_rate=float(args.fee_rate),
                            fill_buffer_bps=float(args.fill_buffer_bps),
                            decision_lag=int(args.decision_lag),
                            scores_by_t=scores_by_t,
                        )
                        rows.append(row)
                        writer.writerow(row)
                        fh.flush()
            print(f"evaluated {idx}/{len(experiments)} {exp.name}", flush=True)

    for days in eval_days:
        subset = [
            row for row in rows
            if int(row["eval_days"]) == int(days) and float(row["slip_bps"]) == max(slippages)
        ]
        subset.sort(key=lambda row: float(row["median_pct"]), reverse=True)
        print(f"\n=== Best XGB {days}d slip{max(slippages):g} ===")
        for row in subset[:10]:
            print(
                f"{row['median_pct']:+7.2f}% p10={row['p10_pct']:+7.2f}% "
                f"neg={row['neg_windows']}/{row['windows']} dd90={row['p90_dd_pct']:.2f}% "
                f"{row['experiment']} lev={row['max_leverage']} h={row['horizon']} "
                f"{row['label']} {row['mode']} rb={row['rebalance_days']}"
            )
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
