#!/usr/bin/env python3
"""Fast deterministic Binance33 rule sweep on MKTD holdout windows.

This is a sanity baseline for the PPO sweep: if simple lagged feature rules also
fail on 120-day unseen windows, the issue is likely signal/regime rather than
just PPO optimization.
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

from pufferlib_market.hourly_replay import MktdData, read_mktd, simulate_daily_policy
from src.robust_trading_metrics import compute_pnl_smoothness_from_equity, compute_ulcer_index


FEATURES = {
    "return_1d": 0,
    "return_5d": 1,
    "return_20d": 2,
    "volatility_5d": 3,
    "volatility_20d": 4,
    "ma_delta_5d": 5,
    "ma_delta_20d": 6,
    "ma_delta_60d": 7,
    "atr_pct_14d": 8,
    "range_pct_1d": 9,
    "rsi_14": 10,
    "trend_60d": 11,
    "drawdown_20d": 12,
    "drawdown_60d": 13,
    "log_volume_z20d": 14,
    "log_volume_delta_5d": 15,
}


@dataclass(frozen=True)
class ScoreSpec:
    name: str
    feature: str
    sign: float = 1.0
    vol_adjust: bool = False
    volume_bonus: float = 0.0


@dataclass(frozen=True)
class RuleConfig:
    score: ScoreSpec
    mode: str
    rebalance_days: int
    min_abs_score: float
    btc_min_return_20d: float

    @property
    def name(self) -> str:
        gate = "nogate" if self.btc_min_return_20d <= -9.0 else f"btcmom20gt{self.btc_min_return_20d:g}"
        return (
            f"{self.score.name}_{self.mode}_rb{self.rebalance_days}"
            f"_thr{self.min_abs_score:g}_{gate}"
        )


def _slice_window(data: MktdData, start: int, steps: int) -> MktdData:
    end = int(start) + int(steps) + 1
    return MktdData(
        version=data.version,
        symbols=list(data.symbols),
        features=data.features[start:end].copy(),
        prices=data.prices[start:end].copy(),
        tradable=data.tradable[start:end].copy() if data.tradable is not None else None,
    )


def _candidate_starts(data: MktdData, steps: int, *, stride: int = 1) -> list[int]:
    window_len = int(steps) + 1
    if window_len > data.num_timesteps:
        raise ValueError(f"data too short for {steps}d windows: {data.num_timesteps} timesteps")
    return list(range(0, data.num_timesteps - window_len + 1, max(1, int(stride))))


def _score_vector(features: np.ndarray, spec: ScoreSpec) -> np.ndarray:
    values = np.asarray(features[:, FEATURES[spec.feature]], dtype=np.float64) * float(spec.sign)
    if spec.vol_adjust:
        vol = np.asarray(features[:, FEATURES["volatility_20d"]], dtype=np.float64)
        values = values / np.maximum(vol, 0.01)
    if spec.volume_bonus:
        volume = np.asarray(features[:, FEATURES["log_volume_z20d"]], dtype=np.float64)
        values = values + float(spec.volume_bonus) * volume
    values[~np.isfinite(values)] = -np.inf
    return values


def _make_policy(
    data: MktdData,
    config: RuleConfig,
    *,
    decision_lag: int,
) -> object:
    symbols = [s.upper() for s in data.symbols]
    btc_idx = next((idx for idx, sym in enumerate(symbols) if sym.startswith("BTC")), 0)
    num_symbols = int(data.num_symbols)
    pending: list[int] = []
    state = {"step": 0, "last_action": 0}

    def raw_policy(_obs: np.ndarray) -> int:
        step = int(state["step"])
        state["step"] = step + 1
        if step > 0 and int(config.rebalance_days) > 1 and step % int(config.rebalance_days) != 0:
            return int(state["last_action"])

        # Mirror simulator observation lag: features at t-1, never current-bar future features.
        feat_idx = max(0, min(step - 1, data.num_timesteps - 1))
        feat = data.features[feat_idx]
        btc_mom = float(feat[btc_idx, FEATURES["return_20d"]])
        if config.mode != "regime_long_top_short_bottom" and config.btc_min_return_20d > -9.0:
            if not np.isfinite(btc_mom) or btc_mom < float(config.btc_min_return_20d):
                state["last_action"] = 0
                return 0

        scores = _score_vector(feat, config.score)
        if data.tradable is not None:
            tradable_idx = max(0, min(step, data.num_timesteps - 1))
            tradable = np.asarray(data.tradable[tradable_idx], dtype=bool)
            scores = np.where(tradable, scores, -np.inf)
        if not np.isfinite(scores).any():
            state["last_action"] = 0
            return 0

        best_idx = int(np.nanargmax(scores))
        best = float(scores[best_idx])
        worst_idx = int(np.nanargmin(scores))
        worst = float(scores[worst_idx])
        threshold = float(config.min_abs_score)

        action = 0
        if config.mode == "long_top":
            if best >= threshold:
                action = 1 + best_idx
        elif config.mode == "short_bottom":
            if -worst >= threshold:
                action = 1 + num_symbols + worst_idx
        elif config.mode == "long_or_short":
            if best >= threshold and best >= -worst:
                action = 1 + best_idx
            elif -worst >= threshold:
                action = 1 + num_symbols + worst_idx
        elif config.mode == "regime_long_top_short_bottom":
            cutoff = 0.0 if config.btc_min_return_20d <= -9.0 else float(config.btc_min_return_20d)
            if np.isfinite(btc_mom) and btc_mom >= cutoff:
                if best >= threshold:
                    action = 1 + best_idx
            elif -worst >= threshold:
                action = 1 + num_symbols + worst_idx
        else:
            raise ValueError(f"unknown mode: {config.mode}")
        state["last_action"] = int(action)
        return int(action)

    if decision_lag <= 0:
        return raw_policy

    def lagged_policy(obs: np.ndarray) -> int:
        action_now = int(raw_policy(obs))
        pending.append(action_now)
        if len(pending) <= int(decision_lag):
            return 0
        return int(pending.pop(0))

    return lagged_policy


def _eval_config(
    data: MktdData,
    config: RuleConfig,
    *,
    eval_days: int,
    slippage_bps: float,
    fee_rate: float,
    fill_buffer_bps: float,
    decision_lag: int,
    max_leverage: float,
    periods_per_year: float,
    stride: int,
) -> dict[str, float | int | str]:
    returns: list[float] = []
    sortinos: list[float] = []
    maxdds: list[float] = []
    smooths: list[float] = []
    ulcers: list[float] = []
    trades: list[int] = []
    for start in _candidate_starts(data, eval_days, stride=int(stride)):
        window = _slice_window(data, start, eval_days)
        policy = _make_policy(window, config, decision_lag=decision_lag)
        result = simulate_daily_policy(
            window,
            policy,
            max_steps=int(eval_days),
            fee_rate=float(fee_rate),
            slippage_bps=float(slippage_bps),
            fill_buffer_bps=float(fill_buffer_bps),
            max_leverage=float(max_leverage),
            periods_per_year=float(periods_per_year),
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
        "config": config.name,
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


def _default_configs() -> list[RuleConfig]:
    specs = [
        ScoreSpec("mom5", "return_5d"),
        ScoreSpec("mom20", "return_20d"),
        ScoreSpec("mom20_voladj", "return_20d", vol_adjust=True),
        ScoreSpec("trend60", "trend_60d"),
        ScoreSpec("ma20", "ma_delta_20d"),
        ScoreSpec("rsi", "rsi_14"),
        ScoreSpec("dip20", "drawdown_20d", sign=-1.0),
        ScoreSpec("dip60", "drawdown_60d", sign=-1.0),
        ScoreSpec("reversal5", "return_5d", sign=-1.0),
        ScoreSpec("vol_mom5", "return_5d", volume_bonus=0.03),
    ]
    configs: list[RuleConfig] = []
    for spec in specs:
        for mode in ("long_top", "short_bottom", "long_or_short", "regime_long_top_short_bottom"):
            for rebalance in (1, 3, 7, 14):
                for threshold in (0.0, 0.02, 0.05):
                    for btc_gate in (-99.0, -0.05, 0.0):
                        configs.append(
                            RuleConfig(
                                score=spec,
                                mode=mode,
                                rebalance_days=rebalance,
                                min_abs_score=threshold,
                                btc_min_return_20d=btc_gate,
                            )
                        )
    return configs


def main() -> int:
    parser = argparse.ArgumentParser(description="Sweep deterministic Binance33 feature rules.")
    parser.add_argument("--data-path", type=Path, default=Path("pufferlib_market/data/binance33_daily_val.bin"))
    parser.add_argument("--out", type=Path, default=Path("analysis/binance33_rule_sweep.csv"))
    parser.add_argument("--eval-days", default="30,120", help="Comma-separated holdout window lengths")
    parser.add_argument("--slippage-bps", default="5,20", help="Comma-separated slippage cells")
    parser.add_argument("--fee-rate", type=float, default=0.001)
    parser.add_argument("--fill-buffer-bps", type=float, default=5.0)
    parser.add_argument("--decision-lag", type=int, default=2)
    parser.add_argument("--max-leverage", type=float, default=1.0)
    parser.add_argument("--periods-per-year", type=float, default=365.0)
    parser.add_argument("--stride", type=int, default=1, help="Window start stride; 1 is exhaustive")
    parser.add_argument("--max-configs", type=int, default=0, help="Debug cap; 0 means all configs")
    parser.add_argument(
        "--config-contains",
        default="",
        help="Comma-separated substrings; keep configs whose generated name contains any substring",
    )
    args = parser.parse_args()

    data = read_mktd(args.data_path)
    eval_days = [int(part.strip()) for part in str(args.eval_days).split(",") if part.strip()]
    slippages = [float(part.strip()) for part in str(args.slippage_bps).split(",") if part.strip()]
    configs = _default_configs()
    filters = [part.strip() for part in str(args.config_contains).split(",") if part.strip()]
    if filters:
        configs = [config for config in configs if any(token in config.name for token in filters)]
    if args.max_configs > 0:
        configs = configs[: int(args.max_configs)]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "config",
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
        for idx, config in enumerate(configs, start=1):
            for days in eval_days:
                for slip in slippages:
                    row = _eval_config(
                        data,
                        config,
                        eval_days=int(days),
                        slippage_bps=float(slip),
                        fee_rate=float(args.fee_rate),
                        fill_buffer_bps=float(args.fill_buffer_bps),
                        decision_lag=int(args.decision_lag),
                        max_leverage=float(args.max_leverage),
                        periods_per_year=float(args.periods_per_year),
                        stride=int(args.stride),
                    )
                    rows.append(row)
                    writer.writerow(row)
            if idx % 25 == 0:
                print(f"evaluated {idx}/{len(configs)} configs")

    best_120 = sorted(
        (row for row in rows if int(row["eval_days"]) == 120 and float(row["slip_bps"]) == 20.0),
        key=lambda row: float(row["median_pct"]),
        reverse=True,
    )[:10]
    best_30 = sorted(
        (row for row in rows if int(row["eval_days"]) == 30 and float(row["slip_bps"]) == 20.0),
        key=lambda row: float(row["median_pct"]),
        reverse=True,
    )[:10]

    print("\n=== Best 120d slip20 ===")
    for row in best_120:
        print(
            f"{row['median_pct']:+7.2f}% p10={row['p10_pct']:+7.2f}% "
            f"neg={row['neg_windows']}/{row['windows']} dd90={row['p90_dd_pct']:.2f}% {row['config']}"
        )
    print("\n=== Best 30d slip20 ===")
    for row in best_30:
        print(
            f"{row['median_pct']:+7.2f}% p10={row['p10_pct']:+7.2f}% "
            f"neg={row['neg_windows']}/{row['windows']} dd90={row['p90_dd_pct']:.2f}% {row['config']}"
        )
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
