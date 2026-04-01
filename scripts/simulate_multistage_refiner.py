#!/usr/bin/env python3
"""Synthetic market-sim harness for the multi-stage daily refiner."""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING, Iterable

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from llm_hourly_trader.gemini_wrapper import TradePlan
from src.daily_mixed_hybrid import (
    RLSignal,
    build_candidate_plan,
    normalize_trade_plan,
    refine_trade_plan_multistage,
    snapshot_plan,
)

if TYPE_CHECKING:
    from binanceneural.marketsimulator import SimulationConfig


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Simulate the multi-stage daily refiner on synthetic data")
    parser.add_argument("--symbols", nargs="+", default=["AAPL", "MSFT", "NVDA"])
    parser.add_argument("--bars-per-symbol", type=int, default=240)
    parser.add_argument("--warmup-bars", type=int, default=48)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output-dir", default="analysis/multistage_refiner_sim")
    parser.add_argument("--wandb-project", default="stock")
    parser.add_argument("--wandb-entity", default=None)
    parser.add_argument("--wandb-mode", default="offline")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--no-wandb", action="store_true")
    parser.add_argument("--r2-dest", default=None, help="Optional rclone destination, e.g. r2:model/stock/multistage_refiner/")
    return parser.parse_args(argv)


def _synthetic_bars(symbol: str, *, bars: int, seed: int, phase: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed + phase * 97)
    timestamps = pd.date_range("2026-01-01T00:00:00Z", periods=bars, freq="h", tz="UTC")
    price = 100.0 + phase * 35.0
    rows: list[dict[str, float | str]] = []
    for idx, ts in enumerate(timestamps):
        regime = ((idx + phase * 11) // 72) % 3
        drift = (0.0012, -0.0010, 0.0007)[regime]
        seasonal = 0.0008 * math.sin((idx + phase * 5) / 8.0)
        noise = float(rng.normal(0.0, 0.0035))
        ret = drift + seasonal + noise
        next_close = max(5.0, price * (1.0 + ret))
        intrabar = abs(float(rng.normal(0.0045, 0.0015)))
        high = max(price, next_close) * (1.0 + intrabar)
        low = min(price, next_close) * max(0.8, 1.0 - intrabar)
        rows.append(
            {
                "timestamp": ts,
                "symbol": symbol,
                "open": price,
                "high": high,
                "low": low,
                "close": next_close,
                "volume": float(1_000_000 + phase * 50_000 + idx * 1_000),
            }
        )
        price = next_close
    return pd.DataFrame(rows)


def _build_synthetic_forecast(frame: pd.DataFrame, idx: int, horizon: int, *, noise_scale: float, rng: np.random.Generator) -> dict[str, float]:
    current = float(frame.iloc[idx]["close"])
    future_idx = min(len(frame) - 1, idx + horizon)
    future_close = float(frame.iloc[future_idx]["close"])
    noisy_close = max(0.01, future_close * (1.0 + float(rng.normal(0.0, noise_scale))))
    high = max(current, noisy_close) * (1.0 + 0.003)
    low = min(current, noisy_close) * (1.0 - 0.003)
    spread = abs(noisy_close - current)
    return {
        "predicted_close_p50": noisy_close,
        "predicted_close_p10": max(0.01, noisy_close - 0.75 * spread),
        "predicted_close_p90": noisy_close + 0.75 * spread,
        "predicted_high_p50": high,
        "predicted_low_p50": low,
    }


def _build_rl_signal(frame: pd.DataFrame, idx: int, symbol_idx: int, symbol: str) -> RLSignal:
    current = float(frame.iloc[idx]["close"])
    lookback = max(0, idx - 6)
    past = float(frame.iloc[lookback]["close"])
    momentum = (current / max(past, 1e-6)) - 1.0
    direction = "long" if momentum > -0.0005 else "hold"
    confidence = float(min(0.95, max(0.15, 0.30 + abs(momentum) * 12.0)))
    allocation = float(min(1.5, max(0.25, 0.55 + abs(momentum) * 18.0)))
    return RLSignal(
        symbol_idx=symbol_idx,
        symbol_name=symbol,
        direction=direction,
        confidence=confidence,
        logit_gap=float(abs(momentum) * 150.0),
        allocation_pct=allocation,
        level_offset_bps=0.0,
    )


def _baseline_plan_from_rl(signal: RLSignal, *, current_price: float) -> TradePlan:
    if signal.direction != "long":
        return TradePlan(
            direction="hold",
            buy_price=0.0,
            sell_price=0.0,
            confidence=float(signal.confidence * 0.75),
            reasoning="baseline_hold",
            allocation_pct=0.0,
        )
    allocation_pct = float(min(100.0, max(20.0, signal.allocation_pct * 40.0)))
    return TradePlan(
        direction="long",
        buy_price=float(current_price * 0.9985),
        sell_price=float(current_price * 1.0075),
        confidence=float(signal.confidence),
        reasoning="baseline_rl_long",
        allocation_pct=allocation_pct,
    )


def _make_llm_refiner(
    *,
    current_price: float,
    forecast_1h: dict[str, float] | None,
    forecast_24h: dict[str, float] | None,
    base_plan: TradePlan,
) -> callable:
    def _refine(_prompt: str) -> TradePlan:
        forecast_targets = [
            float(forecast["predicted_close_p50"])
            for forecast in (forecast_1h, forecast_24h)
            if forecast and float(forecast.get("predicted_close_p50", 0.0) or 0.0) > 0.0
        ]
        if not forecast_targets:
            return base_plan
        weighted_target = float(sum(forecast_targets) / len(forecast_targets))
        edge = weighted_target / max(current_price, 1e-6) - 1.0
        if edge <= 0.0005:
            return TradePlan(
                direction="hold",
                buy_price=0.0,
                sell_price=max(0.0, current_price * 1.002),
                confidence=max(0.1, float(base_plan.confidence) * 0.8),
                reasoning="synthetic_llm_hold",
                allocation_pct=0.0,
            )
        return TradePlan(
            direction="long",
            buy_price=float(current_price * (1.0 - min(0.006, max(0.001, 0.35 * edge)))),
            sell_price=float(max(current_price * (1.0 + min(0.02, max(0.004, 0.90 * edge))), base_plan.sell_price)),
            confidence=float(min(0.99, base_plan.confidence + min(0.20, edge * 8.0))),
            reasoning="synthetic_llm_refine",
            allocation_pct=float(min(100.0, max(base_plan.allocation_pct, 25.0 + edge * 7000.0))),
        )

    return _refine


def _plan_to_action_row(ts: pd.Timestamp, symbol: str, plan: TradePlan) -> dict[str, float | str | pd.Timestamp]:
    buy_amount = float(max(0.0, min(100.0, float(getattr(plan, "allocation_pct", 0.0) or 0.0))))
    sell_amount = 100.0 if float(plan.sell_price) > 0.0 else 0.0
    if plan.direction != "long":
        buy_amount = 0.0
    return {
        "timestamp": ts,
        "symbol": symbol,
        "buy_price": float(plan.buy_price),
        "sell_price": float(plan.sell_price),
        "buy_amount": buy_amount,
        "sell_amount": sell_amount,
        "trade_amount": max(buy_amount, sell_amount),
        "confidence": float(plan.confidence),
        "plan_direction": str(plan.direction),
        "reasoning": str(plan.reasoning),
    }


def _trade_summary(result) -> dict[str, float]:
    trades = [trade for symbol_result in result.per_symbol.values() for trade in symbol_result.trades]
    realized = [float(trade.realized_pnl) for trade in trades if trade.side == "sell"]
    wins = [pnl for pnl in realized if pnl > 0.0]
    return {
        "trade_count": float(len(trades)),
        "round_trips": float(len(realized)),
        "win_rate": float(len(wins) / len(realized)) if realized else 0.0,
        "realized_pnl": float(sum(realized)),
    }


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _build_actions(
    *,
    bars: pd.DataFrame,
    warmup_bars: int,
    mode: str,
    seed: int,
) -> tuple[pd.DataFrame, list[dict[str, object]]]:
    action_rows: list[dict[str, object]] = []
    trace_rows: list[dict[str, object]] = []
    previous_plans: dict[str, dict[str, object]] = {}
    grouped = {symbol: frame.reset_index(drop=True) for symbol, frame in bars.groupby("symbol", sort=False)}

    for symbol_idx, symbol in enumerate(grouped):
        frame = grouped[symbol]
        rng = np.random.default_rng(seed + symbol_idx * 313)
        for idx in range(warmup_bars, len(frame) - 24):
            current_price = float(frame.iloc[idx]["close"])
            history = frame.iloc[max(0, idx - warmup_bars):idx]
            history_rows = history.loc[:, ["timestamp", "open", "high", "low", "close", "volume"]].copy()
            history_rows["timestamp"] = history_rows["timestamp"].astype(str)
            history_payload = history_rows.to_dict("records")
            forecast_1h = _build_synthetic_forecast(frame, idx, 1, noise_scale=0.0025, rng=rng)
            forecast_24h = _build_synthetic_forecast(frame, idx, 24, noise_scale=0.0045, rng=rng)
            rl_signal = _build_rl_signal(frame, idx, symbol_idx, symbol)
            baseline_plan = normalize_trade_plan(
                _baseline_plan_from_rl(rl_signal, current_price=current_price),
                current_price=current_price,
                asset_class="stock",
            )
            baseline_candidate = build_candidate_plan(
                symbol=symbol,
                asset_class="stock",
                current_price=current_price,
                rl_signal=rl_signal,
                plan=baseline_plan,
                current_position=None,
                equity=10_000.0,
                forecast_1h=forecast_1h,
                forecast_24h=forecast_24h,
                previous_forecast_error=None,
            )
            llm_refiner = _make_llm_refiner(
                current_price=current_price,
                forecast_1h=forecast_1h,
                forecast_24h=forecast_24h,
                base_plan=baseline_plan,
            )
            refined_plan, trace = refine_trade_plan_multistage(
                baseline_plan,
                symbol=symbol,
                asset_class="stock",
                current_price=current_price,
                rl_signal=rl_signal,
                history_rows=history_payload,
                previous_plan=previous_plans.get(symbol),
                forecast_1h=forecast_1h,
                forecast_24h=forecast_24h,
                target_allocation=baseline_candidate.target_allocation,
                llm_refiner=llm_refiner,
                return_trace=True,
            )
            final_plan = refined_plan if mode == "refined" else baseline_plan
            timestamp = pd.Timestamp(frame.iloc[idx]["timestamp"])
            action_rows.append(_plan_to_action_row(timestamp, symbol, final_plan))
            previous_plans[symbol] = snapshot_plan(
                timestamp=timestamp,
                symbol=symbol,
                rl_signal=rl_signal,
                plan=final_plan,
                current_price=current_price,
                forecast_1h=forecast_1h,
                forecast_24h=forecast_24h,
                score=baseline_candidate.score,
                target_allocation=baseline_candidate.target_allocation,
                overnight_allocation=baseline_candidate.overnight_allocation,
            )
            if mode == "refined":
                for step_idx, entry in enumerate(trace):
                    trace_rows.append(
                        {
                            "timestamp": str(timestamp),
                            "symbol": symbol,
                            "step_idx": step_idx,
                            "stage": entry.stage,
                            "driver": entry.driver,
                            "energy_budget": entry.energy_budget,
                            "alpha": entry.alpha,
                            "target_distance": entry.target_distance,
                            "plan": asdict(entry.plan),
                            "prompt": entry.prompt,
                        }
                    )

    return pd.DataFrame(action_rows), trace_rows


def _sync_to_r2(out_dir: Path, r2_dest: str) -> None:
    subprocess.run(
        ["rclone", "copy", str(out_dir), r2_dest, "--progress"],
        check=True,
    )


def main(argv: list[str] | None = None) -> None:
    from binanceneural.marketsimulator import SimulationConfig, run_shared_cash_simulation
    from wandboard import WandBoardLogger

    args = _parse_args(argv)
    out_dir = Path(args.output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    run_name = args.run_name or f"multistage_refiner_seed{args.seed}"

    bars = pd.concat(
        [
            _synthetic_bars(symbol, bars=args.bars_per_symbol, seed=args.seed, phase=idx)
            for idx, symbol in enumerate(args.symbols)
        ],
        ignore_index=True,
    )
    baseline_actions, _ = _build_actions(
        bars=bars,
        warmup_bars=args.warmup_bars,
        mode="baseline",
        seed=args.seed,
    )
    refined_actions, trace_rows = _build_actions(
        bars=bars,
        warmup_bars=args.warmup_bars,
        mode="refined",
        seed=args.seed,
    )

    sim_config = SimulationConfig(
        maker_fee=0.0008,
        initial_cash=10_000.0,
        decision_lag_bars=1,
        max_hold_hours=24,
        fill_buffer_bps=3.0,
    )
    baseline_result = run_shared_cash_simulation(bars, baseline_actions, config=sim_config)
    refined_result = run_shared_cash_simulation(bars, refined_actions, config=sim_config)

    baseline_summary = {**baseline_result.metrics, **_trade_summary(baseline_result)}
    refined_summary = {**refined_result.metrics, **_trade_summary(refined_result)}
    comparison = {
        "delta_total_return": refined_summary["total_return"] - baseline_summary["total_return"],
        "delta_sortino": refined_summary["sortino"] - baseline_summary["sortino"],
        "delta_win_rate": refined_summary["win_rate"] - baseline_summary["win_rate"],
        "delta_realized_pnl": refined_summary["realized_pnl"] - baseline_summary["realized_pnl"],
        "delta_round_trips": refined_summary["round_trips"] - baseline_summary["round_trips"],
    }
    summary = {
        "seed": args.seed,
        "symbols": list(args.symbols),
        "bars_per_symbol": int(args.bars_per_symbol),
        "warmup_bars": int(args.warmup_bars),
        "baseline": baseline_summary,
        "refined": refined_summary,
        "comparison": comparison,
    }

    bars.to_csv(out_dir / "synthetic_bars.csv", index=False)
    baseline_actions.to_csv(out_dir / "baseline_actions.csv", index=False)
    refined_actions.to_csv(out_dir / "refined_actions.csv", index=False)
    pd.DataFrame(trace_rows).to_json(out_dir / "refined_trace.jsonl", orient="records", lines=True)
    baseline_result.combined_equity.rename("equity").to_csv(out_dir / "baseline_equity.csv")
    refined_result.combined_equity.rename("equity").to_csv(out_dir / "refined_equity.csv")
    _write_json(out_dir / "summary.json", summary)

    with WandBoardLogger(
        run_name=run_name,
        project=args.wandb_project,
        entity=args.wandb_entity,
        mode=args.wandb_mode,
        enable_wandb=not args.no_wandb,
        log_dir=out_dir / "tensorboard",
        tensorboard_subdir="tb",
        config={
            "seed": args.seed,
            "symbols": list(args.symbols),
            "bars_per_symbol": args.bars_per_symbol,
            "warmup_bars": args.warmup_bars,
        },
    ) as logger:
        logger.log(
            {
                "baseline/total_return": baseline_summary["total_return"],
                "baseline/sortino": baseline_summary["sortino"],
                "baseline/win_rate": baseline_summary["win_rate"],
                "baseline/realized_pnl": baseline_summary["realized_pnl"],
                "refined/total_return": refined_summary["total_return"],
                "refined/sortino": refined_summary["sortino"],
                "refined/win_rate": refined_summary["win_rate"],
                "refined/realized_pnl": refined_summary["realized_pnl"],
                "comparison/delta_total_return": comparison["delta_total_return"],
                "comparison/delta_sortino": comparison["delta_sortino"],
                "comparison/delta_win_rate": comparison["delta_win_rate"],
                "comparison/delta_realized_pnl": comparison["delta_realized_pnl"],
            },
            step=0,
        )
        logger.log_table(
            "summary_rows",
            columns=["mode", "total_return", "sortino", "win_rate", "realized_pnl", "round_trips", "trade_count"],
            data=[
                ["baseline", baseline_summary["total_return"], baseline_summary["sortino"], baseline_summary["win_rate"], baseline_summary["realized_pnl"], baseline_summary["round_trips"], baseline_summary["trade_count"]],
                ["refined", refined_summary["total_return"], refined_summary["sortino"], refined_summary["win_rate"], refined_summary["realized_pnl"], refined_summary["round_trips"], refined_summary["trade_count"]],
            ],
            step=0,
        )
        logger.log_text("summary/json", json.dumps(summary, indent=2, sort_keys=True), step=0)

    if args.r2_dest:
        _sync_to_r2(out_dir, args.r2_dest)

    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
