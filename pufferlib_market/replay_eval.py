"""Hourly replay evaluation CLI for daily-trained pufferlib_market policies.

This tool:
1) Runs the policy on the daily MKTD (v2) file to generate a daily action trace.
2) Replays that daily trace on aligned hourly bars to compute higher-resolution
   risk metrics (hourly Sortino, intraday max drawdown) and order counts.

It is meant to catch "looks good on daily close-to-close" artifacts where the
equity path is much rougher intraday or where the policy flips too frequently
if executed more often.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from pufferlib_market.evaluate_multiperiod import LoadedPolicy, load_policy, make_policy_fn
from pufferlib_market.hourly_replay import (
    InitialPositionSpec,
    load_hourly_market,
    read_mktd,
    replay_hourly_frozen_daily_actions,
    simulate_hourly_policy,
    simulate_daily_policy,
)
from pufferlib_market.metrics import annualize_total_return
from src.robust_trading_metrics import (
    compute_market_sim_goodness_score,
    compute_pnl_smoothness_from_equity,
    compute_ulcer_index,
)


def _order_day_stats(orders_by_day: dict[str, int], num_days: int) -> dict[str, object]:
    if num_days <= 0:
        num_days = max(len(orders_by_day), 1)
    counts = list(orders_by_day.values())
    max_in_day = int(max(counts, default=0))
    mean_per_day = float(sum(counts) / num_days) if num_days > 0 else 0.0
    nonzero_days = int(sum(1 for v in counts if v > 0))
    top = sorted(orders_by_day.items(), key=lambda kv: kv[1], reverse=True)[:5]
    return {
        "max_orders_in_day": max_in_day,
        "mean_orders_per_day": mean_per_day,
        "nonzero_order_days": nonzero_days,
        "top_order_days": [{"date": k, "orders": int(v)} for k, v in top],
    }


def _section_metrics(
    *,
    total_return: float,
    sortino: float,
    max_drawdown: float,
    num_trades: int,
    win_rate: float,
    equity_curve: np.ndarray | None,
    periods_per_year: float,
    num_orders: int | None = None,
    avg_hold_steps: float | None = None,
    extra: dict[str, object] | None = None,
) -> dict[str, object]:
    equity = np.asarray(equity_curve if equity_curve is not None else [], dtype=np.float64)
    pnl_smoothness = compute_pnl_smoothness_from_equity(equity) if equity.size >= 2 else 0.0
    ulcer_index = compute_ulcer_index(equity) if equity.size >= 1 else 0.0
    period_count = max(int(equity.size) - 1, 0)
    section = {
        "total_return": float(total_return),
        "annualized_return": annualize_total_return(
            total_return,
            periods=max(period_count, 1),
            periods_per_year=periods_per_year,
        ),
        "sortino": float(sortino),
        "max_drawdown": float(max_drawdown),
        "num_trades": int(num_trades),
        "win_rate": float(win_rate),
        "pnl_smoothness": float(pnl_smoothness),
        "ulcer_index": float(ulcer_index),
        "goodness_score": float(
            compute_market_sim_goodness_score(
                total_return=float(total_return),
                sortino=float(sortino),
                max_drawdown=float(max_drawdown),
                pnl_smoothness=float(pnl_smoothness),
                ulcer_index=float(ulcer_index),
                trade_count=int(num_trades),
                period_count=max(period_count, 1),
            )
        ),
    }
    if num_orders is not None:
        section["num_orders"] = int(num_orders)
    if avg_hold_steps is not None:
        section["avg_hold_steps"] = float(avg_hold_steps)
    if extra:
        section.update(extra)
    return section


def _serialize_initial_position(spec: InitialPositionSpec | None) -> dict[str, object] | None:
    if spec is None:
        return None
    return {
        "symbol": str(spec.symbol).upper(),
        "side": str(spec.side).lower(),
        "allocation_pct": float(spec.allocation_pct),
    }


def _parse_robust_start_states(raw: str) -> list[tuple[str, InitialPositionSpec | None]]:
    text = str(raw or "").strip()
    if not text:
        return []

    scenarios: list[tuple[str, InitialPositionSpec | None]] = []
    for token in text.split(","):
        item = token.strip()
        if not item:
            continue
        if item.lower() == "flat":
            scenarios.append(("flat", None))
            continue
        parts = [part.strip() for part in item.split(":")]
        if len(parts) != 3:
            raise ValueError(
                "robust start states must use 'flat' or '<long|short>:<SYMBOL>:<allocation_pct>'"
            )
        side, symbol, alloc_text = parts
        side = side.lower()
        if side not in {"long", "short"}:
            raise ValueError(f"Unsupported robust start side {side!r}; expected 'long' or 'short'")
        allocation_pct = float(alloc_text)
        spec = InitialPositionSpec(symbol=symbol.upper(), side=side, allocation_pct=allocation_pct)
        scenarios.append((f"{side}:{symbol.upper()}:{allocation_pct:g}", spec))
    return scenarios


def _summarize_robust_section(
    scenarios: list[dict[str, object]],
    *,
    section: str,
) -> dict[str, object] | None:
    rows: list[tuple[str, dict[str, object]]] = []
    for scenario in scenarios:
        row = scenario.get(section)
        if isinstance(row, dict):
            rows.append((str(scenario.get("name", section)), row))
    if not rows:
        return None

    returns = np.asarray([float(row.get("total_return", 0.0) or 0.0) for _, row in rows], dtype=np.float64)
    sortinos = np.asarray([float(row.get("sortino", 0.0) or 0.0) for _, row in rows], dtype=np.float64)
    drawdowns = np.asarray([float(row.get("max_drawdown", 0.0) or 0.0) for _, row in rows], dtype=np.float64)

    worst_return_idx = int(np.argmin(returns))
    worst_sortino_idx = int(np.argmin(sortinos))
    worst_drawdown_idx = int(np.argmax(drawdowns))
    return {
        "scenario_count": len(rows),
        "median_total_return": float(np.median(returns)),
        "worst_total_return": float(returns[worst_return_idx]),
        "worst_total_return_scenario": rows[worst_return_idx][0],
        "worst_sortino": float(sortinos[worst_sortino_idx]),
        "worst_sortino_scenario": rows[worst_sortino_idx][0],
        "worst_max_drawdown": float(drawdowns[worst_drawdown_idx]),
        "worst_max_drawdown_scenario": rows[worst_drawdown_idx][0],
    }


def _summarize_robust_scenarios(scenarios: list[dict[str, object]]) -> dict[str, object]:
    summary: dict[str, object] = {
        "scenario_count": len(scenarios),
        "scenario_names": [str(scenario.get("name", "")) for scenario in scenarios],
    }
    for section in ("daily", "hourly_replay", "hourly_policy"):
        section_summary = _summarize_robust_section(scenarios, section=section)
        if section_summary is not None:
            summary[section] = section_summary
    return summary


def _coerce_loaded_policy(value: object) -> LoadedPolicy:
    if isinstance(value, LoadedPolicy):
        return value
    if isinstance(value, tuple) and len(value) == 3:
        policy, metadata, num_actions = value
        meta = metadata if isinstance(metadata, dict) else {}
        return LoadedPolicy(
            policy=policy,
            arch=str(meta.get("arch", "unknown")),
            hidden_size=int(meta.get("hidden_size", 0) or 0),
            action_allocation_bins=int(meta.get("action_allocation_bins", 1) or 1),
            action_level_bins=int(meta.get("action_level_bins", 1) or 1),
            action_max_offset_bps=float(meta.get("action_max_offset_bps", 0.0) or 0.0),
            num_actions=int(num_actions),
        )
    raise TypeError(f"Unsupported load_policy result: {type(value)!r}")


def main() -> None:
    p = argparse.ArgumentParser(description="Replay daily RL actions on hourly prices")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--daily-data-path", required=True)
    p.add_argument("--hourly-data-root", default="trainingdatahourly")
    p.add_argument("--start-date", required=True, help="UTC date used to export the daily MKTD (inclusive)")
    p.add_argument("--end-date", required=True, help="UTC date used to export the daily MKTD (inclusive)")
    p.add_argument("--max-steps", type=int, required=True, help="Episode steps (days), e.g. 50")
    p.add_argument("--fee-rate", type=float, default=0.001)
    p.add_argument(
        "--fill-buffer-bps",
        type=float,
        default=5.0,
        help="Require the daily bar to trade through each limit by this many bps before fill.",
    )
    p.add_argument("--max-leverage", type=float, default=1.0)
    p.add_argument("--short-borrow-apr", type=float, default=0.0)
    p.add_argument("--daily-periods-per-year", type=float, default=365.0)
    p.add_argument("--hourly-periods-per-year", type=float, default=8760.0)
    p.add_argument("--arch", choices=["auto", "mlp", "resmlp"], default="auto")
    p.add_argument("--hidden-size", type=int, default=None)
    p.add_argument("--deterministic", action="store_true")
    p.add_argument("--cpu", action="store_true")
    p.add_argument(
        "--run-hourly-policy",
        action="store_true",
        help="Also run a stress-test where the daily-trained policy acts every hour "
        "(daily features frozen per day; portfolio fields update hourly).",
    )
    p.add_argument(
        "--robust-start-states",
        default="",
        help="Optional comma-separated start states like 'flat,long:AAPL:0.25,short:MSFT:0.25'",
    )
    p.add_argument("--output-json", default=None, help="Optional path to write a JSON report")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")

    daily_data = read_mktd(args.daily_data_path)
    S = daily_data.num_symbols
    features_per_sym = int(daily_data.features.shape[2])
    loaded = _coerce_loaded_policy(load_policy(
        args.checkpoint,
        S,
        arch=args.arch,
        hidden_size=args.hidden_size,
        device=device,
        features_per_sym=features_per_sym,
    ))

    policy_fn = make_policy_fn(
        loaded.policy,
        num_symbols=S,
        deterministic=bool(args.deterministic),
        device=device,
    )

    effective_max_steps = min(args.max_steps, max(1, daily_data.num_timesteps - 1))
    daily = simulate_daily_policy(
        daily_data,
        policy_fn,
        max_steps=effective_max_steps,
        fee_rate=args.fee_rate,
        fill_buffer_bps=args.fill_buffer_bps,
        max_leverage=args.max_leverage,
        short_borrow_apr=args.short_borrow_apr,
        periods_per_year=args.daily_periods_per_year,
        action_allocation_bins=loaded.action_allocation_bins,
        action_level_bins=loaded.action_level_bins,
        action_max_offset_bps=loaded.action_max_offset_bps,
    )

    market = load_hourly_market(
        daily_data.symbols,
        args.hourly_data_root,
        start=f"{args.start_date} 00:00",
        end=f"{args.end_date} 23:00",
    )

    hourly = replay_hourly_frozen_daily_actions(
        data=daily_data,
        actions=daily.actions,
        market=market,
        start_date=args.start_date,
        end_date=args.end_date,
        max_steps=effective_max_steps,
        fee_rate=args.fee_rate,
        max_leverage=args.max_leverage,
        short_borrow_apr=args.short_borrow_apr,
        periods_per_year=args.hourly_periods_per_year,
        action_allocation_bins=loaded.action_allocation_bins,
        action_level_bins=loaded.action_level_bins,
        action_max_offset_bps=loaded.action_max_offset_bps,
    )

    hourly_policy = None
    if args.run_hourly_policy:
        hourly_policy = simulate_hourly_policy(
            data=daily_data,
            policy_fn=policy_fn,
            market=market,
            start_date=args.start_date,
            end_date=args.end_date,
            max_steps_days=effective_max_steps,
            fee_rate=args.fee_rate,
            max_leverage=args.max_leverage,
            short_borrow_apr=args.short_borrow_apr,
            periods_per_year=args.hourly_periods_per_year,
            action_allocation_bins=loaded.action_allocation_bins,
            action_level_bins=loaded.action_level_bins,
            action_max_offset_bps=loaded.action_max_offset_bps,
        )

    report = {
        "checkpoint": str(args.checkpoint),
        "daily_data_path": str(args.daily_data_path),
        "hourly_data_root": str(args.hourly_data_root),
        "date_range": {"start": args.start_date, "end": args.end_date},
        "symbols": list(daily_data.symbols),
        "fill_buffer_bps": float(args.fill_buffer_bps),
        "daily": _section_metrics(
            total_return=daily.total_return,
            sortino=daily.sortino,
            max_drawdown=daily.max_drawdown,
            num_trades=daily.num_trades,
            win_rate=daily.win_rate,
            equity_curve=daily.equity_curve,
            periods_per_year=args.daily_periods_per_year,
            avg_hold_steps=daily.avg_hold_steps,
        ),
        "hourly_replay": _section_metrics(
            total_return=hourly.total_return,
            sortino=hourly.sortino,
            max_drawdown=hourly.max_drawdown,
            num_trades=hourly.num_trades,
            num_orders=hourly.num_orders,
            win_rate=hourly.win_rate,
            equity_curve=hourly.equity_curve,
            periods_per_year=args.hourly_periods_per_year,
            extra=_order_day_stats(hourly.orders_by_day, num_days=effective_max_steps + 1),
        ),
    }
    if hourly_policy is not None:
        report["hourly_policy"] = _section_metrics(
            total_return=hourly_policy.total_return,
            sortino=hourly_policy.sortino,
            max_drawdown=hourly_policy.max_drawdown,
            num_trades=hourly_policy.num_trades,
            num_orders=hourly_policy.num_orders,
            win_rate=hourly_policy.win_rate,
            equity_curve=hourly_policy.equity_curve,
            periods_per_year=args.hourly_periods_per_year,
            extra=_order_day_stats(hourly_policy.orders_by_day, num_days=effective_max_steps + 1),
        )

    robust_start_states = _parse_robust_start_states(args.robust_start_states)
    if robust_start_states:
        scenarios: list[dict[str, object]] = []
        for name, initial_position in robust_start_states:
            scenario_daily = simulate_daily_policy(
                daily_data,
                policy_fn,
                max_steps=effective_max_steps,
                fee_rate=args.fee_rate,
                fill_buffer_bps=args.fill_buffer_bps,
                max_leverage=args.max_leverage,
                short_borrow_apr=args.short_borrow_apr,
                periods_per_year=args.daily_periods_per_year,
                initial_position=initial_position,
                action_allocation_bins=loaded.action_allocation_bins,
                action_level_bins=loaded.action_level_bins,
                action_max_offset_bps=loaded.action_max_offset_bps,
            )
            scenario: dict[str, object] = {
                "name": name,
                "initial_position": _serialize_initial_position(initial_position),
                "daily": _section_metrics(
                    total_return=scenario_daily.total_return,
                    sortino=scenario_daily.sortino,
                    max_drawdown=scenario_daily.max_drawdown,
                    num_trades=scenario_daily.num_trades,
                    win_rate=scenario_daily.win_rate,
                    equity_curve=scenario_daily.equity_curve,
                    periods_per_year=args.daily_periods_per_year,
                    avg_hold_steps=scenario_daily.avg_hold_steps,
                ),
            }
            scenario_hourly = replay_hourly_frozen_daily_actions(
                data=daily_data,
                actions=scenario_daily.actions,
                market=market,
                start_date=args.start_date,
                end_date=args.end_date,
                max_steps=effective_max_steps,
                fee_rate=args.fee_rate,
                max_leverage=args.max_leverage,
                short_borrow_apr=args.short_borrow_apr,
                periods_per_year=args.hourly_periods_per_year,
                initial_position=initial_position,
                action_allocation_bins=loaded.action_allocation_bins,
                action_level_bins=loaded.action_level_bins,
                action_max_offset_bps=loaded.action_max_offset_bps,
            )
            scenario["hourly_replay"] = _section_metrics(
                total_return=scenario_hourly.total_return,
                sortino=scenario_hourly.sortino,
                max_drawdown=scenario_hourly.max_drawdown,
                num_trades=scenario_hourly.num_trades,
                num_orders=scenario_hourly.num_orders,
                win_rate=scenario_hourly.win_rate,
                equity_curve=scenario_hourly.equity_curve,
                periods_per_year=args.hourly_periods_per_year,
                extra=_order_day_stats(scenario_hourly.orders_by_day, num_days=effective_max_steps + 1),
            )
            if args.run_hourly_policy:
                scenario_hourly_policy = simulate_hourly_policy(
                    data=daily_data,
                    policy_fn=policy_fn,
                    market=market,
                    start_date=args.start_date,
                    end_date=args.end_date,
                    max_steps_days=effective_max_steps,
                    fee_rate=args.fee_rate,
                    max_leverage=args.max_leverage,
                    short_borrow_apr=args.short_borrow_apr,
                    periods_per_year=args.hourly_periods_per_year,
                    initial_position=initial_position,
                    action_allocation_bins=loaded.action_allocation_bins,
                    action_level_bins=loaded.action_level_bins,
                    action_max_offset_bps=loaded.action_max_offset_bps,
                )
                scenario["hourly_policy"] = _section_metrics(
                    total_return=scenario_hourly_policy.total_return,
                    sortino=scenario_hourly_policy.sortino,
                    max_drawdown=scenario_hourly_policy.max_drawdown,
                    num_trades=scenario_hourly_policy.num_trades,
                    num_orders=scenario_hourly_policy.num_orders,
                    win_rate=scenario_hourly_policy.win_rate,
                    equity_curve=scenario_hourly_policy.equity_curve,
                    periods_per_year=args.hourly_periods_per_year,
                    extra=_order_day_stats(
                        scenario_hourly_policy.orders_by_day,
                        num_days=effective_max_steps + 1,
                    ),
                )
            scenarios.append(scenario)
        # Keep the legacy key for downstream tooling while exposing the
        # clearer name used by the newer replay/autoresearch pipeline.
        report["robust_start_states"] = scenarios
        report["robust_start_scenarios"] = scenarios
        report["robust_start_summary"] = _summarize_robust_scenarios(scenarios)

    print(json.dumps(report, indent=2, sort_keys=True))

    if args.output_json:
        out = Path(args.output_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
