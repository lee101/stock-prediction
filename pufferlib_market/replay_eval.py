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

from pufferlib_market.evaluate_multiperiod import load_policy, make_policy_fn
from pufferlib_market.hourly_replay import (
    load_hourly_market,
    read_mktd,
    replay_hourly_frozen_daily_actions,
    simulate_hourly_policy,
    simulate_daily_policy,
)
from pufferlib_market.metrics import annualize_total_return


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
    p.add_argument("--output-json", default=None, help="Optional path to write a JSON report")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")

    daily_data = read_mktd(args.daily_data_path)
    S = daily_data.num_symbols
    policy, _, _ = load_policy(
        args.checkpoint,
        S,
        arch=args.arch,
        hidden_size=args.hidden_size,
        device=device,
    )

    policy_fn = make_policy_fn(
        policy,
        num_symbols=S,
        deterministic=bool(args.deterministic),
        device=device,
    )

    daily = simulate_daily_policy(
        daily_data,
        policy_fn,
        max_steps=args.max_steps,
        fee_rate=args.fee_rate,
        fill_buffer_bps=args.fill_buffer_bps,
        max_leverage=args.max_leverage,
        short_borrow_apr=args.short_borrow_apr,
        periods_per_year=args.daily_periods_per_year,
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
        max_steps=args.max_steps,
        fee_rate=args.fee_rate,
        max_leverage=args.max_leverage,
        short_borrow_apr=args.short_borrow_apr,
        periods_per_year=args.hourly_periods_per_year,
    )

    # Annualize using calendar days (max_steps) for comparability.
    daily_ann = annualize_total_return(
        daily.total_return,
        periods=args.max_steps,
        periods_per_year=args.daily_periods_per_year,
    )
    hourly_ann = annualize_total_return(
        hourly.total_return,
        periods=args.max_steps,
        periods_per_year=args.daily_periods_per_year,
    )

    hourly_policy = None
    if args.run_hourly_policy:
        hourly_policy = simulate_hourly_policy(
            data=daily_data,
            policy_fn=policy_fn,
            market=market,
            start_date=args.start_date,
            end_date=args.end_date,
            max_steps_days=args.max_steps,
            fee_rate=args.fee_rate,
            max_leverage=args.max_leverage,
            short_borrow_apr=args.short_borrow_apr,
            periods_per_year=args.hourly_periods_per_year,
        )

    report = {
        "checkpoint": str(args.checkpoint),
        "daily_data_path": str(args.daily_data_path),
        "hourly_data_root": str(args.hourly_data_root),
        "date_range": {"start": args.start_date, "end": args.end_date},
        "symbols": list(daily_data.symbols),
        "fill_buffer_bps": float(args.fill_buffer_bps),
        "daily": {
            "total_return": daily.total_return,
            "annualized_return": daily_ann,
            "sortino": daily.sortino,
            "max_drawdown": daily.max_drawdown,
            "num_trades": daily.num_trades,
            "win_rate": daily.win_rate,
            "avg_hold_steps": daily.avg_hold_steps,
        },
        "hourly_replay": {
            "total_return": hourly.total_return,
            "annualized_return": hourly_ann,
            "sortino": hourly.sortino,
            "max_drawdown": hourly.max_drawdown,
            "num_trades": hourly.num_trades,
            "num_orders": hourly.num_orders,
            "win_rate": hourly.win_rate,
            **_order_day_stats(hourly.orders_by_day, num_days=args.max_steps + 1),
        },
    }
    if hourly_policy is not None:
        hourly_policy_ann = annualize_total_return(
            hourly_policy.total_return,
            periods=args.max_steps,
            periods_per_year=args.daily_periods_per_year,
        )
        report["hourly_policy"] = {
            "total_return": hourly_policy.total_return,
            "annualized_return": hourly_policy_ann,
            "sortino": hourly_policy.sortino,
            "max_drawdown": hourly_policy.max_drawdown,
            "num_trades": hourly_policy.num_trades,
            "num_orders": hourly_policy.num_orders,
            "win_rate": hourly_policy.win_rate,
            **_order_day_stats(hourly_policy.orders_by_day, num_days=args.max_steps + 1),
        }

    print(json.dumps(report, indent=2, sort_keys=True))

    if args.output_json:
        out = Path(args.output_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
