#!/usr/bin/env python
"""Render an intra-bar (hourly-fill) eval video for a daily-trained policy.

Differs from `render_prod_stocks_video.py` in two important ways:

1. Execution is simulated on hourly OHLC bars via
   `pufferlib_market.intrabar_replay.replay_intrabar`. Stops, take-profits,
   max-hold-hours, and limit-price entries fire at the actual hour the bar's
   [low, high] crosses the trigger level — not at the daily close.

2. The rendered MP4 shows hourly candlesticks (24× the bar count of the
   daily video) and gold order markers at the actual hour each fill happened,
   so heavy intraday oscillations are visible.

The hourly OHLC source can be either real CSV files under `--hourly-root`
(crypto/<SYM>.csv or stocks/<SYM>.csv) or, with `--synthetic-hourly`, a
24-bar-per-day synthetic walk built from the daily MKTD itself. Use
synthetic for fast smoke renders without external data.

Output goes to models/artifacts/<run>/videos/ so artifacts_server.py can
serve it at /files/<run>/videos/<file>.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from pufferlib_market.checkpoint_loader import load_checkpoint_payload
from pufferlib_market.evaluate import _load_policy_with_metadata
from pufferlib_market.evaluate_holdout import _slice_window
from pufferlib_market.evaluate_sliding import _build_policy_fn
from pufferlib_market.hourly_replay import read_mktd, simulate_daily_policy
from pufferlib_market.intrabar_replay import (
    build_hourly_marketsim_trace,
    load_hourly_ohlc,
    replay_intrabar,
    synthetic_hourly_ohlc_from_daily,
)
from src.marketsim_video import render_html_plotly, render_mp4


def _record_actions(policy_fn, window, max_steps: int, fee_rate: float) -> np.ndarray:
    actions: list[int] = []

    def rec(obs):
        a = int(policy_fn(obs))
        actions.append(a)
        return a

    simulate_daily_policy(
        window,
        rec,
        max_steps=max_steps,
        fee_rate=fee_rate,
        fill_buffer_bps=5.0,
        periods_per_year=252.0,
        enable_drawdown_profit_early_exit=False,
    )
    return np.asarray(actions, dtype=np.int32)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default="pufferlib_market/prod_ensemble/s15.pt")
    p.add_argument("--data-path", default="pufferlib_market/data/stocks12_daily_val_20260401.bin")
    p.add_argument("--window-start", type=int, default=0)
    p.add_argument("--window-steps", type=int, default=30)
    p.add_argument("--start-date", default="2025-11-01",
                   help="UTC date the FIRST step of the window aligns to.")
    p.add_argument("--hourly-root", default="data/hourly",
                   help="Root containing crypto/<SYM>.csv and/or stocks/<SYM>.csv")
    p.add_argument("--synthetic-hourly", action="store_true",
                   help="Build synthetic 24h-per-day bars from the daily MKTD; useful for smoke tests.")
    p.add_argument("--num-pairs", type=int, default=4)
    p.add_argument("--fps", type=int, default=8)
    p.add_argument("--frames-per-bar", type=int, default=1)
    p.add_argument("--fee-rate", type=float, default=0.001)
    p.add_argument("--fill-buffer-bps", type=float, default=5.0)
    p.add_argument("--stop-loss-pct", type=float, default=0.0)
    p.add_argument("--take-profit-pct", type=float, default=0.0)
    p.add_argument("--max-hold-hours", type=int, default=0)
    p.add_argument("--out", default="models/artifacts/intrabar_eval/videos/run.mp4")
    p.add_argument("--cpu", action="store_true")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    data = read_mktd(args.data_path)
    print(f"data: {len(data.symbols)} symbols × {data.num_timesteps} steps")

    features_per_sym = int(data.features.shape[2])
    obs_size = data.num_symbols * features_per_sym + 5 + data.num_symbols

    ckpt = load_checkpoint_payload(args.checkpoint, map_location=device)
    state_dict = ckpt.get("model", ckpt)
    num_actions = int(state_dict["actor.2.bias"].shape[0])
    hidden = int(state_dict["encoder.0.bias"].shape[0])
    policy, _, _ = _load_policy_with_metadata(
        ckpt, obs_size, num_actions, hidden, "mlp", device, features_per_sym=features_per_sym
    )
    policy.eval()
    policy_fn = _build_policy_fn(policy, device, deterministic=True, disable_shorts=False)

    window = _slice_window(data, start=int(args.window_start), steps=int(args.window_steps))
    actions = _record_actions(policy_fn, window, int(args.window_steps), float(args.fee_rate))
    print(f"recorded {len(actions)} daily actions, nonzero={int((actions != 0).sum())}")

    if args.synthetic_hourly:
        hourly = synthetic_hourly_ohlc_from_daily(window, start=args.start_date)
    else:
        end_date = pd.Timestamp(args.start_date, tz="UTC") + pd.Timedelta(
            hours=24 * int(args.window_steps) - 1
        )
        hourly = load_hourly_ohlc(
            window.symbols,
            args.hourly_root,
            start=args.start_date,
            end=end_date.isoformat(),
        )

    res = replay_intrabar(
        data=window,
        actions=actions,
        hourly=hourly,
        start_date=args.start_date,
        max_steps=int(args.window_steps),
        fee_rate=float(args.fee_rate),
        fill_buffer_bps=float(args.fill_buffer_bps),
        stop_loss_pct=(float(args.stop_loss_pct) or None),
        take_profit_pct=(float(args.take_profit_pct) or None),
        max_hold_hours=(int(args.max_hold_hours) or None),
    )
    print(
        f"intrabar replay: ret={res.total_return*100:+.2f}% "
        f"sortino={res.sortino:.2f} max_dd={res.max_drawdown*100:.2f}% "
        f"trades={res.num_trades} fills={res.num_orders}"
    )

    trace = build_hourly_marketsim_trace(
        hourly=hourly,
        fills=res.fills,
        equity_curve=res.equity_curve,
        initial_equity=res.initial_equity,
    )

    out = Path(args.out)
    title = (
        f"intrabar · {Path(args.checkpoint).stem} · {args.window_steps}d window · "
        f"ret={res.total_return*100:+.2f}% trades={res.num_trades} fills={res.num_orders}"
    )
    render_mp4(
        trace, out,
        num_pairs=args.num_pairs,
        fps=args.fps,
        frames_per_bar=args.frames_per_bar,
        title=title,
        fee_rate=args.fee_rate,
        periods_per_year=8760.0,
    )
    print(f"wrote {out}  ({out.stat().st_size} bytes)")
    json_path = out.with_suffix(".json")
    trace.to_json(json_path)
    print(f"wrote {json_path}")
    try:
        html_path = out.with_suffix(".html")
        render_html_plotly(trace, html_path, num_pairs=args.num_pairs, title=title)
        print(f"wrote {html_path}")
    except Exception as e:
        print(f"  plotly html skipped: {e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
