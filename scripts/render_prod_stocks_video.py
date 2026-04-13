#!/usr/bin/env python
"""Render a realistic prod-stocks marketsim video.

Loads a stocks12 prod-ensemble checkpoint, runs simulate_daily_policy on a 90d
window of the latest val bin, and writes an MP4 with price/buy/sell/equity/fees
to models/artifacts/<run>/videos/.

This mirrors what the live ensemble sees on prod stocks (AAPL, MSFT, NVDA,
GOOG, META, TSLA, SPY, QQQ, JPM, V, AMZN, PLTR) using the same val bin used
for promotion gating.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from pufferlib_market.checkpoint_loader import load_checkpoint_payload
from pufferlib_market.evaluate import _load_policy_with_metadata
from pufferlib_market.evaluate_sliding import _build_policy_fn
from pufferlib_market.hourly_replay import (
    P_OPEN, P_HIGH, P_LOW, P_CLOSE,
    read_mktd,
    simulate_daily_policy,
)
from pufferlib_market.evaluate_holdout import _slice_window
from src.marketsim_video import MarketsimTrace, OrderTick, render_mp4, render_html_plotly


def _find_best_window_start(
    num_timesteps: int,
    window_steps: int,
    evaluate_start,
) -> dict:
    """Find the window start that maximises total_return, breaking ties by sortino.

    Args:
        num_timesteps: Total number of available timesteps in the data.
        window_steps: Number of steps in each evaluation window.
        evaluate_start: Callable(start: int) -> (total_return, sortino, max_drawdown, num_trades).

    Returns:
        dict with keys: total_return, sortino, max_drawdown, num_trades, window_start.

    Raises:
        ValueError: If window_steps >= num_timesteps (no valid start).
    """
    n_starts = num_timesteps - window_steps
    if n_starts <= 0:
        raise ValueError(
            f"window_steps={window_steps} is too large for num_timesteps={num_timesteps} "
            f"(need window_steps < num_timesteps)"
        )

    best_start = None
    best_total_return = float("-inf")
    best_sortino = float("-inf")
    best_max_drawdown = 0.0
    best_num_trades = 0

    for start in range(n_starts):
        total_return, sortino, max_drawdown, num_trades = evaluate_start(start)
        if (total_return > best_total_return or
                (total_return == best_total_return and sortino > best_sortino)):
            best_start = start
            best_total_return = total_return
            best_sortino = sortino
            best_max_drawdown = max_drawdown
            best_num_trades = num_trades

    return {
        "total_return": best_total_return,
        "sortino": best_sortino,
        "max_drawdown": best_max_drawdown,
        "num_trades": best_num_trades,
        "window_start": best_start,
    }


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default="pufferlib_market/prod_ensemble/s15.pt")
    p.add_argument("--data-path", default="pufferlib_market/data/stocks12_daily_val_20260401.bin")
    p.add_argument("--window-start", type=int, default=0)
    p.add_argument("--window-steps", type=int, default=90)
    p.add_argument("--num-pairs", type=int, default=4)
    p.add_argument("--fps", type=int, default=2)
    p.add_argument("--frames-per-bar", type=int, default=4,
                   help="Repeat each simulator bar N video frames for smoother playback")
    p.add_argument("--fee-rate", type=float, default=0.001)
    p.add_argument("--periods-per-year", type=float, default=252.0)
    p.add_argument("--out", default="models/artifacts/prod_stocks12/videos/s15_window0.mp4")
    p.add_argument("--cpu", action="store_true")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")

    data = read_mktd(args.data_path)
    print(f"data: {len(data.symbols)} symbols × {data.num_timesteps} steps  symbols={data.symbols}")

    features_per_sym = int(data.features.shape[2])
    obs_size = data.num_symbols * features_per_sym + 5 + data.num_symbols

    ckpt = load_checkpoint_payload(args.checkpoint, map_location=device)
    state_dict = ckpt.get("model", ckpt)
    num_actions = int(state_dict["actor.2.bias"].shape[0])
    hidden = int(state_dict["encoder.0.bias"].shape[0])
    print(f"checkpoint: obs_size={obs_size} num_actions={num_actions} hidden={hidden}")

    policy, arch_used, hidden_used = _load_policy_with_metadata(
        ckpt,
        obs_size,
        num_actions,
        hidden,
        "mlp",
        device,
        features_per_sym=features_per_sym,
    )
    policy.eval()
    policy_fn = _build_policy_fn(policy, device, deterministic=True, disable_shorts=False)

    window = _slice_window(data, start=int(args.window_start), steps=int(args.window_steps))
    print(f"window: start={args.window_start} steps={args.window_steps}")

    rec_actions: list[int] = []

    def rec_policy(obs):
        a = int(policy_fn(obs))
        rec_actions.append(a)
        return a

    result = simulate_daily_policy(
        window,
        rec_policy,
        max_steps=int(args.window_steps),
        fee_rate=float(args.fee_rate),
        fill_buffer_bps=5.0,
        periods_per_year=float(args.periods_per_year),
        enable_drawdown_profit_early_exit=False,
    )

    # Reconstruct per-step (position_sym, is_short) from action stream.
    # For 1-bin alloc/level: action 0 = flat, [1..S] = long sym, [S+1..2S] = short sym.
    S = window.num_symbols
    pos_sym = -1
    pos_short = False
    eq_curve = np.asarray(result.equity_curve, dtype=np.float64) if result.equity_curve is not None else np.array([])
    initial_equity = float(eq_curve[0]) if len(eq_curve) else 1.0

    close_prices = window.prices[:, :, P_CLOSE].astype(np.float32, copy=False)
    ohlc = window.prices[:, :, [P_OPEN, P_HIGH, P_LOW, P_CLOSE]].astype(np.float32, copy=False)
    trace = MarketsimTrace(
        symbols=list(window.symbols), prices=close_prices, prices_ohlc=ohlc
    )

    # Per-bar long/short target lines: for the 1-bin prod policy these are
    # always at the current bar's close (no level offset). They give a clear
    # visual of "this is where the bot would buy / where it would short" on
    # every bar regardless of whether it actually traded.
    for i, a in enumerate(rec_actions):
        target_sym, target_short = -1, False
        if a == 0:
            target_sym, target_short = -1, False
        elif 1 <= a <= S:
            target_sym, target_short = a - 1, False
        elif S + 1 <= a <= 2 * S:
            target_sym, target_short = a - S - 1, True
        eq = float(eq_curve[i + 1]) if (i + 1) < len(eq_curve) else float(eq_curve[-1] if len(eq_curve) else initial_equity)
        orders = []
        if target_sym >= 0:
            orders.append(OrderTick(sym=target_sym, price=float(close_prices[i, target_sym]), is_short=target_short))
        # Daily long/short target at this bar's close (or target sym only).
        lt = np.full(S, np.nan, dtype=np.float32)
        st = np.full(S, np.nan, dtype=np.float32)
        if target_sym >= 0:
            if target_short:
                st[target_sym] = float(close_prices[i, target_sym])
            else:
                lt[target_sym] = float(close_prices[i, target_sym])
        pos_sym, pos_short = target_sym, target_short
        trace.record(
            step=i,
            action_id=a,
            position_sym=pos_sym,
            position_is_short=pos_short,
            equity=eq * 10000.0 / initial_equity,
            orders=orders,
            long_target=lt,
            short_target=st,
        )

    out = Path(args.out)
    title = (
        f"prod stocks12 · {Path(args.checkpoint).stem} · "
        f"window {args.window_start} · ret={result.total_return*100:+.2f}% · "
        f"trades={result.num_trades} · sortino={result.sortino:.2f}"
    )
    render_mp4(
        trace, out,
        num_pairs=args.num_pairs,
        fps=args.fps,
        frames_per_bar=args.frames_per_bar,
        title=title,
        fee_rate=args.fee_rate,
        periods_per_year=args.periods_per_year,
    )
    print(f"wrote {out}  ({out.stat().st_size} bytes)")
    # JSON sidecar (so any consumer — incl. the C training loop — can dump
    # the same shape and reuse the renderer) and an interactive Plotly HTML.
    json_path = out.with_suffix(".json")
    trace.to_json(json_path)
    print(f"wrote {json_path}")
    try:
        html_path = out.with_suffix(".html")
        render_html_plotly(trace, html_path, num_pairs=args.num_pairs, title=title)
        print(f"wrote {html_path}")
    except Exception as e:
        print(f"  plotly html skipped: {e}")
    print(f"return={result.total_return*100:+.2f}%  trades={result.num_trades}  sortino={result.sortino:.2f}  max_dd={result.max_drawdown*100:.2f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
