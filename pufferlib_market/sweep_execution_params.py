"""Sweep execution parameters on top of the best RL models.

Varies: slippage_bps, trailing_stop_pct, max_hold_bars, fill_buffer_bps
to find optimal execution settings for Binance production.

Usage:
    source .venv313/bin/activate
    python pufferlib_market/sweep_execution_params.py
"""
from __future__ import annotations

import itertools
import json
import sys
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pufferlib_market.hourly_replay import read_mktd, simulate_daily_policy
from pufferlib_market.evaluate_tail import TradingPolicy, ResidualTradingPolicy, _slice_tail
from pufferlib_market.evaluate_holdout import _infer_arch, _infer_hidden_size, _infer_resmlp_blocks

CKPT_DIR = Path("pufferlib_market/checkpoints/mixed23_a40_sweep")
VAL_DATA = Path("pufferlib_market/data/mixed23_daily_val.bin")

CHECKPOINTS = [
    "robust_reg_tp005_dd002",
    "robust_reg_tp005_ent",
    "gspo_like_smooth_mix15",
]

SLIPPAGE_BPS = [0, 1, 3, 5, 8, 10]
TRAILING_STOP_PCT = [0.0, 0.001, 0.003, 0.005, 0.01]
MAX_HOLD_BARS = [4, 6, 8, 12, 24]
FILL_BUFFER_BPS = [0, 3, 5, 8, 12]

TARGET_PERIOD = 120


def load_model(ckpt_name: str, data, device: str = "cpu"):
    ckpt_path = CKPT_DIR / ckpt_name / "best.pt"
    ckpt = torch.load(str(ckpt_path), map_location=device, weights_only=False)
    sd = ckpt.get("model", ckpt)
    arch = _infer_arch(sd)
    obs_size = data.num_symbols * 16 + 5 + data.num_symbols
    num_actions = 1 + 2 * data.num_symbols
    hidden = _infer_hidden_size(sd, arch=arch)
    if arch == "resmlp":
        blocks = _infer_resmlp_blocks(sd)
        policy = ResidualTradingPolicy(obs_size, num_actions, hidden=hidden, num_blocks=blocks)
    else:
        policy = TradingPolicy(obs_size, num_actions, hidden=hidden)
    policy.load_state_dict(sd, strict=False)
    policy.eval()
    return policy


def make_policy_fn(policy):
    @torch.no_grad()
    def fn(obs_np):
        obs_t = torch.from_numpy(obs_np).float().unsqueeze(0)
        logits, val = policy(obs_t)
        return logits.argmax(dim=-1).item()
    return fn


def eval_with_params(policy_fn, data, period, slippage, trailing_stop, max_hold, fill_buffer):
    tail_data = _slice_tail(data, period)
    result = simulate_daily_policy(
        tail_data, policy_fn,
        max_steps=period,
        fee_rate=0.0,
        slippage_bps=slippage,
        fill_buffer_bps=fill_buffer,
        max_leverage=1.0,
        trailing_stop_pct=trailing_stop,
        max_hold_bars=max_hold,
        min_notional_usd=12.0,
        periods_per_year=365,
        enable_drawdown_profit_early_exit=False,
    )
    return {
        "return_pct": result.total_return * 100,
        "sortino": result.sortino,
        "max_dd_pct": result.max_drawdown * 100,
        "num_trades": result.num_trades,
        "win_rate": result.win_rate * 100 if result.win_rate is not None else 0,
    }


def main():
    data = read_mktd(str(VAL_DATA))
    print(f"Loaded val data: {data.num_symbols} symbols, {data.num_timesteps} steps")

    for ckpt_name in CHECKPOINTS:
        print(f"\n{'='*70}")
        print(f"CHECKPOINT: {ckpt_name}")
        print(f"{'='*70}")

        policy = load_model(ckpt_name, data)
        policy_fn = make_policy_fn(policy)

        # Phase 1: Sweep slippage (other params fixed at production values)
        print(f"\n  Phase 1: Slippage sweep (trail=0.003, hold=6, buffer=8)")
        for slip in SLIPPAGE_BPS:
            m = eval_with_params(policy_fn, data, TARGET_PERIOD, slip, 0.003, 6, 8)
            print(f"    slip={slip:2d}bps: ret={m['return_pct']:+6.1f}% sort={m['sortino']:+.2f} "
                  f"dd={m['max_dd_pct']:.1f}% trades={m['num_trades']} wr={m['win_rate']:.0f}%")

        # Phase 2: Sweep trailing stop
        print(f"\n  Phase 2: Trailing stop sweep (slip=3, hold=6, buffer=8)")
        for ts in TRAILING_STOP_PCT:
            m = eval_with_params(policy_fn, data, TARGET_PERIOD, 3, ts, 6, 8)
            label = f"{ts*100:.1f}%" if ts > 0 else "none"
            print(f"    trail={label:>5}: ret={m['return_pct']:+6.1f}% sort={m['sortino']:+.2f} "
                  f"dd={m['max_dd_pct']:.1f}% trades={m['num_trades']} wr={m['win_rate']:.0f}%")

        # Phase 3: Sweep max hold
        print(f"\n  Phase 3: Max hold sweep (slip=3, trail=0.003, buffer=8)")
        for mh in MAX_HOLD_BARS:
            m = eval_with_params(policy_fn, data, TARGET_PERIOD, 3, 0.003, mh, 8)
            print(f"    hold={mh:2d}: ret={m['return_pct']:+6.1f}% sort={m['sortino']:+.2f} "
                  f"dd={m['max_dd_pct']:.1f}% trades={m['num_trades']} wr={m['win_rate']:.0f}%")

        # Phase 4: Sweep fill buffer
        print(f"\n  Phase 4: Fill buffer sweep (slip=3, trail=0.003, hold=6)")
        for fb in FILL_BUFFER_BPS:
            m = eval_with_params(policy_fn, data, TARGET_PERIOD, 3, 0.003, 6, fb)
            print(f"    buf={fb:2d}bps: ret={m['return_pct']:+6.1f}% sort={m['sortino']:+.2f} "
                  f"dd={m['max_dd_pct']:.1f}% trades={m['num_trades']} wr={m['win_rate']:.0f}%")

        # Phase 5: Fine-tune best combo
        print(f"\n  Phase 5: Fine-tune grid (top combos)")
        best_sort = -999
        best_combo = None
        combos = list(itertools.product(
            [0, 3, 5],       # slippage
            [0.0, 0.003, 0.005],  # trailing stop
            [6, 8, 12],      # max hold
            [0, 5, 8],       # fill buffer
        ))
        for slip, ts, mh, fb in combos:
            m = eval_with_params(policy_fn, data, TARGET_PERIOD, slip, ts, mh, fb)
            if m["sortino"] > best_sort:
                best_sort = m["sortino"]
                best_combo = (slip, ts, mh, fb)
                best_m = m

        print(f"    BEST: slip={best_combo[0]}bps trail={best_combo[1]*100:.1f}% "
              f"hold={best_combo[2]} buf={best_combo[3]}bps")
        print(f"    -> ret={best_m['return_pct']:+.1f}% sort={best_m['sortino']:+.2f} "
              f"dd={best_m['max_dd_pct']:.1f}% trades={best_m['num_trades']} wr={best_m['win_rate']:.0f}%")


if __name__ == "__main__":
    main()
