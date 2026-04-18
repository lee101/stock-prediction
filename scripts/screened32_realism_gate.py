"""Fill-buffer × leverage realism gate for the prod screened32 ensemble.

Sweeps the production ensemble (DEFAULT_CHECKPOINT + DEFAULT_EXTRA_CHECKPOINTS)
across a grid of (fill_buffer_bps, max_leverage) and reports per-cell median /
p10 monthly returns plus negative-window count on the published 263-window
single-offset val set. Use this BEFORE deploying any ensemble change to
verify that the proposed config still passes at the live-equivalent fill
buffer (5bps minimum; live limits sit at +5/+25 bps from open).

Why this exists: scripts/eval_100d.py runs the daily fp4 path with
fill_buffer_bps=0 by default — that lets the policy "fill" anywhere inside
[low, high] of the daily bar. Production limit orders need the bar to trade
*through* the limit by 5+ bps before they get hit. The 19.57%/mo headline is
the lookahead-tolerant (bps=0) number; this script tells you whether the
ensemble actually clears the bar at production-realistic execution.

Usage::

    python scripts/screened32_realism_gate.py \
        --val-data pufferlib_market/data/screened32_single_offset_val_full.bin \
        --window-days 50 \
        --fill-buffer-bps-grid 0,5,10,20 \
        --max-leverage-grid 1.0,1.5,2.0 \
        --decision-lag 2 \
        --out-dir docs/realism_gate
"""
from __future__ import annotations

import argparse
import collections
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from torch.distributions import Categorical

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from pufferlib_market.evaluate_holdout import (  # noqa: E402
    _mask_all_shorts,
    _slice_window,
    load_policy,
)
from pufferlib_market.batched_ensemble import StackedEnsemble, can_batch  # noqa: E402
from pufferlib_market.hourly_replay import read_mktd, simulate_daily_policy  # noqa: E402
from src.daily_stock_defaults import (  # noqa: E402
    DEFAULT_CHECKPOINT,
    DEFAULT_EXTRA_CHECKPOINTS,
)


@dataclass(frozen=True)
class CellResult:
    fill_buffer_bps: float
    max_leverage: float
    median_total_return: float
    p10_total_return: float
    p90_total_return: float
    median_monthly_return: float
    p10_monthly_return: float
    median_sortino: float
    median_max_dd: float
    n_neg: int
    n_windows: int


def _monthly_from_total(total: float, days: int, trading_days_per_month: float = 21.0) -> float:
    if days <= 0:
        return 0.0
    try:
        return math.expm1(math.log1p(float(total)) * (trading_days_per_month / float(days)))
    except (ValueError, OverflowError):
        return 0.0


def _percentile(values: Sequence[float], q: float) -> float:
    if not values:
        return 0.0
    return float(np.percentile(np.asarray(values, dtype=np.float64), q))


def _build_ensemble_policy_fn(
    *,
    checkpoints: Sequence[Path],
    num_symbols: int,
    features_per_sym: int,
    decision_lag: int,
    disable_shorts: bool,
    device: torch.device,
    deterministic: bool,
    ensemble_mode: str = "softmax_avg",
):
    """Load N checkpoints and return an ensemble policy_fn matching prod.

    ensemble_mode:
      - "softmax_avg" (default, current prod): average per-policy softmax probs, then argmax.
        Conservative members with high flat-probability pull ensemble argmax toward flat.
      - "logit_avg": average raw logits, then argmax. Sharper non-flat decisions when
        member confidences on the argmax winner dominate the few flat-leaning members.
    """
    if ensemble_mode not in ("softmax_avg", "logit_avg"):
        raise ValueError(f"ensemble_mode must be 'softmax_avg' or 'logit_avg', got {ensemble_mode!r}")
    loaded = [
        load_policy(str(p), num_symbols, features_per_sym=features_per_sym, device=device)
        for p in checkpoints
    ]
    head = loaded[0]
    alloc_bins = int(head.action_allocation_bins)
    level_bins = int(head.action_level_bins)
    per_symbol_actions = max(1, alloc_bins) * max(1, level_bins)
    policies = [lp.policy.eval() for lp in loaded]
    n_ensemble = len(policies)
    pending: collections.deque[int] = collections.deque(maxlen=max(1, decision_lag + 1))

    # Fast path: stack all members' weights into one bmm-based forward.
    # ~13× speedup on per-step inference vs the serial for-loop at batch=1.
    # Golden-tested in tests/test_batched_ensemble.py for argmax parity.
    # Env kill-switch: BATCHED_ENSEMBLE_DISABLE=1 forces the serial loop (for debugging).
    import os as _os
    stacked = None
    if n_ensemble > 1 and can_batch(policies) and not _os.environ.get("BATCHED_ENSEMBLE_DISABLE"):
        stacked = StackedEnsemble.from_policies(policies, device)

    def reset_buffer() -> None:
        pending.clear()

    def policy_fn(obs: np.ndarray) -> int:
        obs_t = torch.from_numpy(obs.astype(np.float32, copy=False)).to(device=device).view(1, -1)
        with torch.no_grad():
            if n_ensemble == 1:
                logits, _ = policies[0](obs_t)
            elif stacked is not None:
                all_logits = stacked.forward(obs_t)  # [N, 1, A]
                if ensemble_mode == "softmax_avg":
                    probs_avg = torch.softmax(all_logits, dim=-1).mean(dim=0)
                    logits = torch.log(probs_avg + 1e-8)
                else:  # logit_avg
                    logits = all_logits.mean(dim=0)
            elif ensemble_mode == "softmax_avg":
                probs_sum: torch.Tensor | None = None
                for p in policies:
                    lg, _ = p(obs_t)
                    pr = torch.softmax(lg, dim=-1)
                    probs_sum = pr if probs_sum is None else probs_sum + pr
                assert probs_sum is not None
                logits = torch.log(probs_sum / float(n_ensemble) + 1e-8)
            else:  # logit_avg
                logit_sum: torch.Tensor | None = None
                for p in policies:
                    lg, _ = p(obs_t)
                    logit_sum = lg if logit_sum is None else logit_sum + lg
                assert logit_sum is not None
                logits = logit_sum / float(n_ensemble)
        if disable_shorts:
            logits = _mask_all_shorts(
                logits,
                num_symbols=int(num_symbols),
                per_symbol_actions=int(per_symbol_actions),
            )
        if deterministic:
            action_now = int(torch.argmax(logits, dim=-1).item())
        else:
            action_now = int(Categorical(logits=logits).sample().item())
        if decision_lag <= 0:
            return action_now
        pending.append(action_now)
        if len(pending) <= decision_lag:
            return 0
        return pending.popleft()

    return policy_fn, reset_buffer, head


def _run_cell(
    *,
    data,
    checkpoints: Sequence[Path],
    num_symbols: int,
    features_per_sym: int,
    decision_lag: int,
    disable_shorts: bool,
    deterministic: bool,
    device: torch.device,
    fill_buffer_bps: float,
    max_leverage: float,
    fee_rate: float,
    slippage_bps: float,
    window_days: int,
    start_indices: Sequence[int],
    ensemble_mode: str = "softmax_avg",
) -> CellResult:
    policy_fn, reset_buffer, head = _build_ensemble_policy_fn(
        checkpoints=checkpoints,
        num_symbols=num_symbols,
        features_per_sym=features_per_sym,
        decision_lag=decision_lag,
        disable_shorts=disable_shorts,
        device=device,
        deterministic=deterministic,
        ensemble_mode=ensemble_mode,
    )
    rets: list[float] = []
    sortinos: list[float] = []
    maxdds: list[float] = []
    for start in start_indices:
        window = _slice_window(data, start=int(start), steps=int(window_days))
        reset_buffer()
        result = simulate_daily_policy(
            window,
            policy_fn,
            max_steps=int(window_days),
            fee_rate=float(fee_rate),
            slippage_bps=float(slippage_bps),
            max_leverage=float(max_leverage),
            periods_per_year=365.0,
            fill_buffer_bps=float(fill_buffer_bps),
            action_allocation_bins=int(head.action_allocation_bins),
            action_level_bins=int(head.action_level_bins),
            action_max_offset_bps=float(head.action_max_offset_bps),
            enable_drawdown_profit_early_exit=False,
        )
        rets.append(float(result.total_return))
        sortinos.append(float(result.sortino))
        maxdds.append(float(result.max_drawdown))
    median_ret = _percentile(rets, 50)
    p10_ret = _percentile(rets, 10)
    p90_ret = _percentile(rets, 90)
    return CellResult(
        fill_buffer_bps=float(fill_buffer_bps),
        max_leverage=float(max_leverage),
        median_total_return=median_ret,
        p10_total_return=p10_ret,
        p90_total_return=p90_ret,
        median_monthly_return=_monthly_from_total(median_ret, int(window_days)),
        p10_monthly_return=_monthly_from_total(p10_ret, int(window_days)),
        median_sortino=_percentile(sortinos, 50),
        median_max_dd=_percentile(maxdds, 50),
        n_neg=int(sum(1 for r in rets if r < 0.0)),
        n_windows=int(len(rets)),
    )


def _render_md(
    cells: list[CellResult],
    *,
    val_path: Path,
    window_days: int,
    fee_rate: float,
    slippage_bps: float,
    decision_lag: int,
    monthly_target: float,
    checkpoint_names: list[str] | None = None,
) -> str:
    lines: list[str] = []
    lines.append(f"# Screened32 realism gate — `{val_path.name}`")
    lines.append("")
    lines.append(
        f"- window_days={window_days}, fee_rate={fee_rate}, slippage_bps={slippage_bps}, "
        f"decision_lag={decision_lag}, monthly_target={monthly_target * 100:.1f}%/mo"
    )
    if checkpoint_names is not None:
        names_summary = ", ".join(checkpoint_names) if len(checkpoint_names) <= 16 \
            else f"{len(checkpoint_names)} checkpoints"
        lines.append(f"- ensemble: {len(checkpoint_names)}-model softmax_avg ({names_summary})")
    else:
        lines.append("- ensemble: softmax_avg (checkpoint list unspecified)")
    lines.append("")

    fill_buffers = sorted({c.fill_buffer_bps for c in cells})
    leverages = sorted({c.max_leverage for c in cells})

    def cell_for(fb: float, lev: float) -> CellResult | None:
        for c in cells:
            if c.fill_buffer_bps == fb and c.max_leverage == lev:
                return c
        return None

    lines.append("## Median monthly return (worst on each row is the realistic deploy gate)")
    lines.append("")
    header = "| fill_bps \\ leverage | " + " | ".join(f"{lev:g}x" for lev in leverages) + " |"
    sep = "|---:" * (len(leverages) + 1) + "|"
    lines.append(header)
    lines.append(sep)
    for fb in fill_buffers:
        row = [f"{fb:g}"]
        for lev in leverages:
            c = cell_for(fb, lev)
            if c is None:
                row.append("—")
            else:
                marker = "✅" if c.median_monthly_return >= monthly_target else ("⚠️" if c.median_monthly_return >= 0.10 else "❌")
                row.append(f"{c.median_monthly_return * 100:+.2f}% {marker}")
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")

    lines.append("## p10 monthly return (tail risk)")
    lines.append("")
    lines.append(header)
    lines.append(sep)
    for fb in fill_buffers:
        row = [f"{fb:g}"]
        for lev in leverages:
            c = cell_for(fb, lev)
            row.append("—" if c is None else f"{c.p10_monthly_return * 100:+.2f}%")
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")

    lines.append("## Negative-window count (out of 263)")
    lines.append("")
    lines.append(header)
    lines.append(sep)
    for fb in fill_buffers:
        row = [f"{fb:g}"]
        for lev in leverages:
            c = cell_for(fb, lev)
            row.append("—" if c is None else f"{c.n_neg}/{c.n_windows}")
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")

    lines.append("## Per-cell raw")
    lines.append("")
    lines.append("| fill_bps | leverage | median_total | p10_total | sortino | max_dd | n_neg |")
    lines.append("|---:|---:|---:|---:|---:|---:|---:|")
    for c in cells:
        lines.append(
            f"| {c.fill_buffer_bps:g} | {c.max_leverage:g} "
            f"| {c.median_total_return * 100:+.2f}% | {c.p10_total_return * 100:+.2f}% "
            f"| {c.median_sortino:.2f} | {c.median_max_dd * 100:.2f}% "
            f"| {c.n_neg}/{c.n_windows} |"
        )
    lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--val-data", default="pufferlib_market/data/screened32_single_offset_val_full.bin")
    ap.add_argument("--window-days", type=int, default=50)
    ap.add_argument("--fill-buffer-bps-grid", default="0,5,10,20",
                    help="Comma-separated fill_buffer_bps cells (default: 0,5,10,20)")
    ap.add_argument("--max-leverage-grid", default="1.0,1.5,2.0",
                    help="Comma-separated max_leverage cells (default: 1.0,1.5,2.0)")
    ap.add_argument("--fee-rate", type=float, default=0.001,
                    help="Per-fill fee rate (default 0.001 = 10bps).")
    ap.add_argument("--slippage-bps", type=float, default=5.0,
                    help="Adverse fill slippage in bps; symmetric on open and close.")
    ap.add_argument("--decision-lag", type=int, default=2,
                    help="Defer the policy's action by N bars (production-safe default 2).")
    ap.add_argument("--monthly-target", type=float, default=0.27)
    ap.add_argument("--no-disable-shorts", dest="disable_shorts", action="store_false",
                    help="Allow short actions; default is to mask them like prod (longs-only, 33-action ensemble).")
    ap.set_defaults(disable_shorts=True)
    ap.add_argument("--max-windows", type=int, default=None,
                    help="Cap the number of windows per cell (debug). Default: all 263.")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--deterministic", action="store_true", default=True)
    ap.add_argument("--out-dir", default="docs/realism_gate")
    ap.add_argument("--checkpoints", nargs="*", default=None,
                    help="Override prod ensemble (default: DEFAULT_CHECKPOINT + DEFAULT_EXTRA_CHECKPOINTS).")
    ap.add_argument("--ensemble-mode", choices=["softmax_avg", "logit_avg"], default="softmax_avg",
                    help="How to combine per-policy outputs. softmax_avg (prod) averages probs; "
                         "logit_avg averages raw logits (sharper, fewer aggregation-artifact flats).")
    args = ap.parse_args(argv)

    val_path = Path(args.val_data).resolve()
    if not val_path.exists():
        print(f"realism_gate: val data not found: {val_path}", file=sys.stderr)
        return 2
    fill_buffers = [float(x) for x in args.fill_buffer_bps_grid.split(",") if x.strip()]
    leverages = [float(x) for x in args.max_leverage_grid.split(",") if x.strip()]
    if not fill_buffers or not leverages:
        print("realism_gate: empty grid", file=sys.stderr)
        return 2

    if args.checkpoints:
        ckpts = [Path(c) for c in args.checkpoints]
    else:
        ckpts = [Path(DEFAULT_CHECKPOINT), *(Path(p) for p in DEFAULT_EXTRA_CHECKPOINTS)]
    missing = [c for c in ckpts if not (REPO / c).exists()]
    if missing:
        print(f"realism_gate: missing checkpoints: {missing}", file=sys.stderr)
        return 2
    abs_ckpts = [REPO / c for c in ckpts]

    data = read_mktd(val_path)
    num_symbols = int(data.num_symbols)
    features_per_sym = int(data.features.shape[2])
    window_len = int(args.window_days) + 1
    if window_len > data.num_timesteps:
        print(
            f"realism_gate: val too short for window_days={args.window_days} "
            f"(T={data.num_timesteps})",
            file=sys.stderr,
        )
        return 2
    candidate_count = data.num_timesteps - window_len + 1
    start_indices = list(range(candidate_count))
    if args.max_windows is not None:
        start_indices = start_indices[: int(args.max_windows)]

    device = torch.device(args.device)
    cells: list[CellResult] = []
    for fb in fill_buffers:
        for lev in leverages:
            print(f"[{fb:g} bps × {lev:g}x] running {len(start_indices)} windows...", flush=True)
            cell = _run_cell(
                data=data,
                checkpoints=abs_ckpts,
                num_symbols=num_symbols,
                features_per_sym=features_per_sym,
                decision_lag=int(args.decision_lag),
                disable_shorts=bool(args.disable_shorts),
                deterministic=bool(args.deterministic),
                device=device,
                fill_buffer_bps=float(fb),
                max_leverage=float(lev),
                fee_rate=float(args.fee_rate),
                slippage_bps=float(args.slippage_bps),
                window_days=int(args.window_days),
                start_indices=start_indices,
                ensemble_mode=str(args.ensemble_mode),
            )
            cells.append(cell)
            print(
                f"  med_monthly={cell.median_monthly_return * 100:+.2f}%  "
                f"p10_monthly={cell.p10_monthly_return * 100:+.2f}%  "
                f"neg={cell.n_neg}/{cell.n_windows}  sortino={cell.median_sortino:.2f}",
                flush=True,
            )

    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = REPO / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    val_stem = val_path.stem
    payload = {
        "val_data": str(val_path),
        "window_days": int(args.window_days),
        "fee_rate": float(args.fee_rate),
        "slippage_bps": float(args.slippage_bps),
        "decision_lag": int(args.decision_lag),
        "disable_shorts": bool(args.disable_shorts),
        "monthly_target": float(args.monthly_target),
        "ensemble_size": len(abs_ckpts),
        "ensemble_mode": str(args.ensemble_mode),
        "checkpoints": [str(c) for c in ckpts],
        "n_windows_per_cell": len(start_indices),
        "fill_buffer_bps_grid": fill_buffers,
        "max_leverage_grid": leverages,
        "cells": [c.__dict__ for c in cells],
    }
    json_path = out_dir / f"{val_stem}_realism_gate.json"
    md_path = out_dir / f"{val_stem}_realism_gate.md"
    json_path.write_text(json.dumps(payload, indent=2, default=str))
    md = _render_md(
        cells,
        val_path=val_path,
        window_days=int(args.window_days),
        fee_rate=float(args.fee_rate),
        slippage_bps=float(args.slippage_bps),
        decision_lag=int(args.decision_lag),
        monthly_target=float(args.monthly_target),
        checkpoint_names=[Path(c).stem for c in ckpts],
    )
    md_path.write_text(md)
    print()
    print(md)
    print()
    print(f"wrote {json_path}")
    print(f"wrote {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
