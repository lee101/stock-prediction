#!/usr/bin/env python3
"""Fast GPU screen for screened32 ensemble composition experiments.

This is a research filter, not a production gate. It uses the parity-tested
GPU realism sim with binary fills, decision_lag=2, fill-through buffers, fees,
and adverse slippage, but keeps short_borrow_apr/margin APR at zero so many
candidate ensembles can be ranked quickly. Send only the best candidates to
scripts/screened32_realism_gate.py or scripts/eval_100d.py for full-cost proof.
"""
from __future__ import annotations

import argparse
import gc
import itertools
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import torch


REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from pufferlib_market.batched_ensemble import StackedEnsemble, can_batch  # noqa: E402
from pufferlib_market.evaluate_holdout import load_policy  # noqa: E402
from pufferlib_market.gpu_realism_gate import (  # noqa: E402
    _P_CLOSE,
    _P_LOW,
    _argmax_with_short_mask,
    _build_obs_batch,
    _stage_windows,
)
from pufferlib_market.hourly_replay import INITIAL_CASH, read_mktd  # noqa: E402
from xgbnew.artifacts import write_json_atomic  # noqa: E402

from src.daily_stock_defaults import DEFAULT_CHECKPOINT, DEFAULT_EXTRA_CHECKPOINTS  # noqa: E402


@dataclass(frozen=True)
class Candidate:
    label: str
    paths: tuple[Path, ...]


@dataclass(frozen=True)
class ScreenResult:
    label: str
    members: tuple[str, ...]
    ensemble_size: int
    median_total_return: float
    p10_total_return: float
    p90_total_return: float
    median_monthly_return: float
    p10_monthly_return: float
    max_drawdown: float
    median_drawdown: float
    n_neg: int
    n_windows: int
    score: float


def _monthly_from_total(total_return: float, window_days: int) -> float:
    if window_days <= 0:
        return 0.0
    try:
        return math.expm1(math.log1p(float(total_return)) * (21.0 / float(window_days)))
    except (ValueError, OverflowError):
        return -1.0


def _percentile(values: Sequence[float], q: float) -> float:
    if not values:
        return 0.0
    return float(np.percentile(np.asarray(values, dtype=np.float64), q))


def _safe_label(text: str) -> str:
    out = []
    for ch in str(text):
        if ch.isalnum() or ch in {"_", "-", "."}:
            out.append(ch)
        else:
            out.append("_")
    return "".join(out).strip("._") or "candidate"


def _resolve_repo_path(path: str | Path) -> Path:
    p = Path(path)
    if not p.is_absolute():
        p = REPO / p
    return p.resolve()


def _default_baseline_paths() -> tuple[Path, ...]:
    return tuple(_resolve_repo_path(p) for p in (DEFAULT_CHECKPOINT, *DEFAULT_EXTRA_CHECKPOINTS))


def _all_checkpoint_paths(root: Path) -> tuple[Path, ...]:
    return tuple(sorted(root.glob("*.pt"), key=lambda p: p.stem))


def _dedupe_candidates(candidates: Iterable[Candidate]) -> list[Candidate]:
    out: list[Candidate] = []
    seen_members: set[tuple[str, ...]] = set()
    used_labels: set[str] = set()
    for cand in candidates:
        member_key = tuple(str(p) for p in cand.paths)
        if member_key in seen_members:
            continue
        seen_members.add(member_key)
        label = cand.label
        if label in used_labels:
            suffix = 2
            while f"{label}_{suffix}" in used_labels:
                suffix += 1
            label = f"{label}_{suffix}"
        used_labels.add(label)
        out.append(Candidate(label=label, paths=cand.paths))
    return out


def build_initial_candidates(
    *,
    baseline_paths: Sequence[Path],
    all_paths: Sequence[Path],
    include_singles: bool,
    include_drop_one: bool,
    include_add_one: bool,
    include_dup_one: bool,
) -> list[Candidate]:
    baseline = tuple(baseline_paths)
    baseline_set = set(baseline)
    candidates: list[Candidate] = [Candidate("baseline_v7", baseline)]

    if include_singles:
        for path in all_paths:
            candidates.append(Candidate(f"single_{path.stem}", (path,)))

    if include_drop_one:
        for i, path in enumerate(baseline):
            label = f"drop{i + 1}_{path.stem}"
            candidates.append(Candidate(label, baseline[:i] + baseline[i + 1 :]))

    if include_add_one:
        for path in all_paths:
            if path not in baseline_set:
                candidates.append(Candidate(f"add_{path.stem}", baseline + (path,)))

    if include_dup_one:
        for path in all_paths:
            candidates.append(Candidate(f"dup_{path.stem}", baseline + (path,)))

    return _dedupe_candidates(candidates)


def build_pair_candidates(single_results: Sequence[ScreenResult], *, top_n: int) -> list[Candidate]:
    if top_n <= 1:
        return []
    top = list(single_results[:top_n])
    out: list[Candidate] = []
    for left, right in itertools.combinations(top, 2):
        left_path = Path(left.members[0])
        right_path = Path(right.members[0])
        out.append(
            Candidate(
                label=f"pair_{left_path.stem}_{right_path.stem}",
                paths=(left_path, right_path),
            )
        )
    return _dedupe_candidates(out)


def _result_from_returns(
    *,
    label: str,
    paths: Sequence[Path],
    total_returns: np.ndarray,
    max_drawdowns: np.ndarray,
    window_days: int,
    neg_penalty: float,
    dd_penalty: float,
) -> ScreenResult:
    rets = [float(x) for x in np.asarray(total_returns, dtype=np.float64)]
    dds = [float(x) for x in np.asarray(max_drawdowns, dtype=np.float64)]
    median_total = _percentile(rets, 50)
    p10_total = _percentile(rets, 10)
    p90_total = _percentile(rets, 90)
    median_monthly = _monthly_from_total(median_total, int(window_days))
    p10_monthly = _monthly_from_total(p10_total, int(window_days))
    n_neg = int(sum(1 for value in rets if value < 0.0))
    max_dd = max(dds) if dds else 0.0
    # Rank for research: median PnL matters, but candidates that get there
    # through many losing windows or ugly drawdown should fall down the queue.
    score = (
        float(median_monthly)
        + 0.5 * float(p10_monthly)
        - float(neg_penalty) * float(n_neg)
        - float(dd_penalty) * float(max_dd)
    )
    return ScreenResult(
        label=str(label),
        members=tuple(str(p) for p in paths),
        ensemble_size=len(paths),
        median_total_return=float(median_total),
        p10_total_return=float(p10_total),
        p90_total_return=float(p90_total),
        median_monthly_return=float(median_monthly),
        p10_monthly_return=float(p10_monthly),
        max_drawdown=float(max_dd),
        median_drawdown=_percentile(dds, 50),
        n_neg=int(n_neg),
        n_windows=len(rets),
        score=float(score),
    )


def _evaluate_candidate_gpu(
    *,
    label: str,
    paths: Sequence[Path],
    policies_by_path: dict[Path, object],
    prices: torch.Tensor,
    features: torch.Tensor,
    tradable: torch.Tensor,
    num_symbols: int,
    features_per_sym: int,
    window_days: int,
    fill_buffer_bps: float,
    max_leverage: float,
    fee_rate: float,
    slippage_bps: float,
    decision_lag: int,
    ensemble_mode: str,
    neg_penalty: float,
    dd_penalty: float,
) -> ScreenResult:
    if decision_lag != 2:
        raise ValueError("GPU screen currently requires decision_lag=2")
    if ensemble_mode not in {"softmax_avg", "logit_avg"}:
        raise ValueError(f"unsupported ensemble_mode={ensemble_mode!r}")
    if not paths:
        raise ValueError("candidate must contain at least one checkpoint")

    loaded = [policies_by_path[Path(p)] for p in paths]
    head = loaded[0]
    alloc_bins = int(head.action_allocation_bins)
    level_bins = int(head.action_level_bins)
    action_max_offset_bps = float(head.action_max_offset_bps)
    per_symbol_actions = max(1, alloc_bins) * max(1, level_bins)
    if alloc_bins != 1 or level_bins != 1 or action_max_offset_bps != 0.0:
        raise ValueError(
            "GPU screen restricted to alloc_bins=level_bins=1 and action_max_offset_bps=0"
        )
    selected_policies = [lp.policy.eval() for lp in loaded]
    if not can_batch(selected_policies):
        raise ValueError(f"candidate {label!r} has policies that are not stack-compatible")
    device = prices.device
    stacked = StackedEnsemble.from_policies(selected_policies, device)

    n_windows = int(prices.shape[0])
    slip_frac = max(0.0, float(slippage_bps)) / 10_000.0
    effective_fee = float(fee_rate) + slip_frac
    fill_buffer_frac = max(0.0, float(fill_buffer_bps)) / 10_000.0
    init_cash = float(INITIAL_CASH)
    cash = torch.full((n_windows,), init_cash, device=device, dtype=torch.float32)
    pos_sym = torch.full((n_windows,), -1, device=device, dtype=torch.int32)
    pos_qty = torch.zeros(n_windows, device=device, dtype=torch.float32)
    pos_entry = torch.zeros(n_windows, device=device, dtype=torch.float32)
    hold_steps = torch.zeros(n_windows, device=device, dtype=torch.int32)
    peak_equity = torch.full((n_windows,), init_cash, device=device, dtype=torch.float32)
    max_dd = torch.zeros(n_windows, device=device, dtype=torch.float32)
    lag = int(decision_lag)
    action_buf = torch.zeros(n_windows, lag + 1, device=device, dtype=torch.int32)
    buf_count = 0
    row_idx = torch.arange(n_windows, device=device)

    for step in range(int(window_days)):
        obs = _build_obs_batch(
            features,
            prices,
            int(step),
            cash,
            pos_sym,
            pos_qty,
            pos_entry,
            hold_steps,
            int(step),
            int(window_days),
            init_cash,
        )
        with torch.no_grad():
            all_logits = stacked.forward(obs)
            if ensemble_mode == "softmax_avg":
                probs = torch.softmax(all_logits, dim=-1).mean(dim=0)
                logits = torch.log(probs + 1e-8)
            else:
                logits = all_logits.mean(dim=0)
        action_now = _argmax_with_short_mask(
            logits,
            int(num_symbols),
            int(per_symbol_actions),
            disable_shorts=True,
        ).to(torch.int32)

        slot = buf_count % (lag + 1)
        action_buf[:, slot] = action_now
        buf_count += 1
        if buf_count <= lag:
            action = torch.zeros(n_windows, device=device, dtype=torch.int32)
        else:
            emit_slot = (buf_count - 1 - lag) % (lag + 1)
            action = action_buf[:, emit_slot]

        held = pos_sym >= 0
        price_cur = torch.zeros(n_windows, device=device, dtype=torch.float32)
        if held.any():
            sym_idx = pos_sym.clamp_min(0).long()
            price_cur = prices[:, step, :, _P_CLOSE].gather(1, sym_idx.unsqueeze(1)).squeeze(1)
            price_cur = torch.where(held, price_cur, torch.zeros_like(price_cur))
        pos_sym_pre = pos_sym.clone()
        was_held = pos_sym_pre >= 0
        flat_mask = action == 0
        long_mask = (action >= 1) & (action <= int(num_symbols))
        target_sym = (action - 1).clamp_min(0).long()
        same_sym = long_mask & was_held & (target_sym == pos_sym_pre.long())

        trad_t = tradable[:, step, :]
        cur_sym_safe = pos_sym_pre.clamp_min(0).long()
        cur_tradable = trad_t.gather(1, cur_sym_safe.unsqueeze(1)).squeeze(1)
        cur_tradable = torch.where(was_held, cur_tradable, torch.ones_like(cur_tradable))
        target_tradable = trad_t.gather(1, target_sym.unsqueeze(1)).squeeze(1)

        flat_close = flat_mask & was_held & cur_tradable
        switch_ready = long_mask & was_held & ~same_sym & cur_tradable & target_tradable
        switch_close = switch_ready
        flat_hold = flat_mask & was_held & ~cur_tradable
        switch_blocked_hold = long_mask & was_held & ~same_sym & (~cur_tradable | ~target_tradable)

        closing = flat_close | switch_close
        proceeds = pos_qty * price_cur * (1.0 - effective_fee)
        cash = torch.where(closing, cash + proceeds, cash)
        pos_sym = torch.where(closing, torch.full_like(pos_sym, -1), pos_sym)
        pos_qty = torch.where(closing, torch.zeros_like(pos_qty), pos_qty)
        pos_entry = torch.where(closing, torch.zeros_like(pos_entry), pos_entry)
        hold_steps = torch.where(flat_close, torch.zeros_like(hold_steps), hold_steps)

        want_open = (long_mask & ~was_held & target_tradable) | switch_close
        close_tgt = prices[row_idx, step, target_sym, _P_CLOSE]
        low_tgt = prices[row_idx, step, target_sym, _P_LOW]
        fillable = low_tgt <= close_tgt * (1.0 - fill_buffer_frac)
        denom = close_tgt * (1.0 + effective_fee)
        qty_new = torch.where(denom > 0, (cash * float(max_leverage)) / denom, torch.zeros_like(denom))
        cost_new = qty_new * denom
        can_open = want_open & fillable & (close_tgt > 0) & (cash > 0) & (qty_new > 0) & (cost_new > 0)
        cash = torch.where(can_open, cash - cost_new, cash)
        pos_sym = torch.where(can_open, target_sym.to(torch.int32), pos_sym)
        pos_qty = torch.where(can_open, qty_new, pos_qty)
        pos_entry = torch.where(can_open, close_tgt, pos_entry)
        hold_steps = torch.where(can_open, torch.zeros_like(hold_steps), hold_steps)

        carry_hold = same_sym | flat_hold | switch_blocked_hold
        hold_steps = torch.where(carry_hold, hold_steps + 1, hold_steps)

        t_new = min(step + 1, int(prices.size(1)) - 1)
        held2 = pos_sym >= 0
        price_new = torch.zeros(n_windows, device=device, dtype=torch.float32)
        if held2.any():
            sym_idx = pos_sym.clamp_min(0).long()
            price_new = prices[:, t_new, :, _P_CLOSE].gather(1, sym_idx.unsqueeze(1)).squeeze(1)
            price_new = torch.where(held2, price_new, torch.zeros_like(price_new))
        equity_after = torch.where(held2, cash + pos_qty * price_new, cash)
        peak_equity = torch.maximum(peak_equity, equity_after)
        dd = torch.where(
            peak_equity > 0,
            (peak_equity - equity_after) / peak_equity.clamp_min(1e-12),
            torch.zeros_like(peak_equity),
        )
        max_dd = torch.maximum(max_dd, dd)

    held_final = pos_sym >= 0
    if held_final.any():
        t_last = int(prices.size(1)) - 1
        sym_idx = pos_sym.clamp_min(0).long()
        price_end = prices[:, t_last, :, _P_CLOSE].gather(1, sym_idx.unsqueeze(1)).squeeze(1)
        proceeds_final = pos_qty * price_end * (1.0 - effective_fee)
        cash = torch.where(held_final, cash + proceeds_final, cash)

    total_return = (cash / init_cash) - 1.0
    result = _result_from_returns(
        label=label,
        paths=paths,
        total_returns=total_return.detach().to(torch.float64).cpu().numpy(),
        max_drawdowns=max_dd.detach().to(torch.float64).cpu().numpy(),
        window_days=int(window_days),
        neg_penalty=float(neg_penalty),
        dd_penalty=float(dd_penalty),
    )
    del stacked
    return result


def _as_json(result: ScreenResult) -> dict:
    return {
        "label": result.label,
        "members": list(result.members),
        "ensemble_size": result.ensemble_size,
        "median_total_return": result.median_total_return,
        "p10_total_return": result.p10_total_return,
        "p90_total_return": result.p90_total_return,
        "median_monthly_return": result.median_monthly_return,
        "p10_monthly_return": result.p10_monthly_return,
        "max_drawdown": result.max_drawdown,
        "median_drawdown": result.median_drawdown,
        "n_neg": result.n_neg,
        "n_windows": result.n_windows,
        "score": result.score,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--val-data", default="pufferlib_market/data/screened32_single_offset_val_full.bin")
    parser.add_argument("--checkpoint-root", default="pufferlib_market/prod_ensemble_screened32")
    parser.add_argument("--window-days", type=int, default=100)
    parser.add_argument("--max-windows", type=int, default=None)
    parser.add_argument("--fill-buffer-bps", type=float, default=5.0)
    parser.add_argument("--slippage-bps", type=float, default=20.0)
    parser.add_argument("--leverage", type=float, default=2.0)
    parser.add_argument("--fee-rate", type=float, default=0.001)
    parser.add_argument("--decision-lag", type=int, default=2)
    parser.add_argument("--ensemble-mode", choices=["softmax_avg", "logit_avg"], default="softmax_avg")
    parser.add_argument(
        "--pair-top",
        type=int,
        default=0,
        help="After singles, evaluate all pairs among the top N singles.",
    )
    parser.add_argument("--no-singles", action="store_true")
    parser.add_argument("--no-drop-one", action="store_true")
    parser.add_argument("--no-add-one", action="store_true")
    parser.add_argument("--no-dup-one", action="store_true")
    parser.add_argument("--neg-penalty", type=float, default=0.002)
    parser.add_argument("--dd-penalty", type=float, default=0.15)
    parser.add_argument("--top-k", type=int, default=30)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out", default="analysis/screened32_gpu_ensemble_sweep/sweep.json")
    return parser


def _validate_finite_nonnegative(value: object, name: str) -> str | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return f"{name} must be finite and non-negative"
    if not math.isfinite(parsed) or parsed < 0.0:
        return f"{name} must be finite and non-negative"
    return None


def _validate_finite_positive(value: object, name: str) -> str | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return f"{name} must be finite and positive"
    if not math.isfinite(parsed) or parsed <= 0.0:
        return f"{name} must be finite and positive"
    return None


def validate_args(args: argparse.Namespace) -> list[str]:
    failures: list[str] = []
    for attr in ("fill_buffer_bps", "slippage_bps", "fee_rate", "neg_penalty", "dd_penalty"):
        failure = _validate_finite_nonnegative(getattr(args, attr), attr)
        if failure is not None:
            failures.append(failure)
    for attr in ("window_days", "leverage"):
        failure = _validate_finite_positive(getattr(args, attr), attr)
        if failure is not None:
            failures.append(failure)
    if int(args.decision_lag) != 2:
        failures.append("decision_lag must be exactly 2 for the GPU screen")
    if args.max_windows is not None:
        try:
            max_windows = int(args.max_windows)
        except (TypeError, ValueError):
            failures.append("max_windows must be a positive integer when provided")
        else:
            if max_windows <= 0:
                failures.append("max_windows must be a positive integer when provided")
    for attr in ("pair_top", "top_k"):
        try:
            value = int(getattr(args, attr))
        except (TypeError, ValueError):
            failures.append(f"{attr} must be a non-negative integer")
            continue
        if value < 0:
            failures.append(f"{attr} must be a non-negative integer")
    return failures


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    validation_failures = validate_args(args)
    if validation_failures:
        for failure in validation_failures:
            print(f"sweep_screened32_gpu_ensemble: {failure}", file=sys.stderr)
        return 2
    if not torch.cuda.is_available() and str(args.device).startswith("cuda"):
        print("sweep_screened32_gpu_ensemble: CUDA is unavailable", file=sys.stderr)
        return 2

    val_path = _resolve_repo_path(args.val_data)
    checkpoint_root = _resolve_repo_path(args.checkpoint_root)
    if not val_path.exists():
        print(f"sweep_screened32_gpu_ensemble: val data not found: {val_path}", file=sys.stderr)
        return 2
    if not checkpoint_root.exists():
        print(f"sweep_screened32_gpu_ensemble: checkpoint root not found: {checkpoint_root}", file=sys.stderr)
        return 2

    baseline_paths = _default_baseline_paths()
    all_paths = _all_checkpoint_paths(checkpoint_root)
    missing = [p for p in baseline_paths if not p.exists()]
    if missing:
        print(f"sweep_screened32_gpu_ensemble: missing baseline checkpoints: {missing}", file=sys.stderr)
        return 2
    if not all_paths:
        print(f"sweep_screened32_gpu_ensemble: no .pt checkpoints under {checkpoint_root}", file=sys.stderr)
        return 2

    data = read_mktd(val_path)
    window_len = int(args.window_days) + 1
    if window_len > int(data.num_timesteps):
        print("sweep_screened32_gpu_ensemble: window is longer than val data", file=sys.stderr)
        return 2
    starts = list(range(int(data.num_timesteps) - window_len + 1))
    if args.max_windows is not None:
        starts = starts[: max(1, int(args.max_windows))]

    device = torch.device(str(args.device))
    prices, features, tradable = _stage_windows(data, starts, int(args.window_days), device)
    print(
        f"[screen] staged {len(starts)} windows x {args.window_days}d, "
        f"slip={float(args.slippage_bps):g}bps fill={float(args.fill_buffer_bps):g}bps "
        f"lev={float(args.leverage):g}x"
    )
    print(f"[screen] loading {len(all_paths)} checkpoints from {checkpoint_root}")
    policies_by_path = {
        path: load_policy(
            path,
            int(data.num_symbols),
            features_per_sym=int(data.features.shape[2]),
            device=device,
        )
        for path in all_paths
    }

    candidates = build_initial_candidates(
        baseline_paths=baseline_paths,
        all_paths=all_paths,
        include_singles=not bool(args.no_singles),
        include_drop_one=not bool(args.no_drop_one),
        include_add_one=not bool(args.no_add_one),
        include_dup_one=not bool(args.no_dup_one),
    )
    results: list[ScreenResult] = []
    seen_member_sets: set[tuple[str, ...]] = set()

    def evaluate_batch(batch: Sequence[Candidate]) -> None:
        for i, cand in enumerate(batch, start=1):
            member_key = tuple(str(p) for p in cand.paths)
            if member_key in seen_member_sets:
                continue
            seen_member_sets.add(member_key)
            result = _evaluate_candidate_gpu(
                label=cand.label,
                paths=cand.paths,
                policies_by_path=policies_by_path,
                prices=prices,
                features=features,
                tradable=tradable,
                num_symbols=int(data.num_symbols),
                features_per_sym=int(data.features.shape[2]),
                window_days=int(args.window_days),
                fill_buffer_bps=float(args.fill_buffer_bps),
                max_leverage=float(args.leverage),
                fee_rate=float(args.fee_rate),
                slippage_bps=float(args.slippage_bps),
                decision_lag=int(args.decision_lag),
                ensemble_mode=str(args.ensemble_mode),
                neg_penalty=float(args.neg_penalty),
                dd_penalty=float(args.dd_penalty),
            )
            results.append(result)
            if i % 10 == 0 or cand.label == "baseline_v7":
                print(
                    f"[screen] {len(results):4d} {result.label:28s} "
                    f"med={result.median_monthly_return * 100:+6.2f}% "
                    f"p10={result.p10_monthly_return * 100:+6.2f}% "
                    f"neg={result.n_neg:3d}/{result.n_windows} "
                    f"dd={result.max_drawdown * 100:5.1f}%"
                )
            if i % 25 == 0:
                gc.collect()
                if device.type == "cuda":
                    torch.cuda.empty_cache()

    evaluate_batch(candidates)
    single_results = sorted(
        [r for r in results if r.label.startswith("single_")],
        key=lambda r: r.score,
        reverse=True,
    )
    pair_candidates = build_pair_candidates(single_results, top_n=int(args.pair_top))
    if pair_candidates:
        print(f"[screen] evaluating {len(pair_candidates)} pair candidates from top {int(args.pair_top)} singles")
        evaluate_batch(pair_candidates)

    ranked = sorted(results, key=lambda r: r.score, reverse=True)
    payload = {
        "val_data": str(val_path),
        "checkpoint_root": str(checkpoint_root),
        "window_days": int(args.window_days),
        "n_windows": len(starts),
        "starts": starts,
        "fill_buffer_bps": float(args.fill_buffer_bps),
        "slippage_bps": float(args.slippage_bps),
        "leverage": float(args.leverage),
        "fee_rate": float(args.fee_rate),
        "short_borrow_apr": 0.0,
        "decision_lag": int(args.decision_lag),
        "ensemble_mode": str(args.ensemble_mode),
        "ranking": {
            "score": "median_monthly + 0.5*p10_monthly - neg_penalty*n_neg - dd_penalty*max_drawdown",
            "neg_penalty": float(args.neg_penalty),
            "dd_penalty": float(args.dd_penalty),
        },
        "baseline_members": [str(p) for p in baseline_paths],
        "results": [_as_json(r) for r in ranked],
    }
    out_path = Path(args.out)
    if not out_path.is_absolute():
        out_path = REPO / out_path
    write_json_atomic(out_path, payload, sort_keys=True)

    print(f"\nTop {max(1, int(args.top_k))} GPU-screen candidates:")
    for r in ranked[: max(1, int(args.top_k))]:
        print(
            f"{r.score:+.4f}  {r.median_monthly_return * 100:+6.2f}%/mo "
            f"p10={r.p10_monthly_return * 100:+6.2f}% "
            f"neg={r.n_neg:3d}/{r.n_windows} "
            f"dd={r.max_drawdown * 100:5.1f}% "
            f"size={r.ensemble_size:2d}  {r.label}"
        )
    print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
