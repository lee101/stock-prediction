#!/usr/bin/env python3
"""Batched CUDA search over screened32 integer ensemble weights.

This is a research accelerator, not a production gate. Unlike the one-candidate
GPU screens, this simulates many candidate ensembles concurrently:

    candidates x validation_windows x days

For each day, all candidate/window observations are flattened into one large
batch, every checkpoint policy is forwarded once, then per-candidate integer
weights combine the per-policy probabilities. This keeps the GPU much busier
when searching static ensemble weights.

The simulation uses binary fills, decision_lag=2, fill buffer, fees, and
adverse slippage. It intentionally omits borrow/margin APR for fast ranking.
Confirm top candidates with scripts/screened32_realism_gate.py before any
production consideration.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Sequence

import numpy as np
import torch


REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from pufferlib_market.batched_ensemble import StackedEnsemble, can_batch  # noqa: E402
from pufferlib_market.evaluate_holdout import load_policy  # noqa: E402
from pufferlib_market.gpu_realism_gate import _P_CLOSE, _P_LOW, _stage_windows  # noqa: E402
from pufferlib_market.hourly_replay import INITIAL_CASH, read_mktd  # noqa: E402
from xgbnew.artifacts import write_json_atomic  # noqa: E402

from scripts.sweep_screened32_gpu_ensemble import _monthly_from_total  # noqa: E402
from src.daily_stock_defaults import DEFAULT_CHECKPOINT, DEFAULT_EXTRA_CHECKPOINTS  # noqa: E402


def _resolve_repo_path(path: str | Path) -> Path:
    p = Path(path)
    if not p.is_absolute():
        p = REPO / p
    return p.resolve()


def _checkpoint_pool(root: Path, include: Sequence[str] | None) -> tuple[Path, ...]:
    paths = tuple(sorted(root.glob("*.pt"), key=lambda p: p.stem))
    if not include:
        return paths
    wanted = {x.strip() for x in include if x.strip()}
    selected = tuple(path for path in paths if path.stem in wanted or path.name in wanted)
    found = {p.stem for p in selected} | {p.name for p in selected}
    missing = sorted(wanted - found)
    if missing:
        raise ValueError(f"unknown checkpoint(s): {', '.join(missing)}")
    return selected


def _counts_for_members(pool: Sequence[Path], members: Sequence[str | Path]) -> tuple[int, ...]:
    by_stem = {path.stem: i for i, path in enumerate(pool)}
    counts = [0] * len(pool)
    for member in members:
        stem = Path(member).stem
        if stem in by_stem:
            counts[by_stem[stem]] += 1
    return tuple(counts)


def _default_baseline_counts(pool: Sequence[Path]) -> tuple[int, ...]:
    return _counts_for_members(pool, (DEFAULT_CHECKPOINT, *DEFAULT_EXTRA_CHECKPOINTS))


def _weighted_v8_counts(pool: Sequence[Path]) -> tuple[int, ...]:
    counts = list(_default_baseline_counts(pool))
    by_stem = {path.stem: i for i, path in enumerate(pool)}
    if "D_s42" in by_stem:
        counts[by_stem["D_s42"]] = 0
    for stem in ("D_s28", "D_s24", "D_s57", "D_s72"):
        if stem in by_stem:
            counts[by_stem[stem]] += 1
    return tuple(counts)


def _load_seed_counts(
    pool: Sequence[Path],
    artifacts: Sequence[str],
    *,
    top_per_artifact: int,
) -> list[tuple[int, ...]]:
    seeds = [_default_baseline_counts(pool), _weighted_v8_counts(pool)]
    for artifact in artifacts:
        if not artifact:
            continue
        path = _resolve_repo_path(artifact)
        data = json.loads(path.read_text(encoding="utf-8"))
        for item in list(data.get("results", []))[: max(0, int(top_per_artifact))]:
            if "members" in item:
                seeds.append(_counts_for_members(pool, item["members"]))
            elif "params" in item:
                params = item["params"]
                seeds.append(tuple(int(params.get(p.stem, 0)) for p in pool))
    out = []
    seen = set()
    for counts in seeds:
        key = tuple(int(x) for x in counts)
        if sum(key) <= 0 or key in seen:
            continue
        seen.add(key)
        out.append(key)
    return out


def _repair_counts(
    counts: np.ndarray,
    rng: np.random.Generator,
    *,
    max_weight: int,
    min_members: int,
    max_members: int,
    preferred: np.ndarray | None = None,
) -> np.ndarray:
    counts = np.clip(counts.astype(np.int16, copy=True), 0, int(max_weight))
    total = int(counts.sum())
    if preferred is None or preferred.size == 0:
        preferred = np.arange(counts.size)
    while total < int(min_members):
        choices = preferred[counts[preferred] < int(max_weight)]
        if choices.size == 0:
            choices = np.flatnonzero(counts < int(max_weight))
        if choices.size == 0:
            break
        idx = int(rng.choice(choices))
        counts[idx] += 1
        total += 1
    while total > int(max_members):
        choices = np.flatnonzero(counts > 0)
        if choices.size == 0:
            break
        idx = int(rng.choice(choices))
        counts[idx] -= 1
        total -= 1
    return counts


def make_population(
    *,
    pool: Sequence[Path],
    seeds: Sequence[Sequence[int]],
    n_candidates: int,
    rng: np.random.Generator,
    min_members: int,
    max_members: int,
    max_weight: int,
    mutation_prob: float,
    random_frac: float,
) -> np.ndarray:
    """Generate deduped integer-weight candidates."""
    if not pool:
        raise ValueError("empty checkpoint pool")
    preferred_counts = np.asarray(seeds, dtype=np.int16) if seeds else np.zeros((0, len(pool)), dtype=np.int16)
    preferred = np.flatnonzero(preferred_counts.sum(axis=0) > 0) if preferred_counts.size else np.arange(len(pool))
    out: list[tuple[int, ...]] = []
    seen: set[tuple[int, ...]] = set()

    def add(counts: np.ndarray) -> None:
        counts = _repair_counts(
            counts,
            rng,
            max_weight=max_weight,
            min_members=min_members,
            max_members=max_members,
            preferred=preferred,
        )
        key = tuple(int(x) for x in counts)
        if sum(key) <= 0 or key in seen:
            return
        seen.add(key)
        out.append(key)

    for seed in seeds:
        add(np.asarray(seed, dtype=np.int16))

    target = max(int(n_candidates), len(out))
    max_attempts = max(10_000, target * 30)
    attempts = 0
    while len(out) < target and attempts < max_attempts:
        attempts += 1
        if seeds and rng.random() > float(random_frac):
            base = np.asarray(seeds[int(rng.integers(0, len(seeds)))], dtype=np.int16).copy()
            mask = rng.random(len(pool)) < float(mutation_prob)
            deltas = rng.choice(np.array([-1, 1], dtype=np.int16), size=len(pool))
            base[mask] += deltas[mask]
            if rng.random() < 0.35:
                base[int(rng.integers(0, len(pool)))] += 1
            if rng.random() < 0.25:
                nz = np.flatnonzero(base > 0)
                if nz.size:
                    base[int(rng.choice(nz))] -= 1
            add(base)
        else:
            counts = np.zeros(len(pool), dtype=np.int16)
            k = int(rng.integers(int(min_members), int(max_members) + 1))
            draws = rng.choice(len(pool), size=k, replace=True)
            for idx in draws:
                if counts[int(idx)] < int(max_weight):
                    counts[int(idx)] += 1
            add(counts)
    return np.asarray(out, dtype=np.int16)


def _build_obs_flat(
    features: torch.Tensor,
    prices: torch.Tensor,
    step: int,
    cash: torch.Tensor,
    pos_sym: torch.Tensor,
    pos_qty: torch.Tensor,
    pos_entry: torch.Tensor,
    hold_steps: torch.Tensor,
    window_days: int,
) -> torch.Tensor:
    C, N = cash.shape
    _, _, S, F = features.shape
    base = int(S) * int(F)
    obs_dim = base + 5 + int(S)
    device = features.device
    obs = torch.zeros(C, N, obs_dim, device=device, dtype=torch.float32)
    t_obs = max(0, int(step) - 1)
    obs[:, :, :base] = features[:, t_obs].reshape(N, base).unsqueeze(0)

    held = pos_sym >= 0
    price_at_tobs = prices[:, t_obs, :, _P_CLOSE].unsqueeze(0).expand(C, N, S)
    sym_safe = pos_sym.clamp_min(0).long()
    held_price = price_at_tobs.gather(2, sym_safe.unsqueeze(-1)).squeeze(-1)
    held_price = torch.where(held, held_price, torch.zeros_like(held_price))
    pos_val = torch.where(held, pos_qty * held_price, torch.zeros_like(held_price))
    unreal = torch.where(held, pos_qty * (held_price - pos_entry), torch.zeros_like(held_price))
    denom = max(abs(float(INITIAL_CASH)), 1e-12)
    obs[:, :, base + 0] = cash / denom
    obs[:, :, base + 1] = pos_val / denom
    obs[:, :, base + 2] = unreal / denom
    obs[:, :, base + 3] = hold_steps.to(torch.float32) / max(int(window_days), 1)
    obs[:, :, base + 4] = float(step) / max(int(window_days), 1)

    obs_flat = obs.reshape(C * N, obs_dim)
    held_flat = held.reshape(-1)
    if bool(held_flat.any().item()):
        rows = torch.arange(C * N, device=device)
        syms = sym_safe.reshape(-1)
        obs_flat[rows[held_flat], base + 5 + syms[held_flat]] = 1.0
    return obs_flat


def _argmax_long_only(logits: torch.Tensor, *, num_symbols: int, per_symbol_actions: int) -> torch.Tensor:
    side_block = int(num_symbols) * int(per_symbol_actions)
    logits = logits.clone()
    logits[:, 1 + side_block :] = float("-inf")
    return logits.argmax(dim=-1)


def evaluate_weight_batch(
    *,
    weights_np: np.ndarray,
    stacked: StackedEnsemble,
    prices: torch.Tensor,
    features: torch.Tensor,
    tradable: torch.Tensor,
    num_symbols: int,
    per_symbol_actions: int,
    window_days: int,
    fill_buffer_bps: float,
    max_leverage: float,
    fee_rate: float,
    slippage_bps: float,
) -> tuple[np.ndarray, np.ndarray]:
    device = prices.device
    weights = torch.as_tensor(weights_np, device=device, dtype=torch.float32)
    weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(1.0)
    C = int(weights.shape[0])
    N = int(prices.shape[0])
    S = int(num_symbols)
    init_cash = float(INITIAL_CASH)
    cash = torch.full((C, N), init_cash, device=device, dtype=torch.float32)
    pos_sym = torch.full((C, N), -1, device=device, dtype=torch.int32)
    pos_qty = torch.zeros(C, N, device=device, dtype=torch.float32)
    pos_entry = torch.zeros(C, N, device=device, dtype=torch.float32)
    hold_steps = torch.zeros(C, N, device=device, dtype=torch.int32)
    peak_equity = torch.full((C, N), init_cash, device=device, dtype=torch.float32)
    max_dd = torch.zeros(C, N, device=device, dtype=torch.float32)
    action_buf = torch.zeros(C, N, 3, device=device, dtype=torch.int32)
    buf_count = 0
    rows_n = torch.arange(N, device=device)
    fill_buffer_frac = max(0.0, float(fill_buffer_bps)) / 10_000.0
    effective_fee = float(fee_rate) + max(0.0, float(slippage_bps)) / 10_000.0

    for step in range(int(window_days)):
        obs = _build_obs_flat(
            features,
            prices,
            step,
            cash,
            pos_sym,
            pos_qty,
            pos_entry,
            hold_steps,
            int(window_days),
        )
        with torch.no_grad():
            logits_all = stacked.forward(obs)  # [M, C*N, A]
            probs_all = torch.softmax(logits_all, dim=-1).reshape(
                int(weights.shape[1]), C, N, int(logits_all.shape[-1])
            )
            probs = torch.einsum("cm,mcna->cna", weights, probs_all)
        action_now = _argmax_long_only(
            torch.log(probs.reshape(C * N, -1).clamp_min(1e-8)),
            num_symbols=S,
            per_symbol_actions=int(per_symbol_actions),
        ).reshape(C, N).to(torch.int32)

        slot = buf_count % 3
        action_buf[:, :, slot] = action_now
        buf_count += 1
        if buf_count <= 2:
            action = torch.zeros(C, N, device=device, dtype=torch.int32)
        else:
            action = action_buf[:, :, (buf_count - 3) % 3]

        held = pos_sym >= 0
        price_cur_all = prices[:, step, :, _P_CLOSE].unsqueeze(0).expand(C, N, S)
        price_cur = price_cur_all.gather(2, pos_sym.clamp_min(0).long().unsqueeze(-1)).squeeze(-1)
        price_cur = torch.where(held, price_cur, torch.zeros_like(price_cur))

        pos_sym_pre = pos_sym.clone()
        was_held = pos_sym_pre >= 0
        flat_mask = action == 0
        long_mask = (action >= 1) & (action <= S)
        target_sym = (action - 1).clamp_min(0).long()
        same_sym = long_mask & was_held & (target_sym == pos_sym_pre.long())

        trad_t = tradable[:, step, :].unsqueeze(0).expand(C, N, S)
        cur_tradable = trad_t.gather(2, pos_sym_pre.clamp_min(0).long().unsqueeze(-1)).squeeze(-1)
        cur_tradable = torch.where(was_held, cur_tradable, torch.ones_like(cur_tradable))
        target_tradable = trad_t.gather(2, target_sym.unsqueeze(-1)).squeeze(-1)

        flat_close = flat_mask & was_held & cur_tradable
        switch_close = long_mask & was_held & ~same_sym & cur_tradable & target_tradable
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
        close_t = prices[rows_n, step, :, _P_CLOSE].unsqueeze(0).expand(C, N, S)
        low_t = prices[rows_n, step, :, _P_LOW].unsqueeze(0).expand(C, N, S)
        close_tgt = close_t.gather(2, target_sym.unsqueeze(-1)).squeeze(-1)
        low_tgt = low_t.gather(2, target_sym.unsqueeze(-1)).squeeze(-1)
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
        price_new_all = prices[:, t_new, :, _P_CLOSE].unsqueeze(0).expand(C, N, S)
        price_new = price_new_all.gather(2, pos_sym.clamp_min(0).long().unsqueeze(-1)).squeeze(-1)
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
    if bool(held_final.any().item()):
        t_last = int(prices.size(1)) - 1
        price_end_all = prices[:, t_last, :, _P_CLOSE].unsqueeze(0).expand(C, N, S)
        price_end = price_end_all.gather(2, pos_sym.clamp_min(0).long().unsqueeze(-1)).squeeze(-1)
        proceeds_final = pos_qty * price_end * (1.0 - effective_fee)
        cash = torch.where(held_final, cash + proceeds_final, cash)

    total_return = (cash / init_cash) - 1.0
    return (
        total_return.detach().to(torch.float64).cpu().numpy(),
        max_dd.detach().to(torch.float64).cpu().numpy(),
    )


def _summarize_batch(
    *,
    weights: np.ndarray,
    pool: Sequence[Path],
    total_returns: np.ndarray,
    max_drawdowns: np.ndarray,
    window_days: int,
    neg_penalty: float,
    dd_penalty: float,
    start_index: int,
) -> list[dict]:
    med_total = np.percentile(total_returns, 50, axis=1)
    p10_total = np.percentile(total_returns, 10, axis=1)
    p90_total = np.percentile(total_returns, 90, axis=1)
    med_dd = np.percentile(max_drawdowns, 50, axis=1)
    max_dd = np.max(max_drawdowns, axis=1)
    n_neg = np.sum(total_returns < 0.0, axis=1).astype(int)
    out = []
    for i in range(weights.shape[0]):
        med_monthly = _monthly_from_total(float(med_total[i]), int(window_days))
        p10_monthly = _monthly_from_total(float(p10_total[i]), int(window_days))
        score = (
            float(med_monthly)
            + 0.5 * float(p10_monthly)
            - float(neg_penalty) * float(n_neg[i])
            - float(dd_penalty) * float(max_dd[i])
        )
        counts = {pool[j].stem: int(weights[i, j]) for j in range(weights.shape[1]) if int(weights[i, j]) > 0}
        members = [str(pool[j]) for j in range(weights.shape[1]) for _ in range(int(weights[i, j]))]
        out.append(
            {
                "candidate_index": int(start_index + i),
                "score": float(score),
                "counts": counts,
                "members": members,
                "ensemble_size": int(weights[i].sum()),
                "median_total_return": float(med_total[i]),
                "p10_total_return": float(p10_total[i]),
                "p90_total_return": float(p90_total[i]),
                "median_monthly_return": float(med_monthly),
                "p10_monthly_return": float(p10_monthly),
                "max_drawdown": float(max_dd[i]),
                "median_drawdown": float(med_dd[i]),
                "n_neg": int(n_neg[i]),
                "n_windows": int(total_returns.shape[1]),
            }
        )
    return out


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--val-data", default="pufferlib_market/data/screened32_single_offset_val_full.bin")
    parser.add_argument("--checkpoint-root", default="pufferlib_market/prod_ensemble_screened32")
    parser.add_argument("--include-checkpoints", default=None)
    parser.add_argument("--seed-artifact", action="append", default=[])
    parser.add_argument("--seed-top-k", type=int, default=25)
    parser.add_argument("--candidates", type=int, default=4096)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=20260430)
    parser.add_argument("--min-members", type=int, default=8)
    parser.add_argument("--max-members", type=int, default=18)
    parser.add_argument("--max-weight", type=int, default=3)
    parser.add_argument("--mutation-prob", type=float, default=0.18)
    parser.add_argument("--random-frac", type=float, default=0.20)
    parser.add_argument("--window-days", type=int, default=100)
    parser.add_argument("--max-windows", type=int, default=None)
    parser.add_argument("--fill-buffer-bps", type=float, default=5.0)
    parser.add_argument("--slippage-bps", type=float, default=20.0)
    parser.add_argument("--leverage", type=float, default=2.0)
    parser.add_argument("--fee-rate", type=float, default=0.001)
    parser.add_argument("--neg-penalty", type=float, default=0.002)
    parser.add_argument("--dd-penalty", type=float, default=0.15)
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out", default="analysis/screened32_gpu_batch_weight_search/search.json")
    return parser


def _finite_nonnegative(value: float, name: str) -> str | None:
    if not math.isfinite(float(value)) or float(value) < 0.0:
        return f"{name} must be finite and non-negative"
    return None


def _finite_positive(value: float, name: str) -> str | None:
    if not math.isfinite(float(value)) or float(value) <= 0.0:
        return f"{name} must be finite and positive"
    return None


def validate_args(args: argparse.Namespace) -> list[str]:
    failures: list[str] = []
    for attr in ("fill_buffer_bps", "slippage_bps", "fee_rate", "neg_penalty", "dd_penalty"):
        failure = _finite_nonnegative(getattr(args, attr), attr)
        if failure is not None:
            failures.append(failure)
    for attr in ("leverage",):
        failure = _finite_positive(getattr(args, attr), attr)
        if failure is not None:
            failures.append(failure)
    for attr in (
        "seed_top_k",
        "candidates",
        "batch_size",
        "min_members",
        "max_members",
        "max_weight",
        "window_days",
        "top_k",
    ):
        if int(getattr(args, attr)) <= 0:
            failures.append(f"{attr} must be positive")
    if args.max_windows is not None and int(args.max_windows) <= 0:
        failures.append("max_windows must be positive when provided")
    if not 0.0 <= float(args.mutation_prob) <= 1.0:
        failures.append("mutation_prob must be between 0 and 1")
    if not 0.0 <= float(args.random_frac) <= 1.0:
        failures.append("random_frac must be between 0 and 1")
    if int(args.max_members) < int(args.min_members):
        failures.append("max_members must be >= min_members")
    return failures


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    validation_failures = validate_args(args)
    if validation_failures:
        for failure in validation_failures:
            print(f"gpu_batch_screened32_weight_search: {failure}", file=sys.stderr)
        return 2
    if not torch.cuda.is_available() and str(args.device).startswith("cuda"):
        print("gpu_batch_screened32_weight_search: CUDA is unavailable", file=sys.stderr)
        return 2

    include = None
    if args.include_checkpoints:
        include = [x.strip() for x in str(args.include_checkpoints).split(",") if x.strip()]
    try:
        pool = _checkpoint_pool(_resolve_repo_path(args.checkpoint_root), include)
    except ValueError as exc:
        print(f"gpu_batch_screened32_weight_search: {exc}", file=sys.stderr)
        return 2
    if not pool:
        print("gpu_batch_screened32_weight_search: empty checkpoint pool", file=sys.stderr)
        return 2
    if int(args.min_members) > int(args.max_weight) * len(pool):
        print(
            "gpu_batch_screened32_weight_search: min_members exceeds maximum possible weighted pool size",
            file=sys.stderr,
        )
        return 2

    val_path = _resolve_repo_path(args.val_data)
    data = read_mktd(val_path)
    window_len = int(args.window_days) + 1
    if window_len > int(data.num_timesteps):
        print("gpu_batch_screened32_weight_search: window longer than val data", file=sys.stderr)
        return 2
    starts = list(range(int(data.num_timesteps) - window_len + 1))
    if args.max_windows is not None:
        starts = starts[: max(1, int(args.max_windows))]

    rng = np.random.default_rng(int(args.seed))
    seeds = _load_seed_counts(pool, args.seed_artifact, top_per_artifact=int(args.seed_top_k))
    weights = make_population(
        pool=pool,
        seeds=seeds,
        n_candidates=int(args.candidates),
        rng=rng,
        min_members=int(args.min_members),
        max_members=int(args.max_members),
        max_weight=int(args.max_weight),
        mutation_prob=float(args.mutation_prob),
        random_frac=float(args.random_frac),
    )
    print(
        f"[batch-search] pool={len(pool)} seeds={len(seeds)} candidates={len(weights)} "
        f"batch={int(args.batch_size)} windows={len(starts)}"
    )

    device = torch.device(str(args.device))
    prices, features, tradable = _stage_windows(data, starts, int(args.window_days), device)
    loaded = [
        load_policy(path, int(data.num_symbols), features_per_sym=int(data.features.shape[2]), device=device)
        for path in pool
    ]
    policies = [lp.policy.eval() for lp in loaded]
    if not can_batch(policies):
        print("gpu_batch_screened32_weight_search: policies are not stack-compatible", file=sys.stderr)
        return 2
    head = loaded[0]
    alloc_bins = int(head.action_allocation_bins)
    level_bins = int(head.action_level_bins)
    action_max_offset_bps = float(head.action_max_offset_bps)
    if alloc_bins != 1 or level_bins != 1 or action_max_offset_bps != 0.0:
        print("gpu_batch_screened32_weight_search: unsupported action head for GPU path", file=sys.stderr)
        return 2
    stacked = StackedEnsemble.from_policies(policies, device)
    per_symbol_actions = max(1, alloc_bins) * max(1, level_bins)

    results: list[dict] = []
    for start in range(0, len(weights), int(args.batch_size)):
        chunk = weights[start : start + int(args.batch_size)]
        total_returns, max_drawdowns = evaluate_weight_batch(
            weights_np=chunk,
            stacked=stacked,
            prices=prices,
            features=features,
            tradable=tradable,
            num_symbols=int(data.num_symbols),
            per_symbol_actions=int(per_symbol_actions),
            window_days=int(args.window_days),
            fill_buffer_bps=float(args.fill_buffer_bps),
            max_leverage=float(args.leverage),
            fee_rate=float(args.fee_rate),
            slippage_bps=float(args.slippage_bps),
        )
        batch_results = _summarize_batch(
            weights=chunk,
            pool=pool,
            total_returns=total_returns,
            max_drawdowns=max_drawdowns,
            window_days=int(args.window_days),
            neg_penalty=float(args.neg_penalty),
            dd_penalty=float(args.dd_penalty),
            start_index=start,
        )
        results.extend(batch_results)
        best = max(batch_results, key=lambda x: float(x["score"]))
        print(
            f"[batch-search] {min(start + len(chunk), len(weights)):5d}/{len(weights)} "
            f"best_batch={best['median_monthly_return'] * 100:+6.2f}% "
            f"p10={best['p10_monthly_return'] * 100:+6.2f}% "
            f"neg={best['n_neg']:3d}/{best['n_windows']} "
            f"score={best['score']:+.4f}",
            flush=True,
        )

    results.sort(key=lambda x: float(x["score"]), reverse=True)
    payload = {
        "val_data": str(val_path),
        "checkpoint_root": str(_resolve_repo_path(args.checkpoint_root)),
        "pool": [str(p) for p in pool],
        "seed_artifact": list(args.seed_artifact),
        "seed": int(args.seed),
        "window_days": int(args.window_days),
        "n_windows": len(starts),
        "starts": starts,
        "fill_buffer_bps": float(args.fill_buffer_bps),
        "slippage_bps": float(args.slippage_bps),
        "leverage": float(args.leverage),
        "fee_rate": float(args.fee_rate),
        "short_borrow_apr": 0.0,
        "decision_lag": 2,
        "population": {
            "candidates_requested": int(args.candidates),
            "candidates_evaluated": len(weights),
            "batch_size": int(args.batch_size),
            "min_members": int(args.min_members),
            "max_members": int(args.max_members),
            "max_weight": int(args.max_weight),
            "mutation_prob": float(args.mutation_prob),
            "random_frac": float(args.random_frac),
        },
        "ranking": {
            "score": "median_monthly + 0.5*p10_monthly - neg_penalty*n_neg - dd_penalty*max_drawdown",
            "neg_penalty": float(args.neg_penalty),
            "dd_penalty": float(args.dd_penalty),
        },
        "best": results[0] if results else None,
        "results": results,
    }
    out_path = Path(args.out)
    if not out_path.is_absolute():
        out_path = REPO / out_path
    write_json_atomic(out_path, payload)
    print("\nTop candidates:")
    for item in results[: max(1, int(args.top_k))]:
        print(
            f"{item['score']:+.4f}  {item['median_monthly_return'] * 100:+6.2f}%/mo "
            f"p10={item['p10_monthly_return'] * 100:+6.2f}% "
            f"neg={item['n_neg']:3d}/{item['n_windows']} "
            f"dd={item['max_drawdown'] * 100:5.1f}% "
            f"size={item['ensemble_size']:2d} idx={item['candidate_index']}"
        )
    print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
