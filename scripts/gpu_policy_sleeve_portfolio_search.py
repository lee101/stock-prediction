#!/usr/bin/env python3
"""GPU search for multi-sleeve policy portfolios.

Unlike flat-score portfolio packing, this keeps each checkpoint inside the
state/action semantics it was trained for: every policy trades an independent
single-position sleeve with its own cash, held symbol, entry, and hold counter.
The script then searches capital weights over those sleeve equity curves.

This is a research screen only. Promising candidates still need the normal
realism gate before any production consideration.
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

from scripts.gpu_portfolio_pack_screen import _monthly_from_total  # noqa: E402
from src.daily_stock_defaults import DEFAULT_CHECKPOINT, DEFAULT_EXTRA_CHECKPOINTS  # noqa: E402


def _resolve(path: str | Path) -> Path:
    p = Path(path)
    if not p.is_absolute():
        p = REPO / p
    return p.resolve()


def _label(path: Path) -> str:
    return path.stem


def _load_pool(args: argparse.Namespace) -> list[Path]:
    if args.checkpoints:
        paths = [_resolve(p) for p in args.checkpoints]
    else:
        root = _resolve(args.checkpoint_root)
        paths = sorted(root.glob("*.pt"))
    if not paths:
        raise FileNotFoundError("empty checkpoint pool")
    missing = [str(p) for p in paths if not p.exists()]
    if missing:
        raise FileNotFoundError(f"missing checkpoints: {missing}")
    return paths


def _artifact_counts(path: str | Path, candidate_index: int | None, pool: Sequence[Path]) -> np.ndarray:
    data = json.loads(_resolve(path).read_text(encoding="utf-8"))
    results = list(data.get("results", []))
    if not results:
        raise ValueError(f"artifact has no results: {path}")
    if candidate_index is None:
        item = results[0]
    else:
        matches = [
            row
            for row in results
            if int(row.get("candidate_index", row.get("trial", -1))) == int(candidate_index)
        ]
        if not matches:
            raise ValueError(f"candidate_index={candidate_index} not found in {path}")
        item = matches[0]
    members = [Path(member).name for member in item.get("members", [])]
    counts = np.zeros(len(pool), dtype=np.float32)
    name_to_idx = {p.name: i for i, p in enumerate(pool)}
    for member in members:
        if member in name_to_idx:
            counts[name_to_idx[member]] += 1.0
    return counts


def _build_obs_sleeves(
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
    m, n = cash.shape
    _, _, s, f = features.shape
    base = int(s) * int(f)
    obs_dim = base + 5 + int(s)
    obs = torch.zeros(m, n, obs_dim, device=features.device, dtype=torch.float32)
    t_obs = max(0, int(step) - 1)
    obs[:, :, :base] = features[:, t_obs].reshape(n, base).unsqueeze(0)

    held = pos_sym >= 0
    sym_safe = pos_sym.clamp_min(0).long()
    price_at_tobs = prices[:, t_obs, :, _P_CLOSE].unsqueeze(0).expand(m, n, s)
    held_price = price_at_tobs.gather(2, sym_safe.unsqueeze(-1)).squeeze(-1)
    held_price = torch.where(held, held_price, torch.zeros_like(held_price))
    pos_val = torch.where(held, pos_qty * held_price, torch.zeros_like(held_price))
    unreal = torch.where(held, pos_qty * (held_price - pos_entry), torch.zeros_like(held_price))

    obs[:, :, base + 0] = cash / float(INITIAL_CASH)
    obs[:, :, base + 1] = pos_val / float(INITIAL_CASH)
    obs[:, :, base + 2] = unreal / float(INITIAL_CASH)
    obs[:, :, base + 3] = hold_steps.to(torch.float32) / max(int(window_days), 1)
    obs[:, :, base + 4] = float(step) / max(int(window_days), 1)

    flat = obs.reshape(m * n, obs_dim)
    held_flat = held.reshape(-1)
    if bool(held_flat.any().item()):
        rows = torch.arange(m * n, device=features.device)
        syms = sym_safe.reshape(-1)
        flat[rows[held_flat], base + 5 + syms[held_flat]] = 1.0
    return flat


def simulate_sleeves(
    *,
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
    margin_apr: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    device = prices.device
    m = int(stacked.n_members)
    n = int(prices.shape[0])
    s = int(num_symbols)
    cash = torch.full((m, n), float(INITIAL_CASH), device=device, dtype=torch.float32)
    pos_sym = torch.full((m, n), -1, device=device, dtype=torch.int32)
    pos_qty = torch.zeros(m, n, device=device, dtype=torch.float32)
    pos_entry = torch.zeros(m, n, device=device, dtype=torch.float32)
    hold_steps = torch.zeros(m, n, device=device, dtype=torch.int32)
    equity_curve = torch.empty(m, n, int(window_days) + 1, device=device, dtype=torch.float32)
    equity_curve[:, :, 0] = float(INITIAL_CASH)
    turnover = torch.zeros(m, n, device=device, dtype=torch.float32)
    action_buf = torch.zeros(m, n, 3, device=device, dtype=torch.int32)
    buf_count = 0
    rows_n = torch.arange(n, device=device)
    rows_m = torch.arange(m, device=device)
    obs_offsets = rows_m.view(m, 1) * n + rows_n.view(1, n)
    fill_buffer_frac = max(0.0, float(fill_buffer_bps)) / 10_000.0
    effective_fee = float(fee_rate) + max(0.0, float(slippage_bps)) / 10_000.0
    margin_rate = max(0.0, float(margin_apr)) / 365.0

    for step in range(int(window_days)):
        obs = _build_obs_sleeves(
            features,
            prices,
            int(step),
            cash,
            pos_sym,
            pos_qty,
            pos_entry,
            hold_steps,
            int(window_days),
        )
        with torch.no_grad():
            logits_all = stacked.forward(obs)
            logits = logits_all[rows_m.view(m, 1), obs_offsets, :]
        logits = logits.clone()
        logits[:, :, 1 + s * int(per_symbol_actions) :] = float("-inf")
        action_now = logits.argmax(dim=-1).to(torch.int32)

        action_buf[:, :, buf_count % 3] = action_now
        buf_count += 1
        if buf_count <= 2:
            action = torch.zeros(m, n, device=device, dtype=torch.int32)
        else:
            action = action_buf[:, :, (buf_count - 3) % 3]

        held = pos_sym >= 0
        price_cur_all = prices[:, step, :, _P_CLOSE].unsqueeze(0).expand(m, n, s)
        price_cur = price_cur_all.gather(2, pos_sym.clamp_min(0).long().unsqueeze(-1)).squeeze(-1)
        price_cur = torch.where(held, price_cur, torch.zeros_like(price_cur))

        pos_sym_pre = pos_sym.clone()
        was_held = pos_sym_pre >= 0
        flat_mask = action == 0
        long_mask = (action >= 1) & (action <= s)
        target_sym = (action - 1).clamp_min(0).long()
        same_sym = long_mask & was_held & (target_sym == pos_sym_pre.long())

        trad_t = tradable[:, step, :].unsqueeze(0).expand(m, n, s)
        cur_tradable = trad_t.gather(2, pos_sym_pre.clamp_min(0).long().unsqueeze(-1)).squeeze(-1)
        cur_tradable = torch.where(was_held, cur_tradable, torch.ones_like(cur_tradable))
        target_tradable = trad_t.gather(2, target_sym.unsqueeze(-1)).squeeze(-1)

        flat_close = flat_mask & was_held & cur_tradable
        switch_close = long_mask & was_held & ~same_sym & cur_tradable & target_tradable
        flat_hold = flat_mask & was_held & ~cur_tradable
        switch_blocked_hold = long_mask & was_held & ~same_sym & (~cur_tradable | ~target_tradable)
        closing = flat_close | switch_close

        proceeds = pos_qty * price_cur * (1.0 - effective_fee)
        turnover = turnover + (
            torch.where(closing, pos_qty * price_cur, torch.zeros_like(price_cur)) / float(INITIAL_CASH)
        )
        cash = torch.where(closing, cash + proceeds, cash)
        pos_sym = torch.where(closing, torch.full_like(pos_sym, -1), pos_sym)
        pos_qty = torch.where(closing, torch.zeros_like(pos_qty), pos_qty)
        pos_entry = torch.where(closing, torch.zeros_like(pos_entry), pos_entry)
        hold_steps = torch.where(flat_close, torch.zeros_like(hold_steps), hold_steps)

        want_open = (long_mask & ~was_held & target_tradable) | switch_close
        close_t = prices[rows_n, step, :, _P_CLOSE].unsqueeze(0).expand(m, n, s)
        low_t = prices[rows_n, step, :, _P_LOW].unsqueeze(0).expand(m, n, s)
        close_tgt = close_t.gather(2, target_sym.unsqueeze(-1)).squeeze(-1)
        low_tgt = low_t.gather(2, target_sym.unsqueeze(-1)).squeeze(-1)
        fillable = low_tgt <= close_tgt * (1.0 - fill_buffer_frac)
        denom = close_tgt * (1.0 + effective_fee)
        qty_new = torch.where(denom > 0, (cash * float(max_leverage)) / denom, torch.zeros_like(denom))
        cost_new = qty_new * denom
        can_open = want_open & fillable & (close_tgt > 0) & (cash > 0) & (qty_new > 0) & (cost_new > 0)
        turnover = turnover + (
            torch.where(can_open, qty_new * close_tgt, torch.zeros_like(close_tgt)) / float(INITIAL_CASH)
        )
        cash = torch.where(can_open, cash - cost_new, cash)
        pos_sym = torch.where(can_open, target_sym.to(torch.int32), pos_sym)
        pos_qty = torch.where(can_open, qty_new, pos_qty)
        pos_entry = torch.where(can_open, close_tgt, pos_entry)
        hold_steps = torch.where(can_open, torch.zeros_like(hold_steps), hold_steps)

        carry_hold = same_sym | flat_hold | switch_blocked_hold
        hold_steps = torch.where(carry_hold, hold_steps + 1, hold_steps)
        if margin_rate > 0.0:
            cash = cash - torch.clamp(-cash, min=0.0) * margin_rate

        close_new = prices[:, min(step + 1, int(prices.size(1)) - 1), :, _P_CLOSE].unsqueeze(0).expand(m, n, s)
        price_new = close_new.gather(2, pos_sym.clamp_min(0).long().unsqueeze(-1)).squeeze(-1)
        held2 = pos_sym >= 0
        equity = torch.where(held2, cash + pos_qty * price_new, cash)
        equity_curve[:, :, step + 1] = equity

    held_final = pos_sym >= 0
    if bool(held_final.any().item()):
        price_end_all = prices[:, -1, :, _P_CLOSE].unsqueeze(0).expand(m, n, s)
        price_end = price_end_all.gather(2, pos_sym.clamp_min(0).long().unsqueeze(-1)).squeeze(-1)
        proceeds_final = pos_qty * price_end * (1.0 - effective_fee)
        cash = torch.where(held_final, cash + proceeds_final, cash)
        equity_curve[:, :, -1] = cash
    return equity_curve, turnover


def _default_counts(pool: Sequence[Path]) -> np.ndarray:
    wanted = [Path(DEFAULT_CHECKPOINT).name, *(Path(p).name for p in DEFAULT_EXTRA_CHECKPOINTS)]
    idx = {p.name: i for i, p in enumerate(pool)}
    counts = np.zeros(len(pool), dtype=np.float32)
    for name in wanted:
        if name in idx:
            counts[idx[name]] += 1.0
    return counts


def build_candidate_weights(
    *,
    pool: Sequence[Path],
    sleeve_returns: np.ndarray,
    sleeve_p10: np.ndarray,
    random_candidates: int,
    min_members: int,
    max_members: int,
    max_count: int,
    seed: int,
    artifact_counts: Sequence[np.ndarray],
) -> np.ndarray:
    m = len(pool)
    rows: list[np.ndarray] = []
    rows.append(np.ones(m, dtype=np.float32))
    rows.extend(np.eye(m, dtype=np.float32))
    default = _default_counts(pool)
    if default.sum() > 0:
        rows.append(default)
    rows.extend(np.asarray(c, dtype=np.float32) for c in artifact_counts if np.asarray(c).sum() > 0)
    for metric in (sleeve_returns, sleeve_p10):
        order = np.argsort(metric)[::-1]
        for k in range(max(1, min_members), min(max_members, m) + 1):
            counts = np.zeros(m, dtype=np.float32)
            counts[order[:k]] = 1.0
            rows.append(counts)
    rng = np.random.default_rng(int(seed))
    for _ in range(max(0, int(random_candidates))):
        k = int(rng.integers(max(1, min_members), min(max_members, m) + 1))
        idx = rng.choice(m, size=k, replace=False)
        counts = np.zeros(m, dtype=np.float32)
        counts[idx] = rng.integers(1, max(1, int(max_count)) + 1, size=k).astype(np.float32)
        rows.append(counts)
    mat = np.stack(rows)
    mat = mat[mat.sum(axis=1) > 0]
    mat = mat / mat.sum(axis=1, keepdims=True)
    rounded = np.round(mat, 6)
    _, unique_idx = np.unique(rounded, axis=0, return_index=True)
    return mat[np.sort(unique_idx)]


def _eval_weight_candidates(
    *,
    weights: np.ndarray,
    equity_curve: torch.Tensor,
    turnover: torch.Tensor,
    labels: Sequence[str],
    window_days: int,
    batch_size: int,
    neg_penalty: float,
    dd_penalty: float,
    turnover_penalty: float,
    top_k: int,
) -> list[dict]:
    device = equity_curve.device
    init_cash = float(INITIAL_CASH)
    best: list[dict] = []
    for start in range(0, int(weights.shape[0]), int(batch_size)):
        w_np = weights[start : start + int(batch_size)]
        w = torch.as_tensor(w_np, device=device, dtype=torch.float32)
        port = torch.einsum("cm,mnt->cnt", w, equity_curve)
        final = port[:, :, -1]
        total_returns = (final / init_cash) - 1.0
        peak = torch.cummax(port, dim=2).values
        dd = torch.where(peak > 0, (peak - port) / peak.clamp_min(1e-12), torch.zeros_like(port))
        max_dd = dd.max(dim=2).values
        port_turn = torch.einsum("cm,mn->cn", w, turnover)
        tr_np = total_returns.detach().to(torch.float64).cpu().numpy()
        dd_np = max_dd.detach().to(torch.float64).cpu().numpy()
        turn_np = port_turn.detach().to(torch.float64).cpu().numpy()
        for local_i in range(w_np.shape[0]):
            med_total = float(np.percentile(tr_np[local_i], 50))
            p10_total = float(np.percentile(tr_np[local_i], 10))
            med_monthly = _monthly_from_total(med_total, int(window_days))
            p10_monthly = _monthly_from_total(p10_total, int(window_days))
            n_neg = int(np.sum(tr_np[local_i] < 0.0))
            max_dd_i = float(np.max(dd_np[local_i]))
            med_turn = float(np.percentile(turn_np[local_i], 50))
            score = (
                med_monthly
                + 0.5 * p10_monthly
                - float(neg_penalty) * n_neg
                - float(dd_penalty) * max_dd_i
                - float(turnover_penalty) * med_turn
            )
            nonzero = [(labels[j], float(w_np[local_i, j])) for j in np.flatnonzero(w_np[local_i] > 1e-7)]
            best.append(
                {
                    "candidate_index": start + local_i,
                    "score": float(score),
                    "median_total_return": med_total,
                    "p10_total_return": p10_total,
                    "median_monthly_return": float(med_monthly),
                    "p10_monthly_return": float(p10_monthly),
                    "max_drawdown": max_dd_i,
                    "median_turnover_x_initial": med_turn,
                    "n_neg": n_neg,
                    "n_windows": int(tr_np.shape[1]),
                    "members": nonzero,
                }
            )
        best.sort(key=lambda item: float(item["score"]), reverse=True)
        del best[int(top_k) * 4 :]
    best.sort(key=lambda item: float(item["score"]), reverse=True)
    return best[: int(top_k)]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--val-data", default="pufferlib_market/data/screened32_single_offset_val_full.bin")
    parser.add_argument("--checkpoint-root", default="pufferlib_market/prod_ensemble_screened32")
    parser.add_argument("--checkpoints", nargs="*", default=None)
    parser.add_argument("--seed-artifact", action="append", default=[])
    parser.add_argument("--candidate-index", action="append", type=int, default=[])
    parser.add_argument("--window-days", type=int, default=100)
    parser.add_argument("--max-windows", type=int, default=None)
    parser.add_argument("--fill-buffer-bps", type=float, default=5.0)
    parser.add_argument("--slippage-bps", type=float, default=20.0)
    parser.add_argument("--leverage", type=float, default=2.0)
    parser.add_argument("--fee-rate", type=float, default=0.001)
    parser.add_argument("--margin-apr", type=float, default=0.0625)
    parser.add_argument("--random-candidates", type=int, default=32768)
    parser.add_argument("--min-members", type=int, default=2)
    parser.add_argument("--max-members", type=int, default=16)
    parser.add_argument("--max-count", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260430)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--neg-penalty", type=float, default=0.002)
    parser.add_argument("--dd-penalty", type=float, default=0.15)
    parser.add_argument("--turnover-penalty", type=float, default=0.00005)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out", default="analysis/screened32_policy_sleeves/search.json")
    return parser


def _finite_nonnegative(value: float, name: str) -> str | None:
    if not math.isfinite(float(value)) or float(value) < 0.0:
        return f"{name} must be finite and non-negative"
    return None


def _finite_positive(value: float, name: str) -> str | None:
    if not math.isfinite(float(value)) or float(value) <= 0.0:
        return f"{name} must be finite and positive"
    return None


def validate_sleeve_args(args: argparse.Namespace) -> list[str]:
    failures: list[str] = []
    for attr in (
        "fill_buffer_bps",
        "slippage_bps",
        "fee_rate",
        "margin_apr",
        "neg_penalty",
        "dd_penalty",
        "turnover_penalty",
    ):
        failure = _finite_nonnegative(getattr(args, attr), attr)
        if failure is not None:
            failures.append(failure)
    failure = _finite_positive(args.leverage, "leverage")
    if failure is not None:
        failures.append(failure)
    for attr in ("window_days", "min_members", "max_members", "max_count", "batch_size", "top_k"):
        if int(getattr(args, attr)) <= 0:
            failures.append(f"{attr} must be positive")
    if hasattr(args, "top_per_artifact") and int(args.top_per_artifact) <= 0:
        failures.append("top_per_artifact must be positive")
    if int(args.random_candidates) < 0:
        failures.append("random_candidates must be non-negative")
    if args.max_windows is not None and int(args.max_windows) <= 0:
        failures.append("max_windows must be positive when provided")
    for candidate_index in getattr(args, "candidate_index", []) or []:
        if candidate_index is not None and int(candidate_index) < 0:
            failures.append("candidate_index values must be non-negative")
            break
    if int(args.max_members) < int(args.min_members):
        failures.append("max_members must be >= min_members")
    return failures


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    validation_failures = validate_sleeve_args(args)
    if validation_failures:
        for failure in validation_failures:
            print(f"gpu_policy_sleeve_portfolio_search: {failure}", file=sys.stderr)
        return 2
    if not torch.cuda.is_available() and str(args.device).startswith("cuda"):
        print("gpu_policy_sleeve_portfolio_search: CUDA is unavailable", file=sys.stderr)
        return 2
    try:
        pool = _load_pool(args)
        candidate_indexes = list(args.candidate_index)
        if len(candidate_indexes) < len(args.seed_artifact):
            candidate_indexes.extend([None] * (len(args.seed_artifact) - len(candidate_indexes)))
        artifact_counts = [
            _artifact_counts(path, idx, pool)
            for path, idx in zip(args.seed_artifact, candidate_indexes)
        ]
    except (OSError, ValueError) as exc:
        print(f"gpu_policy_sleeve_portfolio_search: {exc}", file=sys.stderr)
        return 2
    if int(args.min_members) > len(pool) * int(args.max_count):
        print("gpu_policy_sleeve_portfolio_search: min_members exceeds maximum possible pool size", file=sys.stderr)
        return 2
    val_path = _resolve(args.val_data)
    data = read_mktd(val_path)
    starts = list(range(int(data.num_timesteps) - int(args.window_days)))
    if args.max_windows is not None:
        starts = starts[: max(1, int(args.max_windows))]
    device = torch.device(str(args.device))
    prices, features, tradable = _stage_windows(data, starts, int(args.window_days), device)
    loaded = [
        load_policy(p, int(data.num_symbols), features_per_sym=int(data.features.shape[2]), device=device)
        for p in pool
    ]
    policies = [lp.policy.eval() for lp in loaded]
    if not can_batch(policies):
        print("gpu_policy_sleeve_portfolio_search: policies are not stack-compatible", file=sys.stderr)
        return 2
    head = loaded[0]
    alloc_bins = int(head.action_allocation_bins)
    level_bins = int(head.action_level_bins)
    action_max_offset_bps = float(head.action_max_offset_bps)
    if alloc_bins != 1 or level_bins != 1 or action_max_offset_bps != 0.0:
        print("gpu_policy_sleeve_portfolio_search: unsupported action grid", file=sys.stderr)
        return 2
    labels = [_label(p) for p in pool]
    stacked = StackedEnsemble.from_policies(policies, device)
    print(
        f"[sleeves] policies={len(pool)} windows={len(starts)} "
        f"slip={float(args.slippage_bps):g}bps lev={float(args.leverage):g}x"
    )
    equity_curve, turnover = simulate_sleeves(
        stacked=stacked,
        prices=prices,
        features=features,
        tradable=tradable,
        num_symbols=int(data.num_symbols),
        per_symbol_actions=max(1, alloc_bins) * max(1, level_bins),
        window_days=int(args.window_days),
        fill_buffer_bps=float(args.fill_buffer_bps),
        max_leverage=float(args.leverage),
        fee_rate=float(args.fee_rate),
        slippage_bps=float(args.slippage_bps),
        margin_apr=float(args.margin_apr),
    )
    sleeve_final = (equity_curve[:, :, -1] / float(INITIAL_CASH)) - 1.0
    sleeve_returns = np.percentile(sleeve_final.detach().cpu().numpy(), 50, axis=1)
    sleeve_p10 = np.percentile(sleeve_final.detach().cpu().numpy(), 10, axis=1)
    weights = build_candidate_weights(
        pool=pool,
        sleeve_returns=sleeve_returns,
        sleeve_p10=sleeve_p10,
        random_candidates=int(args.random_candidates),
        min_members=int(args.min_members),
        max_members=int(args.max_members),
        max_count=int(args.max_count),
        seed=int(args.seed),
        artifact_counts=artifact_counts,
    )
    print(f"[sleeves] candidate_weights={weights.shape[0]}")
    results = _eval_weight_candidates(
        weights=weights,
        equity_curve=equity_curve,
        turnover=turnover,
        labels=labels,
        window_days=int(args.window_days),
        batch_size=int(args.batch_size),
        neg_penalty=float(args.neg_penalty),
        dd_penalty=float(args.dd_penalty),
        turnover_penalty=float(args.turnover_penalty),
        top_k=int(args.top_k),
    )
    payload = {
        "val_data": str(val_path),
        "pool": [str(p) for p in pool],
        "labels": labels,
        "window_days": int(args.window_days),
        "n_windows": len(starts),
        "starts": starts,
        "fill_buffer_bps": float(args.fill_buffer_bps),
        "slippage_bps": float(args.slippage_bps),
        "leverage": float(args.leverage),
        "fee_rate": float(args.fee_rate),
        "margin_apr": float(args.margin_apr),
        "decision_lag": 2,
        "random_candidates": int(args.random_candidates),
        "candidate_weights": int(weights.shape[0]),
        "solo": [
            {
                "label": labels[i],
                "median_monthly_return": float(_monthly_from_total(float(sleeve_returns[i]), int(args.window_days))),
                "p10_monthly_return": float(_monthly_from_total(float(sleeve_p10[i]), int(args.window_days))),
            }
            for i in range(len(labels))
        ],
        "results": results,
        "best": results[0] if results else None,
    }
    out_path = Path(args.out)
    if not out_path.is_absolute():
        out_path = REPO / out_path
    write_json_atomic(out_path, payload)
    print("\nTop sleeve portfolios:")
    for item in results:
        print(
            f"{item['score']:+.4f} med={item['median_monthly_return'] * 100:+6.2f}% "
            f"p10={item['p10_monthly_return'] * 100:+6.2f}% "
            f"neg={item['n_neg']:3d}/{item['n_windows']} "
            f"dd={item['max_drawdown'] * 100:5.1f}% "
            f"turn={item['median_turnover_x_initial']:6.1f}x "
            f"members={len(item['members'])}"
        )
        print("  " + ", ".join(f"{name}:{weight:.3f}" for name, weight in item["members"][:16]))
    print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
