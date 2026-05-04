#!/usr/bin/env python3
"""GPU portfolio search over stateful weighted-ensemble sleeves.

Each sleeve is a full softmax-averaged checkpoint ensemble with its own
single-position state. This preserves the profitable held-state dynamics of
the current simulator while allowing multiple ensemble variants to hold
different symbols at the same time.
"""
from __future__ import annotations

import argparse
import json
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

from scripts.gpu_policy_sleeve_portfolio_search import (  # noqa: E402
    _build_obs_sleeves,
    _eval_weight_candidates,
    build_candidate_weights,
    validate_sleeve_args,
)
from scripts.gpu_portfolio_pack_screen import _monthly_from_total  # noqa: E402
from src.daily_stock_defaults import DEFAULT_CHECKPOINT, DEFAULT_EXTRA_CHECKPOINTS  # noqa: E402


def _resolve(path: str | Path) -> Path:
    p = Path(path)
    if not p.is_absolute():
        p = REPO / p
    return p.resolve()


def _label(path: Path) -> str:
    return path.stem


def _counts_from_names(names: Sequence[str], pool: Sequence[Path]) -> np.ndarray:
    idx = {p.stem: i for i, p in enumerate(pool)}
    counts = np.zeros(len(pool), dtype=np.float32)
    for name in names:
        stem = Path(name).stem
        if stem in idx:
            counts[idx[stem]] += 1.0
    return counts


def _default_counts(pool: Sequence[Path]) -> np.ndarray:
    return _counts_from_names([str(DEFAULT_CHECKPOINT), *(str(p) for p in DEFAULT_EXTRA_CHECKPOINTS)], pool)


def load_ensemble_specs(
    artifact_paths: Sequence[str | Path],
    *,
    top_per_artifact: int,
) -> tuple[list[Path], list[tuple[str, np.ndarray]]]:
    pool_by_name: dict[str, Path] = {}
    raw_specs: list[tuple[str, dict[str, int]]] = []
    for artifact_path in artifact_paths:
        path = _resolve(artifact_path)
        data = json.loads(path.read_text(encoding="utf-8"))
        for p in data.get("pool", []):
            pp = _resolve(p)
            pool_by_name[pp.stem] = pp
        rows = list(data.get("results", []))[: max(1, int(top_per_artifact))]
        for row in rows:
            counts = {str(k): int(v) for k, v in dict(row.get("counts", {})).items() if int(v) > 0}
            if counts:
                raw_specs.append((f"{path.stem}:{row.get('candidate_index', len(raw_specs))}", counts))
    pool = [pool_by_name[name] for name in sorted(pool_by_name)]
    specs: list[tuple[str, np.ndarray]] = []
    default = _default_counts(pool)
    if default.sum() > 0:
        specs.append(("default_prod", default))
    for name, counts_map in raw_specs:
        counts = np.zeros(len(pool), dtype=np.float32)
        idx = {p.stem: i for i, p in enumerate(pool)}
        for label, count in counts_map.items():
            if label in idx:
                counts[idx[label]] += float(count)
        if counts.sum() > 0:
            specs.append((name, counts))
    dedup: dict[tuple[float, ...], tuple[str, np.ndarray]] = {}
    for name, counts in specs:
        weights = counts / counts.sum()
        key = tuple(np.round(weights, 6).tolist())
        dedup.setdefault(key, (name, counts))
    return pool, list(dedup.values())


def simulate_ensemble_sleeves(
    *,
    stacked: StackedEnsemble,
    ensemble_weights: torch.Tensor,
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
    e = int(ensemble_weights.shape[0])
    n = int(prices.shape[0])
    s = int(num_symbols)
    cash = torch.full((e, n), float(INITIAL_CASH), device=device, dtype=torch.float32)
    pos_sym = torch.full((e, n), -1, device=device, dtype=torch.int32)
    pos_qty = torch.zeros(e, n, device=device, dtype=torch.float32)
    pos_entry = torch.zeros(e, n, device=device, dtype=torch.float32)
    hold_steps = torch.zeros(e, n, device=device, dtype=torch.int32)
    equity_curve = torch.empty(e, n, int(window_days) + 1, device=device, dtype=torch.float32)
    equity_curve[:, :, 0] = float(INITIAL_CASH)
    turnover = torch.zeros(e, n, device=device, dtype=torch.float32)
    action_buf = torch.zeros(e, n, 3, device=device, dtype=torch.int32)
    buf_count = 0
    rows_n = torch.arange(n, device=device)
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
            probs_all = torch.softmax(logits_all, dim=-1).reshape(
                int(ensemble_weights.shape[1]), e, n, int(logits_all.shape[-1])
            )
            probs = torch.einsum("em,mena->ena", ensemble_weights, probs_all)
        probs = probs.clone()
        probs[:, :, 1 + s * int(per_symbol_actions) :] = 0.0
        action_now = probs.argmax(dim=-1).to(torch.int32)

        action_buf[:, :, buf_count % 3] = action_now
        buf_count += 1
        if buf_count <= 2:
            action = torch.zeros(e, n, device=device, dtype=torch.int32)
        else:
            action = action_buf[:, :, (buf_count - 3) % 3]

        held = pos_sym >= 0
        price_cur_all = prices[:, step, :, _P_CLOSE].unsqueeze(0).expand(e, n, s)
        price_cur = price_cur_all.gather(2, pos_sym.clamp_min(0).long().unsqueeze(-1)).squeeze(-1)
        price_cur = torch.where(held, price_cur, torch.zeros_like(price_cur))

        pos_sym_pre = pos_sym.clone()
        was_held = pos_sym_pre >= 0
        flat_mask = action == 0
        long_mask = (action >= 1) & (action <= s)
        target_sym = (action - 1).clamp_min(0).long()
        same_sym = long_mask & was_held & (target_sym == pos_sym_pre.long())

        trad_t = tradable[:, step, :].unsqueeze(0).expand(e, n, s)
        cur_tradable = trad_t.gather(2, pos_sym_pre.clamp_min(0).long().unsqueeze(-1)).squeeze(-1)
        cur_tradable = torch.where(was_held, cur_tradable, torch.ones_like(cur_tradable))
        target_tradable = trad_t.gather(2, target_sym.unsqueeze(-1)).squeeze(-1)

        flat_close = flat_mask & was_held & cur_tradable
        switch_close = long_mask & was_held & ~same_sym & cur_tradable & target_tradable
        flat_hold = flat_mask & was_held & ~cur_tradable
        switch_blocked_hold = long_mask & was_held & ~same_sym & (~cur_tradable | ~target_tradable)
        closing = flat_close | switch_close

        turnover = turnover + (
            torch.where(closing, pos_qty * price_cur, torch.zeros_like(price_cur)) / float(INITIAL_CASH)
        )
        cash = torch.where(closing, cash + pos_qty * price_cur * (1.0 - effective_fee), cash)
        pos_sym = torch.where(closing, torch.full_like(pos_sym, -1), pos_sym)
        pos_qty = torch.where(closing, torch.zeros_like(pos_qty), pos_qty)
        pos_entry = torch.where(closing, torch.zeros_like(pos_entry), pos_entry)
        hold_steps = torch.where(flat_close, torch.zeros_like(hold_steps), hold_steps)

        want_open = (long_mask & ~was_held & target_tradable) | switch_close
        close_t = prices[rows_n, step, :, _P_CLOSE].unsqueeze(0).expand(e, n, s)
        low_t = prices[rows_n, step, :, _P_LOW].unsqueeze(0).expand(e, n, s)
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

        hold_steps = torch.where(same_sym | flat_hold | switch_blocked_hold, hold_steps + 1, hold_steps)
        if margin_rate > 0.0:
            cash = cash - torch.clamp(-cash, min=0.0) * margin_rate

        close_new = prices[:, min(step + 1, int(prices.size(1)) - 1), :, _P_CLOSE].unsqueeze(0).expand(e, n, s)
        price_new = close_new.gather(2, pos_sym.clamp_min(0).long().unsqueeze(-1)).squeeze(-1)
        held2 = pos_sym >= 0
        equity_curve[:, :, step + 1] = torch.where(held2, cash + pos_qty * price_new, cash)

    held_final = pos_sym >= 0
    if bool(held_final.any().item()):
        price_end_all = prices[:, -1, :, _P_CLOSE].unsqueeze(0).expand(e, n, s)
        price_end = price_end_all.gather(2, pos_sym.clamp_min(0).long().unsqueeze(-1)).squeeze(-1)
        cash = torch.where(held_final, cash + pos_qty * price_end * (1.0 - effective_fee), cash)
        equity_curve[:, :, -1] = cash
    return equity_curve, turnover


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--val-data", default="pufferlib_market/data/screened32_single_offset_val_full.bin")
    parser.add_argument("--artifact", action="append", required=True)
    parser.add_argument("--top-per-artifact", type=int, default=64)
    parser.add_argument("--window-days", type=int, default=100)
    parser.add_argument("--max-windows", type=int, default=None)
    parser.add_argument("--fill-buffer-bps", type=float, default=5.0)
    parser.add_argument("--slippage-bps", type=float, default=20.0)
    parser.add_argument("--leverage", type=float, default=2.0)
    parser.add_argument("--fee-rate", type=float, default=0.001)
    parser.add_argument("--margin-apr", type=float, default=0.0625)
    parser.add_argument("--random-candidates", type=int, default=32768)
    parser.add_argument("--min-members", type=int, default=2)
    parser.add_argument("--max-members", type=int, default=12)
    parser.add_argument("--max-count", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260430)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--neg-penalty", type=float, default=0.002)
    parser.add_argument("--dd-penalty", type=float, default=0.15)
    parser.add_argument("--turnover-penalty", type=float, default=0.00005)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out", default="analysis/screened32_ensemble_sleeves/search.json")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    validation_failures = validate_sleeve_args(args)
    if validation_failures:
        for failure in validation_failures:
            print(f"gpu_ensemble_sleeve_portfolio_search: {failure}", file=sys.stderr)
        return 2
    if not torch.cuda.is_available() and str(args.device).startswith("cuda"):
        print("gpu_ensemble_sleeve_portfolio_search: CUDA is unavailable", file=sys.stderr)
        return 2
    pool, specs = load_ensemble_specs(args.artifact, top_per_artifact=int(args.top_per_artifact))
    if not pool or not specs:
        print("gpu_ensemble_sleeve_portfolio_search: empty pool/specs", file=sys.stderr)
        return 2
    if int(args.min_members) > len(specs) * int(args.max_count):
        print(
            "gpu_ensemble_sleeve_portfolio_search: min_members exceeds maximum possible sleeve count",
            file=sys.stderr,
        )
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
        print("gpu_ensemble_sleeve_portfolio_search: policies are not stack-compatible", file=sys.stderr)
        return 2
    head = loaded[0]
    alloc_bins = int(head.action_allocation_bins)
    level_bins = int(head.action_level_bins)
    action_max_offset_bps = float(head.action_max_offset_bps)
    if alloc_bins != 1 or level_bins != 1 or action_max_offset_bps != 0.0:
        print("gpu_ensemble_sleeve_portfolio_search: unsupported action grid", file=sys.stderr)
        return 2
    labels = [name for name, _ in specs]
    counts_np = np.stack([counts for _, counts in specs]).astype(np.float32)
    ensemble_weights_np = counts_np / counts_np.sum(axis=1, keepdims=True)
    ensemble_weights = torch.as_tensor(ensemble_weights_np, device=device, dtype=torch.float32)
    print(
        f"[ensemble-sleeves] base_policies={len(pool)} sleeves={len(specs)} windows={len(starts)} "
        f"slip={float(args.slippage_bps):g}bps lev={float(args.leverage):g}x"
    )
    equity_curve, turnover = simulate_ensemble_sleeves(
        stacked=StackedEnsemble.from_policies(policies, device),
        ensemble_weights=ensemble_weights,
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
    fake_pool = [Path(f"{label}.pt") for label in labels]
    portfolio_weights = build_candidate_weights(
        pool=fake_pool,
        sleeve_returns=sleeve_returns,
        sleeve_p10=sleeve_p10,
        random_candidates=int(args.random_candidates),
        min_members=int(args.min_members),
        max_members=int(args.max_members),
        max_count=int(args.max_count),
        seed=int(args.seed),
        artifact_counts=[],
    )
    print(f"[ensemble-sleeves] portfolio_weights={portfolio_weights.shape[0]}")
    results = _eval_weight_candidates(
        weights=portfolio_weights,
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
        "sleeves": [{"label": labels[i], "counts": counts_np[i].tolist()} for i in range(len(labels))],
        "window_days": int(args.window_days),
        "n_windows": len(starts),
        "starts": starts,
        "fill_buffer_bps": float(args.fill_buffer_bps),
        "slippage_bps": float(args.slippage_bps),
        "leverage": float(args.leverage),
        "fee_rate": float(args.fee_rate),
        "margin_apr": float(args.margin_apr),
        "decision_lag": 2,
        "results": results,
        "best": results[0] if results else None,
        "solo_sleeves": [
            {
                "label": labels[i],
                "median_monthly_return": float(_monthly_from_total(float(sleeve_returns[i]), int(args.window_days))),
                "p10_monthly_return": float(_monthly_from_total(float(sleeve_p10[i]), int(args.window_days))),
            }
            for i in range(len(labels))
        ],
    }
    out_path = Path(args.out)
    if not out_path.is_absolute():
        out_path = REPO / out_path
    write_json_atomic(out_path, payload)
    print("\nTop ensemble-sleeve portfolios:")
    for item in results:
        print(
            f"{item['score']:+.4f} med={item['median_monthly_return'] * 100:+6.2f}% "
            f"p10={item['p10_monthly_return'] * 100:+6.2f}% "
            f"neg={item['n_neg']:3d}/{item['n_windows']} "
            f"dd={item['max_drawdown'] * 100:5.1f}% "
            f"turn={item['median_turnover_x_initial']:6.1f}x members={len(item['members'])}"
        )
        print("  " + ", ".join(f"{name}:{weight:.3f}" for name, weight in item["members"][:10]))
    print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
