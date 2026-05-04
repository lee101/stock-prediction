#!/usr/bin/env python3
"""GPU portfolio-packing overlay for screened32 policy scores.

This is a research simulator, not a production gate. It reuses a trained
single-symbol screened32 ensemble only as an alpha surface: each day we compute
long probabilities for every symbol from a flat portfolio observation, then
pack the top-ranked names into a long-only portfolio under binary-fill,
decision_lag=2 execution.

The design mirrors the useful part of NVIDIA's portfolio-optimization
blueprint: cardinality (top-K), concentration/gross limits, turnover throttles,
and risk-aware weights. We intentionally keep it solver-free for this first
pass so thousands of parameter cells can run directly on CUDA before any slow
full-gate proof.
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
import sys
from dataclasses import asdict, dataclass
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


@dataclass(frozen=True)
class PackConfig:
    pack_size: int
    score_power: float
    vol_power: float
    flat_gate: float
    gross_scale: float
    rebalance_every: int
    rebalance_threshold: float


def _resolve_repo_path(path: str | Path) -> Path:
    p = Path(path)
    if not p.is_absolute():
        p = REPO / p
    return p.resolve()


def _parse_float_list(raw: str) -> list[float]:
    values = [float(part.strip()) for part in str(raw).split(",") if part.strip()]
    if not values:
        raise ValueError("list must contain at least one value")
    if any(not math.isfinite(v) for v in values):
        raise ValueError("list values must be finite")
    return values


def _parse_int_list(raw: str) -> list[int]:
    values = [int(part.strip()) for part in str(raw).split(",") if part.strip()]
    if not values:
        raise ValueError("list must contain at least one value")
    if any(v <= 0 for v in values):
        raise ValueError("integer list values must be positive")
    return values


def _finite_nonnegative(value: float, name: str) -> str | None:
    if not math.isfinite(float(value)) or float(value) < 0.0:
        return f"{name} must be finite and non-negative"
    return None


def _finite_positive(value: float, name: str) -> str | None:
    if not math.isfinite(float(value)) or float(value) <= 0.0:
        return f"{name} must be finite and positive"
    return None


def validate_common_pack_args(args: argparse.Namespace, *, gate_attr: str) -> list[str]:
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
    for attr in ("window_days", "vol_lookback", "top_k"):
        if int(getattr(args, attr)) <= 0:
            failures.append(f"{attr} must be positive")
    if args.max_windows is not None and int(args.max_windows) <= 0:
        failures.append("max_windows must be positive when provided")
    candidate_index = getattr(args, "candidate_index", None)
    if candidate_index is not None and int(candidate_index) < 0:
        failures.append("candidate_index must be non-negative when provided")

    int_lists = ("pack_sizes", "rebalance_everys")
    float_lists = ("score_powers", "vol_powers", gate_attr, "gross_scales", "rebalance_thresholds")
    for attr in int_lists:
        try:
            _parse_int_list(getattr(args, attr))
        except ValueError as exc:
            failures.append(f"{attr}: {exc}")
    for attr in float_lists:
        try:
            values = _parse_float_list(getattr(args, attr))
        except ValueError as exc:
            failures.append(f"{attr}: {exc}")
            continue
        if any(value < 0.0 for value in values):
            failures.append(f"{attr} must contain only non-negative values")
    return failures


def build_config_grid(
    *,
    pack_sizes: Sequence[int],
    score_powers: Sequence[float],
    vol_powers: Sequence[float],
    flat_gates: Sequence[float],
    gross_scales: Sequence[float],
    rebalance_everys: Sequence[int],
    rebalance_thresholds: Sequence[float],
) -> list[PackConfig]:
    configs = []
    for item in itertools.product(
            pack_sizes,
            score_powers,
            vol_powers,
            flat_gates,
            gross_scales,
            rebalance_everys,
            rebalance_thresholds,
    ):
        k, score_power, vol_power, flat_gate, gross_scale, rebalance_every, rebalance_threshold = item
        configs.append(
            PackConfig(
                pack_size=int(k),
                score_power=float(score_power),
                vol_power=float(vol_power),
                flat_gate=float(flat_gate),
                gross_scale=float(gross_scale),
                rebalance_every=int(rebalance_every),
                rebalance_threshold=float(rebalance_threshold),
            )
        )
    if not configs:
        raise ValueError("empty config grid")
    return configs


def _members_from_artifact(path: str | Path, candidate_index: int | None) -> list[str]:
    data = json.loads(_resolve_repo_path(path).read_text(encoding="utf-8"))
    results = list(data.get("results", []))
    if not results:
        raise ValueError(f"artifact has no results: {path}")
    if candidate_index is None:
        item = results[0]
    else:
        matches = [
            item
            for item in results
            if int(item.get("candidate_index", item.get("trial", -1))) == int(candidate_index)
        ]
        if not matches:
            raise ValueError(f"candidate_index={candidate_index} not found in {path}")
        item = matches[0]
    members = item.get("members")
    if not members:
        raise ValueError(f"selected artifact item has no members: {path}")
    return [str(member) for member in members]


def _load_member_paths(args: argparse.Namespace) -> list[Path]:
    if args.members_artifact:
        members = _members_from_artifact(args.members_artifact, args.candidate_index)
    elif args.checkpoints:
        members = list(args.checkpoints)
    else:
        members = [str(DEFAULT_CHECKPOINT), *(str(p) for p in DEFAULT_EXTRA_CHECKPOINTS)]
    paths = [_resolve_repo_path(member) for member in members]
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"missing checkpoints: {missing}")
    return paths


def _build_flat_obs(
    features: torch.Tensor,
    step: int,
    *,
    window_days: int,
    initial_cash: float,
) -> torch.Tensor:
    n_windows, _, n_symbols, n_features = features.shape
    base = int(n_symbols) * int(n_features)
    obs = torch.zeros(n_windows, base + 5 + int(n_symbols), device=features.device, dtype=torch.float32)
    t_obs = max(0, int(step) - 1)
    obs[:, :base] = features[:, t_obs].reshape(n_windows, base)
    obs[:, base + 0] = 1.0
    obs[:, base + 3] = 0.0
    obs[:, base + 4] = float(step) / max(int(window_days), 1)
    _ = initial_cash
    return obs


def _target_weights_from_scores(
    *,
    long_probs: torch.Tensor,
    flat_probs: torch.Tensor,
    tradable: torch.Tensor,
    vol: torch.Tensor,
    configs: Sequence[PackConfig],
) -> torch.Tensor:
    """Build target portfolio weights [C, N, S] for one decision step."""
    device = long_probs.device
    c = len(configs)
    n_windows, n_symbols = long_probs.shape
    pack_sizes = torch.tensor([cfg.pack_size for cfg in configs], device=device, dtype=torch.long)
    score_power = torch.tensor([cfg.score_power for cfg in configs], device=device, dtype=torch.float32)
    vol_power = torch.tensor([cfg.vol_power for cfg in configs], device=device, dtype=torch.float32)
    flat_gate = torch.tensor([cfg.flat_gate for cfg in configs], device=device, dtype=torch.float32)
    gross_scale = torch.tensor([cfg.gross_scale for cfg in configs], device=device, dtype=torch.float32)

    score = long_probs.unsqueeze(0) - flat_gate.view(c, 1, 1) * flat_probs.view(1, n_windows, 1)
    score = torch.clamp(score, min=0.0)
    risk = torch.pow(vol.clamp_min(1e-4).unsqueeze(0), vol_power.view(c, 1, 1))
    score = score / risk.clamp_min(1e-6)
    score = torch.where(tradable.unsqueeze(0), score, torch.zeros_like(score))

    max_k = int(min(max(int(cfg.pack_size) for cfg in configs), int(n_symbols)))
    top_vals, top_idx = torch.topk(score, k=max_k, dim=2)
    rank_mask = torch.arange(max_k, device=device).view(1, 1, max_k) < pack_sizes.clamp_max(max_k).view(c, 1, 1)
    positive_mask = top_vals > 0.0
    selected = rank_mask & positive_mask

    raw_vals = torch.where(
        score_power.view(c, 1, 1) <= 0.0,
        torch.ones_like(top_vals),
        torch.pow(top_vals.clamp_min(1e-12), score_power.view(c, 1, 1)),
    )
    raw_vals = torch.where(selected, raw_vals, torch.zeros_like(raw_vals))
    weights_top = raw_vals / raw_vals.sum(dim=2, keepdim=True).clamp_min(1e-12)
    weights = torch.zeros(c, n_windows, n_symbols, device=device, dtype=torch.float32)
    weights.scatter_add_(2, top_idx, weights_top)
    weights = weights * gross_scale.view(c, 1, 1)
    return weights


def _rolling_vol(
    prices: torch.Tensor,
    step: int,
    *,
    lookback: int,
) -> torch.Tensor:
    close = prices[:, :, :, _P_CLOSE]
    t_obs = max(0, int(step) - 1)
    if t_obs <= 0:
        return torch.full_like(close[:, 0, :], 0.02)
    start = max(1, t_obs - int(lookback) + 1)
    prev = close[:, start - 1 : t_obs, :]
    cur = close[:, start : t_obs + 1, :]
    rets = torch.where(prev > 0, (cur / prev.clamp_min(1e-12)) - 1.0, torch.zeros_like(cur))
    if rets.size(1) <= 1:
        return rets.abs().squeeze(1).clamp_min(0.005)
    return rets.std(dim=1, unbiased=False).clamp_min(0.005)


def run_pack_sim(
    *,
    stacked: StackedEnsemble,
    prices: torch.Tensor,
    features: torch.Tensor,
    tradable: torch.Tensor,
    configs: Sequence[PackConfig],
    num_symbols: int,
    per_symbol_actions: int,
    window_days: int,
    fill_buffer_bps: float,
    max_leverage: float,
    fee_rate: float,
    slippage_bps: float,
    margin_apr: float,
    vol_lookback: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    device = prices.device
    c = len(configs)
    n = int(prices.shape[0])
    s = int(num_symbols)
    init_cash = float(INITIAL_CASH)
    cash = torch.full((c, n), init_cash, device=device, dtype=torch.float32)
    qty = torch.zeros(c, n, s, device=device, dtype=torch.float32)
    peak_equity = torch.full((c, n), init_cash, device=device, dtype=torch.float32)
    max_dd = torch.zeros(c, n, device=device, dtype=torch.float32)
    turnover = torch.zeros(c, n, device=device, dtype=torch.float32)
    target_buf = torch.zeros(c, n, s, 3, device=device, dtype=torch.float32)
    desired_target = torch.zeros(c, n, s, device=device, dtype=torch.float32)
    rebalance_every = torch.tensor(
        [max(1, int(cfg.rebalance_every)) for cfg in configs],
        device=device,
        dtype=torch.long,
    ).view(c, 1, 1)
    rebalance_threshold = torch.tensor(
        [cfg.rebalance_threshold for cfg in configs],
        device=device,
        dtype=torch.float32,
    ).view(c, 1, 1)
    fill_buffer_frac = max(0.0, float(fill_buffer_bps)) / 10_000.0
    effective_fee = float(fee_rate) + max(0.0, float(slippage_bps)) / 10_000.0
    margin_rate = max(0.0, float(margin_apr)) / 365.0

    for step in range(int(window_days)):
        obs = _build_flat_obs(features, int(step), window_days=int(window_days), initial_cash=init_cash)
        with torch.no_grad():
            logits_all = stacked.forward(obs)
            probs = torch.softmax(logits_all, dim=-1).mean(dim=0)
        flat_probs = probs[:, 0]
        long_probs = probs[:, 1 : 1 + s * int(per_symbol_actions) : int(per_symbol_actions)]
        vol = _rolling_vol(prices, int(step), lookback=int(vol_lookback))
        target_now = _target_weights_from_scores(
            long_probs=long_probs,
            flat_probs=flat_probs,
            tradable=tradable[:, step, :],
            vol=vol,
            configs=configs,
        )
        should_rebalance = (torch.tensor(step, device=device) % rebalance_every) == 0
        target_now = torch.where(should_rebalance, target_now, desired_target)
        desired_target = target_now
        slot = step % 3
        target_buf[:, :, :, slot] = target_now
        if step < 2:
            target = torch.zeros(c, n, s, device=device, dtype=torch.float32)
        else:
            target = target_buf[:, :, :, (step - 2) % 3]

        close_t = prices[:, step, :, _P_CLOSE].unsqueeze(0).expand(c, n, s)
        low_t = prices[:, step, :, _P_LOW].unsqueeze(0).expand(c, n, s)
        trad_t = tradable[:, step, :].unsqueeze(0).expand(c, n, s)
        current_value = qty * close_t
        equity = cash + current_value.sum(dim=2)
        desired_value = equity.clamp_min(0.0).unsqueeze(2) * float(max_leverage) * target

        sell_value = torch.clamp(current_value - desired_value, min=0.0)
        sell_mask = (sell_value > current_value * rebalance_threshold) & trad_t & (close_t > 0.0)
        sell_qty = torch.where(sell_mask, sell_value / close_t.clamp_min(1e-12), torch.zeros_like(sell_value))
        sell_qty = torch.minimum(sell_qty, qty)
        proceeds = sell_qty * close_t * (1.0 - effective_fee)
        qty = qty - sell_qty
        cash = cash + proceeds.sum(dim=2)
        turnover = turnover + (sell_qty * close_t).sum(dim=2) / init_cash

        current_value = qty * close_t
        buy_value = torch.clamp(desired_value - current_value, min=0.0)
        buy_mask = buy_value > desired_value * rebalance_threshold
        fillable = low_t <= close_t * (1.0 - fill_buffer_frac)
        can_buy = buy_mask & fillable & trad_t & (close_t > 0.0) & (equity.unsqueeze(2) > 0.0)
        buy_qty = torch.where(
            can_buy,
            buy_value / (close_t * (1.0 + effective_fee)).clamp_min(1e-12),
            torch.zeros_like(buy_value),
        )
        cost = buy_qty * close_t * (1.0 + effective_fee)
        qty = qty + buy_qty
        cash = cash - cost.sum(dim=2)
        turnover = turnover + (buy_qty * close_t).sum(dim=2) / init_cash

        if margin_rate > 0.0:
            cash = cash - torch.clamp(-cash, min=0.0) * margin_rate

        t_new = min(step + 1, int(prices.size(1)) - 1)
        close_new = prices[:, t_new, :, _P_CLOSE].unsqueeze(0).expand(c, n, s)
        equity_after = cash + (qty * close_new).sum(dim=2)
        peak_equity = torch.maximum(peak_equity, equity_after)
        dd = torch.where(
            peak_equity > 0,
            (peak_equity - equity_after) / peak_equity.clamp_min(1e-12),
            torch.zeros_like(peak_equity),
        )
        max_dd = torch.maximum(max_dd, dd)

    close_end = prices[:, -1, :, _P_CLOSE].unsqueeze(0).expand(c, n, s)
    proceeds_final = qty * close_end * (1.0 - effective_fee)
    cash = cash + proceeds_final.sum(dim=2)
    total_return = (cash / init_cash) - 1.0
    return (
        total_return.detach().to(torch.float64).cpu().numpy(),
        max_dd.detach().to(torch.float64).cpu().numpy(),
        turnover.detach().to(torch.float64).cpu().numpy(),
    )


def _summarize(
    *,
    configs: Sequence[PackConfig],
    total_returns: np.ndarray,
    max_drawdowns: np.ndarray,
    turnover: np.ndarray,
    window_days: int,
    neg_penalty: float,
    dd_penalty: float,
    turnover_penalty: float,
) -> list[dict]:
    med_total = np.percentile(total_returns, 50, axis=1)
    p10_total = np.percentile(total_returns, 10, axis=1)
    p90_total = np.percentile(total_returns, 90, axis=1)
    med_dd = np.percentile(max_drawdowns, 50, axis=1)
    max_dd = np.max(max_drawdowns, axis=1)
    med_turnover = np.percentile(turnover, 50, axis=1)
    n_neg = np.sum(total_returns < 0.0, axis=1).astype(int)
    out: list[dict] = []
    for i, cfg in enumerate(configs):
        med_monthly = _monthly_from_total(float(med_total[i]), int(window_days))
        p10_monthly = _monthly_from_total(float(p10_total[i]), int(window_days))
        score = (
            float(med_monthly)
            + 0.5 * float(p10_monthly)
            - float(neg_penalty) * float(n_neg[i])
            - float(dd_penalty) * float(max_dd[i])
            - float(turnover_penalty) * float(med_turnover[i])
        )
        out.append(
            {
                "config": asdict(cfg),
                "score": float(score),
                "median_total_return": float(med_total[i]),
                "p10_total_return": float(p10_total[i]),
                "p90_total_return": float(p90_total[i]),
                "median_monthly_return": float(med_monthly),
                "p10_monthly_return": float(p10_monthly),
                "max_drawdown": float(max_dd[i]),
                "median_drawdown": float(med_dd[i]),
                "median_turnover_x_initial": float(med_turnover[i]),
                "n_neg": int(n_neg[i]),
                "n_windows": int(total_returns.shape[1]),
            }
        )
    out.sort(key=lambda item: float(item["score"]), reverse=True)
    return out


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--val-data", default="pufferlib_market/data/screened32_single_offset_val_full.bin")
    parser.add_argument("--members-artifact", default=None)
    parser.add_argument("--candidate-index", type=int, default=None)
    parser.add_argument("--checkpoints", nargs="*", default=None)
    parser.add_argument("--window-days", type=int, default=100)
    parser.add_argument("--max-windows", type=int, default=None)
    parser.add_argument("--pack-sizes", default="1,2,3,4,5,6,8")
    parser.add_argument("--score-powers", default="0,0.5,1,2")
    parser.add_argument("--vol-powers", default="0,0.5,1")
    parser.add_argument("--flat-gates", default="0,0.25,0.5,0.75,1.0")
    parser.add_argument("--gross-scales", default="0.25,0.5,0.75,1.0")
    parser.add_argument("--rebalance-everys", default="1,2,3,5,10")
    parser.add_argument("--rebalance-thresholds", default="0,0.05,0.10,0.20")
    parser.add_argument("--vol-lookback", type=int, default=20)
    parser.add_argument("--fill-buffer-bps", type=float, default=5.0)
    parser.add_argument("--slippage-bps", type=float, default=20.0)
    parser.add_argument("--leverage", type=float, default=2.0)
    parser.add_argument("--fee-rate", type=float, default=0.001)
    parser.add_argument("--margin-apr", type=float, default=0.0625)
    parser.add_argument("--neg-penalty", type=float, default=0.002)
    parser.add_argument("--dd-penalty", type=float, default=0.15)
    parser.add_argument("--turnover-penalty", type=float, default=0.00005)
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out", default="analysis/screened32_portfolio_pack/screen.json")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    validation_failures = validate_common_pack_args(args, gate_attr="flat_gates")
    if validation_failures:
        for failure in validation_failures:
            print(f"gpu_portfolio_pack_screen: {failure}", file=sys.stderr)
        return 2
    if not torch.cuda.is_available() and str(args.device).startswith("cuda"):
        print("gpu_portfolio_pack_screen: CUDA is unavailable", file=sys.stderr)
        return 2
    try:
        paths = _load_member_paths(args)
        configs = build_config_grid(
            pack_sizes=_parse_int_list(args.pack_sizes),
            score_powers=_parse_float_list(args.score_powers),
            vol_powers=_parse_float_list(args.vol_powers),
            flat_gates=_parse_float_list(args.flat_gates),
            gross_scales=_parse_float_list(args.gross_scales),
            rebalance_everys=_parse_int_list(args.rebalance_everys),
            rebalance_thresholds=_parse_float_list(args.rebalance_thresholds),
        )
    except (OSError, ValueError) as exc:
        print(f"gpu_portfolio_pack_screen: {exc}", file=sys.stderr)
        return 2

    val_path = _resolve_repo_path(args.val_data)
    data = read_mktd(val_path)
    window_len = int(args.window_days) + 1
    if window_len > int(data.num_timesteps):
        print("gpu_portfolio_pack_screen: window is longer than val data", file=sys.stderr)
        return 2
    starts = list(range(int(data.num_timesteps) - window_len + 1))
    if args.max_windows is not None:
        starts = starts[: max(1, int(args.max_windows))]

    device = torch.device(str(args.device))
    prices, features, tradable = _stage_windows(data, starts, int(args.window_days), device)
    loaded = [
        load_policy(path, int(data.num_symbols), features_per_sym=int(data.features.shape[2]), device=device)
        for path in paths
    ]
    policies = [lp.policy.eval() for lp in loaded]
    if not can_batch(policies):
        print("gpu_portfolio_pack_screen: policies are not stack-compatible", file=sys.stderr)
        return 2
    head = loaded[0]
    alloc_bins = int(head.action_allocation_bins)
    level_bins = int(head.action_level_bins)
    action_max_offset_bps = float(head.action_max_offset_bps)
    if alloc_bins != 1 or level_bins != 1 or action_max_offset_bps != 0.0:
        print("gpu_portfolio_pack_screen: unsupported action grid for score extraction", file=sys.stderr)
        return 2
    stacked = StackedEnsemble.from_policies(policies, device)
    print(
        f"[pack] members={len(paths)} configs={len(configs)} windows={len(starts)} "
        f"slip={float(args.slippage_bps):g}bps lev={float(args.leverage):g}x"
    )
    total_returns, max_drawdowns, turnover = run_pack_sim(
        stacked=stacked,
        prices=prices,
        features=features,
        tradable=tradable,
        configs=configs,
        num_symbols=int(data.num_symbols),
        per_symbol_actions=max(1, alloc_bins) * max(1, level_bins),
        window_days=int(args.window_days),
        fill_buffer_bps=float(args.fill_buffer_bps),
        max_leverage=float(args.leverage),
        fee_rate=float(args.fee_rate),
        slippage_bps=float(args.slippage_bps),
        margin_apr=float(args.margin_apr),
        vol_lookback=int(args.vol_lookback),
    )
    results = _summarize(
        configs=configs,
        total_returns=total_returns,
        max_drawdowns=max_drawdowns,
        turnover=turnover,
        window_days=int(args.window_days),
        neg_penalty=float(args.neg_penalty),
        dd_penalty=float(args.dd_penalty),
        turnover_penalty=float(args.turnover_penalty),
    )
    payload = {
        "val_data": str(val_path),
        "members": [str(path) for path in paths],
        "window_days": int(args.window_days),
        "n_windows": len(starts),
        "starts": starts,
        "fill_buffer_bps": float(args.fill_buffer_bps),
        "slippage_bps": float(args.slippage_bps),
        "leverage": float(args.leverage),
        "fee_rate": float(args.fee_rate),
        "margin_apr": float(args.margin_apr),
        "decision_lag": 2,
        "rebalance_everys": _parse_int_list(args.rebalance_everys),
        "vol_lookback": int(args.vol_lookback),
        "results": results,
        "best": results[0] if results else None,
    }
    out_path = Path(args.out)
    if not out_path.is_absolute():
        out_path = REPO / out_path
    write_json_atomic(out_path, payload)
    print("\nTop portfolio-pack configs:")
    for item in results[: max(1, int(args.top_k))]:
        cfg = item["config"]
        print(
            f"{item['score']:+.4f} med={item['median_monthly_return'] * 100:+6.2f}% "
            f"p10={item['p10_monthly_return'] * 100:+6.2f}% "
            f"neg={item['n_neg']:3d}/{item['n_windows']} "
            f"dd={item['max_drawdown'] * 100:5.1f}% "
            f"turn={item['median_turnover_x_initial']:6.1f}x "
            f"K={cfg['pack_size']} sp={cfg['score_power']:g} vp={cfg['vol_power']:g} "
            f"gate={cfg['flat_gate']:g} gross={cfg['gross_scale']:g} "
            f"every={cfg['rebalance_every']} thr={cfg['rebalance_threshold']:g}"
        )
    print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
