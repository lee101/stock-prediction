#!/usr/bin/env python3
"""Rolling-window GPU scorer for daily crypto plan templates.

This is the more scalable inference-time path:

1. Generate many candidate daily templates across historical day-starts.
2. Simulate each template's next-day outcome.
3. Train a GPU Torch scorer on market features + template vector.
4. At inference, sample many candidates, score them in one GPU batch, optionally
   do surrogate-gradient refinement, then execute only the selected plan.

No live orders are placed.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from experiments.binance_deepseek_planner.deepseek_energy_refit_agent import (
    PlanTemplate,
    TemplateLeg,
    clamp_vector_array,
    template_to_vector,
    vector_to_template,
)
from experiments.binance_openai_planner.gpt55_week_replanner import DEFAULT_INITIAL_CASH, _metrics, simulate_day
from experiments.binance_transition_graph.graph_policy import DEFAULT_SYMBOLS, close_matrix, load_hourly_panel


def market_features(frames: Mapping[str, pd.DataFrame], *, day_start_i: int, symbols: Sequence[str]) -> np.ndarray:
    close = close_matrix(frames)
    hist = close.iloc[:day_start_i]
    returns = hist.pct_change().replace([np.inf, -np.inf], np.nan)
    feats: list[float] = []
    for symbol in symbols:
        series = hist[str(symbol)].dropna()
        if len(series) < 169:
            feats.extend([0.0] * 8)
            continue
        ret_1 = float(series.iloc[-1] / series.iloc[-2] - 1.0)
        ret_6 = float(series.iloc[-1] / series.iloc[-7] - 1.0)
        ret_24 = float(series.iloc[-1] / series.iloc[-25] - 1.0)
        ret_72 = float(series.iloc[-1] / series.iloc[-73] - 1.0)
        ret_168 = float(series.iloc[-1] / series.iloc[-169] - 1.0)
        vol_24 = float(returns[str(symbol)].tail(24).std(ddof=0) or 0.0)
        vol_168 = float(returns[str(symbol)].tail(168).std(ddof=0) or 0.0)
        high_168 = float(series.tail(168).max())
        dd_168 = float(series.iloc[-1] / high_168 - 1.0) if high_168 > 0 else 0.0
        feats.extend([ret_1, ret_6, ret_24, ret_72, ret_168, vol_24, vol_168, dd_168])
    corr = returns[list(symbols)].tail(168).corr().fillna(0.0).to_numpy(dtype=np.float64)
    upper = corr[np.triu_indices(len(symbols), k=1)]
    feats.extend([float(np.mean(upper)), float(np.std(upper)), float(np.min(upper)), float(np.max(upper))])
    return np.asarray(feats, dtype=np.float64)


def make_template_from_arrays(
    sides: np.ndarray,
    allocs: np.ndarray,
    entries: np.ndarray,
    tps: np.ndarray,
    stops: np.ndarray,
    symbols: Sequence[str],
    *,
    day: int,
    name: str,
    max_gross_exposure: float = 100.0,
) -> PlanTemplate:
    legs: list[TemplateLeg] = []
    for i, symbol in enumerate(symbols):
        side_num = int(sides[i])
        if side_num > 0:
            side = "long"
            entry = min(0.0, float(entries[i]))
        elif side_num < 0:
            side = "short"
            entry = max(0.0, float(entries[i]))
        else:
            side = "flat"
            entry = 0.0
        alloc = 0.0 if side == "flat" else float(max(0.0, allocs[i]))
        legs.append(
            TemplateLeg(
                symbol=str(symbol),
                side=side,
                allocation_pct=alloc,
                entry_offset_bps=entry,
                take_profit_bps=float(max(25.0, tps[i])),
                stop_loss_bps=float(max(25.0, stops[i])),
                confidence=0.5,
                rationale="rolling trainer candidate",
            )
        )
    gross = sum(leg.allocation_pct for leg in legs if leg.side != "flat")
    if gross > float(max_gross_exposure):
        scale = float(max_gross_exposure) / gross
        legs = [
            TemplateLeg(**{**asdict(leg), "allocation_pct": leg.allocation_pct * scale})
            if leg.side != "flat"
            else leg
            for leg in legs
        ]
        gross = float(max_gross_exposure)
    return PlanTemplate(
        name=name,
        rationale="rolling trainer candidate",
        day_index=int(day),
        cash_pct=max(0.0, 100.0 - gross),
        gross_exposure_pct=float(gross),
        risk_notes="rolling trainer candidate",
        legs=legs,
    )


def legacy_random_template(
    rng: np.random.Generator,
    frames: Mapping[str, pd.DataFrame],
    *,
    day_start_i: int,
    day: int,
    symbols: Sequence[str],
    name: str,
    max_active: int,
    max_symbol_allocation: float,
    max_gross_exposure: float,
) -> PlanTemplate:
    close = close_matrix(frames)
    hist = close.iloc[:day_start_i]
    ret_24 = hist.pct_change(24).iloc[-1].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    ret_72 = hist.pct_change(72).iloc[-1].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    score = (0.6 * ret_24 + 0.4 * ret_72).reindex(list(symbols)).fillna(0.0).to_numpy()
    n_active = int(rng.integers(0, int(max_active) + 1))
    sides = np.zeros(len(symbols), dtype=int)
    if n_active > 0:
        mode = rng.choice(["mom", "rev", "random"], p=[0.45, 0.25, 0.30])
        if mode == "mom":
            order = np.argsort(np.abs(score))[::-1]
            chosen = order[:n_active]
            sides[chosen] = np.where(score[chosen] >= 0, 1, -1)
        elif mode == "rev":
            order = np.argsort(np.abs(score))[::-1]
            chosen = order[:n_active]
            sides[chosen] = np.where(score[chosen] >= 0, -1, 1)
        else:
            chosen = rng.choice(len(symbols), size=n_active, replace=False)
            sides[chosen] = rng.choice([-1, 1], size=n_active)
    allocs = np.where(sides != 0, rng.uniform(3.0, float(max_symbol_allocation), size=len(symbols)), 0.0)
    gross = float(allocs.sum())
    if gross > max_gross_exposure:
        allocs *= float(max_gross_exposure) / gross
    entries = np.where(sides > 0, -rng.uniform(0.0, 120.0, len(symbols)), rng.uniform(0.0, 120.0, len(symbols)))
    tps = rng.uniform(60.0, 450.0, len(symbols))
    stops = rng.uniform(50.0, 350.0, len(symbols))
    return make_template_from_arrays(
        sides,
        allocs,
        entries,
        tps,
        stops,
        symbols,
        day=day,
        name=name,
        max_gross_exposure=max_gross_exposure,
    )


def random_template(
    rng: np.random.Generator,
    frames: Mapping[str, pd.DataFrame],
    *,
    day_start_i: int,
    day: int,
    symbols: Sequence[str],
    name: str,
    max_active: int,
    max_symbol_allocation: float,
    max_gross_exposure: float,
    candidate_style: str,
) -> PlanTemplate:
    if candidate_style == "legacy" or (candidate_style == "mixed" and rng.random() < 0.50):
        return legacy_random_template(
            rng,
            frames,
            day_start_i=day_start_i,
            day=day,
            symbols=symbols,
            name=name,
            max_active=max_active,
            max_symbol_allocation=max_symbol_allocation,
            max_gross_exposure=max_gross_exposure,
        )
    close = close_matrix(frames)
    hist = close.iloc[:day_start_i]
    n = len(symbols)
    ret_24 = hist.pct_change(24).iloc[-1].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    ret_72 = hist.pct_change(72).iloc[-1].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    ret_168 = hist.pct_change(168).iloc[-1].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    returns = hist.pct_change().replace([np.inf, -np.inf], np.nan)
    score = (0.55 * ret_24 + 0.30 * ret_72 + 0.15 * ret_168).reindex(list(symbols)).fillna(0.0).to_numpy()
    vol = returns.reindex(columns=list(symbols)).tail(168).std(ddof=0).fillna(0.0).to_numpy() * 10_000.0
    range_bps = []
    pos_168 = []
    for symbol in symbols:
        df = frames[str(symbol)].iloc[:day_start_i]
        recent = df.tail(168)
        range_bps.append(float(((recent["high"] - recent["low"]) / recent["close"]).tail(48).mean() * 10_000.0))
        high = float(recent["close"].max())
        low = float(recent["close"].min())
        last = float(recent["close"].iloc[-1])
        pos_168.append((last - low) / max(high - low, 1e-12) - 0.5)
    range_bps_arr = np.nan_to_num(np.asarray(range_bps, dtype=np.float64), nan=80.0, posinf=180.0, neginf=80.0)
    vol_arr = np.maximum(np.nan_to_num(vol, nan=40.0, posinf=120.0, neginf=40.0), 10.0)
    extreme = score * 10_000.0 / np.maximum(vol_arr * math.sqrt(24.0), 1.0)
    max_active = max(0, int(max_active))
    if max_active <= 0 or rng.random() < 0.08:
        n_active = 0
    else:
        choices = np.arange(1, max_active + 1)
        weights = np.linspace(1.0, 0.55, len(choices))
        if len(choices) >= 2:
            weights[1] *= 1.7
        weights = weights / weights.sum()
        n_active = int(rng.choice(choices, p=weights))
    sides = np.zeros(n, dtype=int)
    if n_active > 0:
        # Mix directional, cross-sectional, beta-neutral, and pure exploration priors.
        mode = rng.choice(["mom", "rev", "pair_rev", "pair_mom", "range_rev", "random"], p=[0.24, 0.18, 0.22, 0.14, 0.12, 0.10])
        if mode == "mom":
            order = np.argsort(np.abs(score))[::-1]
            chosen = order[:n_active]
            sides[chosen] = np.where(score[chosen] >= 0, 1, -1)
        elif mode == "rev":
            order = np.argsort(np.abs(score))[::-1]
            chosen = order[:n_active]
            sides[chosen] = np.where(score[chosen] >= 0, -1, 1)
        elif mode == "pair_rev":
            order = np.argsort(extreme)
            long_count = max(1, n_active // 2)
            short_count = max(1, n_active - long_count)
            sides[order[:long_count]] = 1
            sides[order[-short_count:]] = -1
        elif mode == "pair_mom":
            order = np.argsort(score)
            long_count = max(1, n_active // 2)
            short_count = max(1, n_active - long_count)
            sides[order[-long_count:]] = 1
            sides[order[:short_count]] = -1
        elif mode == "range_rev":
            order = np.argsort(np.asarray(pos_168, dtype=np.float64))
            long_count = max(1, n_active // 2)
            short_count = max(1, n_active - long_count)
            sides[order[:long_count]] = 1
            sides[order[-short_count:]] = -1
        else:
            chosen = rng.choice(n, size=n_active, replace=False)
            sides[chosen] = rng.choice([-1, 1], size=n_active)
    allocs = np.where(sides != 0, rng.uniform(4.0, float(max_symbol_allocation), size=n), 0.0)
    gross = float(allocs.sum())
    if gross > max_gross_exposure:
        allocs *= float(max_gross_exposure) / gross
    entry_profile = rng.choice(["near", "balanced", "deep"], p=[0.35, 0.50, 0.15])
    base_range = np.clip(0.55 * range_bps_arr + 0.45 * vol_arr, 12.0, 260.0)
    if entry_profile == "near":
        dist = rng.uniform(0.0, 0.45, n) * base_range
    elif entry_profile == "deep":
        dist = rng.uniform(0.95, 2.1, n) * base_range
    else:
        dist = rng.uniform(0.25, 1.15, n) * base_range
    dist = np.clip(dist, 0.0, 250.0)
    entries = np.where(sides > 0, -dist, dist)
    tps = np.clip(rng.uniform(0.85, 2.9, n) * base_range + rng.uniform(20.0, 80.0, n), 35.0, 800.0)
    stops = np.clip(rng.uniform(0.75, 2.4, n) * base_range + rng.uniform(20.0, 70.0, n), 35.0, 700.0)
    return make_template_from_arrays(
        sides,
        allocs,
        entries,
        tps,
        stops,
        symbols,
        day=day,
        name=name,
        max_gross_exposure=max_gross_exposure,
    )


def simulate_template_day(
    frames: Mapping[str, pd.DataFrame],
    template: PlanTemplate,
    *,
    day_start_i: int,
    day: int,
    initial_cash: float,
    fee_bps: float,
    slippage_bps: float,
    return_weight: float,
    drawdown_penalty: float,
    trade_penalty: float,
    loss_penalty: float,
) -> dict[str, Any]:
    from experiments.binance_deepseek_planner.deepseek_iterative_replanner import instantiate_plan

    plan = instantiate_plan(template, frames, day_start_i=day_start_i, day=day)
    equity, trades, rows = simulate_day(
        frames,
        plan,
        start_i=day_start_i,
        end_i=day_start_i + 24,
        starting_equity=float(initial_cash),
        fee_bps=float(fee_bps),
        slippage_bps=float(slippage_bps),
    )
    metrics = _metrics(rows, float(initial_cash))
    score = (
        float(return_weight) * float(metrics["total_return"])
        - float(drawdown_penalty) * float(metrics["max_drawdown"])
        - float(trade_penalty) * len(trades)
        + float(loss_penalty) * min(float(metrics["total_return"]), 0.0)
    )
    return {
        "score": float(score),
        "total_return": float(metrics["total_return"]),
        "max_drawdown": float(metrics["max_drawdown"]),
        "trades": len(trades),
        "final_equity": float(equity),
    }


def build_dataset(
    frames: Mapping[str, pd.DataFrame],
    *,
    symbols: Sequence[str],
    start_i: int,
    end_i: int,
    stride_hours: int,
    templates_per_day: int,
    rng: np.random.Generator,
    initial_cash: float,
    fee_bps: float,
    slippage_bps: float,
    return_weight: float,
    drawdown_penalty: float,
    trade_penalty: float,
    loss_penalty: float,
    label_target: str,
    max_active: int,
    max_symbol_allocation: float,
    max_gross_exposure: float,
    candidate_style: str,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    xs: list[np.ndarray] = []
    ys: list[float] = []
    meta: list[dict[str, Any]] = []
    starts = list(range(int(start_i), int(end_i), int(stride_hours)))
    for day_n, day_start in enumerate(starts):
        mfeat = market_features(frames, day_start_i=day_start, symbols=symbols)
        for j in range(int(templates_per_day)):
            template = random_template(
                rng,
                frames,
                day_start_i=day_start,
                day=day_n,
                symbols=symbols,
                name=f"train_d{day_n}_t{j}",
                max_active=max_active,
                max_symbol_allocation=max_symbol_allocation,
                max_gross_exposure=max_gross_exposure,
                candidate_style=candidate_style,
            )
            result = simulate_template_day(
                frames,
                template,
                day_start_i=day_start,
                day=day_n,
                initial_cash=initial_cash,
                fee_bps=fee_bps,
                slippage_bps=slippage_bps,
                return_weight=return_weight,
                drawdown_penalty=drawdown_penalty,
                trade_penalty=trade_penalty,
                loss_penalty=loss_penalty,
            )
            xs.append(np.concatenate([mfeat, template_to_vector(template, symbols)]))
            ys.append(float(result["total_return"] if label_target == "total_return" else result["score"]))
            meta.append({"day_start_i": day_start, "template": asdict(template), **result})
    return np.vstack(xs).astype(np.float32), np.asarray(ys, dtype=np.float32), meta


class TorchScorer:
    def __init__(
        self,
        model: Any,
        mean: np.ndarray,
        std: np.ndarray,
        device: str,
        *,
        members: Sequence[tuple[Any, np.ndarray, np.ndarray]] | None = None,
        uncertainty_penalty: float = 0.0,
    ):
        self.model = model
        self.mean = mean.astype(np.float32)
        self.std = std.astype(np.float32)
        self.device = device
        self.members = list(members) if members is not None else [(model, self.mean, self.std)]
        self.uncertainty_penalty = float(uncertainty_penalty)

    def predict(self, x: np.ndarray, batch_size: int = 8192) -> np.ndarray:
        import torch

        member_preds: list[np.ndarray] = []
        with torch.no_grad():
            for model, mean, std in self.members:
                outs: list[np.ndarray] = []
                model.eval()
                mean = mean.astype(np.float32)
                std = std.astype(np.float32)
                for i in range(0, len(x), int(batch_size)):
                    xb = torch.tensor((x[i : i + batch_size].astype(np.float32) - mean) / std, device=self.device)
                    outs.append(model(xb).detach().cpu().numpy().reshape(-1))
                member_preds.append(np.concatenate(outs) if outs else np.array([], dtype=np.float32))
        if not member_preds:
            return np.array([], dtype=np.float32)
        stacked = np.vstack(member_preds)
        pred = stacked.mean(axis=0)
        if len(member_preds) > 1 and self.uncertainty_penalty > 0.0:
            pred = pred - self.uncertainty_penalty * stacked.std(axis=0)
        return pred.astype(np.float32)


def train_torch_scorer(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    *,
    device: str,
    epochs: int,
    batch_size: int,
    lr: float,
    seed: int,
    ensemble_size: int,
    uncertainty_penalty: float,
) -> tuple[TorchScorer, dict[str, float]]:
    import torch
    import torch.nn as nn

    mean = x_train.mean(axis=0)
    std = x_train.std(axis=0) + 1e-6
    xtr = torch.tensor((x_train - mean) / std, dtype=torch.float32, device=device)
    ytr = torch.tensor(y_train.reshape(-1, 1), dtype=torch.float32, device=device)
    xva = torch.tensor((x_val - mean) / std, dtype=torch.float32, device=device)
    yva = torch.tensor(y_val.reshape(-1, 1), dtype=torch.float32, device=device)
    n = len(x_train)
    members: list[tuple[Any, np.ndarray, np.ndarray]] = []
    best_vals: list[float] = []
    for member_i in range(max(1, int(ensemble_size))):
        torch.manual_seed(int(seed) + 1009 * member_i)
        model = nn.Sequential(
            nn.Linear(x_train.shape[1], 256),
            nn.SiLU(),
            nn.Dropout(0.05),
            nn.Linear(256, 128),
            nn.SiLU(),
            nn.Linear(128, 64),
            nn.SiLU(),
            nn.Linear(64, 1),
        ).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=float(lr), weight_decay=1e-4)
        best_val = float("inf")
        best_state = None
        for epoch in range(int(epochs)):
            perm = torch.randperm(n, device=device)
            for start in range(0, n, int(batch_size)):
                idx = perm[start : start + int(batch_size)]
                pred = model(xtr[idx])
                loss = torch.nn.functional.smooth_l1_loss(pred, ytr[idx], beta=0.002)
                opt.zero_grad()
                loss.backward()
                opt.step()
            if epoch % 5 == 0 or epoch == int(epochs) - 1:
                with torch.no_grad():
                    val_loss = torch.nn.functional.mse_loss(model(xva), yva).item()
                if val_loss < best_val:
                    best_val = val_loss
                    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if best_state is not None:
            model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
        members.append((model, mean.copy(), std.copy()))
        best_vals.append(float(best_val))
    scorer = TorchScorer(
        members[0][0],
        mean,
        std,
        device,
        members=members,
        uncertainty_penalty=float(uncertainty_penalty),
    )
    pred = scorer.predict(x_val)
    corr = float(np.corrcoef(pred, y_val)[0, 1]) if len(pred) > 2 and np.std(pred) > 0 and np.std(y_val) > 0 else 0.0
    return scorer, {
        "val_mse": float(np.mean((pred - y_val) ** 2)),
        "val_corr": corr,
        "ensemble_size": float(max(1, int(ensemble_size))),
        "member_best_val_mse": float(np.mean(best_vals)) if best_vals else 0.0,
    }


def calibrate_min_pred_score(
    pred: np.ndarray,
    y: np.ndarray,
    meta: Sequence[Mapping[str, Any]],
    *,
    base_min_score: float,
    min_selected_days: int,
    min_selected_mean: float,
    risk_penalty: float,
) -> dict[str, Any]:
    """Pick a validation-derived trading hurdle from top predicted candidate days."""
    by_day: dict[int, list[int]] = {}
    for i, row in enumerate(meta):
        by_day.setdefault(int(row["day_start_i"]), []).append(i)
    top_rows: list[dict[str, float]] = []
    for day_start, idxs in sorted(by_day.items()):
        idx_arr = np.asarray(idxs, dtype=int)
        best_i = int(idx_arr[np.argmax(pred[idx_arr])])
        top_rows.append(
            {
                "day_start_i": float(day_start),
                "pred": float(pred[best_i]),
                "score": float(y[best_i]),
            }
        )
    if not top_rows:
        return {
            "threshold": float(base_min_score),
            "mode": "fixed_empty_validation",
            "selected_days": 0,
            "selected_mean_score": 0.0,
            "objective": 0.0,
        }
    top_pred = np.asarray([row["pred"] for row in top_rows], dtype=np.float64)
    top_score = np.asarray([row["score"] for row in top_rows], dtype=np.float64)
    thresholds = sorted(set([float(base_min_score), *[float(x) for x in np.quantile(top_pred, np.linspace(0.0, 0.90, 19))]]))
    best: dict[str, Any] | None = None
    for threshold in thresholds:
        selected = top_pred >= max(float(base_min_score), float(threshold))
        selected_days = int(selected.sum())
        if selected_days < int(min_selected_days):
            continue
        selected_scores = top_score[selected]
        selected_mean = float(selected_scores.mean()) if selected_days else 0.0
        if selected_mean < float(min_selected_mean):
            continue
        portfolio_scores = np.where(selected, top_score, 0.0)
        mean_score = float(portfolio_scores.mean())
        downside = portfolio_scores[portfolio_scores < 0.0]
        downside_risk = float(np.sqrt(np.mean(np.square(downside)))) if len(downside) else 0.0
        objective = mean_score - float(risk_penalty) * downside_risk
        candidate = {
            "threshold": max(float(base_min_score), float(threshold)),
            "mode": "validation_topday",
            "selected_days": selected_days,
            "validation_days": len(top_rows),
            "selected_mean_score": selected_mean,
            "portfolio_mean_score": mean_score,
            "downside_risk": downside_risk,
            "objective": objective,
            "top_pred_mean": float(top_pred.mean()),
            "top_pred_p50": float(np.quantile(top_pred, 0.50)),
            "top_pred_p75": float(np.quantile(top_pred, 0.75)),
            "top_pred_p90": float(np.quantile(top_pred, 0.90)),
        }
        if best is None or candidate["objective"] > best["objective"]:
            best = candidate
    if best is None:
        return {
            "threshold": 1.0e9,
            "mode": "validation_topday_reject_all",
            "selected_days": 0,
            "validation_days": len(top_rows),
            "selected_mean_score": float(top_score[top_pred >= float(base_min_score)].mean())
            if np.any(top_pred >= float(base_min_score))
            else 0.0,
            "portfolio_mean_score": 0.0,
            "downside_risk": 0.0,
            "objective": 0.0,
            "top_pred_mean": float(top_pred.mean()),
            "top_pred_p50": float(np.quantile(top_pred, 0.50)),
            "top_pred_p75": float(np.quantile(top_pred, 0.75)),
            "top_pred_p90": float(np.quantile(top_pred, 0.90)),
        }
    return best


def candidate_matrix(
    frames: Mapping[str, pd.DataFrame],
    templates: Sequence[PlanTemplate],
    *,
    day_start_i: int,
    symbols: Sequence[str],
) -> np.ndarray:
    mfeat = market_features(frames, day_start_i=day_start_i, symbols=symbols)
    return np.vstack([np.concatenate([mfeat, template_to_vector(template, symbols)]) for template in templates]).astype(np.float32)


def gradient_refine_candidates(
    scorer: TorchScorer,
    frames: Mapping[str, pd.DataFrame],
    templates: Sequence[PlanTemplate],
    *,
    day_start_i: int,
    day: int,
    symbols: Sequence[str],
    steps: int,
    lr: float,
    take: int,
    max_symbol_allocation: float,
    max_gross_exposure: float,
    move_penalty: float,
) -> list[PlanTemplate]:
    if int(steps) <= 0 or not templates:
        return []
    import torch

    mfeat = market_features(frames, day_start_i=day_start_i, symbols=symbols).astype(np.float32)
    base = np.vstack([template_to_vector(template, symbols) for template in templates]).astype(np.float32)
    vec = torch.tensor(base, dtype=torch.float32, device=scorer.device, requires_grad=True)
    vec0 = torch.tensor(base, dtype=torch.float32, device=scorer.device)
    mfeat_t = torch.tensor(np.tile(mfeat, (len(base), 1)), dtype=torch.float32, device=scorer.device)
    mean_t = torch.tensor(scorer.mean, dtype=torch.float32, device=scorer.device)
    std_t = torch.tensor(scorer.std, dtype=torch.float32, device=scorer.device)
    opt = torch.optim.Adam([vec], lr=float(lr))
    for _ in range(int(steps)):
        full = torch.cat([mfeat_t, vec], dim=1)
        pred = scorer.model((full - mean_t) / std_t).reshape(-1)
        movement = torch.mean(((vec - vec0) / std_t[-vec.shape[1] :]) ** 2, dim=1)
        loss = -(pred - float(move_penalty) * movement).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        with torch.no_grad():
            clipped = clamp_vector_array(
                vec.detach().cpu().numpy(),
                symbols,
                max_symbol_allocation=max_symbol_allocation,
                max_gross_exposure=max_gross_exposure,
            )
            vec.copy_(torch.tensor(clipped, dtype=torch.float32, device=scorer.device))
    final = vec.detach().cpu().numpy()
    full = np.hstack([np.tile(mfeat, (len(final), 1)), final])
    pred = scorer.predict(full)
    chosen = np.argsort(pred)[-int(take) :][::-1]
    return [
        vector_to_template(
            final[i],
            symbols,
            day=day,
            name=f"rolling_grad_{rank:03d}",
            rationale=f"rolling scorer gradient pred {pred[i]:.6f}",
        )
        for rank, i in enumerate(chosen)
    ]


def run_eval(
    frames: Mapping[str, pd.DataFrame],
    scorer: TorchScorer,
    *,
    symbols: Sequence[str],
    eval_start_i: int,
    eval_days: int,
    candidates_per_day: int,
    gradient_top: int,
    gradient_steps: int,
    rng: np.random.Generator,
    initial_cash: float,
    fee_bps: float,
    slippage_bps: float,
    max_active: int,
    max_symbol_allocation: float,
    max_gross_exposure: float,
    min_pred_score: float,
    return_weight: float,
    drawdown_penalty: float,
    trade_penalty: float,
    loss_penalty: float,
    rerank_calibration_days: int,
    rerank_top: int,
    rerank_calibration_weight: float,
    rerank_worst_weight: float,
    rerank_min_mean_score: float,
    rerank_require_latest_positive: bool,
    candidate_style: str,
) -> dict[str, Any]:
    equity = float(initial_cash)
    all_rows: list[dict[str, Any]] = []
    all_trades: list[dict[str, Any]] = []
    selected: list[dict[str, Any]] = []
    common = close_matrix(frames).index
    for day in range(int(eval_days)):
        day_start = int(eval_start_i) + day * 24
        templates = [
            random_template(
                rng,
                frames,
                day_start_i=day_start,
                day=day,
                symbols=symbols,
                name=f"eval_d{day}_t{i}",
                max_active=max_active,
                max_symbol_allocation=max_symbol_allocation,
                max_gross_exposure=max_gross_exposure,
                candidate_style=candidate_style,
            )
            for i in range(int(candidates_per_day))
        ]
        x = candidate_matrix(frames, templates, day_start_i=day_start, symbols=symbols)
        pred = scorer.predict(x)
        top_idx = np.argsort(pred)[-max(1, int(gradient_top)) :][::-1]
        grad_templates = gradient_refine_candidates(
            scorer,
            frames,
            [templates[i] for i in top_idx],
            day_start_i=day_start,
            day=day,
            symbols=symbols,
            steps=int(gradient_steps),
            lr=0.01,
            take=max(1, int(gradient_top)),
            max_symbol_allocation=max_symbol_allocation,
            max_gross_exposure=max_gross_exposure,
            move_penalty=0.02,
        )
        all_templates = templates + grad_templates
        pred2 = scorer.predict(candidate_matrix(frames, all_templates, day_start_i=day_start, symbols=symbols))
        combined_scores = pred2.astype(np.float64).copy()
        calibration_rows: dict[int, dict[str, float]] = {}
        if int(rerank_calibration_days) > 0 and int(rerank_top) > 0:
            top_for_rerank = np.argsort(pred2)[-min(int(rerank_top), len(pred2)) :][::-1]
            for candidate_i in top_for_rerank:
                cal_scores: list[float] = []
                cal_returns: list[float] = []
                for lag_day in range(1, int(rerank_calibration_days) + 1):
                    cal_start = day_start - lag_day * 24
                    if cal_start < 1:
                        continue
                    result = simulate_template_day(
                        frames,
                        all_templates[int(candidate_i)],
                        day_start_i=cal_start,
                        day=day - lag_day,
                        initial_cash=initial_cash,
                        fee_bps=fee_bps,
                        slippage_bps=slippage_bps,
                        return_weight=return_weight,
                        drawdown_penalty=drawdown_penalty,
                        trade_penalty=trade_penalty,
                        loss_penalty=loss_penalty,
                    )
                    cal_scores.append(float(result["score"]))
                    cal_returns.append(float(result["total_return"]))
                if cal_scores:
                    mean_score = float(np.mean(cal_scores))
                    worst_score = float(np.min(cal_scores))
                    latest_score = float(cal_scores[0])
                    combined = (
                        float(pred2[int(candidate_i)])
                        + float(rerank_calibration_weight) * mean_score
                        + float(rerank_worst_weight) * worst_score
                    )
                    if mean_score < float(rerank_min_mean_score):
                        combined = -1.0e9
                    if rerank_require_latest_positive and latest_score <= 0.0:
                        combined = -1.0e9
                    combined_scores[int(candidate_i)] = combined
                    calibration_rows[int(candidate_i)] = {
                        "mean_score": mean_score,
                        "worst_score": worst_score,
                        "latest_score": latest_score,
                        "mean_return": float(np.mean(cal_returns)) if cal_returns else 0.0,
                        "combined_score": combined,
                    }
                else:
                    combined_scores[int(candidate_i)] = -1.0e9
            not_reranked = np.ones(len(pred2), dtype=bool)
            not_reranked[top_for_rerank] = False
            combined_scores[not_reranked] = -1.0e9
        best_i = int(np.argmax(combined_scores))
        best_pred = float(pred2[best_i])
        chosen = all_templates[best_i]
        if best_pred < float(min_pred_score) or combined_scores[best_i] <= -1.0e8:
            chosen = make_template_from_arrays(
                np.zeros(len(symbols)),
                np.zeros(len(symbols)),
                np.zeros(len(symbols)),
                np.full(len(symbols), 100.0),
                np.full(len(symbols), 100.0),
                symbols,
                day=day,
                name="cash_pred_hurdle",
            )
        from experiments.binance_deepseek_planner.deepseek_iterative_replanner import instantiate_plan

        plan = instantiate_plan(chosen, frames, day_start_i=day_start, day=day)
        start_equity = equity
        equity, trades, rows = simulate_day(
            frames,
            plan,
            start_i=day_start,
            end_i=day_start + 24,
            starting_equity=equity,
            fee_bps=fee_bps,
            slippage_bps=slippage_bps,
        )
        day_metrics = _metrics(rows, start_equity)
        all_rows.extend(rows)
        all_trades.extend([asdict(trade) for trade in trades])
        selected.append(
            {
                "day": day,
                "timestamp": common[day_start].isoformat(),
                "template": asdict(chosen),
                "pred_score": best_pred,
                "combined_score": float(combined_scores[best_i]),
                "calibration": calibration_rows.get(best_i, {}),
                "actual_return": float(day_metrics["total_return"]),
                "actual_max_drawdown": float(day_metrics["max_drawdown"]),
                "trades": len(trades),
                "equity": float(equity),
            }
        )
        print(
            f"eval_day={day} selected={chosen.name} pred={best_pred:.5f} "
            f"actual={day_metrics['total_return'] * 100:+.2f}% equity=${equity:.2f}",
            flush=True,
        )
    metrics = _metrics(all_rows, initial_cash)
    return {
        "initial_cash": float(initial_cash),
        "final_equity": float(equity),
        "metrics": metrics,
        "selected": selected,
        "trades": all_trades,
        "equity_curve": all_rows,
    }


def write_report(path: Path, result: Mapping[str, Any]) -> None:
    metrics = result["eval"]["metrics"]
    train = result["training"]
    calibration = train.get("calibration", {})
    threshold = float(calibration.get("threshold", result["config"].get("min_pred_score", 0.0)))
    lines = [
        "# Rolling Energy Trainer",
        "",
        f"- Device: `{train['device']}`",
        f"- Train samples: `{train['train_samples']}`",
        f"- Validation samples: `{train['val_samples']}`",
        f"- Validation MSE: `{train['val_mse']:.8f}`",
        f"- Validation correlation: `{train['val_corr']:.4f}`",
        f"- Ensemble size: `{int(train.get('ensemble_size', 1))}`",
        f"- Candidate style: `{result['config'].get('candidate_style', 'legacy')}`",
        f"- Label target: `{result['config'].get('label_target', 'score')}`",
        f"- Threshold mode: `{calibration.get('mode', result['config'].get('threshold_mode', 'fixed'))}`",
        f"- Active prediction threshold: `{threshold:.6f}`",
        f"- Calibration selected days: `{calibration.get('selected_days', 'n/a')}`",
        f"- Initial cash: `${result['eval']['initial_cash']:,.2f}`",
        f"- Final equity: `${result['eval']['final_equity']:,.2f}`",
        f"- Total return: `{metrics['total_return'] * 100:+.2f}%`",
        f"- Monthly-equivalent return: `{metrics['monthly_return'] * 100:+.2f}%`",
        f"- Max drawdown: `{metrics['max_drawdown'] * 100:.2f}%`",
        f"- Sortino: `{metrics['sortino']:.2f}`",
        f"- Trades: `{len(result['eval']['trades'])}`",
        "",
        "## Selected Days",
        "",
    ]
    for row in result["eval"]["selected"]:
        active = [leg for leg in row["template"]["legs"] if leg["side"] != "flat" and leg["allocation_pct"] > 0]
        summary = ", ".join(f"{leg['side']} {leg['symbol']} {leg['allocation_pct']:.1f}% off={leg['entry_offset_bps']:.0f}" for leg in active) or "flat"
        cal = row.get("calibration", {})
        cal_text = ""
        if cal:
            cal_text = f" cal_mean={cal.get('mean_score', 0.0):+.5f} cal_worst={cal.get('worst_score', 0.0):+.5f}"
        lines.append(
            f"- Day {row['day']} `{row['template']['name']}` pred={row['pred_score']:.5f} "
            f"combined={row.get('combined_score', row['pred_score']):.5f}{cal_text} "
            f"actual={row['actual_return'] * 100:+.2f}% {summary}"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default="trainingdatahourlybinance")
    parser.add_argument("--symbols", nargs="+", default=list(DEFAULT_SYMBOLS))
    parser.add_argument("--train-days", type=int, default=240)
    parser.add_argument("--val-days", type=int, default=45)
    parser.add_argument("--eval-days", type=int, default=7)
    parser.add_argument("--history-hours", type=int, default=240)
    parser.add_argument("--stride-hours", type=int, default=24)
    parser.add_argument("--templates-per-day", type=int, default=48)
    parser.add_argument("--eval-candidates", type=int, default=2048)
    parser.add_argument("--gradient-top", type=int, default=64)
    parser.add_argument("--gradient-steps", type=int, default=40)
    parser.add_argument("--max-active", type=int, default=3)
    parser.add_argument("--max-symbol-allocation", type=float, default=25.0)
    parser.add_argument("--max-gross-exposure", type=float, default=70.0)
    parser.add_argument("--candidate-style", choices=["legacy", "frontier", "mixed"], default="legacy")
    parser.add_argument("--min-pred-score", type=float, default=0.0005)
    parser.add_argument("--threshold-mode", choices=["fixed", "validation_topday"], default="validation_topday")
    parser.add_argument("--calibration-min-selected-days", type=int, default=0)
    parser.add_argument("--calibration-min-selected-mean", type=float, default=0.0)
    parser.add_argument("--calibration-risk-penalty", type=float, default=0.25)
    parser.add_argument("--initial-cash", type=float, default=DEFAULT_INITIAL_CASH)
    parser.add_argument("--fee-bps", type=float, default=10.0)
    parser.add_argument("--slippage-bps", type=float, default=5.0)
    parser.add_argument("--return-weight", type=float, default=1.0)
    parser.add_argument("--drawdown-penalty", type=float, default=0.75)
    parser.add_argument("--trade-penalty", type=float, default=0.00015)
    parser.add_argument("--loss-penalty", type=float, default=0.05)
    parser.add_argument("--label-target", choices=["score", "total_return"], default="score")
    parser.add_argument("--rerank-calibration-days", type=int, default=0)
    parser.add_argument("--rerank-top", type=int, default=24)
    parser.add_argument("--rerank-calibration-weight", type=float, default=1.0)
    parser.add_argument("--rerank-worst-weight", type=float, default=0.5)
    parser.add_argument("--rerank-min-mean-score", type=float, default=-1.0e9)
    parser.add_argument("--rerank-require-latest-positive", action="store_true")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--ensemble-size", type=int, default=1)
    parser.add_argument("--uncertainty-penalty", type=float, default=0.0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=20260427)
    parser.add_argument("--output-dir", default="experiments/binance_deepseek_planner/results")
    args = parser.parse_args(argv)

    rng = np.random.default_rng(int(args.seed))
    total_hours = int(args.history_hours) + 24 * (int(args.train_days) + int(args.val_days) + int(args.eval_days)) + 24
    frames = load_hourly_panel(args.data_root, args.symbols, tail_hours=total_hours)
    close = close_matrix(frames)
    train_start = int(args.history_hours)
    train_end = train_start + int(args.train_days) * 24
    val_end = train_end + int(args.val_days) * 24
    eval_end = val_end + int(args.eval_days) * 24
    if eval_end + 24 > len(close):
        raise RuntimeError("not enough loaded data for requested split")
    print(f"loaded {len(close)} rows {close.index[0]} -> {close.index[-1]}", flush=True)
    print("building train dataset", flush=True)
    x_train, y_train, train_meta = build_dataset(
        frames,
        symbols=args.symbols,
        start_i=train_start,
        end_i=train_end,
        stride_hours=int(args.stride_hours),
        templates_per_day=int(args.templates_per_day),
        rng=rng,
        initial_cash=float(args.initial_cash),
        fee_bps=float(args.fee_bps),
        slippage_bps=float(args.slippage_bps),
        return_weight=float(args.return_weight),
        drawdown_penalty=float(args.drawdown_penalty),
        trade_penalty=float(args.trade_penalty),
        loss_penalty=float(args.loss_penalty),
        label_target=str(args.label_target),
        max_active=int(args.max_active),
        max_symbol_allocation=float(args.max_symbol_allocation),
        max_gross_exposure=float(args.max_gross_exposure),
        candidate_style=str(args.candidate_style),
    )
    print("building validation dataset", flush=True)
    x_val, y_val, val_meta = build_dataset(
        frames,
        symbols=args.symbols,
        start_i=train_end,
        end_i=val_end,
        stride_hours=int(args.stride_hours),
        templates_per_day=int(args.templates_per_day),
        rng=rng,
        initial_cash=float(args.initial_cash),
        fee_bps=float(args.fee_bps),
        slippage_bps=float(args.slippage_bps),
        return_weight=float(args.return_weight),
        drawdown_penalty=float(args.drawdown_penalty),
        trade_penalty=float(args.trade_penalty),
        loss_penalty=float(args.loss_penalty),
        label_target=str(args.label_target),
        max_active=int(args.max_active),
        max_symbol_allocation=float(args.max_symbol_allocation),
        max_gross_exposure=float(args.max_gross_exposure),
        candidate_style=str(args.candidate_style),
    )
    import torch

    device = str(args.device)
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
    print(f"training scorer on {device}", flush=True)
    scorer, train_stats = train_torch_scorer(
        x_train,
        y_train,
        x_val,
        y_val,
        device=device,
        epochs=int(args.epochs),
        batch_size=int(args.batch_size),
        lr=float(args.lr),
        seed=int(args.seed),
        ensemble_size=int(args.ensemble_size),
        uncertainty_penalty=float(args.uncertainty_penalty),
    )
    if args.threshold_mode == "validation_topday":
        val_pred = scorer.predict(x_val)
        min_selected_days = int(args.calibration_min_selected_days)
        if min_selected_days <= 0:
            validation_day_count = max(1, int(math.ceil((val_end - train_end) / max(1, int(args.stride_hours)))))
            min_selected_days = max(3, int(math.ceil(validation_day_count * 0.20)))
        calibration = calibrate_min_pred_score(
            val_pred,
            y_val,
            val_meta,
            base_min_score=float(args.min_pred_score),
            min_selected_days=min_selected_days,
            min_selected_mean=float(args.calibration_min_selected_mean),
            risk_penalty=float(args.calibration_risk_penalty),
        )
    else:
        calibration = {
            "threshold": float(args.min_pred_score),
            "mode": "fixed",
            "selected_days": 0,
            "selected_mean_score": 0.0,
            "objective": 0.0,
        }
    active_threshold = float(calibration["threshold"])
    print(
        f"active threshold={active_threshold:.6f} mode={calibration['mode']} "
        f"selected_days={calibration.get('selected_days', 0)}",
        flush=True,
    )
    eval_result = run_eval(
        frames,
        scorer,
        symbols=args.symbols,
        eval_start_i=val_end,
        eval_days=int(args.eval_days),
        candidates_per_day=int(args.eval_candidates),
        gradient_top=int(args.gradient_top),
        gradient_steps=int(args.gradient_steps),
        rng=rng,
        initial_cash=float(args.initial_cash),
        fee_bps=float(args.fee_bps),
        slippage_bps=float(args.slippage_bps),
        max_active=int(args.max_active),
        max_symbol_allocation=float(args.max_symbol_allocation),
        max_gross_exposure=float(args.max_gross_exposure),
        min_pred_score=active_threshold,
        return_weight=float(args.return_weight),
        drawdown_penalty=float(args.drawdown_penalty),
        trade_penalty=float(args.trade_penalty),
        loss_penalty=float(args.loss_penalty),
        rerank_calibration_days=int(args.rerank_calibration_days),
        rerank_top=int(args.rerank_top),
        rerank_calibration_weight=float(args.rerank_calibration_weight),
        rerank_worst_weight=float(args.rerank_worst_weight),
        rerank_min_mean_score=float(args.rerank_min_mean_score),
        rerank_require_latest_positive=bool(args.rerank_require_latest_positive),
        candidate_style=str(args.candidate_style),
    )
    result = {
        "training": {
            **train_stats,
            "calibration": calibration,
            "device": device,
            "train_samples": int(len(x_train)),
            "val_samples": int(len(x_val)),
            "train_start": close.index[train_start].isoformat(),
            "train_end": close.index[train_end - 1].isoformat(),
            "val_start": close.index[train_end].isoformat(),
            "val_end": close.index[val_end - 1].isoformat(),
            "eval_start": close.index[val_end].isoformat(),
            "eval_end": close.index[eval_end - 1].isoformat(),
            "score_objective": {
                "return_weight": float(args.return_weight),
                "drawdown_penalty": float(args.drawdown_penalty),
                "trade_penalty": float(args.trade_penalty),
                "loss_penalty": float(args.loss_penalty),
                "label_target": str(args.label_target),
            },
        },
        "config": vars(args),
        "eval": eval_result,
    }
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    json_path = output_dir / f"rolling_energy_trainer_{stamp}.json"
    md_path = output_dir / f"rolling_energy_trainer_{stamp}.md"
    json_path.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    write_report(md_path, result)
    metrics = eval_result["metrics"]
    print(
        f"final=${eval_result['final_equity']:.2f} "
        f"return={metrics['total_return'] * 100:+.2f}% "
        f"monthly={metrics['monthly_return'] * 100:+.2f}% "
        f"dd={metrics['max_drawdown'] * 100:.2f}% "
        f"trades={len(eval_result['trades'])} val_corr={train_stats['val_corr']:.4f}",
        flush=True,
    )
    print(f"wrote {json_path}")
    print(f"wrote {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
