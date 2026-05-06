#!/usr/bin/env python3
"""DeepSeek seed -> boosted refit -> DeepSeek refit -> neural refit agent.

The intent is to make LLM calls propose compact trading "energy" templates, then
let local simulator-guided models move the parameters less aggressively at each
stage.  Research defaults to DeepSeek flash/no-think JSON prompts and never
places live orders.
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

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from experiments.binance_deepseek_planner.deepseek_iterative_replanner import (
    DEFAULT_DEEPSEEK_MODEL,
    TemplateLeg,
    PlanTemplate,
    _call_and_parse_templates,
    _load_env_real,
    _mutate_templates,
    _parse_templates,
    _prompt_payload,
    call_deepseek_templates,
    instantiate_plan,
    score_templates,
)
from experiments.binance_openai_planner.gpt55_week_replanner import DEFAULT_INITIAL_CASH, _metrics, simulate_day
from experiments.binance_transition_graph.graph_policy import DEFAULT_SYMBOLS, close_matrix, load_hourly_panel


DEFAULT_FLASH_MODEL = "deepseek-v4-flash"


def template_to_vector(template: PlanTemplate, symbols: Sequence[str]) -> np.ndarray:
    by_symbol = {leg.symbol: leg for leg in template.legs}
    values: list[float] = []
    for symbol in symbols:
        leg = by_symbol.get(str(symbol))
        if leg is None or leg.side == "flat":
            values.extend([0.0, 0.0, 0.0, 120.0, 90.0])
            continue
        side = 1.0 if leg.side == "long" else -1.0
        values.extend(
            [
                side,
                float(leg.allocation_pct),
                float(leg.entry_offset_bps),
                float(leg.take_profit_bps),
                float(leg.stop_loss_bps),
            ]
        )
    return np.asarray(values, dtype=np.float64)


def vector_to_template(vec: np.ndarray, symbols: Sequence[str], *, day: int, name: str, rationale: str) -> PlanTemplate:
    legs: list[TemplateLeg] = []
    arr = np.asarray(vec, dtype=np.float64).reshape(len(symbols), 5).copy()
    for idx, symbol in enumerate(symbols):
        side_val, alloc, entry, tp, sl = arr[idx]
        if side_val > 0.35:
            side = "long"
            entry = min(0.0, entry)
        elif side_val < -0.35:
            side = "short"
            entry = max(0.0, entry)
        else:
            side = "flat"
            alloc = 0.0
            entry = 0.0
        legs.append(
            TemplateLeg(
                symbol=str(symbol),
                side=side,
                allocation_pct=max(0.0, min(60.0, float(alloc))),
                entry_offset_bps=max(-250.0, min(250.0, float(entry))),
                take_profit_bps=max(25.0, min(1_500.0, float(tp))),
                stop_loss_bps=max(25.0, min(1_500.0, float(sl))),
                confidence=0.5,
                rationale=rationale,
            )
        )
    gross = sum(leg.allocation_pct for leg in legs if leg.side != "flat")
    if gross > 150.0:
        scale = 150.0 / gross
        legs = [
            TemplateLeg(**{**asdict(leg), "allocation_pct": leg.allocation_pct * scale})
            if leg.side != "flat"
            else leg
            for leg in legs
        ]
        gross = 150.0
    return PlanTemplate(
        name=name,
        rationale=rationale,
        day_index=int(day),
        cash_pct=max(0.0, 100.0 - gross),
        gross_exposure_pct=float(gross),
        risk_notes=rationale,
        legs=legs,
    )


def mutate_vector(vec: np.ndarray, rng: np.random.Generator, *, aggression: float, symbols: Sequence[str]) -> np.ndarray:
    out = np.asarray(vec, dtype=np.float64).reshape(len(symbols), 5).copy()
    for i in range(len(symbols)):
        if rng.random() < 0.04 * aggression:
            out[i, 0] = rng.choice([-1.0, 0.0, 1.0])
            if abs(out[i, 0]) > 0.35 and out[i, 1] <= 0.0:
                out[i, 1] = rng.uniform(5.0, 25.0)
        if abs(out[i, 0]) <= 0.35:
            out[i, 1] = 0.0
            out[i, 2] = 0.0
            continue
        out[i, 1] += rng.normal(0.0, 14.0 * aggression)
        out[i, 2] += rng.normal(0.0, 35.0 * aggression)
        out[i, 3] *= math.exp(rng.normal(0.0, 0.28 * aggression))
        out[i, 4] *= math.exp(rng.normal(0.0, 0.28 * aggression))
        out[i, 1] = np.clip(out[i, 1], 0.0, 60.0)
        out[i, 3] = np.clip(out[i, 3], 25.0, 1_500.0)
        out[i, 4] = np.clip(out[i, 4], 25.0, 1_500.0)
        if out[i, 0] > 0:
            out[i, 2] = np.clip(min(0.0, out[i, 2]), -250.0, 0.0)
        else:
            out[i, 2] = np.clip(max(0.0, out[i, 2]), 0.0, 250.0)
    gross = float(np.sum(np.where(np.abs(out[:, 0]) > 0.35, out[:, 1], 0.0)))
    if gross > 150.0:
        out[:, 1] *= 150.0 / gross
    return out.reshape(-1)


def clamp_vector_array(
    arr: np.ndarray,
    symbols: Sequence[str],
    *,
    max_symbol_allocation: float,
    max_gross_exposure: float,
) -> np.ndarray:
    out = np.asarray(arr, dtype=np.float64).reshape(-1, len(symbols), 5).copy()
    out[:, :, 0] = np.clip(out[:, :, 0], -1.25, 1.25)
    out[:, :, 1] = np.clip(out[:, :, 1], 0.0, float(max_symbol_allocation))
    out[:, :, 2] = np.clip(out[:, :, 2], -250.0, 250.0)
    out[:, :, 3] = np.clip(out[:, :, 3], 25.0, 1_500.0)
    out[:, :, 4] = np.clip(out[:, :, 4], 25.0, 1_500.0)
    active = np.abs(out[:, :, 0]) > 0.35
    out[:, :, 1] *= active
    long_mask = out[:, :, 0] > 0.35
    short_mask = out[:, :, 0] < -0.35
    out[:, :, 2] = np.where(long_mask, np.minimum(out[:, :, 2], 0.0), out[:, :, 2])
    out[:, :, 2] = np.where(short_mask, np.maximum(out[:, :, 2], 0.0), out[:, :, 2])
    out[:, :, 2] = np.where(active, out[:, :, 2], 0.0)
    gross = out[:, :, 1].sum(axis=1)
    over = gross > float(max_gross_exposure)
    if np.any(over):
        scales = np.ones_like(gross)
        scales[over] = float(max_gross_exposure) / np.maximum(gross[over], 1e-12)
        out[:, :, 1] *= scales[:, None]
    return out.reshape(np.asarray(arr).shape)


def evaluate_population(
    frames: Mapping[str, Any],
    templates: Sequence[PlanTemplate],
    *,
    current_day_start_i: int,
    day: int,
    calibration_days: int,
    initial_cash: float,
    fee_bps: float,
    slippage_bps: float,
) -> list[dict[str, Any]]:
    return score_templates(
        frames,
        templates,
        current_day_start_i=current_day_start_i,
        day=day,
        calibration_days=calibration_days,
        initial_cash=initial_cash,
        fee_bps=fee_bps,
        slippage_bps=slippage_bps,
    )


def fit_boosted_model(x: np.ndarray, y: np.ndarray) -> Any:
    try:
        from xgboost import XGBRegressor  # type: ignore

        model = XGBRegressor(
            n_estimators=120,
            max_depth=3,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            objective="reg:squarederror",
            random_state=17,
        )
    except Exception:
        from sklearn.ensemble import HistGradientBoostingRegressor

        model = HistGradientBoostingRegressor(max_iter=160, learning_rate=0.04, l2_regularization=0.01, random_state=17)
    model.fit(x, y)
    return model


def fit_neural_model(x: np.ndarray, y: np.ndarray, rng: np.random.Generator) -> tuple[Any, np.ndarray, np.ndarray] | None:
    if len(x) < 20:
        return None
    try:
        import torch
        import torch.nn as nn
    except Exception:
        return None
    mean = x.mean(axis=0)
    std = x.std(axis=0) + 1e-6
    xt = torch.tensor((x - mean) / std, dtype=torch.float32)
    yt = torch.tensor(y.reshape(-1, 1), dtype=torch.float32)
    torch.manual_seed(int(rng.integers(1, 2**31 - 1)))
    model = nn.Sequential(nn.Linear(x.shape[1], 64), nn.SiLU(), nn.Linear(64, 32), nn.SiLU(), nn.Linear(32, 1))
    opt = torch.optim.AdamW(model.parameters(), lr=0.01, weight_decay=0.001)
    for _ in range(180):
        pred = model(xt)
        loss = torch.nn.functional.mse_loss(pred, yt)
        opt.zero_grad()
        loss.backward()
        opt.step()
    return model, mean, std


def neural_predict(bundle: tuple[Any, np.ndarray, np.ndarray], x: np.ndarray) -> np.ndarray:
    import torch

    model, mean, std = bundle
    with torch.no_grad():
        pred = model(torch.tensor((x - mean) / std, dtype=torch.float32)).detach().cpu().numpy().reshape(-1)
    return pred


def top_templates_from_scores(templates: Sequence[PlanTemplate], scores: Sequence[dict[str, Any]], n: int) -> list[PlanTemplate]:
    by_name = {template.name: template for template in templates}
    out: list[PlanTemplate] = []
    for row in sorted(scores, key=lambda r: r.get("selection_score", r["score"]), reverse=True):
        template = by_name.get(str(row["template_name"]))
        if template is not None:
            out.append(template)
        if len(out) >= n:
            break
    return out


def rank_scores(scores: list[dict[str, Any]], args: argparse.Namespace) -> list[dict[str, Any]]:
    for row in scores:
        evals = list(row.get("calibration_evals", []) or [])
        returns = np.asarray([float(item.get("total_return", 0.0)) for item in evals], dtype=np.float64)
        if len(returns) == 0:
            row["positive_calibration_days"] = 0
            row["latest_calibration_return"] = 0.0
            row["selection_score"] = float(row.get("score", 0.0))
            continue
        positive_days = int(np.sum(returns > 0.0))
        latest = float(returns[-1])
        worst = float(np.min(returns))
        volatility = float(np.std(returns))
        score = float(row.get("score", 0.0))
        if args.selection_mode == "robust":
            score = score + 0.50 * latest + 0.25 * worst - 0.25 * volatility
        if "cash" not in str(row.get("template_name", "")).lower():
            if positive_days < int(args.min_positive_calibration_days):
                score -= 1.0
            if bool(args.require_latest_calibration_positive) and latest <= 0.0:
                score -= 1.0
        row["positive_calibration_days"] = positive_days
        row["latest_calibration_return"] = latest
        row["selection_score"] = float(score)
    scores.sort(
        key=lambda row: (
            row.get("selection_score", row["score"]),
            row.get("score", 0.0),
            row.get("median_return", 0.0),
            -row.get("max_drawdown", 0.0),
        ),
        reverse=True,
    )
    return scores


def propose_with_model(
    model: Any,
    base_templates: Sequence[PlanTemplate],
    symbols: Sequence[str],
    *,
    day: int,
    rng: np.random.Generator,
    aggression: float,
    pool_size: int,
    take: int,
    name_prefix: str,
) -> list[PlanTemplate]:
    pool_vecs: list[np.ndarray] = []
    bases = [template_to_vector(t, symbols) for t in base_templates]
    for idx in range(int(pool_size)):
        base = bases[idx % len(bases)]
        pool_vecs.append(mutate_vector(base, rng, aggression=aggression, symbols=symbols))
    x = np.vstack(pool_vecs)
    pred = np.asarray(model.predict(x), dtype=float)
    chosen = np.argsort(pred)[-int(take) :][::-1]
    return [
        vector_to_template(
            x[i],
            symbols,
            day=day,
            name=f"{name_prefix}_{rank:03d}",
            rationale=f"{name_prefix} predicted score {pred[i]:.6f}",
        )
        for rank, i in enumerate(chosen)
    ]


def propose_with_neural(
    bundle: tuple[Any, np.ndarray, np.ndarray],
    base_templates: Sequence[PlanTemplate],
    symbols: Sequence[str],
    *,
    day: int,
    rng: np.random.Generator,
    aggression: float,
    pool_size: int,
    take: int,
) -> list[PlanTemplate]:
    pool_vecs: list[np.ndarray] = []
    bases = [template_to_vector(t, symbols) for t in base_templates]
    for idx in range(int(pool_size)):
        pool_vecs.append(mutate_vector(bases[idx % len(bases)], rng, aggression=aggression, symbols=symbols))
    x = np.vstack(pool_vecs)
    pred = neural_predict(bundle, x)
    chosen = np.argsort(pred)[-int(take) :][::-1]
    return [
        vector_to_template(
            x[i],
            symbols,
            day=day,
            name=f"neural_refit_{rank:03d}",
            rationale=f"neural predicted score {pred[i]:.6f}",
        )
        for rank, i in enumerate(chosen)
    ]


def propose_with_neural_gradient(
    bundle: tuple[Any, np.ndarray, np.ndarray],
    base_templates: Sequence[PlanTemplate],
    symbols: Sequence[str],
    *,
    day: int,
    rng: np.random.Generator,
    steps: int,
    lr: float,
    restarts_per_base: int,
    take: int,
    max_symbol_allocation: float,
    max_gross_exposure: float,
    move_penalty: float,
) -> list[PlanTemplate]:
    if int(steps) <= 0 or int(restarts_per_base) <= 0 or not base_templates:
        return []
    import torch

    model, mean, std = bundle
    starts: list[np.ndarray] = []
    for template in base_templates:
        base = template_to_vector(template, symbols)
        starts.append(base)
        for _ in range(int(restarts_per_base) - 1):
            starts.append(mutate_vector(base, rng, aggression=0.25, symbols=symbols))
    start_np = np.vstack(starts)
    start_np = clamp_vector_array(
        start_np,
        symbols,
        max_symbol_allocation=max_symbol_allocation,
        max_gross_exposure=max_gross_exposure,
    )
    x = torch.tensor(start_np, dtype=torch.float32, requires_grad=True)
    x0 = torch.tensor(start_np, dtype=torch.float32)
    mean_t = torch.tensor(mean, dtype=torch.float32)
    std_t = torch.tensor(std, dtype=torch.float32)
    opt = torch.optim.Adam([x], lr=float(lr))
    for _ in range(int(steps)):
        pred = model((x - mean_t) / std_t).reshape(-1)
        movement = torch.mean(((x - x0) / std_t) ** 2, dim=1)
        # Encourage predicted score, but keep the local move small.
        loss = -(pred - float(move_penalty) * movement).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        with torch.no_grad():
            clipped = clamp_vector_array(
                x.detach().cpu().numpy(),
                symbols,
                max_symbol_allocation=max_symbol_allocation,
                max_gross_exposure=max_gross_exposure,
            )
            x.copy_(torch.tensor(clipped, dtype=torch.float32))
    final_np = x.detach().cpu().numpy()
    pred = neural_predict(bundle, final_np)
    chosen = np.argsort(pred)[-int(take) :][::-1]
    return [
        vector_to_template(
            final_np[i],
            symbols,
            day=day,
            name=f"gradient_refit_{rank:03d}",
            rationale=f"neural gradient predicted score {pred[i]:.6f}",
        )
        for rank, i in enumerate(chosen)
    ]


def cem_refine_templates(
    frames: Mapping[str, Any],
    base_templates: Sequence[PlanTemplate],
    symbols: Sequence[str],
    *,
    day: int,
    current_day_start_i: int,
    calibration_days: int,
    initial_cash: float,
    fee_bps: float,
    slippage_bps: float,
    rng: np.random.Generator,
    iterations: int,
    population: int,
    elites: int,
    aggression: float,
    take: int,
    max_symbol_allocation: float,
    max_gross_exposure: float,
) -> tuple[list[PlanTemplate], list[dict[str, Any]]]:
    if int(iterations) <= 0 or int(population) <= 0 or not base_templates:
        return [], []
    bases = np.vstack([template_to_vector(template, symbols) for template in base_templates])
    mean = bases[0].copy()
    std = np.tile(np.array([0.45, 10.0, 35.0, 80.0, 70.0], dtype=np.float64), len(symbols)) * float(aggression)
    best_templates: list[PlanTemplate] = []
    best_scores: list[dict[str, Any]] = []
    for iteration in range(int(iterations)):
        samples = rng.normal(mean, std, size=(int(population), mean.shape[0]))
        samples[: len(bases)] = bases[: min(len(bases), len(samples))]
        samples = clamp_vector_array(
            samples,
            symbols,
            max_symbol_allocation=max_symbol_allocation,
            max_gross_exposure=max_gross_exposure,
        )
        templates = [
            vector_to_template(
                samples[i],
                symbols,
                day=day,
                name=f"cem_i{iteration}_{i:03d}",
                rationale=f"CEM iteration {iteration}",
            )
            for i in range(len(samples))
        ]
        scores = evaluate_population(
            frames,
            templates,
            current_day_start_i=current_day_start_i,
            day=day,
            calibration_days=calibration_days,
            initial_cash=initial_cash,
            fee_bps=fee_bps,
            slippage_bps=slippage_bps,
        )
        by_name = {template.name: template for template in templates}
        scores.sort(key=lambda row: (row["score"], row["median_return"], -row["max_drawdown"]), reverse=True)
        best_scores.extend(scores[: int(take)])
        best_templates.extend([by_name[row["template_name"]] for row in scores[: int(take)]])
        elite_scores = scores[: max(1, min(int(elites), len(scores)))]
        elite_vecs = np.vstack([template_to_vector(by_name[row["template_name"]], symbols) for row in elite_scores])
        mean = elite_vecs.mean(axis=0)
        std = np.maximum(elite_vecs.std(axis=0), np.tile(np.array([0.12, 2.0, 8.0, 18.0, 18.0]), len(symbols)))
    return best_templates[: int(take)], best_scores[: int(take)]


def deepseek_refit(
    args: argparse.Namespace,
    *,
    frames: Mapping[str, Any],
    day: int,
    day_start_i: int,
    equity: float,
    top_rows: Sequence[dict[str, Any]],
    best_template: PlanTemplate,
) -> list[PlanTemplate]:
    payload = _prompt_payload(
        frames,
        day=day,
        day_start_i=day_start_i,
        history_hours=int(args.history_hours),
        equity=equity,
        candidates=1,
        calibration_days=int(args.calibration_days),
        feature_style=str(args.feature_style),
        line_bars=int(args.line_bars),
        prompt_mode=str(args.prompt_mode),
        previous_feedback=[],
    )
    payload["task"] = "json only: mutate the current best daily template with small changes."
    payload["current_best_template"] = asdict(best_template)
    payload["top_simulated_scores"] = [
        {
            "name": row["template_name"],
            "score": row["score"],
            "median_return": row["median_return"],
            "worst_return": row["worst_return"],
            "max_drawdown": row["max_drawdown"],
            "avg_trades": row["avg_trades"],
        }
        for row in list(top_rows)[:5]
    ]
    payload["mutation_constraints"] = {
        "max_allocation_change_pct_points": 10,
        "max_entry_offset_change_bps": 25,
        "max_take_profit_change_bps": 40,
        "max_stop_loss_change_bps": 40,
        "prefer_less_aggressive_than_current": True,
        "return_exactly_one_candidate": True,
    }
    raw = call_deepseek_templates(
        payload,
        model=str(args.model),
        max_tokens=int(args.max_tokens),
        temperature=float(args.temperature),
        timeout_s=float(args.api_timeout_s),
        prompt_mode=str(args.prompt_mode),
    )
    return _parse_templates(raw, expected_day=day, allowed_symbols=args.symbols)


def fallback_seed_templates(symbols: Sequence[str], *, day: int) -> list[PlanTemplate]:
    legs: list[TemplateLeg] = []
    for idx, symbol in enumerate(symbols):
        if idx == 0:
            side, alloc, entry = "long", 15.0, -35.0
        elif idx == 1:
            side, alloc, entry = "short", 10.0, 35.0
        else:
            side, alloc, entry = "flat", 0.0, 0.0
        legs.append(
            TemplateLeg(
                symbol=str(symbol),
                side=side,
                allocation_pct=alloc,
                entry_offset_bps=entry,
                take_profit_bps=160.0,
                stop_loss_bps=110.0,
                confidence=0.4,
                rationale="fallback seed",
            )
        )
    return [
        PlanTemplate(
            name="fallback_pair_seed",
            rationale="fallback deterministic pair seed",
            day_index=day,
            cash_pct=75.0,
            gross_exposure_pct=25.0,
            risk_notes="fallback deterministic pair seed",
            legs=legs,
        )
    ]


def drop_example_templates(templates: Sequence[PlanTemplate]) -> list[PlanTemplate]:
    return [template for template in templates if "example" not in template.name.lower()]


def clamp_template_risk(
    template: PlanTemplate,
    *,
    risk_scale: float,
    max_symbol_allocation: float,
    max_gross_exposure: float,
) -> PlanTemplate:
    legs: list[TemplateLeg] = []
    for leg in template.legs:
        if leg.side == "flat":
            legs.append(leg)
            continue
        legs.append(
            TemplateLeg(
                **{
                    **asdict(leg),
                    "allocation_pct": min(float(max_symbol_allocation), max(0.0, leg.allocation_pct * float(risk_scale))),
                }
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
        name=template.name,
        rationale=template.rationale,
        day_index=template.day_index,
        cash_pct=max(0.0, 100.0 - gross),
        gross_exposure_pct=gross,
        risk_notes=template.risk_notes,
        legs=legs,
    )


def clamp_templates(templates: Sequence[PlanTemplate], args: argparse.Namespace) -> list[PlanTemplate]:
    return [
        clamp_template_risk(
            template,
            risk_scale=float(args.risk_scale),
            max_symbol_allocation=float(args.max_symbol_allocation),
            max_gross_exposure=float(args.max_gross_exposure),
        )
        for template in templates
    ]


def run_experiment(args: argparse.Namespace) -> dict[str, Any]:
    if args.load_env_real:
        _load_env_real()
    rng = np.random.default_rng(int(args.seed))
    total_hours = int(args.history_hours) + int(args.days) * 24 + int(args.decision_lag_hours)
    frames = load_hourly_panel(args.data_root, args.symbols, tail_hours=total_hours)
    common = close_matrix(frames).index
    start_i = int(args.history_hours) + int(args.decision_lag_hours)
    equity = float(args.initial_cash)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    raw_paths: list[str] = []
    all_trades: list[dict[str, Any]] = []
    all_equity_rows: list[dict[str, Any]] = []
    day_summaries: list[dict[str, Any]] = []
    selected_templates: list[dict[str, Any]] = []
    selected_plans: list[dict[str, Any]] = []

    for day in range(int(args.days)):
        day_start = start_i + day * 24
        day_end = min(day_start + 24, len(common))
        if day_end <= day_start:
            break
        print(f"day={day} energy-agent equity=${equity:.2f} window={common[day_start].isoformat()}", flush=True)
        payload = _prompt_payload(
            frames,
            day=day,
            day_start_i=day_start,
            history_hours=int(args.history_hours),
            equity=equity,
            candidates=int(args.candidates),
            calibration_days=int(args.calibration_days),
            feature_style=str(args.feature_style),
            line_bars=int(args.line_bars),
            prompt_mode=str(args.prompt_mode),
            previous_feedback=day_summaries[-5:],
        )
        try:
            seed_templates = _call_and_parse_templates(
                args,
                payload,
                output_dir=output_dir,
                stamp=stamp,
                day=day,
                raw_paths=raw_paths,
            )
            seed_templates = drop_example_templates(seed_templates)
            if not seed_templates:
                raise RuntimeError("DeepSeek copied the JSON example instead of producing a candidate")
        except Exception as exc:
            print(f"day={day} deepseek seed failed, using fallback: {exc}", flush=True)
            seed_templates = fallback_seed_templates(args.symbols, day=day)
        seed_templates = clamp_templates(seed_templates, args)

        templates: list[PlanTemplate] = _mutate_templates(
            seed_templates,
            args.symbols,
            day=day,
            include_cash_guard=bool(args.include_cash_guard),
        )
        templates = clamp_templates(templates, args)
        base_vecs = [template_to_vector(t, args.symbols) for t in templates]
        for idx in range(int(args.population)):
            base = base_vecs[idx % len(base_vecs)]
            templates.append(
                vector_to_template(
                    mutate_vector(base, rng, aggression=float(args.seed_aggression), symbols=args.symbols),
                    args.symbols,
                    day=day,
                    name=f"seed_mut_{idx:03d}",
                    rationale="random mutation around DeepSeek seed",
                )
            )
        scores = evaluate_population(
            frames,
            templates,
            current_day_start_i=day_start,
            day=day,
            calibration_days=int(args.calibration_days),
            initial_cash=float(args.initial_cash),
            fee_bps=float(args.fee_bps),
            slippage_bps=float(args.slippage_bps),
        )
        rank_scores(scores, args)
        evaluated_templates = {t.name: t for t in templates}
        x = np.vstack([template_to_vector(evaluated_templates[row["template_name"]], args.symbols) for row in scores])
        y = np.asarray([row["score"] for row in scores], dtype=np.float64)
        stages: list[dict[str, Any]] = [{"stage": "seed_mutation", "best": scores[0]}]

        top = top_templates_from_scores(templates, scores, int(args.top_k))
        tree = fit_boosted_model(x, y)
        boosted_templates = propose_with_model(
            tree,
            top,
            args.symbols,
            day=day,
            rng=rng,
            aggression=float(args.boost_aggression),
            pool_size=int(args.boost_pool),
            take=int(args.boost_take),
            name_prefix="boosted_refit",
        )
        boosted_templates = clamp_templates(boosted_templates, args)
        boosted_scores = evaluate_population(
            frames,
            boosted_templates,
            current_day_start_i=day_start,
            day=day,
            calibration_days=int(args.calibration_days),
            initial_cash=float(args.initial_cash),
            fee_bps=float(args.fee_bps),
            slippage_bps=float(args.slippage_bps),
        )
        templates.extend(boosted_templates)
        scores.extend(boosted_scores)
        evaluated_templates.update({t.name: t for t in boosted_templates})
        rank_scores(scores, args)
        stages.append({"stage": "boosted_refit", "best": scores[0]})

        if args.deepseek_refit:
            try:
                ds_templates = deepseek_refit(
                    args,
                    frames=frames,
                    day=day,
                    day_start_i=day_start,
                    equity=equity,
                    top_rows=scores[:5],
                    best_template=evaluated_templates[scores[0]["template_name"]],
                )
                ds_templates = drop_example_templates(ds_templates)
                if not ds_templates:
                    raise RuntimeError("DeepSeek refit copied the JSON example")
                ds_templates = clamp_templates(ds_templates, args)
                ds_scores = evaluate_population(
                    frames,
                    ds_templates,
                    current_day_start_i=day_start,
                    day=day,
                    calibration_days=int(args.calibration_days),
                    initial_cash=float(args.initial_cash),
                    fee_bps=float(args.fee_bps),
                    slippage_bps=float(args.slippage_bps),
                )
                templates.extend(ds_templates)
                scores.extend(ds_scores)
                evaluated_templates.update({t.name: t for t in ds_templates})
                rank_scores(scores, args)
                stages.append({"stage": "deepseek_refit", "best": scores[0]})
            except Exception as exc:
                stages.append({"stage": "deepseek_refit_failed", "error": str(exc)})
                print(f"day={day} deepseek refit failed: {exc}", flush=True)

        x2 = np.vstack([template_to_vector(evaluated_templates[row["template_name"]], args.symbols) for row in scores])
        y2 = np.asarray([row["score"] for row in scores], dtype=np.float64)
        neural = fit_neural_model(x2, y2, rng)
        if neural is not None:
            top = top_templates_from_scores(list(evaluated_templates.values()), scores, int(args.top_k))
            neural_templates = propose_with_neural(
                neural,
                top,
                args.symbols,
                day=day,
                rng=rng,
                aggression=float(args.neural_aggression),
                pool_size=int(args.neural_pool),
                take=int(args.neural_take),
            )
            neural_templates = clamp_templates(neural_templates, args)
            neural_scores = evaluate_population(
                frames,
                neural_templates,
                current_day_start_i=day_start,
                day=day,
                calibration_days=int(args.calibration_days),
                initial_cash=float(args.initial_cash),
                fee_bps=float(args.fee_bps),
                slippage_bps=float(args.slippage_bps),
            )
            templates.extend(neural_templates)
            scores.extend(neural_scores)
            evaluated_templates.update({t.name: t for t in neural_templates})
            rank_scores(scores, args)
            stages.append({"stage": "neural_refit", "best": scores[0]})

            gradient_templates = propose_with_neural_gradient(
                neural,
                top_templates_from_scores(list(evaluated_templates.values()), scores, int(args.top_k)),
                args.symbols,
                day=day,
                rng=rng,
                steps=int(args.gradient_steps),
                lr=float(args.gradient_lr),
                restarts_per_base=int(args.gradient_restarts),
                take=int(args.gradient_take),
                max_symbol_allocation=float(args.max_symbol_allocation),
                max_gross_exposure=float(args.max_gross_exposure),
                move_penalty=float(args.gradient_move_penalty),
            )
            if gradient_templates:
                gradient_scores = evaluate_population(
                    frames,
                    gradient_templates,
                    current_day_start_i=day_start,
                    day=day,
                    calibration_days=int(args.calibration_days),
                    initial_cash=float(args.initial_cash),
                    fee_bps=float(args.fee_bps),
                    slippage_bps=float(args.slippage_bps),
                )
                templates.extend(gradient_templates)
                scores.extend(gradient_scores)
                evaluated_templates.update({t.name: t for t in gradient_templates})
                rank_scores(scores, args)
                stages.append({"stage": "gradient_refit", "best": scores[0]})

        cem_templates, cem_scores = cem_refine_templates(
            frames,
            top_templates_from_scores(list(evaluated_templates.values()), scores, int(args.top_k)),
            args.symbols,
            day=day,
            current_day_start_i=day_start,
            calibration_days=int(args.calibration_days),
            initial_cash=float(args.initial_cash),
            fee_bps=float(args.fee_bps),
            slippage_bps=float(args.slippage_bps),
            rng=rng,
            iterations=int(args.cem_iterations),
            population=int(args.cem_population),
            elites=int(args.cem_elites),
            aggression=float(args.cem_aggression),
            take=int(args.cem_take),
            max_symbol_allocation=float(args.max_symbol_allocation),
            max_gross_exposure=float(args.max_gross_exposure),
        )
        if cem_templates:
            templates.extend(cem_templates)
            scores.extend(cem_scores)
            evaluated_templates.update({t.name: t for t in cem_templates})
            rank_scores(scores, args)
            stages.append({"stage": "cem_rl_refit", "best": scores[0]})

        chosen_row = scores[0]
        chosen = evaluated_templates[chosen_row["template_name"]]
        if args.include_cash_guard and float(args.min_calibration_score) > 0.0 and chosen_row["score"] < float(args.min_calibration_score):
            chosen = PlanTemplate(
                name="cash_hurdle",
                rationale=f"Best active calibration score {chosen_row['score']:.6f} below hurdle {args.min_calibration_score:.6f}.",
                day_index=day,
                cash_pct=100.0,
                gross_exposure_pct=0.0,
                risk_notes="cash hurdle",
                legs=[
                    TemplateLeg(
                        symbol=str(symbol),
                        side="flat",
                        allocation_pct=0.0,
                        entry_offset_bps=0.0,
                        take_profit_bps=100.0,
                        stop_loss_bps=100.0,
                        confidence=1.0,
                        rationale="cash hurdle",
                    )
                    for symbol in args.symbols
                ],
            )
            evaluated_templates[chosen.name] = chosen
            chosen_row = {
                "template_name": chosen.name,
                "score": 0.0,
                "median_return": 0.0,
                "p25_return": 0.0,
                "worst_return": 0.0,
                "max_drawdown": 0.0,
                "avg_trades": 0.0,
                "hurdle_applied": True,
            }
        plan = instantiate_plan(chosen, frames, day_start_i=day_start, day=day)
        day_start_equity = equity
        equity, trades, rows = simulate_day(
            frames,
            plan,
            start_i=day_start,
            end_i=day_end,
            starting_equity=equity,
            fee_bps=float(args.fee_bps),
            slippage_bps=float(args.slippage_bps),
        )
        day_metrics = _metrics(rows, day_start_equity)
        all_trades.extend([asdict(trade) for trade in trades])
        all_equity_rows.extend(rows)
        selected_templates.append(asdict(chosen))
        selected_plans.append({**asdict(plan), "legs": [asdict(leg) for leg in plan.legs]})
        summary = {
            "day": day,
            "selected_template": chosen.name,
            "calibration_score": chosen_row["score"],
            "calibration_median_return": chosen_row["median_return"],
            "actual_day_return": day_metrics["total_return"],
            "actual_day_max_drawdown": day_metrics["max_drawdown"],
            "actual_trades": len(trades),
            "stages": stages,
            "top5": scores[:5],
        }
        day_summaries.append(summary)
        print(
            f"day={day} selected={chosen.name} "
            f"actual={day_metrics['total_return'] * 100:+.2f}% equity=${equity:.2f}",
            flush=True,
        )

    metrics = _metrics(all_equity_rows, float(args.initial_cash))
    gate_passes = metrics["monthly_return"] >= float(args.monthly_target) and metrics["max_drawdown"] <= float(args.max_drawdown_target)
    return {
        "model": str(args.model),
        "prompt_mode": str(args.prompt_mode),
        "feature_style": str(args.feature_style),
        "line_bars": int(args.line_bars),
        "symbols": list(args.symbols),
        "days": int(args.days),
        "calibration_days": int(args.calibration_days),
        "initial_cash": float(args.initial_cash),
        "final_equity": float(equity),
        "metrics": metrics,
        "production_gate": {
            "passes_gate": bool(gate_passes),
            "monthly_target": float(args.monthly_target),
            "max_drawdown_target": float(args.max_drawdown_target),
            "prod_model": "gpt-5.5",
            "prod_reasoning_effort": "low",
            "route_to_prod_model_if_gate_passes": bool(gate_passes),
        },
        "day_summaries": day_summaries,
        "selected_templates": selected_templates,
        "selected_plans": selected_plans,
        "trades": all_trades,
        "equity_curve": all_equity_rows,
        "raw_response_paths": raw_paths,
        "notes": {
            "tree_refit": "xgboost if installed, otherwise sklearn HistGradientBoostingRegressor",
            "neural_refit": "small Torch MLP regressor over simulated template scores",
            "gradient_refit": "gradient ascent through the Torch score surrogate over side/allocation/entry/take-profit/stop vectors",
            "cem_rl_refit": "cross-entropy evolution-strategy refinement evaluated through the simulator",
            "no_thinking": "implemented as low/no-think prompting; DeepSeek chat API exposes no explicit reasoning_effort knob here",
            "risk_scale": float(args.risk_scale),
            "max_symbol_allocation": float(args.max_symbol_allocation),
            "max_gross_exposure": float(args.max_gross_exposure),
        },
    }


def write_report(path: Path, result: Mapping[str, Any]) -> None:
    metrics = result["metrics"]
    lines = [
        "# DeepSeek Energy Refit Agent",
        "",
        f"- Model: `{result['model']}`",
        f"- Prompt mode: `{result['prompt_mode']}`",
        f"- Feature style: `{result['feature_style']}`",
        f"- Initial cash: `${result['initial_cash']:,.2f}`",
        f"- Final equity: `${result['final_equity']:,.2f}`",
        f"- Total return: `{metrics['total_return'] * 100:+.2f}%`",
        f"- Monthly-equivalent return: `{metrics['monthly_return'] * 100:+.2f}%`",
        f"- Max drawdown: `{metrics['max_drawdown'] * 100:.2f}%`",
        f"- Sortino: `{metrics['sortino']:.2f}`",
        f"- Trades: `{len(result['trades'])}`",
        f"- Production gate passed: `{result['production_gate']['passes_gate']}`",
        "",
        "## Daily Selections",
        "",
    ]
    for template, day in zip(result["selected_templates"], result["day_summaries"], strict=False):
        active = [leg for leg in template["legs"] if leg["side"] != "flat" and leg["allocation_pct"] > 0]
        summary = ", ".join(
            f"{leg['side']} {leg['symbol']} {leg['allocation_pct']:.1f}% off={leg['entry_offset_bps']:.0f}"
            for leg in active
        ) or "flat"
        lines.append(
            f"- Day {day['day']}: `{template['name']}` {summary}; "
            f"actual={day['actual_day_return'] * 100:+.2f}%, "
            f"cal_score={day['calibration_score']:.5f}"
        )
    lines.extend(["", "## Stage Bests", ""])
    for day in result["day_summaries"]:
        stage_bits = []
        for stage in day["stages"]:
            if "best" in stage:
                stage_bits.append(f"{stage['stage']}={stage['best']['template_name']}:{stage['best']['score']:.5f}")
            else:
                stage_bits.append(f"{stage['stage']} failed")
        lines.append(f"- Day {day['day']}: " + "; ".join(stage_bits))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_FLASH_MODEL)
    parser.add_argument("--provider", choices=["deepseek"], default="deepseek")
    parser.add_argument("--prompt-mode", choices=["minimal", "no_think"], default="no_think")
    parser.add_argument("--feature-style", choices=["summary", "line_pct", "ohlc_mid"], default="ohlc_mid")
    parser.add_argument("--line-bars", type=int, default=48)
    parser.add_argument("--data-root", default="trainingdatahourlybinance")
    parser.add_argument("--symbols", nargs="+", default=list(DEFAULT_SYMBOLS))
    parser.add_argument("--days", type=int, default=7)
    parser.add_argument("--history-hours", type=int, default=168)
    parser.add_argument("--calibration-days", type=int, default=4)
    parser.add_argument("--decision-lag-hours", type=int, default=2)
    parser.add_argument("--initial-cash", type=float, default=DEFAULT_INITIAL_CASH)
    parser.add_argument("--fee-bps", type=float, default=10.0)
    parser.add_argument("--slippage-bps", type=float, default=5.0)
    parser.add_argument("--candidates", type=int, default=1)
    parser.add_argument("--population", type=int, default=96)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--boost-pool", type=int, default=256)
    parser.add_argument("--boost-take", type=int, default=24)
    parser.add_argument("--neural-pool", type=int, default=256)
    parser.add_argument("--neural-take", type=int, default=24)
    parser.add_argument("--gradient-steps", type=int, default=0)
    parser.add_argument("--gradient-restarts", type=int, default=3)
    parser.add_argument("--gradient-take", type=int, default=16)
    parser.add_argument("--gradient-lr", type=float, default=0.02)
    parser.add_argument("--gradient-move-penalty", type=float, default=0.01)
    parser.add_argument("--cem-iterations", type=int, default=0)
    parser.add_argument("--cem-population", type=int, default=64)
    parser.add_argument("--cem-elites", type=int, default=8)
    parser.add_argument("--cem-take", type=int, default=12)
    parser.add_argument("--cem-aggression", type=float, default=0.35)
    parser.add_argument("--seed-aggression", type=float, default=1.0)
    parser.add_argument("--boost-aggression", type=float, default=0.45)
    parser.add_argument("--neural-aggression", type=float, default=0.25)
    parser.add_argument("--deepseek-refit", action="store_true")
    parser.add_argument("--include-cash-guard", action="store_true")
    parser.add_argument("--min-calibration-score", type=float, default=0.0)
    parser.add_argument("--selection-mode", choices=["score", "robust"], default="score")
    parser.add_argument("--min-positive-calibration-days", type=int, default=0)
    parser.add_argument("--require-latest-calibration-positive", action="store_true")
    parser.add_argument("--risk-scale", type=float, default=1.0)
    parser.add_argument("--max-symbol-allocation", type=float, default=60.0)
    parser.add_argument("--max-gross-exposure", type=float, default=150.0)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--api-timeout-s", type=float, default=180.0)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--raw-response-paths", nargs="*")
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--load-env-real", action="store_true")
    parser.add_argument("--monthly-target", type=float, default=0.27)
    parser.add_argument("--max-drawdown-target", type=float, default=0.20)
    parser.add_argument("--seed", type=int, default=20260427)
    parser.add_argument("--output-dir", default="experiments/binance_deepseek_planner/results")
    args = parser.parse_args(argv)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    json_path = output_dir / f"deepseek_energy_refit_{stamp}.json"
    md_path = output_dir / f"deepseek_energy_refit_{stamp}.md"
    try:
        result = run_experiment(args)
    except Exception as exc:
        error = {"model": str(args.model), "prompt_mode": str(args.prompt_mode), "error": str(exc)}
        json_path.write_text(json.dumps(error, indent=2, sort_keys=True), encoding="utf-8")
        print(f"experiment failed: {exc}")
        print(f"wrote {json_path}")
        return 2
    json_path.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    write_report(md_path, result)
    metrics = result["metrics"]
    print(
        f"final=${result['final_equity']:.2f} "
        f"return={metrics['total_return'] * 100:+.2f}% "
        f"monthly={metrics['monthly_return'] * 100:+.2f}% "
        f"dd={metrics['max_drawdown'] * 100:.2f}% "
        f"trades={len(result['trades'])} "
        f"prod_route={result['production_gate']['route_to_prod_model_if_gate_passes']}"
    )
    print(f"wrote {json_path}")
    print(f"wrote {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
