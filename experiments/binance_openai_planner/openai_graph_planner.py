#!/usr/bin/env python3
"""Use OpenAI strict Structured Outputs to propose graph-policy configs.

This is an offline research path. The cheap/default model is used only to prove
the OpenAI structured-output loop works and to generate candidate configs. If a
candidate clears the configured PnL/drawdown gate, the artifact records that
production-time planning should route to the stronger model/reasoning setting.
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
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

from experiments.binance_transition_graph.graph_policy import (
    DEFAULT_SYMBOLS,
    GraphPolicyConfig,
    baseline_configs,
    close_matrix,
    config_from_mapping,
    evaluate_slippage_sweep,
    load_hourly_panel,
    summarize_sweep_rows,
    train_eval_split,
    write_result_json,
)


DEFAULT_TEST_MODEL = "gpt-5.4-nano"
DEFAULT_TEST_REASONING = "low"
DEFAULT_PROD_MODEL = "gpt-5.5"
DEFAULT_PROD_REASONING = "xhigh"

SYSTEM_PROMPT = """You are a quantitative research planner. Return exactly the
requested structured JSON. You are not placing live orders. You propose
parameterized graph-policy configs for an offline Binance crypto simulator with
decision lag, fees, slippage, leverage, and borrow costs.

Known failure mode from prior stockagent-style one-shot planning: coherent plans
still lost money when they became mostly correlated crypto-beta longs with tight
stops. Fees/slippage plus selloff hours overwhelmed the winners. Prefer
simulator-robust edges: explicit cash regimes, selective shorts/hedges, lower
turnover, stop distances that survive normal hourly noise, and post-fee
expectancy over plausible discretionary narratives."""


def _load_env_real() -> None:
    """Best-effort import of env_real.py without printing secrets."""
    try:
        env_real = importlib.import_module("env_real")
    except Exception:
        return
    value = getattr(env_real, "OPENAI_API_KEY", None)
    if value and not os.getenv("OPENAI_API_KEY"):
        os.environ["OPENAI_API_KEY"] = str(value)


def _market_summary(close: pd.DataFrame) -> dict[str, Any]:
    returns = close.pct_change().replace([np.inf, -np.inf], np.nan)
    horizons = {"6h": 6, "24h": 24, "72h": 72, "168h": 168}
    per_symbol: dict[str, dict[str, float]] = {}
    for symbol in close.columns:
        row: dict[str, float] = {}
        for label, n in horizons.items():
            if len(close) > n:
                row[f"ret_{label}"] = float(close[symbol].iloc[-1] / close[symbol].iloc[-n - 1] - 1.0)
        row["vol_24h"] = float(returns[symbol].tail(24).std(ddof=0) or 0.0)
        row["vol_168h"] = float(returns[symbol].tail(168).std(ddof=0) or 0.0)
        rolling_high = close[symbol].tail(168).max()
        row["drawdown_168h"] = float(close[symbol].iloc[-1] / rolling_high - 1.0) if rolling_high > 0 else 0.0
        per_symbol[symbol] = row
    return {
        "rows": int(len(close)),
        "start": close.index[0].isoformat(),
        "end": close.index[-1].isoformat(),
        "symbols": list(close.columns),
        "per_symbol": per_symbol,
        "correlation_tail_168h": returns.tail(168).corr().fillna(0.0).round(3).to_dict(),
    }


def graph_candidate_schema() -> dict[str, Any]:
    config_properties = {
        "name": {"type": "string"},
        "fast_lookback_hours": {"type": "integer"},
        "slow_lookback_hours": {"type": "integer"},
        "regime_lookback_hours": {"type": "integer"},
        "rebalance_hours": {"type": "integer"},
        "target_leverage": {"type": "number"},
        "max_leverage": {"type": "number"},
        "allow_short": {"type": "boolean"},
        "momentum_threshold": {"type": "number"},
        "dip_buy_threshold": {"type": "number"},
        "winner_sell_threshold": {"type": "number"},
        "rotation_spread_threshold": {"type": "number"},
        "short_threshold": {"type": "number"},
        "market_regime_cash_threshold": {"type": "number"},
        "market_regime_short_threshold": {"type": "number"},
        "volatility_cash_threshold": {"type": "number"},
        "drawdown_cash_threshold": {"type": "number"},
        "stop_loss": {"type": "number"},
        "take_profit": {"type": "number"},
        "fee_bps": {"type": "number"},
        "slippage_bps": {"type": "number"},
        "borrow_apr": {"type": "number"},
        "decision_lag_hours": {"type": "integer"},
    }
    config_required = list(config_properties)
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["candidates"],
        "properties": {
            "candidates": {
                "type": "array",
                "minItems": 1,
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["name", "rationale", "config"],
                    "properties": {
                        "name": {"type": "string"},
                        "rationale": {"type": "string"},
                        "config": {
                            "type": "object",
                            "additionalProperties": False,
                            "required": config_required,
                            "properties": config_properties,
                        },
                    },
                },
            }
        },
    }


def build_messages(
    close: pd.DataFrame,
    *,
    max_leverage: float,
    allow_short: bool,
    candidates: int,
) -> list[dict[str, Any]]:
    payload = {
        "task": (
            "Propose compact graph-policy configs that might beat current Binance baselines. "
            "Prefer interpretable edges: rotate winner to dip, go cash in bad regimes, optionally "
            "short weak assets in bearish regimes. Avoid overfit and keep max drawdown under 20%. "
            "Do not default to all-long correlated crypto beta with tight stops; that already failed after fees."
        ),
        "num_candidates": int(candidates),
        "market_summary": _market_summary(close),
        "parameter_guidance": {
            "lookbacks_hours": "fast 3-24, slow 12-72, regime 48-168",
            "threshold_units": "decimal fractions, e.g. 0.03 is 3%",
            "target_leverage": f"0 to {max_leverage}",
            "allow_short": bool(allow_short),
            "decision_lag_hours": "must be 2 or greater",
            "fee_bps": 10.0,
            "slippage_bps": 5.0,
            "borrow_apr": 0.0675,
        },
    }
    return [
        {
            "role": "system",
            "content": [{"type": "input_text", "text": SYSTEM_PROMPT}],
        },
        {
            "role": "user",
            "content": [{"type": "input_text", "text": json.dumps(payload, indent=2)}],
        },
    ]


def _coerce_openai_reasoning_effort(effort: str) -> str:
    normalized = str(effort or "").strip().lower()
    # xhigh is accepted by newer GPT-5.x deployments. For older SDK/model combos,
    # retry logic below records the provider error rather than silently changing it.
    allowed = {"minimal", "low", "medium", "high", "xhigh"}
    return normalized if normalized in allowed else DEFAULT_TEST_REASONING


def call_openai_structured(
    messages: Sequence[Mapping[str, Any]],
    *,
    model: str,
    reasoning_effort: str,
    max_output_tokens: int,
    store: bool = False,
) -> str:
    try:
        from openai import OpenAI
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("openai package is required for OpenAI planner calls") from exc
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set; pass --load-env-real or export it")
    client = OpenAI(api_key=api_key)
    schema = {
        "type": "json_schema",
        "name": "binance_graph_candidates",
        "strict": True,
        "schema": graph_candidate_schema(),
    }
    response = client.responses.create(
        model=str(model),
        input=[dict(message) for message in messages],
        text={"format": schema},
        reasoning={"effort": _coerce_openai_reasoning_effort(reasoning_effort)},
        max_output_tokens=int(max_output_tokens),
        store=bool(store),
    )
    output_text = getattr(response, "output_text", None)
    if not isinstance(output_text, str) or not output_text.strip():
        raise RuntimeError("OpenAI response did not include output_text")
    return output_text


def offline_candidates(max_leverage: float, allow_short: bool) -> list[dict[str, Any]]:
    configs = baseline_configs(max_leverage=max_leverage, allow_short=allow_short)[1:]
    return [
        {
            "name": cfg.name,
            "rationale": "offline baseline candidate for smoke testing",
            "config": asdict(cfg),
        }
        for cfg in configs
    ]


def parse_candidates(payload: Mapping[str, Any]) -> list[tuple[str, str, GraphPolicyConfig]]:
    raw_candidates = payload.get("candidates", [])
    if not isinstance(raw_candidates, list):
        raise ValueError("OpenAI structured payload must include candidates list")
    parsed: list[tuple[str, str, GraphPolicyConfig]] = []
    for idx, item in enumerate(raw_candidates):
        if not isinstance(item, Mapping):
            continue
        name = str(item.get("name") or f"openai_candidate_{idx}")
        rationale = str(item.get("rationale") or "")
        config_payload = item.get("config", {})
        if not isinstance(config_payload, Mapping):
            continue
        cfg = config_from_mapping(config_payload, name=name)
        if cfg.decision_lag_hours < 2:
            cfg = GraphPolicyConfig(**{**asdict(cfg), "decision_lag_hours": 2}).clipped()
        parsed.append((name, rationale, cfg))
    if not parsed:
        raise ValueError("No usable graph-policy candidates found in OpenAI structured payload")
    return parsed


def _gate_payload(
    *,
    best_monthly: float,
    best_drawdown: float,
    monthly_target: float,
    max_drawdown_target: float,
    test_model: str,
    prod_model: str,
    prod_reasoning_effort: str,
) -> dict[str, Any]:
    passes = best_monthly >= monthly_target and best_drawdown <= max_drawdown_target
    return {
        "passes_gate": bool(passes),
        "monthly_target": float(monthly_target),
        "max_drawdown_target": float(max_drawdown_target),
        "best_worst_slippage_monthly": float(best_monthly),
        "best_worst_slippage_max_drawdown": float(best_drawdown),
        "tested_model": str(test_model),
        "production_model_if_gate_passes": str(prod_model),
        "production_reasoning_effort_if_gate_passes": str(prod_reasoning_effort),
        "route_to_production_model": bool(passes),
        "note": (
            "Candidate passed offline gate; production-time planner should use the stronger model/reasoning."
            if passes
            else "Candidate did not pass offline gate; do not route to production model."
        ),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default="trainingdatahourlybinance")
    parser.add_argument("--symbols", nargs="+", default=list(DEFAULT_SYMBOLS))
    parser.add_argument("--train-hours", type=int, default=24 * 30)
    parser.add_argument("--eval-hours", type=int, default=24 * 10)
    parser.add_argument("--eval-warmup-hours", type=int, default=240)
    parser.add_argument("--max-leverage", type=float, default=2.0)
    parser.add_argument("--allow-short", action="store_true")
    parser.add_argument("--candidates", type=int, default=1)
    parser.add_argument("--model", default=DEFAULT_TEST_MODEL)
    parser.add_argument("--reasoning-effort", default=DEFAULT_TEST_REASONING)
    parser.add_argument("--prod-model", default=DEFAULT_PROD_MODEL)
    parser.add_argument("--prod-reasoning-effort", default=DEFAULT_PROD_REASONING)
    parser.add_argument("--monthly-target", type=float, default=0.27)
    parser.add_argument("--max-drawdown-target", type=float, default=0.20)
    parser.add_argument("--max-output-tokens", type=int, default=4096)
    parser.add_argument("--load-env-real", action="store_true")
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--output-dir", default="experiments/binance_openai_planner/results")
    args = parser.parse_args(argv)

    if args.load_env_real:
        _load_env_real()

    eval_warmup_hours = min(int(args.eval_warmup_hours), int(args.train_hours))
    frames = load_hourly_panel(
        args.data_root,
        args.symbols,
        tail_hours=int(args.train_hours) + int(args.eval_hours) + eval_warmup_hours,
    )
    train_frames, eval_frames = train_eval_split(
        frames,
        train_hours=int(args.train_hours),
        eval_hours=int(args.eval_hours),
        eval_warmup_hours=eval_warmup_hours,
    )
    messages = build_messages(
        close_matrix(train_frames),
        max_leverage=float(args.max_leverage),
        allow_short=bool(args.allow_short),
        candidates=int(args.candidates),
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    raw_path = output_dir / f"openai_raw_{stamp}.txt"
    json_path = output_dir / f"openai_eval_{stamp}.json"
    csv_path = output_dir / f"openai_eval_{stamp}.csv"

    try:
        if args.offline:
            raw_text = json.dumps({"candidates": offline_candidates(float(args.max_leverage), bool(args.allow_short))})
        else:
            raw_text = call_openai_structured(
                messages,
                model=str(args.model),
                reasoning_effort=str(args.reasoning_effort),
                max_output_tokens=int(args.max_output_tokens),
            )
    except Exception as exc:
        error_payload = {
            "model": str(args.model),
            "reasoning_effort": str(args.reasoning_effort),
            "offline": bool(args.offline),
            "error": str(exc),
            "structured_outputs": {"type": "json_schema", "strict": True},
        }
        write_result_json(json_path, error_payload)
        print(f"OpenAI structured planner failed: {exc}")
        print(f"wrote {json_path}")
        return 2

    raw_path.write_text(raw_text, encoding="utf-8")
    try:
        parsed_payload = json.loads(raw_text)
        candidates = parse_candidates(parsed_payload)
    except Exception as exc:
        error_payload = {
            "model": str(args.model),
            "reasoning_effort": str(args.reasoning_effort),
            "offline": bool(args.offline),
            "parse_error": str(exc),
            "raw_response_path": str(raw_path),
            "structured_outputs": {"type": "json_schema", "strict": True},
        }
        write_result_json(json_path, error_payload)
        print(f"OpenAI structured response was not usable JSON: {exc}")
        print(f"wrote {json_path}")
        print(f"wrote {raw_path}")
        return 2

    sweeps = [
        evaluate_slippage_sweep(eval_frames, cfg, warmup_rows=eval_warmup_hours)
        for _name, _rationale, cfg in candidates
    ]
    sweeps.sort(
        key=lambda row: (
            row.worst_monthly_return - 0.5 * max(0.0, row.worst_max_drawdown - 0.20),
            -row.worst_max_drawdown,
        ),
        reverse=True,
    )
    summarize_sweep_rows(sweeps).to_csv(csv_path, index=False)
    best = sweeps[0]
    gate = _gate_payload(
        best_monthly=best.worst_monthly_return,
        best_drawdown=best.worst_max_drawdown,
        monthly_target=float(args.monthly_target),
        max_drawdown_target=float(args.max_drawdown_target),
        test_model=str(args.model),
        prod_model=str(args.prod_model),
        prod_reasoning_effort=str(args.prod_reasoning_effort),
    )
    payload = {
        "model": str(args.model),
        "reasoning_effort": str(args.reasoning_effort),
        "offline": bool(args.offline),
        "structured_outputs": {"api": "responses", "type": "json_schema", "strict": True},
        "symbols": list(args.symbols),
        "train_hours": int(args.train_hours),
        "eval_hours": int(args.eval_hours),
        "eval_warmup_hours": eval_warmup_hours,
        "rationales": {name: rationale for name, rationale, _cfg in candidates},
        "production_gate": gate,
        "results": [sweep.to_json_dict() for sweep in sweeps],
        "raw_response_path": str(raw_path),
    }
    write_result_json(json_path, payload)
    print(
        f"best={best.config.name} "
        f"worst_monthly={best.worst_monthly_return * 100:+.2f}% "
        f"worst_dd={best.worst_max_drawdown * 100:.2f}% "
        f"prod_route={gate['route_to_production_model']}"
    )
    print(f"wrote {json_path}")
    print(f"wrote {csv_path}")
    print(f"wrote {raw_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
