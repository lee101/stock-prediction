#!/usr/bin/env python3
"""Use DeepSeek to propose graph-policy configs and evaluate them offline.

This file calls DeepSeek's OpenAI-compatible API directly. It never places live
orders; DeepSeek only emits parameterized graph rules that are then walked
sequentially through the local simulator.
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import re
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


SYSTEM_PROMPT = """You are a quantitative research planner. Return JSON only.
You are not placing live orders. You propose parameterized graph-policy configs
for an offline Binance crypto simulator with decision lag, fees, slippage,
leverage, and borrow costs.

Known failure mode from prior stockagent-style one-shot planning: coherent plans
still lost money when they became mostly correlated crypto-beta longs with tight
stops. Fees/slippage plus selloff hours overwhelmed the winners. Prefer
simulator-robust edges: explicit cash regimes, selective shorts/hedges, lower
turnover, stop distances that survive normal hourly noise, and post-fee
expectancy over plausible discretionary narratives."""

DEFAULT_DEEPSEEK_MODEL = "deepseek-v4-pro"


def _load_env_real() -> None:
    """Best-effort import of env_real.py without printing secrets."""
    try:
        env_real = importlib.import_module("env_real")
    except Exception:
        return
    for key in ("DEEPSEEK_API_KEY", "DEEPSEEK_API_BASE", "DEEPSEEK_MODEL"):
        value = getattr(env_real, key, None)
        if value and not os.getenv(key):
            os.environ[key] = str(value)


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
    corr = returns.tail(168).corr().fillna(0.0)
    return {
        "rows": int(len(close)),
        "start": close.index[0].isoformat(),
        "end": close.index[-1].isoformat(),
        "symbols": list(close.columns),
        "per_symbol": per_symbol,
        "correlation_tail_168h": corr.round(3).to_dict(),
    }


def _schema_text(max_leverage: float, allow_short: bool) -> str:
    template = asdict(
        GraphPolicyConfig(
            name="candidate_name",
            max_leverage=float(max_leverage),
            target_leverage=min(1.0, float(max_leverage)),
            allow_short=bool(allow_short),
        )
    )
    return json.dumps(
        {
            "response_schema": {
                "candidates": [
                    {
                        "name": "short_name",
                        "rationale": "brief reason",
                        "config": template,
                    }
                ]
            },
            "important_ranges": {
                "lookbacks_hours": "fast 3-24, slow 12-72, regime 48-168",
                "thresholds": "returns are decimal fractions, e.g. 0.03 is 3%",
                "target_leverage": f"0 to {max_leverage}",
                "allow_short": bool(allow_short),
                "decision_lag_hours": "must be 2 or greater",
            },
        },
        indent=2,
    )


def build_messages(
    close: pd.DataFrame,
    *,
    max_leverage: float,
    allow_short: bool,
    candidates: int,
) -> list[dict[str, str]]:
    summary = _market_summary(close)
    user = {
        "task": (
            "Propose graph-policy configs that might beat current Binance baselines. "
            "Focus on interpretable transition edges: rotate winner to dip, go cash in bad regimes, "
            "optionally short weak assets in bearish regimes. Avoid overfit and keep max drawdown under 20%. "
            "Do not default to all-long correlated crypto beta with tight stops; that already failed after fees."
        ),
        "num_candidates": int(candidates),
        "market_summary": summary,
        "schema": json.loads(_schema_text(max_leverage, allow_short)),
    }
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": json.dumps(user, indent=2)},
    ]


def call_deepseek_direct(
    messages: Sequence[Mapping[str, str]],
    *,
    model: str | None = None,
    max_tokens: int = 8192,
    temperature: float = 0.2,
) -> str:
    try:
        from openai import OpenAI
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("openai package is required for direct DeepSeek calls") from exc
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        raise RuntimeError("DEEPSEEK_API_KEY is not set; pass --load-env-real or export it")
    base_url = os.getenv("DEEPSEEK_API_BASE", "https://api.deepseek.com")
    chosen_model = model or os.getenv("DEEPSEEK_MODEL", DEFAULT_DEEPSEEK_MODEL)
    client = OpenAI(api_key=api_key, base_url=base_url)
    request = {
        "model": chosen_model,
        "messages": [dict(message) for message in messages],
        "max_tokens": int(max_tokens),
        "temperature": float(temperature),
        "stream": False,
        "response_format": {"type": "json_object"},
    }
    try:
        response = client.chat.completions.create(**request)
    except TypeError:
        request.pop("response_format", None)
        response = client.chat.completions.create(**request)
    content = response.choices[0].message.content
    if not content:
        raise RuntimeError("DeepSeek response was empty")
    return str(content)


def _extract_json(text: str) -> dict[str, Any]:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?", "", stripped).strip()
        stripped = re.sub(r"```$", "", stripped).strip()
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", stripped, flags=re.DOTALL)
        if not match:
            raise
        return json.loads(match.group(0))


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
        raise ValueError("DeepSeek JSON must include a candidates list")
    parsed: list[tuple[str, str, GraphPolicyConfig]] = []
    for idx, item in enumerate(raw_candidates):
        if not isinstance(item, Mapping):
            continue
        name = str(item.get("name") or f"deepseek_candidate_{idx}")
        rationale = str(item.get("rationale") or "")
        config_payload = item.get("config", {})
        if not isinstance(config_payload, Mapping):
            continue
        cfg = config_from_mapping(config_payload, name=name)
        if cfg.decision_lag_hours < 2:
            cfg = GraphPolicyConfig(**{**asdict(cfg), "decision_lag_hours": 2}).clipped()
        parsed.append((name, rationale, cfg))
    if not parsed:
        raise ValueError("No usable graph-policy candidates found in DeepSeek JSON")
    return parsed


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default="trainingdatahourlybinance")
    parser.add_argument("--symbols", nargs="+", default=list(DEFAULT_SYMBOLS))
    parser.add_argument("--train-hours", type=int, default=24 * 30)
    parser.add_argument("--eval-hours", type=int, default=24 * 14)
    parser.add_argument("--max-leverage", type=float, default=2.0)
    parser.add_argument("--allow-short", action="store_true")
    parser.add_argument("--candidates", type=int, default=4)
    parser.add_argument("--eval-warmup-hours", type=int, default=240)
    parser.add_argument("--model", default=DEFAULT_DEEPSEEK_MODEL)
    parser.add_argument("--max-tokens", type=int, default=8192)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--load-env-real", action="store_true")
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--output-dir", default="experiments/binance_deepseek_planner/results")
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
    train_close = close_matrix(train_frames)
    messages = build_messages(
        train_close,
        max_leverage=float(args.max_leverage),
        allow_short=bool(args.allow_short),
        candidates=int(args.candidates),
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    raw_path = output_dir / f"deepseek_raw_{stamp}.txt"
    json_path = output_dir / f"deepseek_eval_{stamp}.json"
    csv_path = output_dir / f"deepseek_eval_{stamp}.csv"

    if args.offline:
        raw_text = json.dumps({"candidates": offline_candidates(float(args.max_leverage), bool(args.allow_short))})
    else:
        raw_text = call_deepseek_direct(
            messages,
            model=args.model,
            max_tokens=int(args.max_tokens),
            temperature=float(args.temperature),
        )
    raw_path.write_text(raw_text, encoding="utf-8")
    try:
        parsed_payload = _extract_json(raw_text)
    except Exception as exc:
        error_payload = {
            "model": args.model or os.getenv("DEEPSEEK_MODEL", DEFAULT_DEEPSEEK_MODEL),
            "offline": bool(args.offline),
            "symbols": list(args.symbols),
            "train_hours": int(args.train_hours),
            "eval_hours": int(args.eval_hours),
            "eval_warmup_hours": eval_warmup_hours,
            "parse_error": str(exc),
            "raw_response_path": str(raw_path),
        }
        write_result_json(json_path, error_payload)
        print(f"DeepSeek response was not valid JSON: {exc}")
        print(f"wrote {json_path}")
        print(f"wrote {raw_path}")
        return 2
    candidates = parse_candidates(parsed_payload)
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
    rationales = {name: rationale for name, rationale, _cfg in candidates}
    payload = {
        "model": args.model or os.getenv("DEEPSEEK_MODEL", DEFAULT_DEEPSEEK_MODEL),
        "offline": bool(args.offline),
        "symbols": list(args.symbols),
        "train_hours": int(args.train_hours),
        "eval_hours": int(args.eval_hours),
        "eval_warmup_hours": eval_warmup_hours,
        "rationales": rationales,
        "results": [sweep.to_json_dict() for sweep in sweeps],
        "raw_response_path": str(raw_path),
    }
    write_result_json(json_path, payload)
    if sweeps:
        best = sweeps[0]
        print(
            f"best={best.config.name} "
            f"worst_monthly={best.worst_monthly_return * 100:+.2f}% "
            f"worst_dd={best.worst_max_drawdown * 100:.2f}%"
        )
    print(f"wrote {json_path}")
    print(f"wrote {csv_path}")
    print(f"wrote {raw_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
