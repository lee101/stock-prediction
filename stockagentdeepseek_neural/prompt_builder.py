"""Prompt helpers that enrich DeepSeek requests with neural forecasts."""

from __future__ import annotations

import json
from datetime import date
from typing import Mapping, Sequence

from stockagent.agentsimulator.data_models import AccountSnapshot
from stockagent.agentsimulator.market_data import MarketDataBundle
from stockagentdeepseek.prompt_builder import build_deepseek_messages as _build_base_messages

from .forecaster import NeuralForecast


def _format_forecast_lines(forecasts: Mapping[str, NeuralForecast]) -> str:
    lines: list[str] = []
    for symbol in sorted(forecasts.keys()):
        forecast = forecasts[symbol]
        combined_bits = ", ".join(f"{key}={value:.2f}" for key, value in forecast.combined.items())
        best_label = forecast.best_model or "blended"
        source_label = f" ({forecast.selection_source})" if forecast.selection_source else ""
        lines.append(
            f"- {symbol}: combined forecast {combined_bits} using {best_label}{source_label}."
        )
        for name, summary in forecast.model_summaries.items():
            model_bits = ", ".join(f"{key}={value:.2f}" for key, value in summary.forecasts.items())
            lines.append(
                f"  * {name} ({summary.config_name}) MAE={summary.average_price_mae:.4f}: {model_bits}"
            )
    return "\n".join(lines)


def build_neural_messages(
    *,
    forecasts: Mapping[str, NeuralForecast],
    market_data: MarketDataBundle,
    target_date: date,
    account_snapshot: AccountSnapshot | None = None,
    account_payload: Mapping[str, object] | None = None,
    symbols: Sequence[str] | None = None,
    include_market_history: bool = True,
) -> list[dict[str, str]]:
    """Build DeepSeek messages augmented with neural forecasts."""
    base_messages = _build_base_messages(
        market_data=market_data,
        target_date=target_date,
        account_snapshot=account_snapshot,
        account_payload=account_payload,
        symbols=symbols,
        include_market_history=include_market_history,
    )

    if len(base_messages) < 3:
        raise ValueError("Expected base messages to include system, prompt, and payload entries.")

    forecast_block = _format_forecast_lines(forecasts)
    if forecast_block:
        base_messages[1]["content"] += "\nNeural forecasts:\n" + forecast_block

    payload = json.loads(base_messages[-1]["content"])
    payload["neural_forecasts"] = {
        symbol: {
            "combined": forecast.combined,
            "best_model": forecast.best_model,
            "selection_source": forecast.selection_source,
            "models": {
                name: {
                    "mae": summary.average_price_mae,
                    "forecasts": summary.forecasts,
                    "config": summary.config_name,
                }
                for name, summary in forecast.model_summaries.items()
            },
        }
        for symbol, forecast in forecasts.items()
    }
    base_messages[-1]["content"] = json.dumps(payload, ensure_ascii=False, indent=2)
    return base_messages


__all__ = ["build_neural_messages"]
