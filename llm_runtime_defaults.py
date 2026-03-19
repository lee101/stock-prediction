from __future__ import annotations

PROD_GEMINI_THINKING_LEVEL = "HIGH"
BACKTEST_GEMINI_THINKING_LEVEL = "LOW"
PROD_REASONING_EFFORT = "high"
BACKTEST_REASONING_EFFORT = "low"


def resolve_runtime_llm_settings(
    *,
    backtest: bool,
    thinking_level: str | None,
    reasoning_effort: str | None,
) -> tuple[str, str]:
    """Resolve live-vs-backtest LLM settings while preserving explicit overrides."""

    default_thinking_level = (
        BACKTEST_GEMINI_THINKING_LEVEL if backtest else PROD_GEMINI_THINKING_LEVEL
    )
    default_reasoning_effort = (
        BACKTEST_REASONING_EFFORT if backtest else PROD_REASONING_EFFORT
    )
    return (
        default_thinking_level if thinking_level is None else thinking_level,
        default_reasoning_effort if reasoning_effort is None else reasoning_effort,
    )
