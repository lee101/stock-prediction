from __future__ import annotations

from src.llm_runtime_defaults import (
    BACKTEST_GEMINI_THINKING_LEVEL,
    BACKTEST_REASONING_EFFORT,
    PROD_GEMINI_THINKING_LEVEL,
    PROD_REASONING_EFFORT,
    resolve_runtime_llm_settings,
)


def test_resolve_runtime_llm_settings_defaults_to_prod_values() -> None:
    thinking_level, reasoning_effort = resolve_runtime_llm_settings(
        backtest=False,
        thinking_level=None,
        reasoning_effort=None,
    )

    assert thinking_level == PROD_GEMINI_THINKING_LEVEL
    assert reasoning_effort == PROD_REASONING_EFFORT


def test_resolve_runtime_llm_settings_defaults_to_backtest_values() -> None:
    thinking_level, reasoning_effort = resolve_runtime_llm_settings(
        backtest=True,
        thinking_level=None,
        reasoning_effort=None,
    )

    assert thinking_level == BACKTEST_GEMINI_THINKING_LEVEL
    assert reasoning_effort == BACKTEST_REASONING_EFFORT


def test_resolve_runtime_llm_settings_preserves_explicit_overrides() -> None:
    thinking_level, reasoning_effort = resolve_runtime_llm_settings(
        backtest=True,
        thinking_level="MEDIUM",
        reasoning_effort="max",
    )

    assert thinking_level == "MEDIUM"
    assert reasoning_effort == "max"
