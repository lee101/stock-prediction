"""Claude Opus 4.5 Stock Trading Agent.

This agent combines neural forecasting with Claude Opus 4.5's advanced reasoning
capabilities for stock trading decisions. It features:
- On-disk caching for reproducibility
- Signal calibration for forecast adjustment
- Max-diff execution for limit-style trading
- Extended thinking for complex market analysis
"""

from .agent import (
    OpusPlanResult,
    OpusPlanStep,
    OpusReplanResult,
    generate_opus_plan,
    simulate_opus_plan,
    simulate_opus_replanning,
)

from .opus_wrapper import (
    call_opus_chat,
    reset_client,
)

__all__ = [
    "OpusPlanResult",
    "OpusPlanStep",
    "OpusReplanResult",
    "generate_opus_plan",
    "simulate_opus_plan",
    "simulate_opus_replanning",
    "call_opus_chat",
    "reset_client",
]
