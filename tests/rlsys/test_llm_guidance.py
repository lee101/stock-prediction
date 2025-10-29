from rlsys.config import LLMConfig
from rlsys.llm_guidance import StrategyLLMGuidance


def test_guidance_disabled_returns_placeholder():
    config = LLMConfig(enabled=False)
    guidance = StrategyLLMGuidance(config)
    result = guidance.summarize({"reward": 1.0, "drawdown": -0.1})
    assert "disabled" in result.response.lower()
    assert "reward" in result.prompt


def test_guidance_uses_custom_generator():
    messages = []

    def generator(prompt: str) -> str:
        messages.append(prompt)
        return "Consider reducing leverage."

    config = LLMConfig(enabled=True)
    guidance = StrategyLLMGuidance(config, generator=generator)
    result = guidance.summarize({"reward": 0.5, "sharpe": 1.2})
    assert messages and messages[0] == result.prompt
    assert "reducing" in result.response.lower()
