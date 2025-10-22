import os

import pytest

from deepseek_wrapper import call_deepseek_chat


@pytest.mark.external
@pytest.mark.skipif(
    not (os.getenv("DEEPSEEK_API_KEY") or os.getenv("OPENROUTER_API_KEY")),
    reason="Requires DEEPSEEK_API_KEY or OPENROUTER_API_KEY",
)
def test_deepseek_live_round_trip():
    messages = [
        {"role": "system", "content": "You are a concise assistant."},
        {"role": "user", "content": "Respond with a single sentence about prudent trading."},
    ]
    output = call_deepseek_chat(messages, max_output_tokens=128, temperature=0.2, cache_ttl=None)
    assert isinstance(output, str)
    assert len(output.strip()) > 0
