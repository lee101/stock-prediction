import json
import types
import sys
import os
from pathlib import Path

# Ensure repository root is importable
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from hyperparamopt.storage import RunLog, RunRecord
from hyperparamopt.optimizer import StructuredOpenAIOptimizer, SuggestionRequest


class _FakeContent:
    def __init__(self, text: str):
        self.text = text


class _FakeOutput:
    def __init__(self, text: str):
        self.content = [_FakeContent(text)]


class _FakeResponse:
    def __init__(self, text: str):
        self.output = [_FakeOutput(text)]
        self.output_text = text


class _FakeResponsesAPI:
    def __init__(self, payload):
        self.payload = payload

    def create(self, **kwargs):
        # Return payload as the model's JSON
        return _FakeResponse(json.dumps(self.payload))


class _FakeOpenAI:
    def __init__(self, api_key: str):
        self.api_key = api_key
        # Provide a default payload; tests can overwrite
        self.responses = _FakeResponsesAPI({
            "suggestions": [
                {"max_positions": 3, "rebalance_frequency": 3, "min_expected_return": 0.02, "position_sizing_method": "equal_weight"},
                {"max_positions": 5, "rebalance_frequency": 5, "min_expected_return": 0.01, "position_sizing_method": "return_weighted"}
            ]
        })


def test_structured_suggestion_with_mocked_openai(tmp_path, monkeypatch):
    # Prepare isolated log file
    log_path = tmp_path / "runs.jsonl"
    log = RunLog(log_path)

    # Log two example runs
    log.append(RunRecord.new(
        params={"max_positions": 2, "rebalance_frequency": 1, "min_expected_return": 0.00, "position_sizing_method": "equal_weight"},
        metrics={"sharpe": 0.9, "return": 0.15},
        score=0.9,
        objective="maximize_sharpe",
        source="manual",
    ))
    log.append(RunRecord.new(
        params={"max_positions": 3, "rebalance_frequency": 3, "min_expected_return": 0.02, "position_sizing_method": "equal_weight"},
        metrics={"sharpe": 1.1, "return": 0.18},
        score=1.1,
        objective="maximize_sharpe",
        source="manual",
    ))

    # Mock openai.OpenAI class
    fake_mod = types.ModuleType("openai")
    fake_mod.OpenAI = _FakeOpenAI  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "openai", fake_mod)

    # Build schema and request
    schema = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "max_positions": {"type": "integer", "minimum": 1, "maximum": 10},
            "rebalance_frequency": {"type": "integer", "enum": [1, 3, 5, 7]},
            "min_expected_return": {"type": "number", "minimum": 0.0, "maximum": 0.2},
            "position_sizing_method": {"type": "string", "enum": ["equal_weight", "return_weighted"]},
        },
        "required": ["max_positions", "rebalance_frequency", "min_expected_return", "position_sizing_method"],
    }

    opt = StructuredOpenAIOptimizer(run_log=log)
    req = SuggestionRequest(
        hyperparam_schema=schema,
        objective="maximize_sharpe",
        guidance="Prefer fewer positions if Sharpe similar.",
        n=2,
        history_limit=50,
        model="gpt5-mini",
    )

    # OPENAI_API_KEY is required by the code path, set a dummy
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    res = opt.suggest(req)
    assert isinstance(res.suggestions, list)
    assert len(res.suggestions) == 2
    assert res.suggestions[0]["max_positions"] in (1, 2, 3, 4, 5, 6, 7, 8, 9, 10)
    assert res.suggestions[0]["position_sizing_method"] in ("equal_weight", "return_weighted")
