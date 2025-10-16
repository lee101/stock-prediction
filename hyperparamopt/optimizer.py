from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from .storage import RunLog


@dataclass
class SuggestionRequest:
    """Inputs for LLM suggestion.

    - hyperparam_schema: JSON Schema dict describing one suggestion object.
    - objective: description like "maximize sharpe_ratio" or "minimize loss".
    - guidance: optional natural language requirements or constraints.
    - n: number of suggestions to return.
    - history_limit: max prior runs to include in context.
    - model: OpenAI model to use (default gpt-5-mini).
    """

    hyperparam_schema: Dict[str, Any]
    objective: str
    guidance: Optional[str] = None
    n: int = 1
    history_limit: int = 100
    model: str = "gpt5-mini"


@dataclass
class SuggestionResponse:
    suggestions: List[Dict[str, Any]]
    raw: Dict[str, Any]


class StructuredOpenAIOptimizer:
    """Uses OpenAI structured outputs to propose the next hyperparameters."""

    def __init__(self, run_log: Optional[RunLog] = None):
        self.run_log = run_log or RunLog()

    def suggest(self, req: SuggestionRequest) -> SuggestionResponse:
        schema_one = req.hyperparam_schema

        # Build a response schema that returns an object with a list of suggestions
        suggestions_schema = {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "suggestions": {
                    "type": "array",
                    "minItems": req.n,
                    "maxItems": req.n,
                    "items": schema_one,
                }
            },
            "required": ["suggestions"],
        }

        history_text = self.run_log.to_prompt_summaries(objective=req.objective, max_items=req.history_limit)

        system_msg = (
            "You are a hyperparameter search strategist. "
            "Study the history of trials and propose the next candidates that optimize the stated objective. "
            "Adhere strictly to the provided JSON schema."
        )

        user_msg = (
            f"Objective: {req.objective}\n\n"
            + (f"Guidance: {req.guidance}\n\n" if req.guidance else "")
            + ("Recent trials:\n" + history_text if history_text else "No prior trials available.")
        )

        # Defer import so the rest of the package works without openai installed
        try:
            from openai import OpenAI  # type: ignore
        except Exception as e:
            raise RuntimeError(
                "openai package not available. Add `openai>=1.0.0` to requirements and set OPENAI_API_KEY."
            ) from e

        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("Missing OPENAI_API_KEY environment variable.")

        client = OpenAI(api_key=api_key)

        response = client.responses.create(
            model=req.model,
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg},
            ],
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "HyperparamSuggestions",
                    "schema": suggestions_schema,
                    "strict": True,
                },
            },
        )

        # Extract JSON content
        try:
            content = response.output[0].content[0].text  # SDK format: text contains JSON
            data = json.loads(content)
        except Exception:
            # Fall back to helper if SDK surface changes
            try:
                data = json.loads(response.output_text)
            except Exception as e:
                raise RuntimeError(f"Failed to parse structured response: {e}")

        suggestions = data.get("suggestions", [])
        if not isinstance(suggestions, list):
            raise RuntimeError("Structured response missing 'suggestions' list.")

        return SuggestionResponse(suggestions=suggestions, raw=data)
