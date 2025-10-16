"""
Hyperparameter optimization helper using LLM structured outputs.

- Logs hyperparameter trials and outcomes as JSONL.
- Generates next suggestions with OpenAI `gpt-5-mini` using JSON schema.

See `hyperparamopt/README.md` and `hyperparamopt/runner.py` for usage.
"""

from .storage import RunLog, RunRecord
from .optimizer import StructuredOpenAIOptimizer, SuggestionRequest, SuggestionResponse

__all__ = [
    "RunLog",
    "RunRecord",
    "StructuredOpenAIOptimizer",
    "SuggestionRequest",
    "SuggestionResponse",
]

