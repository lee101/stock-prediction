"""LLM-powered strategic guidance for RL training."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional

from .config import LLMConfig

try:  # pragma: no cover - optional dependency guard
    from transformers import pipeline
except Exception:  # pragma: no cover - gracefully degrade
    pipeline = None


@dataclass(slots=True)
class GuidanceResult:
    prompt: str
    response: str


class StrategyLLMGuidance:
    """Generates textual insights from LLMs to steer exploration."""

    def __init__(
        self,
        config: LLMConfig,
        generator: Optional[Callable[[str], str]] = None,
    ) -> None:
        self.config = config
        self._generator = generator
        self._pipeline: Optional[Any] = None

    def _ensure_pipeline(self):
        if self._generator is not None or not self.config.enabled:
            return
        if self._pipeline is None:
            if pipeline is None:
                raise RuntimeError(
                    "transformers is required for LLM guidance but is not installed"
                )
            self._pipeline = pipeline(
                "text-generation",
                model=self.config.model_name,
                torch_dtype="auto",
            )

    def summarize(self, metrics: Dict[str, float]) -> GuidanceResult:
        prompt = self.config.strategy_summary_template.format(
            metrics="\n".join(f"{k}: {v:.6f}" for k, v in sorted(metrics.items()))
        )
        if not self.config.enabled:
            return GuidanceResult(prompt=prompt, response="LLM guidance disabled.")
        if self._generator is not None:
            response = self._generator(prompt)
            return GuidanceResult(prompt=prompt, response=response)
        self._ensure_pipeline()
        assert self._pipeline is not None  # for type checkers
        outputs = self._pipeline(
            prompt,
            max_new_tokens=self.config.max_new_tokens,
            temperature=self.config.temperature,
            top_p=self.config.top_p,
            do_sample=True,
        )
        text = outputs[0]["generated_text"] if outputs else ""
        return GuidanceResult(prompt=prompt, response=text)
