from __future__ import annotations

import json
import os
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


DEFAULT_LOG_DIR = Path("hyperparamopt/logs")
DEFAULT_LOG_DIR.mkdir(parents=True, exist_ok=True)
DEFAULT_LOG_FILE = DEFAULT_LOG_DIR / "runs.jsonl"


@dataclass
class RunRecord:
    """A single hyperparameter trial record.

    - params: the hyperparameters used in the trial
    - metrics: any metrics captured during evaluation
    - score: the scalar objective used for ranking/selection
    - source: optional tag (e.g., "manual", "llm", "grid")
    - suggestion_context: optional object describing how params were chosen
    """

    id: str
    timestamp: str
    params: Dict[str, Any]
    metrics: Dict[str, Any]
    score: float
    objective: str
    source: str = "manual"
    suggestion_context: Optional[Dict[str, Any]] = None

    @staticmethod
    def new(
        params: Dict[str, Any],
        metrics: Dict[str, Any],
        score: float,
        objective: str,
        source: str = "manual",
        suggestion_context: Optional[Dict[str, Any]] = None,
    ) -> "RunRecord":
        now = datetime.utcnow().isoformat()
        rid = f"{now.replace(':', '').replace('-', '').replace('.', '')}"
        return RunRecord(
            id=rid,
            timestamp=now,
            params=params,
            metrics=metrics,
            score=score,
            objective=objective,
            source=source,
            suggestion_context=suggestion_context,
        )


class RunLog:
    """Append-only JSONL run log with simple query helpers."""

    def __init__(self, path: os.PathLike | str = DEFAULT_LOG_FILE):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            self.path.touch()

    def append(self, record: RunRecord) -> None:
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(asdict(record), ensure_ascii=False) + "\n")

    def list(self, limit: Optional[int] = None) -> List[RunRecord]:
        out: List[RunRecord] = []
        with self.path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    out.append(RunRecord(**obj))
                except Exception:
                    # Skip bad lines, keep log resilient
                    continue
        if limit is not None:
            return out[-limit:]
        return out

    def best(self, objective: str, maximize: bool = True) -> Optional[RunRecord]:
        runs = [r for r in self.list() if r.objective == objective]
        if not runs:
            return None
        key = (lambda r: r.score)
        return max(runs, key=key) if maximize else min(runs, key=key)

    def to_prompt_summaries(
        self,
        objective: Optional[str] = None,
        max_items: int = 100,
        max_chars: int = 16000,
    ) -> str:
        """Compact textual summary for LLM context.

        Truncates to last `max_items` and `max_chars`.
        """
        runs = self.list()
        if objective is not None:
            runs = [r for r in runs if r.objective == objective]
        runs = runs[-max_items:]

        lines: List[str] = []
        for r in runs:
            params_str = json.dumps(r.params, sort_keys=True)
            metrics_str = json.dumps(r.metrics, sort_keys=True)
            lines.append(
                f"- id={r.id} score={r.score} source={r.source} params={params_str} metrics={metrics_str}"
            )

        text = "\n".join(lines)
        if len(text) > max_chars:
            text = text[-max_chars:]
        return text

