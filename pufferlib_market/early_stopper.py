"""Polynomial-fit early stopper for RL training trials."""
import json
import math
from pathlib import Path
import numpy as np


def combined_score(
    val_return: float | None,
    val_sortino: float | None,
    val_wr: float | None = None,
) -> float | None:
    """Weighted combination of val metrics. Equal weight between sortino and return."""
    parts, weights = [], []
    if val_sortino is not None:
        parts.append(val_sortino); weights.append(0.5)
    if val_return is not None:
        parts.append(val_return); weights.append(0.5)
    if not parts:
        return None
    total = sum(weights)
    return sum(p * w for p, w in zip(parts, weights)) / total


class PolynomialEarlyStopper:
    """Track eval snapshots and project polynomial trajectory for pruning."""

    def __init__(self) -> None:
        self.observations: list[tuple[float, float]] = []

    def add_observation(self, progress: float, score: float) -> None:
        self.observations.append((progress, score))

    def projected_final(self) -> float | None:
        if len(self.observations) < 2:
            return None
        xs = [o[0] for o in self.observations]
        ys = [o[1] for o in self.observations]
        degree = min(len(self.observations) - 1, 2)
        try:
            coeffs = np.polyfit(xs, ys, degree)
            return float(np.polyval(coeffs, 1.0))
        except Exception:
            return None

    def should_prune(
        self,
        best_known: float,
        tolerance: float = 0.75,
        min_obs: int = 2,
    ) -> tuple[bool, float | None]:
        if len(self.observations) < min_obs or best_known <= -1e6:
            return False, None
        proj = self.projected_final()
        if proj is None:
            return False, None
        return proj < best_known * tolerance, proj


class BestKnownTracker:
    """Persist per-track best combined scores across autoresearch runs."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._data: dict = {}
        if self.path.exists():
            try:
                self._data = json.loads(self.path.read_text())
            except Exception:
                pass

    def get_best(self, track: str) -> float:
        return float(self._data.get(track, {}).get("score", -math.inf))

    def update(self, track: str, score: float, description: str = "") -> bool:
        if score <= self.get_best(track):
            return False
        self._data[track] = {"score": score, "description": description}
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self._data, indent=2))
        tmp.replace(self.path)
        return True

    def all_bests(self) -> dict[str, float]:
        return {k: float(v.get("score", -math.inf)) for k, v in self._data.items()}
