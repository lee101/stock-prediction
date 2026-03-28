"""Polynomial early-stopping for RL autoresearch trials.

Tracks mid-training marketsim val metrics and prunes trials whose
projected final metric won't reach a tolerance threshold of the best
known result.
"""

from __future__ import annotations

import json
import os
from math import inf
from pathlib import Path
from typing import Optional

import numpy as np


def combined_score(
    val_return: Optional[float],
    val_sortino: Optional[float],
    val_wr: Optional[float] = None,
) -> Optional[float]:
    """Combine return and sortino with equal weight (0.5 each).

    val_wr is accepted for API completeness but not included in scoring.
    Skips None inputs and re-normalises remaining weights.
    Returns None if all inputs are None.
    """
    components: list[float] = []
    weights: list[float] = []

    if val_return is not None:
        components.append(val_return)
        weights.append(0.5)
    if val_sortino is not None:
        components.append(val_sortino)
        weights.append(0.5)

    if not components:
        return None

    total_weight = sum(weights)
    return sum(c * w for c, w in zip(components, weights)) / total_weight


class PolynomialEarlyStopper:
    """Fits a polynomial to (progress, score) observations and prunes."""

    def __init__(self) -> None:
        self.observations: list[tuple[float, float]] = []

    def add_observation(self, progress: float, score: float) -> None:
        self.observations.append((progress, score))

    def projected_final(self) -> Optional[float]:
        """Fit polynomial and return projected value at progress=1.0.

        Returns None if insufficient data or fitting fails.
        """
        if len(self.observations) < 2:
            return None
        xs = np.array([o[0] for o in self.observations], dtype=float)
        ys = np.array([o[1] for o in self.observations], dtype=float)
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
    ) -> tuple[bool, Optional[float]]:
        """Decide whether to prune this trial.

        Returns (should_prune, projected_score).
        Returns (False, None) when there is insufficient data, no meaningful
        best_known, or polynomial fitting fails.
        """
        if len(self.observations) < min_obs:
            return False, None

        # Never prune during cold start — no meaningful reference yet.
        if best_known <= -1e6:
            return False, None

        proj = self.projected_final()
        if proj is None:
            return False, None

        threshold = best_known * tolerance
        if proj < threshold:
            return True, proj
        return False, proj


class HoldCashDetector:
    """Detects hold-cash (no-trade) policy from training stdout lines.

    Parses lines like:
        [  42/1000] step=   32,768  sps=250000  ret=+0.0000  ...  trades=0  wr=0.50  ...

    If ``trades=0`` appears for ``patience`` consecutive log lines the
    policy is almost certainly stuck in the hold-cash attractor and the
    trial should be killed immediately to save GPU time.
    """

    def __init__(self, patience: int = 6) -> None:
        self.patience = patience
        self._consecutive_zero_trades: int = 0
        self._total_lines_seen: int = 0

    def update(self, line: str) -> bool:
        """Feed one stdout line.  Returns True if hold-cash is detected."""
        if "trades=" not in line:
            return False
        self._total_lines_seen += 1
        try:
            trades_str = line.split("trades=")[1].split()[0]
            trades = float(trades_str)
        except (IndexError, ValueError):
            return False
        if trades == 0.0:
            self._consecutive_zero_trades += 1
        else:
            self._consecutive_zero_trades = 0
        return self._consecutive_zero_trades >= self.patience

    @property
    def consecutive_zero_trades(self) -> int:
        return self._consecutive_zero_trades


class OverfitDetector:
    """Detect obvious train/validation divergence from existing probe metrics.

    This is intentionally cheap: it only uses the latest in-training log line and the
    quick validation probe already run by autoresearch.
    """

    def __init__(
        self,
        *,
        min_progress: float = 0.5,
        min_train_combined: float = 1.0,
        max_val_combined: float = 0.25,
        min_gap: float = 1.0,
    ) -> None:
        self.min_progress = float(min_progress)
        self.min_train_combined = float(min_train_combined)
        self.max_val_combined = float(max_val_combined)
        self.min_gap = float(min_gap)
        self.latest_train_return: float | None = None
        self.latest_train_sortino: float | None = None
        self.latest_train_wr: float | None = None

    def update(self, line: str) -> bool:
        if "ret=" not in line:
            return False

        updated = False
        try:
            for part in line.split():
                if part.startswith("ret="):
                    self.latest_train_return = float(part.split("=")[1])
                    updated = True
                elif part.startswith("sortino="):
                    self.latest_train_sortino = float(part.split("=")[1])
                    updated = True
                elif part.startswith("wr="):
                    self.latest_train_wr = float(part.split("=")[1])
                    updated = True
        except Exception:
            return False
        return updated

    def latest_train_combined(self) -> float | None:
        return combined_score(
            self.latest_train_return,
            self.latest_train_sortino,
            self.latest_train_wr,
        )

    def should_prune(
        self,
        *,
        progress: float,
        val_return: float | None,
        val_sortino: float | None,
        val_wr: float | None = None,
    ) -> tuple[bool, dict[str, float | None]]:
        train_combined = self.latest_train_combined()
        val_combined = combined_score(val_return, val_sortino, val_wr)
        gap = None
        if train_combined is not None and val_combined is not None:
            gap = train_combined - val_combined

        metrics = {
            "train_combined": train_combined,
            "val_combined": val_combined,
            "gap": gap,
            "train_return": self.latest_train_return,
            "val_return": val_return,
        }

        if progress < self.min_progress:
            return False, metrics
        if train_combined is None or val_combined is None or gap is None:
            return False, metrics
        if train_combined < self.min_train_combined:
            return False, metrics
        if val_combined > self.max_val_combined:
            return False, metrics
        if gap < self.min_gap:
            return False, metrics
        return True, metrics


class BestKnownTracker:
    """Persists per-track best combined scores to a JSON file."""

    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)
        self._data: dict[str, dict] = {}
        try:
            with open(self._path) as f:
                self._data = json.load(f)
        except (FileNotFoundError, OSError, ValueError):
            self._data = {}

    def get_best(self, track: str) -> float:
        return float(self._data.get(track, {}).get("score", -inf))

    def update(self, track: str, score: float, description: str = "") -> bool:
        """Update best score for track. Returns True if this is a new best."""
        if score <= self.get_best(track):
            return False
        self._data[track] = {"score": score, "description": description}
        self._atomic_write()
        return True

    def all_bests(self) -> dict[str, float]:
        return {track: float(v["score"]) for track, v in self._data.items()}

    def _atomic_write(self) -> None:
        tmp = Path(str(self._path) + ".tmp")
        with open(tmp, "w") as f:
            json.dump(self._data, f, indent=2)
        os.replace(tmp, self._path)
