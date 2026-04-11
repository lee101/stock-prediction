"""Trainer selection guidance for trading-focused TRL experiments."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class TrainerRecommendation:
    trainer_type: str
    use_vllm: bool
    vllm_mode: str
    rationale: str
    notes: tuple[str, ...]


def recommend_trainer() -> TrainerRecommendation:
    """Return the default TRL trainer recommendation for this repo."""
    return TrainerRecommendation(
        trainer_type="grpo",
        use_vllm=True,
        vllm_mode="colocate",
        rationale=(
            "Use GRPO first because the repo already has an online scalar reward from the Binance market "
            "simulator; we do not have pairwise preference labels, and we want direct optimization for "
            "Sortino, return, and drawdown-aware reward shaping."
        ),
        notes=(
            "Warm-start with SFT on valid JSON plans before GRPO if format drift is high.",
            "Prefer colocated vLLM on a single strong GPU to avoid REST/server overhead during exploration.",
            "Use DPO or OnlineDPO only after collecting preference data from replay or human ranking.",
        ),
    )
