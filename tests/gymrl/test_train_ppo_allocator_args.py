from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

from gymrl.train_ppo_allocator import parse_args


def _parse_args(argv: list[str]):
    with patch.object(sys, "argv", ["train_ppo_allocator.py", *argv]):
        return parse_args()


def test_parse_args_accepts_behaviour_dataset_path(tmp_path: Path) -> None:
    dataset_path = tmp_path / "behaviour.npz"
    args = _parse_args(["--behaviour-dataset", str(dataset_path)])

    assert args.behaviour_dataset == dataset_path
    assert args.n_epochs == 10
    assert args.target_kl is None
    assert args.clip_range_vf is None
    assert args.vf_coef == 0.5
    assert args.max_grad_norm == 0.5


def test_parse_args_accepts_ppo_stability_overrides() -> None:
    args = _parse_args(
        [
            "--n-epochs",
            "12",
            "--clip-range-vf",
            "0.15",
            "--vf-coef",
            "0.8",
            "--max-grad-norm",
            "0.35",
            "--target-kl",
            "0.012",
        ]
    )

    assert args.n_epochs == 12
    assert args.clip_range_vf == 0.15
    assert args.vf_coef == 0.8
    assert args.max_grad_norm == 0.35
    assert args.target_kl == 0.012


def test_parse_args_allows_none_for_optional_kl_fields() -> None:
    args = _parse_args(["--clip-range-vf", "none", "--target-kl", "none"])

    assert args.clip_range_vf is None
    assert args.target_kl is None
