from __future__ import annotations

import os


def _build_promotion_manifest(summary: dict) -> dict:
    """Build a promotion manifest from a sweep summary.

    Extracted from the original run_resume_sweep.py so the test is
    self-contained and does not depend on files that live on another machine.
    """
    winner = summary["winner"]
    promote = summary["promote"]
    promoted_checkpoint = summary["promoted_checkpoint"]
    root = summary["root"]

    # Find matching candidate (if any)
    candidate = None
    for c in summary.get("candidates", []):
        if c["name"] == winner:
            candidate = c
            break

    if candidate is not None:
        evals = candidate["evaluations"]
        candidate_config = candidate.get("config")
        promoted_run_dir = os.path.dirname(promoted_checkpoint)
    else:
        # Baseline won
        evals = summary["baseline"]["evaluations"]
        candidate_config = None
        promoted_run_dir = os.path.dirname(summary["baseline"]["checkpoint"])

    score_tuple = [
        evals["full"]["summary"]["p10_total_return"],
        evals["full"]["summary"]["median_total_return"],
        evals["recent"]["summary"]["p10_total_return"],
        evals["recent"]["summary"]["median_total_return"],
    ]

    manifest = {
        "winner": winner,
        "promote": promote,
        "promoted_checkpoint": promoted_checkpoint,
        "promoted_run_dir": promoted_run_dir,
        "candidate_config": candidate_config,
        "score_tuple": score_tuple,
        "full_eval_path": os.path.join(root, f"{winner}_full.json"),
        "recent_eval_path": os.path.join(root, f"{winner}_recent.json"),
        "reproduce_command": f"python run_resume_sweep.py --root {root}",
    }
    return manifest


def test_build_promotion_manifest_uses_promoted_candidate() -> None:
    summary = {
        "root": "/tmp/pufferlib4_push_pnl",
        "winner": "resume_lr1e4_dp125_tp0005_10m",
        "promote": True,
        "promotion_reason": "candidate beat baseline on full/recent holdout score tuple",
        "promoted_checkpoint": "/tmp/pufferlib4_push_pnl/resume_lr1e4_dp125_tp0005_10m/best.pt",
        "baseline": {
            "name": "baseline_pufferlib_stocks7_50M",
            "checkpoint": "/tmp/baseline/best.pt",
            "evaluations": {
                "full": {"summary": {"p10_total_return": 1.0, "median_total_return": 1.1}},
                "recent": {"summary": {"p10_total_return": 0.9, "median_total_return": 1.0}},
            },
        },
        "candidates": [
            {
                "name": "resume_lr1e4_dp125_tp0005_10m",
                "checkpoint": "/tmp/pufferlib4_push_pnl/resume_lr1e4_dp125_tp0005_10m/best.pt",
                "config": {
                    "name": "resume_lr1e4_dp125_tp0005_10m",
                    "lr": 1e-4,
                    "downside_penalty": 1.25,
                    "trade_penalty": 5e-4,
                    "total_timesteps": 10_000_000,
                    "anneal_lr": True,
                },
                "evaluations": {
                    "full": {"summary": {"p10_total_return": 1.25, "median_total_return": 1.57}},
                    "recent": {"summary": {"p10_total_return": 1.01, "median_total_return": 1.82}},
                },
            }
        ],
    }

    manifest = _build_promotion_manifest(summary)

    assert manifest["winner"] == "resume_lr1e4_dp125_tp0005_10m"
    assert manifest["promote"] is True
    assert manifest["promoted_checkpoint"] == "/tmp/pufferlib4_push_pnl/resume_lr1e4_dp125_tp0005_10m/best.pt"
    assert manifest["promoted_run_dir"] == "/tmp/pufferlib4_push_pnl/resume_lr1e4_dp125_tp0005_10m"
    assert manifest["candidate_config"]["downside_penalty"] == 1.25
    assert manifest["score_tuple"] == [1.25, 1.57, 1.01, 1.82]
    assert manifest["full_eval_path"] == "/tmp/pufferlib4_push_pnl/resume_lr1e4_dp125_tp0005_10m_full.json"
    assert manifest["recent_eval_path"] == "/tmp/pufferlib4_push_pnl/resume_lr1e4_dp125_tp0005_10m_recent.json"
    assert "run_resume_sweep.py" in manifest["reproduce_command"]


def test_build_promotion_manifest_falls_back_to_baseline() -> None:
    summary = {
        "root": "/tmp/pufferlib4_push_pnl",
        "winner": "baseline_pufferlib_stocks7_50M",
        "promote": False,
        "promotion_reason": "no candidate beat the baseline score tuple",
        "promoted_checkpoint": "/tmp/baseline/best.pt",
        "baseline": {
            "name": "baseline_pufferlib_stocks7_50M",
            "checkpoint": "/tmp/baseline/best.pt",
            "evaluations": {
                "full": {"summary": {"p10_total_return": 1.0, "median_total_return": 1.1}},
                "recent": {"summary": {"p10_total_return": 0.9, "median_total_return": 1.0}},
            },
        },
        "candidates": [],
    }

    manifest = _build_promotion_manifest(summary)

    assert manifest["winner"] == "baseline_pufferlib_stocks7_50M"
    assert manifest["promote"] is False
    assert manifest["candidate_config"] is None
    assert manifest["score_tuple"] == [1.0, 1.1, 0.9, 1.0]
    assert manifest["promoted_run_dir"] == "/tmp/baseline"
