from __future__ import annotations

import os


def _build_checkpoint_frontier_manifest(summary: dict) -> dict:
    """Build a promotion manifest for checkpoint frontier analysis.

    Extracted from the original analyze_checkpoint_frontier.py so the test is
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
        checkpoint_label = candidate.get("checkpoint_label", winner)
        checkpoint_meta = candidate.get("checkpoint_meta", {})
    else:
        evals = summary["baseline"]["evaluations"]
        checkpoint_label = summary["baseline"].get("checkpoint_label", winner)
        checkpoint_meta = summary["baseline"].get("checkpoint_meta", {})

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
        "checkpoint_label": checkpoint_label,
        "checkpoint_meta": checkpoint_meta,
        "score_tuple": score_tuple,
        "full_eval_path": os.path.join(root, f"{winner}_full.json"),
        "recent_eval_path": os.path.join(root, f"{winner}_recent.json"),
    }
    return manifest


def _build_followup_sweep_manifest(summary: dict) -> dict:
    """Build a promotion manifest for followup sweep.

    Extracted from the original run_short1x_followup_sweep.py so the test is
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
        "candidate_config": candidate_config,
        "score_tuple": score_tuple,
        "promoted_run_dir": promoted_run_dir,
        "resume_from": summary.get("resume_from"),
    }
    return manifest


def test_checkpoint_frontier_manifest_uses_promoted_checkpoint() -> None:
    summary = {
        "root": "/tmp/pufferlib4_shortcarry_followup",
        "winner": "update_003350",
        "promote": True,
        "promotion_reason": "checkpoint frontier sweep found a better holdout checkpoint inside the current short1x run",
        "promoted_checkpoint": "/tmp/current_short1x/update_003350.pt",
        "baseline": {
            "name": "best",
            "checkpoint": "/tmp/current_short1x/best.pt",
            "checkpoint_label": "best",
            "checkpoint_meta": {"update": 3000},
            "evaluation_config": {"max_leverage": 1.0, "disable_shorts": False, "short_borrow_apr": 0.0625},
            "evaluations": {
                "full": {"summary": {"p10_total_return": 1.67, "median_total_return": 2.23}},
                "recent": {"summary": {"p10_total_return": 1.28, "median_total_return": 2.52}},
            },
        },
        "candidates": [
            {
                "name": "update_003350",
                "checkpoint": "/tmp/current_short1x/update_003350.pt",
                "checkpoint_label": "update_003350",
                "checkpoint_meta": {"update": 3350},
                "evaluation_config": {"max_leverage": 1.0, "disable_shorts": False, "short_borrow_apr": 0.0625},
                "evaluations": {
                    "full": {"summary": {"p10_total_return": 1.7, "median_total_return": 2.3}},
                    "recent": {"summary": {"p10_total_return": 1.31, "median_total_return": 2.6}},
                },
            }
        ],
    }

    manifest = _build_checkpoint_frontier_manifest(summary)

    assert manifest["winner"] == "update_003350"
    assert manifest["promote"] is True
    assert manifest["promoted_checkpoint"] == "/tmp/current_short1x/update_003350.pt"
    assert manifest["checkpoint_label"] == "update_003350"
    assert manifest["checkpoint_meta"]["update"] == 3350
    assert manifest["score_tuple"] == [1.7, 2.3, 1.31, 2.6]
    assert manifest["full_eval_path"] == "/tmp/pufferlib4_shortcarry_followup/update_003350_full.json"


def test_followup_manifest_falls_back_to_baseline() -> None:
    summary = {
        "root": "/tmp/pufferlib4_shortcarry_followup",
        "winner": "baseline_current_short1x_winner",
        "promote": False,
        "promotion_reason": "no continuation candidate beat the current short1x winner score tuple",
        "promoted_checkpoint": "/tmp/current_short1x/best.pt",
        "resume_from": "/tmp/current_short1x/best.pt",
        "baseline": {
            "name": "baseline_current_short1x_winner",
            "checkpoint": "/tmp/current_short1x/best.pt",
            "evaluation_config": {"max_leverage": 1.0, "disable_shorts": False, "short_borrow_apr": 0.0625},
            "evaluations": {
                "full": {"summary": {"p10_total_return": 1.67, "median_total_return": 2.23}},
                "recent": {"summary": {"p10_total_return": 1.28, "median_total_return": 2.52}},
            },
        },
        "candidates": [],
    }

    manifest = _build_followup_sweep_manifest(summary)

    assert manifest["winner"] == "baseline_current_short1x_winner"
    assert manifest["promote"] is False
    assert manifest["candidate_config"] is None
    assert manifest["score_tuple"] == [1.67, 2.23, 1.28, 2.52]
    assert manifest["promoted_run_dir"] == "/tmp/current_short1x"
    assert manifest["resume_from"] == "/tmp/current_short1x/best.pt"
