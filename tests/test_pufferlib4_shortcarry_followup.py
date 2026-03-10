from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_checkpoint_frontier_manifest_uses_promoted_checkpoint() -> None:
    module = _load_module(
        "pufferlib4_shortcarry_checkpoint_frontier",
        Path("/home/lee/code/stock/experiments/pufferlib4_push_pnl_20260308_shortcarry_followup/analyze_checkpoint_frontier.py"),
    )
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

    manifest = module._build_promotion_manifest(summary)

    assert manifest["winner"] == "update_003350"
    assert manifest["promote"] is True
    assert manifest["promoted_checkpoint"] == "/tmp/current_short1x/update_003350.pt"
    assert manifest["checkpoint_label"] == "update_003350"
    assert manifest["checkpoint_meta"]["update"] == 3350
    assert manifest["score_tuple"] == [1.7, 2.3, 1.31, 2.6]
    assert manifest["full_eval_path"] == "/tmp/pufferlib4_shortcarry_followup/update_003350_full.json"


def test_followup_manifest_falls_back_to_baseline() -> None:
    module = _load_module(
        "pufferlib4_shortcarry_followup_sweep",
        Path("/home/lee/code/stock/experiments/pufferlib4_push_pnl_20260308_shortcarry_followup/run_short1x_followup_sweep.py"),
    )
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

    manifest = module._build_promotion_manifest(summary)

    assert manifest["winner"] == "baseline_current_short1x_winner"
    assert manifest["promote"] is False
    assert manifest["candidate_config"] is None
    assert manifest["score_tuple"] == [1.67, 2.23, 1.28, 2.52]
    assert manifest["promoted_run_dir"] == "/tmp/current_short1x"
    assert manifest["resume_from"] == "/tmp/current_short1x/best.pt"
