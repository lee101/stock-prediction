"""Tests for eval_qwen_vs_gemini evaluation logic."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "rl-trainingbinance"))
from eval_qwen_vs_gemini import (
    compute_metrics,
    direction_agrees,
    extract_ground_truth,
    extract_user_prompt,
    extract_system_prompt,
    format_report,
    load_val_examples,
    parse_prediction,
)


def _make_example(action="long", confidence=0.8, entry=100.0, stop=99.0, target=102.0, hold_hours=3):
    plan = {
        "action": action, "confidence": confidence,
        "entry": entry, "stop": stop, "target": target,
        "hold_hours": hold_hours, "reasoning": "test",
    }
    return {
        "messages": [
            {"role": "system", "content": "system prompt"},
            {"role": "user", "content": "BTCUSD | price=100.0 ret1h=0.5%"},
            {"role": "assistant", "content": json.dumps(plan)},
        ]
    }


@pytest.mark.unit
class TestParseHelpers:
    def test_extract_user_prompt(self):
        ex = _make_example()
        assert "BTCUSD" in extract_user_prompt(ex)

    def test_extract_system_prompt(self):
        ex = _make_example()
        assert extract_system_prompt(ex) == "system prompt"

    def test_extract_ground_truth(self):
        ex = _make_example(action="short", confidence=0.7)
        gt = extract_ground_truth(ex)
        assert gt["action"] == "short"
        assert gt["confidence"] == 0.7

    def test_extract_ground_truth_invalid_json(self):
        ex = {"messages": [{"role": "assistant", "content": "not json"}]}
        assert extract_ground_truth(ex) is None

    def test_extract_ground_truth_missing(self):
        ex = {"messages": [{"role": "user", "content": "hi"}]}
        assert extract_ground_truth(ex) is None


@pytest.mark.unit
class TestParsePrediction:
    def test_valid_json(self):
        text = '{"action":"long","confidence":0.8,"entry":100,"stop":99,"target":102,"hold_hours":3,"reasoning":"test"}'
        p = parse_prediction(text)
        assert p["action"] == "long"
        assert p["confidence"] == 0.8

    def test_json_with_whitespace(self):
        text = '  {"action":"short","confidence":0.5}  '
        p = parse_prediction(text)
        assert p["action"] == "short"

    def test_json_embedded_in_text(self):
        text = 'Here is my plan: {"action":"flat","confidence":0.6} end'
        p = parse_prediction(text)
        assert p["action"] == "flat"

    def test_invalid_returns_none(self):
        assert parse_prediction("no json here") is None

    def test_empty_string(self):
        assert parse_prediction("") is None


@pytest.mark.unit
class TestDirectionAgrees:
    def test_same(self):
        assert direction_agrees({"action": "long"}, {"action": "long"})

    def test_different(self):
        assert not direction_agrees({"action": "long"}, {"action": "short"})

    def test_case_insensitive(self):
        assert direction_agrees({"action": "LONG"}, {"action": "long"})


@pytest.mark.unit
class TestComputeMetrics:
    def test_perfect_predictions(self):
        truths = [
            {"action": "long", "confidence": 0.8, "entry": 100, "stop": 99, "target": 102, "hold_hours": 3},
            {"action": "short", "confidence": 0.9, "entry": 200, "stop": 201, "target": 198, "hold_hours": 6},
        ]
        preds = [
            {"action": "long", "confidence": 0.8, "entry": 100, "stop": 99, "target": 102, "hold_hours": 3},
            {"action": "short", "confidence": 0.9, "entry": 200, "stop": 201, "target": 198, "hold_hours": 6},
        ]
        m = compute_metrics(preds, truths)
        assert m["n_total"] == 2
        assert m["n_valid"] == 2
        assert m["direction_agreement"] == 1.0
        assert m["entry_deviation_pct"] == 0.0
        assert m["hold_hours_mae"] == 0.0

    def test_all_wrong_direction(self):
        truths = [{"action": "long", "confidence": 0.8, "entry": 100, "stop": 99, "target": 102, "hold_hours": 3}]
        preds = [{"action": "short", "confidence": 0.8, "entry": 100, "stop": 99, "target": 102, "hold_hours": 3}]
        m = compute_metrics(preds, truths)
        assert m["direction_agreement"] == 0.0

    def test_none_predictions(self):
        truths = [{"action": "long", "confidence": 0.8, "entry": 100, "stop": 99, "target": 102, "hold_hours": 3}] * 3
        preds = [None, None, None]
        m = compute_metrics(preds, truths)
        assert m["n_parsed"] == 0
        assert m["n_valid"] == 0
        assert m["direction_agreement"] == 0.0

    def test_mixed_predictions(self):
        truths = [
            {"action": "long", "confidence": 0.8, "entry": 100, "stop": 99, "target": 102, "hold_hours": 3},
            {"action": "short", "confidence": 0.6, "entry": 200, "stop": 201, "target": 198, "hold_hours": 6},
            {"action": "flat", "confidence": 0.5, "entry": 150, "stop": 0, "target": 0, "hold_hours": 0},
        ]
        preds = [
            {"action": "long", "confidence": 0.9, "entry": 101, "stop": 98, "target": 103, "hold_hours": 3},
            None,
            {"action": "short", "confidence": 0.4, "entry": 150, "stop": 151, "target": 148, "hold_hours": 3},
        ]
        m = compute_metrics(preds, truths)
        assert m["n_total"] == 3
        assert m["n_parsed"] == 2
        assert m["n_valid"] == 2
        assert m["direction_agreement"] == 0.5  # 1 match out of 2 valid

    def test_confidence_correlation(self):
        truths = [{"action": "long", "confidence": c, "entry": 100, "stop": 99, "target": 102, "hold_hours": 3}
                  for c in [0.5, 0.6, 0.7, 0.8, 0.9]]
        preds = [{"action": "long", "confidence": c, "entry": 100, "stop": 99, "target": 102, "hold_hours": 3}
                 for c in [0.5, 0.6, 0.7, 0.8, 0.9]]
        m = compute_metrics(preds, truths)
        assert m["confidence_correlation"] == pytest.approx(1.0, abs=1e-6)

    def test_entry_deviation(self):
        truths = [{"action": "long", "confidence": 0.8, "entry": 100, "stop": 99, "target": 102, "hold_hours": 3}]
        preds = [{"action": "long", "confidence": 0.8, "entry": 101, "stop": 99, "target": 102, "hold_hours": 3}]
        m = compute_metrics(preds, truths)
        assert m["entry_deviation_pct"] == pytest.approx(1.0, abs=0.01)

    def test_action_distribution(self):
        truths = [
            {"action": "long", "confidence": 0.8, "entry": 100, "stop": 99, "target": 102, "hold_hours": 3},
            {"action": "long", "confidence": 0.8, "entry": 100, "stop": 99, "target": 102, "hold_hours": 3},
            {"action": "short", "confidence": 0.8, "entry": 100, "stop": 101, "target": 98, "hold_hours": 3},
        ]
        preds = [
            {"action": "long", "confidence": 0.8, "entry": 100, "stop": 99, "target": 102, "hold_hours": 3},
            {"action": "short", "confidence": 0.8, "entry": 100, "stop": 101, "target": 98, "hold_hours": 3},
            {"action": "flat", "confidence": 0.5, "entry": 100, "stop": 0, "target": 0, "hold_hours": 0},
        ]
        m = compute_metrics(preds, truths)
        assert m["action_distribution_pred"] == {"long": 1, "short": 1, "flat": 1}
        assert m["action_distribution_truth"] == {"long": 2, "short": 1}

    def test_empty(self):
        m = compute_metrics([], [])
        assert m["n_total"] == 0
        assert m["direction_agreement"] == 0.0


@pytest.mark.unit
class TestLoadValExamples:
    def test_load_with_limit(self, tmp_path):
        p = tmp_path / "val.jsonl"
        lines = [json.dumps(_make_example()) for _ in range(10)]
        p.write_text("\n".join(lines))
        examples = load_val_examples(p, limit=3)
        assert len(examples) == 3

    def test_load_all(self, tmp_path):
        p = tmp_path / "val.jsonl"
        lines = [json.dumps(_make_example()) for _ in range(5)]
        p.write_text("\n".join(lines))
        examples = load_val_examples(p)
        assert len(examples) == 5


@pytest.mark.unit
class TestFormatReport:
    def test_report_contains_key_info(self):
        metrics = {
            "n_total": 100, "n_parsed": 95, "n_valid": 90,
            "parse_rate": 0.95, "valid_rate": 0.90,
            "direction_agreement": 0.75,
            "confidence_correlation": 0.82,
            "confidence_mae": 0.05,
            "entry_deviation_pct": 0.1,
            "stop_deviation_pct": 0.2,
            "target_deviation_pct": 0.3,
            "hold_hours_mae": 0.5,
            "action_distribution_pred": {"long": 40, "short": 30, "flat": 20},
            "action_distribution_truth": {"long": 35, "short": 35, "flat": 30},
        }
        report = format_report(metrics)
        assert "75.0%" in report
        assert "0.820" in report
        assert "100" in report
